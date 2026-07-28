import re

import numpy as np

from .config import MVSRules, load_rules
from .inspectors import OCRLabelInspector
from .quality import ImageQualityChecker
from .models import (
    ImageInspection,
    InspectionStatus,
    MVSResult,
)
from .ocr import OCRBackend
from .table_parser import FixedPackingListParser


_SEQUENCE = re.compile(r"-(\d+)$")


def order_images(
    images: list[tuple[str, np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    indexed = []
    for filename, image in images:
        stem = filename.rsplit(".", 1)[0]
        match = _SEQUENCE.search(stem)
        if not match:
            raise ValueError(f"图片文件名缺少 -序号: {filename}")
        indexed.append((int(match.group(1)), filename, image))
    indexed.sort(key=lambda item: item[0])
    if not indexed or indexed[0][0] != 1:
        raise ValueError("必须提供以 -1 命名的装箱清单图片")
    if len({item[0] for item in indexed}) != len(indexed):
        raise ValueError("图片序号不能重复")
    return [(filename, image) for _, filename, image in indexed]


class MVSService:
    def __init__(
        self,
        rules: MVSRules | None = None,
        ocr_backend: OCRBackend | None = None,
        label_inspector: OCRLabelInspector | None = None,
    ) -> None:
        self.rules = rules or load_rules()
        if ocr_backend is None:
            raise ValueError("MVSService 需要注入 OCRBackend")
        self.ocr_backend = ocr_backend
        self.table_parser = FixedPackingListParser(self.rules)
        self.label_inspector = label_inspector
        self.quality_checker = ImageQualityChecker(self.rules.quality)

    def inspect(
        self,
        images: list[tuple[str, np.ndarray]],
        selected_item_key: str | None = None,
    ) -> MVSResult:
        ordered = order_images(images)
        manifest_filename, manifest_image = ordered[0]
        manifest_quality = self.quality_checker.check(manifest_image)
        if not manifest_quality.acceptable:
            return MVSResult(
                status=InspectionStatus.REVIEW,
                manifest_filename=manifest_filename,
                message="装箱清单图片质量不满足要求："
                + "；".join(manifest_quality.reasons),
            )
        manifest_tokens = self.ocr_backend.infer(manifest_image)
        manifest_items = self.table_parser.parse(
            manifest_tokens,
            image_width=manifest_image.shape[1],
            image_height=manifest_image.shape[0],
        )
        if not manifest_items:
            return MVSResult(
                status=InspectionStatus.REVIEW,
                manifest_filename=manifest_filename,
                message="装箱清单未解析到已配置物料",
            )

        inspector_names = {
            self.rules.items[item.item_key].inspector for item in manifest_items
        }
        if selected_item_key and selected_item_key in self.rules.items:
            inspector_names = {self.rules.items[selected_item_key].inspector}
        if len(inspector_names) != 1:
            return MVSResult(
                status=InspectionStatus.REVIEW,
                manifest_filename=manifest_filename,
                manifest_items=tuple(manifest_items),
                message="当前图片需要多个不同类型检验器，无法自动选择",
            )
        inspector_name = next(iter(inspector_names))
        inspector_types = {"ocr_label": OCRLabelInspector}
        inspector_type = inspector_types.get(inspector_name)
        if inspector_type is None:
            return MVSResult(
                status=InspectionStatus.REVIEW,
                manifest_filename=manifest_filename,
                manifest_items=tuple(manifest_items),
                message=f"尚未实现检验器: {inspector_name}",
            )
        inspector = self.label_inspector or inspector_type(self.rules)

        inspections = []
        for filename, image in ordered[1:]:
            observation = inspector.extract(
                image,
                expected_item=None,
                backend=self.ocr_backend,
            )
            result = inspector.evaluate(
                manifest_items,
                observation,
                selected_item_key=selected_item_key,
            )
            inspections.append(ImageInspection(filename=filename, result=result))

        statuses = [inspection.status for inspection in inspections]
        if InspectionStatus.FAIL in statuses:
            overall = InspectionStatus.FAIL
        elif not statuses or InspectionStatus.REVIEW in statuses:
            overall = InspectionStatus.REVIEW
        else:
            overall = InspectionStatus.PASS
        return MVSResult(
            status=overall,
            manifest_filename=manifest_filename,
            manifest_items=tuple(manifest_items),
            inspections=tuple(inspections),
            message=f"完成 {len(inspections)} 张实物标签检验",
        )

    def close(self) -> None:
        close = getattr(self.ocr_backend, "close", None)
        if callable(close):
            close()
