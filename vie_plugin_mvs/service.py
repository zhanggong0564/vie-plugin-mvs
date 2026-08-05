import re

import cv2
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
    if len({item[0] for item in indexed}) != len(indexed):
        raise ValueError("图片序号不能重复")
    if [item[0] for item in indexed] != [1, 2, 3, 4]:
        raise ValueError("必须提供且仅提供以 -1 至 -4 编号的四张图片")
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
        target_names: tuple[str, str, str],
        guidelines: tuple[tuple[float, ...], ...],
    ) -> MVSResult:
        ordered = order_images(images)
        target_keys = self._target_keys(target_names)
        manifest_filename, manifest_image = ordered[0]
        manifest_items = []
        for side, guideline in zip(("左", "右"), guidelines[:2]):
            crop = self._outer_crop(manifest_image, guideline)
            manifest_quality = self.quality_checker.check(crop)
            if not manifest_quality.acceptable:
                return MVSResult(
                    status=InspectionStatus.REVIEW,
                    manifest_filename=manifest_filename,
                    message=f"{side}侧装箱清单图片质量不满足要求："
                    + "；".join(manifest_quality.reasons),
                )
            tokens = self.ocr_backend.infer(crop)
            manifest_items.extend(
                self.table_parser.parse(
                    tokens,
                    image_width=crop.shape[1],
                    image_height=crop.shape[0],
                )
            )
        if not manifest_items:
            return MVSResult(
                status=InspectionStatus.REVIEW,
                manifest_filename=manifest_filename,
                message="装箱清单未解析到已配置物料",
            )

        inspector_names = {
            self.rules.items[item_key].inspector for item_key in target_keys
        }
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
        for (filename, image), item_key, guideline in zip(
            ordered[1:],
            target_keys,
            guidelines[2:],
        ):
            expected_items = [
                item for item in manifest_items if item.item_key == item_key
            ]
            expected_item = expected_items[0] if expected_items else None
            observation = inspector.extract_guided(
                image,
                expected_item=expected_item,
                backend=self.ocr_backend,
                guideline=guideline,
                selected_item_key=item_key,
            )
            result = inspector.evaluate(
                manifest_items,
                observation,
                selected_item_key=item_key,
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

    def _target_keys(
        self,
        target_names: tuple[str, str, str],
    ) -> tuple[str, str, str]:
        keys = tuple(self.rules.item_key_for_name(name) for name in target_names)
        unsupported = [
            name for name, key in zip(target_names, keys) if key is None
        ]
        if unsupported:
            raise ValueError("不支持的检测项: " + "、".join(unsupported))
        if len(set(keys)) != 3:
            raise ValueError("target_names 检测项不能重复")
        return keys  # type: ignore[return-value]

    @staticmethod
    def _outer_crop(
        image: np.ndarray,
        guideline: tuple[float, ...],
    ) -> np.ndarray:
        height, width = image.shape[:2]
        points = np.asarray(
            [
                (guideline[index] * width, guideline[index + 1] * height)
                for index in range(0, 8, 2)
            ],
            dtype=np.float32,
        )
        x, y, crop_width, crop_height = cv2.boundingRect(points)
        right = min(width, x + crop_width)
        bottom = min(height, y + crop_height)
        return image[max(0, y):bottom, max(0, x):right]

    def close(self) -> None:
        close = getattr(self.ocr_backend, "close", None)
        if callable(close):
            close()
