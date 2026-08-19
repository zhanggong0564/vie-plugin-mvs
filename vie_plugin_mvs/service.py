import re
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import MVSRules, MVSSettings, load_rules
from .inspectors import OCRLabelInspector
from .quality import ImageQualityChecker
from .models import (
    ImageInspection,
    InspectionStatus,
    MVSResult,
    PackingListItem,
)
from .ocr import OCRBackend
from schemas.exceptions import InvalidParamsError
from .table_parser import FixedPackingListParser


_SEQUENCE = re.compile(r"-(\d+)$")
_SINGLE_IMAGE_NAME = re.compile(
    r"^(?P<prefix>.+)-(?P<sequence>[1-9]\d*)-(?P<timestamp>\d{13})$"
)
@dataclass(frozen=True)
class ParsedImageName:
    prefix: str
    sequence: int
    timestamp: str


@dataclass(frozen=True)
class ManifestSession:
    signature: tuple
    items: tuple[PackingListItem, ...]
    touched_at: float


def parse_single_image_name(filename: str) -> ParsedImageName:
    stem = filename.rsplit(".", 1)[0]
    match = _SINGLE_IMAGE_NAME.fullmatch(stem)
    if not match:
        raise InvalidParamsError(
            "图片文件名必须为 <业务名称>-<序号>-<13位毫秒时间戳>.<扩展名>"
        )
    return ParsedImageName(
        prefix=match.group("prefix"),
        sequence=int(match.group("sequence")),
        timestamp=match.group("timestamp"),
    )


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
        session_ttl_seconds: int | None = None,
    ) -> None:
        self.rules = rules or load_rules()
        if ocr_backend is None:
            raise ValueError("MVSService 需要注入 OCRBackend")
        self.ocr_backend = ocr_backend
        self.table_parser = FixedPackingListParser(self.rules)
        self.label_inspector = label_inspector
        self.quality_checker = ImageQualityChecker(self.rules.quality)
        if session_ttl_seconds is not None and session_ttl_seconds < 1:
            raise ValueError("session_ttl_seconds 必须大于 0")
        self.session_ttl_seconds = (
            session_ttl_seconds
            if session_ttl_seconds is not None
            else MVSSettings().session_ttl_seconds
        )
        self._manifest_sessions: dict[str, ManifestSession] = {}
        self._session_lock = threading.Lock()

    def inspect_single(self, filename, image, request_params) -> dict:
        parsed = parse_single_image_name(filename)
        model_params = request_params.modelParams
        try:
            target_keys = self._target_keys(model_params.target_names)
        except ValueError as exc:
            raise InvalidParamsError(str(exc)) from exc
        max_sequence = len(target_keys) + 1
        if parsed.sequence > max_sequence:
            raise InvalidParamsError(
                f"图片序号必须在 1 到 {max_sequence} 之间"
            )
        signature = (
            model_params.product_type,
            model_params.target_names,
            model_params.guideline_coordinates,
        )
        if parsed.sequence == 1:
            return self._inspect_manifest_single(
                image,
                request_params.sn,
                model_params.target_names,
                target_keys,
                model_params.guideline_coordinates[:2],
                signature,
            )
        return self._inspect_label_single(
            image,
            request_params.sn,
            parsed.sequence,
            model_params.target_names,
            target_keys,
            model_params.guideline_coordinates[parsed.sequence],
            signature,
        )

    def _inspect_manifest_single(
        self,
        image,
        sn,
        target_names,
        target_keys,
        guidelines,
        signature,
    ) -> dict:
        manifest_items = []
        for side, guideline in zip(("左", "右"), guidelines):
            crop = self._outer_crop(image, guideline)
            quality = self.quality_checker.check(crop)
            if not quality.acceptable:
                self._remove_session(sn)
                reason = f"{side}侧装箱清单图片质量不满足要求：" + "；".join(
                    quality.reasons
                )
                return self._common_result(
                    InspectionStatus.REVIEW,
                    [
                        self._detail(name, "", 0.0, InspectionStatus.REVIEW)
                        for name in target_names
                    ],
                    reason,
                )
            tokens = self.ocr_backend.infer(crop)
            manifest_items.extend(
                self.table_parser.parse(
                    tokens,
                    image_width=crop.shape[1],
                    image_height=crop.shape[0],
                )
            )

        details = []
        verdicts = []
        for name, key in zip(target_names, target_keys):
            matches = [item for item in manifest_items if item.item_key == key]
            if not matches:
                verdict = InspectionStatus.FAIL
                item = None
            else:
                item = matches[0]
                verdict = (
                    InspectionStatus.PASS
                    if item.material_code
                    else InspectionStatus.REVIEW
                )
            verdicts.append(verdict)
            details.append(
                self._detail(
                    name,
                    item.material_code if item and item.material_code else "",
                    item.confidence if item else 0.0,
                    verdict,
                )
            )

        with self._session_lock:
            self._manifest_sessions[sn] = ManifestSession(
                signature=signature,
                items=tuple(manifest_items),
                touched_at=time.monotonic(),
            )
        overall = self._overall_verdict(verdicts)
        return self._common_result(
            overall,
            details,
            "装箱清单检测完成",
        )

    def _inspect_label_single(
        self,
        image,
        sn,
        sequence,
        target_names,
        target_keys,
        guideline,
        signature,
    ) -> dict:
        target_index = sequence - 2
        target_name = target_names[target_index]
        target_key = target_keys[target_index]
        session = self._get_session(sn)
        if session is None:
            return self._common_result(
                InspectionStatus.REVIEW,
                [self._detail(target_name, "", 0.0, InspectionStatus.REVIEW)],
                "未找到有效装箱清单，请先上传序号为 1 的清单图片",
            )
        if session.signature != signature:
            raise InvalidParamsError("同一 SN 的 MVS 检测参数与清单请求不一致")

        expected_items = [
            item for item in session.items if item.item_key == target_key
        ]
        if not expected_items:
            return self._common_result(
                InspectionStatus.REVIEW,
                [self._detail(target_name, "", 0.0, InspectionStatus.REVIEW)],
                f"装箱清单中缺少检测项目：{target_name}",
            )
        inspector = self.label_inspector or OCRLabelInspector(self.rules)
        observation = inspector.extract_guided(
            image,
            expected_item=expected_items[0],
            backend=self.ocr_backend,
            guideline=guideline,
            selected_item_key=target_key,
        )
        result = inspector.evaluate(
            list(session.items),
            observation,
            selected_item_key=target_key,
        )
        actual_code = ""
        if result.actual:
            actual_code = result.actual.get("material_code") or ""
        detail = self._detail(
            target_name,
            actual_code,
            observation.confidence,
            result.status,
        )
        return self._common_result(result.status, [detail], result.reason)

    def _get_session(self, sn: str) -> ManifestSession | None:
        now = time.monotonic()
        with self._session_lock:
            session = self._manifest_sessions.get(sn)
            if session is None:
                return None
            if now - session.touched_at > self.session_ttl_seconds:
                self._manifest_sessions.pop(sn, None)
                return None
            refreshed = ManifestSession(
                signature=session.signature,
                items=session.items,
                touched_at=now,
            )
            self._manifest_sessions[sn] = refreshed
            return refreshed

    def _remove_session(self, sn: str) -> None:
        with self._session_lock:
            self._manifest_sessions.pop(sn, None)

    @staticmethod
    def _detail(scene, name, accuracy, verdict) -> dict:
        passed = verdict is InspectionStatus.PASS
        return {
            "status": "true" if passed else "false",
            "verdict": verdict.value,
            "scene": scene,
            "coordinate": [],
            "accuracy": float(accuracy),
            "name": name,
            "color": "#20ff4f" if passed else "#FFFF00",
        }

    @staticmethod
    def _overall_verdict(verdicts) -> InspectionStatus:
        if InspectionStatus.FAIL in verdicts:
            return InspectionStatus.FAIL
        if InspectionStatus.REVIEW in verdicts:
            return InspectionStatus.REVIEW
        return InspectionStatus.PASS

    @staticmethod
    def _common_result(verdict, details, message) -> dict:
        return {
            "detailList": details,
            "status": "true" if verdict is InspectionStatus.PASS else "false",
            "verdict": verdict.value,
            "error_msg": "",
            "message": message,
        }

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
        target_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        keys = tuple(self.rules.item_key_for_name(name) for name in target_names)
        unsupported = [
            name for name, key in zip(target_names, keys) if key is None
        ]
        if unsupported:
            raise ValueError("不支持的检测项: " + "、".join(unsupported))
        if len(set(keys)) != len(keys):
            raise ValueError("target_names 检测项不能重复")
        return keys

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
