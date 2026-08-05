import re
from dataclasses import asdict

from .config import MVSRules
from .models import (
    InspectionResult,
    InspectionStatus,
    LabelObservation,
    PackingListItem,
)


class MaterialMatcher:
    def __init__(self, rules: MVSRules) -> None:
        self.rules = rules

    def evaluate(
        self,
        expected_items: list[PackingListItem],
        observation: LabelObservation,
        selected_item_key: str | None = None,
    ) -> InspectionResult:
        actual = asdict(observation)
        if observation.review_reasons:
            return self._review("；".join(observation.review_reasons), actual)
        if observation.multiple_labels or len(observation.detected_codes) > 1:
            return self._review("一张图片中识别到多个物料编码", actual)

        qr_patterns = (
            (self.rules.items[selected_item_key].code_pattern,)
            if selected_item_key
            else tuple(
                dict.fromkeys(
                    rule.code_pattern for rule in self.rules.items.values()
                )
            )
        )
        qr_codes = tuple(
            dict.fromkeys(
                match.group(0)
                for pattern in qr_patterns
                for match in re.finditer(
                    pattern,
                    (observation.qr_text or "").upper(),
                )
            )
        )
        if len(qr_codes) > 1:
            return self._review("二维码中包含多个物料编码", actual)
        if (
            observation.material_code
            and qr_codes
            and observation.material_code != qr_codes[0]
        ):
            return self._review("二维码与 OCR 物料编码冲突", actual)

        candidates = expected_items
        if selected_item_key:
            candidates = [
                item for item in expected_items if item.item_key == selected_item_key
            ]
            if not candidates:
                return self._review("指定物料不在本次装箱清单中", actual)

        actual_code = observation.material_code or (qr_codes[0] if qr_codes else None)
        if actual_code:
            matched = [
                item for item in candidates if item.material_code == actual_code
            ]
            if len(matched) > 1:
                return self._review("物料编码匹配到多个清单项目", actual)
            if not matched:
                if any(item.material_code is None for item in candidates):
                    return self._review("装箱清单物料编码缺失或识别不确定", actual)
                return InspectionResult(
                    status=InspectionStatus.FAIL,
                    reason="实物编码与装箱清单不一致",
                    item_key=selected_item_key,
                    expected=self._expected(candidates[0]) if len(candidates) == 1 else None,
                    actual=actual,
                )
            item = matched[0]
            threshold = self.rules.items[item.item_key].confidence_threshold
            if (
                observation.confidence < threshold
                or item.confidence < threshold
            ):
                return self._review(
                    "清单或标签 OCR 置信度低于物料规则阈值",
                    actual,
                    item,
                )
            return InspectionResult(
                status=InspectionStatus.PASS,
                reason="物料编码精确匹配",
                matched_line_no=item.line_no,
                item_key=item.item_key,
                expected=self._expected(item),
                actual=actual,
            )

        if observation.has_ambiguous_code or observation.code_candidates:
            return self._review("物料编码含易混淆字符，需人工复核", actual)

        if selected_item_key and observation.item_name:
            expected_name = self.rules.items[selected_item_key].display_name
            threshold = self.rules.items[selected_item_key].confidence_threshold
            if (
                observation.item_name != expected_name
                and observation.confidence >= threshold
            ):
                return InspectionResult(
                    status=InspectionStatus.FAIL,
                    reason="实物名称与指定清单项目不一致",
                    item_key=selected_item_key,
                    expected=self._expected(candidates[0]),
                    actual=actual,
                )

        name_matched = [
            item
            for item in candidates
            if observation.item_name
            and observation.item_name == self.rules.items[item.item_key].display_name
        ]
        if len(name_matched) > 1:
            return self._review("物料名称匹配到多个清单项目", actual)
        if len(name_matched) == 1:
            rule = self.rules.items[name_matched[0].item_key]
            if "material_code" in rule.required_fields:
                return self._review("已识别物料名称，但缺少必检物料编码", actual)
        return self._review("未识别到可唯一匹配的物料", actual)

    @staticmethod
    def _expected(item: PackingListItem) -> dict:
        return {
            "item_name": item.name_cn,
            "material_code": item.material_code,
        }

    def _review(
        self,
        reason: str,
        actual: dict,
        item: PackingListItem | None = None,
    ) -> InspectionResult:
        return InspectionResult(
            status=InspectionStatus.REVIEW,
            reason=reason,
            matched_line_no=item.line_no if item else None,
            item_key=item.item_key if item else None,
            expected=self._expected(item) if item else None,
            actual=actual,
        )
