import re
from typing import Protocol

import cv2
import numpy as np

from .config import MVSRules
from .matcher import MaterialMatcher
from .models import (
    InspectionResult,
    LabelObservation,
    OCRToken,
    PackingListItem,
)
from .ocr import OCRBackend
from .quality import ImageQualityChecker


_CODE_LIKE = re.compile(r"FQ[0-9A-ZIl]{6}", re.IGNORECASE)
_CONFUSION_MAP = str.maketrans(
    {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"}
)


class Inspector(Protocol):
    def extract(
        self,
        image: np.ndarray,
        expected_item: PackingListItem | None,
        backend: OCRBackend,
    ) -> LabelObservation:
        ...

    def evaluate(
        self,
        expected_items: list[PackingListItem],
        observation: LabelObservation,
        selected_item_key: str | None = None,
    ) -> InspectionResult:
        ...


class OCRLabelInspector:
    def __init__(
        self,
        rules: MVSRules,
        quality_checker: ImageQualityChecker | None = None,
    ) -> None:
        self.rules = rules
        self.quality_checker = quality_checker or ImageQualityChecker(rules.quality)
        self.matcher = MaterialMatcher(rules)

    def extract(
        self,
        image: np.ndarray,
        expected_item: PackingListItem | None,
        backend: OCRBackend,
    ) -> LabelObservation:
        quality = self.quality_checker.check(image)
        tokens = backend.infer(image)
        selected_item_key = expected_item.item_key if expected_item else None
        observation = self.extract_tokens(
            tokens,
            qr_text=backend.decode_qr(image),
            review_reasons=quality.reasons,
            selected_item_key=selected_item_key,
        )
        min_threshold = min(
            rule.confidence_threshold for rule in self.rules.items.values()
        )
        if tokens and (
            observation.material_code is None
            or observation.confidence < min_threshold
        ):
            crop = self._crop_and_enhance(image, tokens)
            retry_tokens = backend.infer(crop)
            observation = self.extract_tokens(
                [*tokens, *retry_tokens],
                qr_text=observation.qr_text,
                review_reasons=quality.reasons,
                multiple_labels=observation.multiple_labels,
                selected_item_key=selected_item_key,
            )
        return observation

    def extract_guided(
        self,
        image: np.ndarray,
        expected_item: PackingListItem | None,
        backend: OCRBackend,
        guideline: tuple[float, ...],
        overlap_threshold: float = 0.9,
        selected_item_key: str | None = None,
    ) -> LabelObservation:
        quality = self.quality_checker.check(image)
        tokens = self.filter_tokens(
            backend.infer(image),
            guideline,
            image.shape[1],
            image.shape[0],
            overlap_threshold,
        )
        selected_item_key = (
            selected_item_key
            or (expected_item.item_key if expected_item else None)
        )
        qr_crop = self._guideline_crop(image, guideline)
        observation = self.extract_tokens(
            tokens,
            qr_text=backend.decode_qr(qr_crop),
            review_reasons=quality.reasons,
            selected_item_key=selected_item_key,
        )
        min_threshold = min(
            rule.confidence_threshold for rule in self.rules.items.values()
        )
        if tokens and (
            observation.material_code is None
            or observation.confidence < min_threshold
        ):
            crop = self._crop_and_enhance(image, tokens)
            retry_tokens = backend.infer(crop)
            observation = self.extract_tokens(
                [*tokens, *retry_tokens],
                qr_text=observation.qr_text,
                review_reasons=quality.reasons,
                multiple_labels=observation.multiple_labels,
                selected_item_key=selected_item_key,
            )
        return observation

    def extract_tokens(
        self,
        tokens: list[OCRToken],
        qr_text: str | None = None,
        review_reasons: tuple[str, ...] = (),
        multiple_labels: bool | None = None,
        selected_item_key: str | None = None,
    ) -> LabelObservation:
        raw_texts = tuple(token.text for token in tokens)
        name_matches = []
        name_scores = []
        matched_rules = []
        candidate_rules = (
            (self.rules.items[selected_item_key],)
            if selected_item_key
            else tuple(self.rules.items.values())
        )
        for token in tokens:
            for rule in candidate_rules:
                if any(
                    alias.casefold() in token.text.casefold()
                    for alias in rule.aliases
                ):
                    name_matches.append(rule.display_name)
                    name_scores.append(token.confidence)
                    matched_rules.append(rule)
        unique_names = tuple(dict.fromkeys(name_matches))
        item_name = unique_names[0] if len(unique_names) == 1 else None

        detected_codes = []
        code_scores = []
        selected_rules = (
            tuple(dict.fromkeys(matched_rules))
            if not selected_item_key and len(set(name_matches)) == 1
            else candidate_rules
        )
        code_patterns = tuple(
            dict.fromkeys(rule.code_pattern for rule in selected_rules)
        )
        for token in tokens:
            for pattern in code_patterns:
                for match in re.finditer(pattern, token.text.upper()):
                    detected_codes.append(match.group(0))
                    code_scores.append(token.confidence)

        unique_codes = tuple(dict.fromkeys(detected_codes))
        exact_code = unique_codes[0] if len(unique_codes) == 1 else None
        if multiple_labels is None:
            multiple_labels = len(detected_codes) > len(unique_codes)
        candidates = []
        for token in tokens:
            for match in _CODE_LIKE.finditer(token.text):
                raw = match.group(0).upper()
                corrected = "FQ" + raw[2:].translate(_CONFUSION_MAP)
                if corrected != raw and re.fullmatch(r"FQ\d{6}", corrected):
                    candidates.append(corrected)

        relevant_scores = [*code_scores, *name_scores]
        confidence = (
            sum(relevant_scores) / len(relevant_scores)
            if relevant_scores
            else max((token.confidence for token in tokens), default=0.0)
        )
        unique_candidates = tuple(dict.fromkeys(candidates))
        return LabelObservation(
            item_name=item_name,
            material_code=exact_code,
            raw_texts=raw_texts,
            qr_text=qr_text,
            confidence=confidence,
            code_candidates=unique_candidates,
            detected_codes=unique_codes,
            has_ambiguous_code=bool(unique_candidates),
            multiple_labels=multiple_labels,
            review_reasons=review_reasons,
        )

    @staticmethod
    def filter_tokens(
        tokens: list[OCRToken],
        guideline: tuple[float, ...],
        image_width: int,
        image_height: int,
        overlap_threshold: float,
    ) -> list[OCRToken]:
        roi = np.asarray(
            [
                (
                    guideline[index] * image_width,
                    guideline[index + 1] * image_height,
                )
                for index in range(0, 8, 2)
            ],
            dtype=np.float32,
        )
        kept = []
        for token in tokens:
            polygon = np.asarray(token.polygon, dtype=np.float32)
            area = cv2.contourArea(polygon)
            if area <= 0:
                continue
            intersection, _ = cv2.intersectConvexConvex(polygon, roi)
            overlap_ratio = min(1.0, intersection / area)
            if intersection > 0 and overlap_ratio >= overlap_threshold:
                kept.append(token)
        return kept

    @staticmethod
    def _guideline_crop(
        image: np.ndarray,
        guideline: tuple[float, ...],
    ) -> np.ndarray:
        width = image.shape[1]
        height = image.shape[0]
        xs = guideline[::2]
        ys = guideline[1::2]
        left = max(0, int(np.floor(min(xs) * width)))
        top = max(0, int(np.floor(min(ys) * height)))
        right = min(width, int(np.ceil(max(xs) * width)))
        bottom = min(height, int(np.ceil(max(ys) * height)))
        return image[top:bottom, left:right]

    def evaluate(
        self,
        expected_items: list[PackingListItem],
        observation: LabelObservation,
        selected_item_key: str | None = None,
    ) -> InspectionResult:
        return self.matcher.evaluate(
            expected_items,
            observation,
            selected_item_key=selected_item_key,
        )

    @staticmethod
    def _crop_and_enhance(
        image: np.ndarray,
        tokens: list[OCRToken],
    ) -> np.ndarray:
        points = np.asarray(
            [point for token in tokens for point in token.polygon],
            dtype=np.float32,
        )
        x, y, width, height = cv2.boundingRect(points)
        pad_x = max(4, int(width * 0.05))
        pad_y = max(4, int(height * 0.15))
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image.shape[1], x + width + pad_x)
        bottom = min(image.shape[0], y + height + pad_y)
        crop = image[top:bottom, left:right]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            lightness
        )
        return cv2.cvtColor(
            cv2.merge((enhanced, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )
