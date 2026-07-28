from dataclasses import dataclass

import cv2
import numpy as np

from .config import QualityRule


@dataclass(frozen=True)
class QualityResult:
    acceptable: bool
    reasons: tuple[str, ...]
    blur_variance: float
    brightness: float


class ImageQualityChecker:
    def __init__(self, rule: QualityRule) -> None:
        self.rule = rule

    def check(self, image: np.ndarray) -> QualityResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        reasons = []
        if blur_variance < self.rule.min_blur_variance:
            reasons.append("图片模糊")
        if brightness < self.rule.min_brightness:
            reasons.append("图片过暗")
        elif brightness > self.rule.max_brightness:
            reasons.append("图片过亮")
        return QualityResult(
            acceptable=not reasons,
            reasons=tuple(reasons),
            blur_variance=blur_variance,
            brightness=brightness,
        )
