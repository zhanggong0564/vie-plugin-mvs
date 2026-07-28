import os
from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np

from .models import OCRToken
from .model_config import MVSModelConfig


class OCRBackend(Protocol):
    def infer(self, image: np.ndarray) -> list[OCRToken]:
        ...

    def decode_qr(self, image: np.ndarray) -> str | None:
        ...


class PaddleOCRv5Backend:
    """PP-OCRv5 检测、识别及文档矫正适配器。"""

    def __init__(
        self,
        device: str | None = None,
        model_config: MVSModelConfig | None = None,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "MVS 需要 paddleocr==3.7.0，请安装插件运行依赖"
            ) from exc

        model_config = model_config or MVSModelConfig.from_env()
        model_config.validate()
        kwargs = {
            "ocr_version": "PP-OCRv5",
            "doc_orientation_classify_model_name": "PP-LCNet_x1_0_doc_ori",
            "doc_unwarping_model_name": "UVDoc",
            "text_detection_model_name": "PP-OCRv5_server_det",
            "textline_orientation_model_name": "PP-LCNet_x1_0_textline_ori",
            "text_recognition_model_name": "PP-OCRv5_server_rec",
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": True,
            "use_textline_orientation": True,
            "text_det_limit_side_len": 1536,
            "text_det_limit_type": "max",
            "engine": "onnxruntime",
            "device": device or os.getenv("MVS_OCR_DEVICE", "gpu:0"),
        }
        kwargs.update(model_config.paddle_kwargs())
        self.pipeline = PaddleOCR(**kwargs)
        self.qr_detector = cv2.QRCodeDetector()

    def infer(self, image: np.ndarray) -> list[OCRToken]:
        pages = list(self.pipeline.predict(image))
        tokens = []
        for page in pages:
            payload = page.json
            result = payload.get("res", payload)
            texts: Sequence = result.get("rec_texts", ())
            scores: Sequence = result.get("rec_scores", ())
            polygons: Sequence = result.get("rec_polys", ())
            if not (len(texts) == len(scores) == len(polygons)):
                raise ValueError("PaddleOCR 返回的文字、置信度和坐标数量不一致")
            for text, score, polygon in zip(texts, scores, polygons):
                stripped = str(text).strip()
                if not stripped:
                    continue
                tokens.append(
                    OCRToken(
                        text=stripped,
                        confidence=float(score),
                        polygon=np.asarray(polygon, dtype=float).tolist(),
                    )
                )
        return tokens

    def decode_qr(self, image: np.ndarray) -> str | None:
        try:
            detected, decoded, _, _ = self.qr_detector.detectAndDecodeMulti(image)
        except cv2.error:
            detected = False
            decoded = ()
        if detected:
            values = [value.strip() for value in decoded if value.strip()]
            if values:
                return "\n".join(values)
        value, _, _ = self.qr_detector.detectAndDecode(image)
        return value.strip() or None

    def close(self) -> None:
        self.pipeline = None
