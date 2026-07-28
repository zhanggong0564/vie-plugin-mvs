import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from services.base import (
    BaseClassificationPipeline,
    BaseCtcRecognitionPipeline,
)
from services.inference import (
    InferenceRunner,
    InferenceRunnerGroup,
    OnnxRuntimeOptions,
    RunnerDefinition,
    RunnerSpec,
)
from utils import vision_logger

from .model_config import MVSModelConfig
from .models import OCRToken


class OCRBackend(Protocol):
    def infer(self, image: np.ndarray) -> list[OCRToken]:
        ...

    def decode_qr(self, image: np.ndarray) -> str | None:
        ...


class _ImageNetClassificationPipeline(BaseClassificationPipeline):
    def __init__(
        self,
        runner: InferenceRunner,
        labels: Sequence[str],
        size: tuple[int, int],
        resize_short: int | None = None,
    ) -> None:
        super().__init__(runner, labels)
        self.size = size
        self.resize_short = resize_short

    def preprocess(self, images: Sequence[np.ndarray]) -> np.ndarray:
        tensors = []
        for image in images:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.resize_short is not None:
                height, width = rgb.shape[:2]
                scale = self.resize_short / min(height, width)
                rgb = cv2.resize(
                    rgb,
                    (int(round(width * scale)), int(round(height * scale))),
                )
                crop_width, crop_height = self.size
                top = (rgb.shape[0] - crop_height) // 2
                left = (rgb.shape[1] - crop_width) // 2
                rgb = rgb[
                    top : top + crop_height,
                    left : left + crop_width,
                ]
            else:
                rgb = cv2.resize(rgb, self.size)
            tensors.append(ONNXRuntimeOCRBackend._imagenet_tensor(rgb))
        return np.stack(tensors).astype(np.float32)


class _PPocrRecognitionPipeline(BaseCtcRecognitionPipeline):
    def _target_width(self, images: Sequence[np.ndarray]) -> int:
        ratios = [image.shape[1] / float(image.shape[0]) for image in images]
        return min(3200, max(320, int(math.ceil(48 * max(ratios)))))

    def preprocess_image(
        self,
        image: np.ndarray,
        target_width: int,
    ) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ratio = image.shape[1] / float(image.shape[0])
        resized_width = min(target_width, int(math.ceil(48 * ratio)))
        resized = cv2.resize(rgb, (resized_width, 48))
        tensor = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5
        padded = np.zeros((3, 48, target_width), dtype=np.float32)
        padded[:, :, :resized_width] = tensor
        return padded


class ONNXRuntimeOCRBackend:
    """使用五个官方 ONNX 模型执行完整 PP-OCRv5 流水线。"""

    _MODEL_KEYS = (
        "doc_orientation",
        "doc_unwarping",
        "text_detection",
        "textline_orientation",
        "text_recognition",
    )
    _MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        runners: dict[str, InferenceRunner] | InferenceRunnerGroup,
        model_config: MVSModelConfig | None = None,
    ) -> None:
        self.model_config = model_config or MVSModelConfig.from_env()
        self.model_config.validate()
        missing = set(self._MODEL_KEYS) - set(runners)
        if missing:
            raise ValueError("MVS 缺少模型 runner: " + ", ".join(sorted(missing)))
        self.runner_group = (
            runners if isinstance(runners, InferenceRunnerGroup) else None
        )
        self.runners = {key: runners[key] for key in self._MODEL_KEYS}
        self.characters = self._load_characters(
            self.model_config.paths["text_recognition"] / "inference.yml"
        )
        self.doc_orientation = _ImageNetClassificationPipeline(
            self.runners["doc_orientation"],
            labels=("0", "90", "180", "270"),
            size=(224, 224),
            resize_short=256,
        )
        self.textline_orientation = _ImageNetClassificationPipeline(
            self.runners["textline_orientation"],
            labels=("0", "180"),
            size=(160, 80),
        )
        self.text_recognition = _PPocrRecognitionPipeline(
            self.runners["text_recognition"],
            characters=self.characters,
            input_height=48,
            max_width=3200,
        )
        self.qr_detector = cv2.QRCodeDetector()

    @classmethod
    def from_settings(
        cls,
        settings,
        device: str | None = None,
        model_config: MVSModelConfig | None = None,
    ) -> "ONNXRuntimeOCRBackend":
        model_config = model_config or MVSModelConfig.from_env()
        model_config.validate()
        options = cls._runtime_options(settings, device)
        runner_group = InferenceRunnerGroup(
            [
                RunnerDefinition(
                    RunnerSpec(
                        scenario="mvs",
                        onnx_path=str(
                            model_config.paths[key] / "inference.onnx"
                        ),
                        model_role=key,
                    ),
                    options,
                )
                for key in cls._MODEL_KEYS
            ]
        )
        return cls(runners=runner_group, model_config=model_config)

    @staticmethod
    def _runtime_options(settings, device: str | None) -> OnnxRuntimeOptions:
        device = device or os.getenv("MVS_OCR_DEVICE", "gpu:0")
        if device == "cpu":
            return OnnxRuntimeOptions.from_settings(
                settings,
                providers=("CPUExecutionProvider",),
                require_cuda=False,
            )
        if not device.startswith(("gpu:", "cuda:")):
            raise ValueError("MVS_OCR_DEVICE 仅支持 cpu、gpu:N 或 cuda:N")
        try:
            device_id = int(device.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("GPU 设备格式应为 gpu:N 或 cuda:N") from exc
        return OnnxRuntimeOptions.from_settings(
            settings,
            require_cuda=True,
            cuda_device_id=device_id,
        )

    @staticmethod
    def _load_characters(config_path: Path) -> list[str]:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        characters = config["PostProcess"]["character_dict"]
        return [*characters, " "]

    def infer(self, image: np.ndarray) -> list[OCRToken]:
        tokens, _ = self.infer_with_visualization(image)
        return tokens

    def infer_with_visualization(
        self,
        image: np.ndarray,
    ) -> tuple[list[OCRToken], np.ndarray]:
        oriented = self._orient_document(image)
        document = self._unwarp_document(oriented)
        polygons = self._detect_text(document)
        crops = [self._crop_text(document, polygon) for polygon in polygons]
        valid = [
            (polygon, crop)
            for polygon, crop in zip(polygons, crops)
            if crop.size and crop.shape[0] > 0 and crop.shape[1] > 0
        ]
        if not valid:
            return [], document

        polygons, crops = map(list, zip(*valid))
        crops = self._orient_textlines(crops)
        texts, scores = self._recognize_text(crops)
        tokens = [
            OCRToken(
                text=text.strip(),
                confidence=float(score),
                polygon=np.asarray(polygon, dtype=float).tolist(),
            )
            for polygon, text, score in zip(polygons, texts, scores)
            if text.strip()
        ]
        return tokens, document.copy()

    def _run(self, key: str, tensor: np.ndarray) -> np.ndarray:
        runner = self.runners[key]
        input_name = runner.input_infos[0].name
        return runner.run({input_name: tensor})[0]

    def _orient_document(self, image: np.ndarray) -> np.ndarray:
        result = self.doc_orientation.predict([image])[0]
        angle = (0, 90, 180, 270)[result.class_id]
        return self._rotate_image(image, angle)

    def _unwarp_document(self, image: np.ndarray) -> np.ndarray:
        tensor = (
            image.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
        )
        output = self._run("doc_unwarping", tensor)[0].transpose(1, 2, 0)
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)

    def _detect_text(self, image: np.ndarray) -> list[np.ndarray]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        ratio = min(1.0, 1536.0 / max(height, width))
        resize_h = max(int(round(height * ratio / 32) * 32), 32)
        resize_w = max(int(round(width * ratio / 32) * 32), 32)
        resized = cv2.resize(rgb, (resize_w, resize_h))
        tensor = self._imagenet_tensor(resized)[None, ...]
        prediction = self._run("text_detection", tensor)[0, 0]
        bitmap = (prediction > 0.3).astype(np.uint8)
        contours, _ = cv2.findContours(
            bitmap * 255,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons = []
        for contour in contours[:1000]:
            box, short_side = self._mini_box(contour)
            if short_side < 3:
                continue
            score = self._box_score(prediction, box)
            if score < 0.6:
                continue
            expanded = self._expand_box(box, 1.5)
            box, short_side = self._mini_box(expanded.reshape(-1, 1, 2))
            if short_side < 5:
                continue
            box[:, 0] = np.clip(
                np.round(box[:, 0] * width / prediction.shape[1]),
                0,
                width,
            )
            box[:, 1] = np.clip(
                np.round(box[:, 1] * height / prediction.shape[0]),
                0,
                height,
            )
            polygons.append(box.astype(np.int16))
        return self._sort_boxes(polygons)

    def _orient_textlines(
        self,
        crops: list[np.ndarray],
    ) -> list[np.ndarray]:
        predictions = self.textline_orientation.predict(crops)
        return [
            self._rotate_image(crop, 180) if pred.class_id == 1 else crop
            for crop, pred in zip(crops, predictions)
        ]

    def _recognize_text(
        self,
        crops: list[np.ndarray],
    ) -> tuple[list[str], list[float]]:
        results = self.text_recognition.predict(crops)
        return (
            [result.text for result in results],
            [result.score for result in results],
        )

    @classmethod
    def _imagenet_tensor(cls, rgb: np.ndarray) -> np.ndarray:
        normalized = (rgb.astype(np.float32) / 255.0 - cls._MEAN) / cls._STD
        return normalized.transpose(2, 0, 1).astype(np.float32)

    @staticmethod
    def _mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
        rectangle = cv2.minAreaRect(contour.astype(np.float32))
        points = sorted(cv2.boxPoints(rectangle).tolist(), key=lambda item: item[0])
        if points[1][1] > points[0][1]:
            first, fourth = 0, 1
        else:
            first, fourth = 1, 0
        if points[3][1] > points[2][1]:
            second, third = 2, 3
        else:
            second, third = 3, 2
        ordered = np.asarray(
            [points[first], points[second], points[third], points[fourth]],
            dtype=np.float32,
        )
        return ordered, min(rectangle[1])

    @staticmethod
    def _box_score(bitmap: np.ndarray, box: np.ndarray) -> float:
        height, width = bitmap.shape
        xmin = max(0, min(math.floor(box[:, 0].min()), width - 1))
        xmax = max(0, min(math.ceil(box[:, 0].max()), width - 1))
        ymin = max(0, min(math.floor(box[:, 1].min()), height - 1))
        ymax = max(0, min(math.ceil(box[:, 1].max()), height - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        local = box.copy()
        local[:, 0] -= xmin
        local[:, 1] -= ymin
        cv2.fillPoly(mask, local.reshape(1, -1, 2).astype(np.int32), 1)
        return float(
            cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
        )

    @staticmethod
    def _expand_box(box: np.ndarray, ratio: float) -> np.ndarray:
        area = abs(cv2.contourArea(box))
        perimeter = cv2.arcLength(box, True)
        if perimeter <= 0:
            return box
        distance = area * ratio / perimeter
        rectangle = cv2.minAreaRect(box.astype(np.float32))
        center, size, angle = rectangle
        expanded = (size[0] + 2 * distance, size[1] + 2 * distance)
        return cv2.boxPoints((center, expanded, angle))

    @classmethod
    def _sort_boxes(cls, polygons: Sequence[np.ndarray]) -> list[np.ndarray]:
        boxes = sorted(polygons, key=lambda box: (box[0][1], box[0][0]))
        for index in range(len(boxes) - 1):
            for cursor in range(index, -1, -1):
                if (
                    abs(boxes[cursor + 1][0][1] - boxes[cursor][0][1]) < 10
                    and boxes[cursor + 1][0][0] < boxes[cursor][0][0]
                ):
                    boxes[cursor], boxes[cursor + 1] = (
                        boxes[cursor + 1],
                        boxes[cursor],
                    )
                else:
                    break
        return boxes

    @classmethod
    def _crop_text(
        cls,
        image: np.ndarray,
        polygon: np.ndarray,
    ) -> np.ndarray:
        points, _ = cls._mini_box(np.asarray(polygon).reshape(-1, 1, 2))
        width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        if width <= 0 or height <= 0:
            return np.empty((0, 0, 3), dtype=image.dtype)
        target = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]]
        )
        matrix = cv2.getPerspectiveTransform(points.astype(np.float32), target)
        crop = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        if crop.shape[0] / float(crop.shape[1]) >= 1.5:
            crop = np.rot90(crop)
        return crop

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return image
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        raise ValueError(f"不支持的旋转角度: {angle}")

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
        if self.runner_group is not None:
            group, self.runner_group = self.runner_group, None
            self.runners = {}
            group.close()
            return
        runners, self.runners = self.runners, {}
        for runner in runners.values():
            try:
                runner.close()
            except Exception as exc:
                vision_logger.warning(f"MVS runner 关闭失败: {exc}")
