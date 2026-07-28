"""Stateless geometry helpers for the MVS OCR pipeline."""

import math
from collections.abc import Sequence

import cv2
import numpy as np


def mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
    rectangle = cv2.minAreaRect(contour.astype(np.float32))
    points = sorted(
        cv2.boxPoints(rectangle).tolist(),
        key=lambda item: item[0],
    )
    first, fourth = (0, 1) if points[1][1] > points[0][1] else (1, 0)
    second, third = (2, 3) if points[3][1] > points[2][1] else (3, 2)
    ordered = np.asarray(
        [points[first], points[second], points[third], points[fourth]],
        dtype=np.float32,
    )
    return ordered, min(rectangle[1])


def box_score(bitmap: np.ndarray, box: np.ndarray) -> float:
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


def expand_box(box: np.ndarray, ratio: float) -> np.ndarray:
    area = abs(cv2.contourArea(box))
    perimeter = cv2.arcLength(box, True)
    if perimeter <= 0:
        return box
    distance = area * ratio / perimeter
    center, size, angle = cv2.minAreaRect(box.astype(np.float32))
    expanded = (size[0] + 2 * distance, size[1] + 2 * distance)
    return cv2.boxPoints((center, expanded, angle))


def sort_text_boxes(
    polygons: Sequence[np.ndarray],
) -> list[np.ndarray]:
    boxes = sorted(polygons, key=lambda box: (box[0][1], box[0][0]))
    for index in range(len(boxes) - 1):
        for cursor in range(index, -1, -1):
            current = boxes[cursor]
            following = boxes[cursor + 1]
            same_row = abs(following[0][1] - current[0][1]) < 10
            if not same_row or following[0][0] >= current[0][0]:
                break
            boxes[cursor], boxes[cursor + 1] = following, current
    return boxes


def crop_text(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points, _ = mini_box(np.asarray(polygon).reshape(-1, 1, 2))
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
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if crop.shape[0] / float(crop.shape[1]) >= 1.5:
        return np.rot90(crop)
    return crop


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"不支持的旋转角度: {angle}")
