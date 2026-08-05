import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from schemas import InspectionVerdict
from schemas.common import VisualReferenceParams


NormalizedQuadrilateral = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


def _validate_quadrilateral(values) -> NormalizedQuadrilateral:
    if not isinstance(values, (list, tuple)) or len(values) != 8:
        raise ValueError("每组 guideline_coordinates 必须包含 8 个四顶点坐标")
    points = tuple(float(value) for value in values)
    if any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in points
    ):
        raise ValueError("guideline_coordinates 必须为 0 到 1 的归一化坐标")

    vertices = list(zip(points[::2], points[1::2]))
    top_left, top_right, bottom_right, bottom_left = vertices
    if not (
        top_left[0] < top_right[0]
        and bottom_left[0] < bottom_right[0]
        and top_left[1] < bottom_left[1]
        and top_right[1] < bottom_right[1]
    ):
        raise ValueError(
            "guideline_coordinates 顶点顺序必须为左上、右上、右下、左下"
        )
    cross_products = []
    for index in range(4):
        current = vertices[index]
        following = vertices[(index + 1) % 4]
        after = vertices[(index + 2) % 4]
        cross_products.append(
            (following[0] - current[0]) * (after[1] - following[1])
            - (following[1] - current[1]) * (after[0] - following[0])
        )
    if any(value <= 0 for value in cross_products):
        raise ValueError(
            "guideline_coordinates 必须按左上、右上、右下、左下顺时针组成凸四边形"
        )
    return points


class MVSModelParams(VisualReferenceParams):
    product_type: str = Field(..., min_length=1)
    target_names: tuple[str, str, str]
    guideline_coordinates: tuple[
        NormalizedQuadrilateral,
        NormalizedQuadrilateral,
        NormalizedQuadrilateral,
        NormalizedQuadrilateral,
        NormalizedQuadrilateral,
    ]

    @field_validator("target_names", mode="before")
    @classmethod
    def _split_target_names(cls, value):
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("target_names 必须包含 3 个检测项")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("target_names 检测项不能为空")
        return tuple(item.strip() for item in value)

    @field_validator("guideline_coordinates", mode="before")
    @classmethod
    def _split_guideline_coordinates(cls, value):
        if isinstance(value, str):
            groups = [
                group.strip()
                for group in value.split(";")
                if group.strip()
            ]
            value = [
                [part.strip() for part in group.split(",") if part.strip()]
                for group in groups
            ]
        if not isinstance(value, (list, tuple)) or len(value) != 5:
            raise ValueError("guideline_coordinates 必须包含 5 组四边形")
        return tuple(_validate_quadrilateral(group) for group in value)


class MVSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modelParams: MVSModelParams


class PackingListItemResponse(BaseModel):
    line_no: str
    item_key: str
    name_cn: str = ""
    name_en: str = ""
    model: str = ""
    unit: str = ""
    quantity: int | None = None
    remarks: str = ""
    material_code: str | None = None
    material_code_source: str | None = None
    confidence: float = 0.0


class InspectionResultResponse(BaseModel):
    status: InspectionVerdict
    verdict: InspectionVerdict
    reason: str
    matched_line_no: str | None = None
    item_key: str | None = None
    expected: dict[str, Any] | None = None
    actual: dict[str, Any] | None = None


class ImageInspectionResponse(BaseModel):
    filename: str
    result: InspectionResultResponse


class MVSResultResponse(BaseModel):
    status: InspectionVerdict
    verdict: InspectionVerdict
    manifest_filename: str
    manifest_items: list[PackingListItemResponse] = Field(default_factory=list)
    inspections: list[ImageInspectionResponse] = Field(default_factory=list)
    message: str = ""


class MVSResponse(BaseModel):
    code: int
    message: str
    result: MVSResultResponse
