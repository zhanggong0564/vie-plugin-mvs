from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from schemas import InspectionVerdict


class MVSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_item_key: str | None = Field(
        default=None,
        description="可选；指定清单项目后检验，省略时自动匹配",
    )


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
