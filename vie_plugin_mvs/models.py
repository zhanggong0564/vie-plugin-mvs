from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class InspectionStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    polygon: list[list[float]]

    @property
    def center(self) -> tuple[float, float]:
        xs = [point[0] for point in self.polygon]
        ys = [point[1] for point in self.polygon]
        return sum(xs) / len(xs), sum(ys) / len(ys)


@dataclass(frozen=True)
class PackingListItem:
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


@dataclass(frozen=True)
class LabelObservation:
    item_name: str | None = None
    material_code: str | None = None
    raw_texts: tuple[str, ...] = ()
    qr_text: str | None = None
    confidence: float = 0.0
    code_candidates: tuple[str, ...] = ()
    detected_codes: tuple[str, ...] = ()
    has_ambiguous_code: bool = False
    multiple_labels: bool = False
    review_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class InspectionResult:
    status: InspectionStatus
    reason: str
    matched_line_no: str | None = None
    item_key: str | None = None
    expected: dict[str, Any] | None = None
    actual: dict[str, Any] | None = None


@dataclass(frozen=True)
class ImageInspection:
    filename: str
    result: InspectionResult

    @property
    def status(self) -> InspectionStatus:
        return self.result.status


@dataclass(frozen=True)
class MVSResult:
    status: InspectionStatus
    manifest_filename: str
    manifest_items: tuple[PackingListItem, ...] = ()
    inspections: tuple[ImageInspection, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        for inspection in data["inspections"]:
            inspection["result"]["status"] = inspection["result"]["status"].value
        return data
