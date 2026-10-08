"""Data model shared by every stage of the pipeline and by the JSON output."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RegionType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE = "image"
    CHART = "chart"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    FOOTNOTE = "footnote"
    SEAL = "seal"
    CODE = "code"
    OTHER = "other"


TEXT_LIKE_TYPES = frozenset(
    {
        RegionType.TITLE,
        RegionType.HEADING,
        RegionType.TEXT,
        RegionType.LIST,
        RegionType.CAPTION,
        RegionType.HEADER,
        RegionType.FOOTER,
        RegionType.PAGE_NUMBER,
        RegionType.FOOTNOTE,
        RegionType.CODE,
        RegionType.OTHER,
    }
)

# Page furniture: kept in JSON, left out of the Markdown by default.
FURNITURE_TYPES = frozenset({RegionType.HEADER, RegionType.FOOTER, RegionType.PAGE_NUMBER})


class Method(str, Enum):
    """Extraction method; also the unit the router chooses between."""

    PDF_TEXT = "pdf_text"  # text layer of a digital PDF (PyMuPDF)
    PDF_TABLE = "pdf_table"  # vector table of a digital PDF (PyMuPDF find_tables)
    DOCX = "docx"  # native Word structure (python-docx)
    OCR = "ocr"  # PP-OCRv5
    TABLE_RECOGNITION = "table_recognition"  # PaddleOCR table pipeline
    FORMULA_RECOGNITION = "formula_recognition"  # PaddleOCR formula module
    VLM = "vlm"  # vision-language model (Qwen-VL, ...)
    NONE = "none"  # nothing could be run


class ValidationStatus(str, Enum):
    PASSED = "passed"  # result checked and accepted
    UNCHECKED = "unchecked"  # no automatic check exists (e.g. image description)
    NEEDS_REVIEW = "needs_review"  # every method failed validation: check by hand
    SKIPPED = "skipped"  # intentionally not extracted (e.g. decorative image)


class PageKind(str, Enum):
    DIGITAL = "digital"  # PDF page with a usable text layer
    SCANNED = "scanned"  # PDF page without (usable) text layer
    IMAGE = "image"  # image file
    DOCX = "docx"  # Word document (no physical pages)


class BBox(BaseModel):
    """Axis-aligned box in page units (PDF points for PDFs, pixels for images)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def scaled(self, factor: float) -> BBox:
        return BBox(x0=self.x0 * factor, y0=self.y0 * factor, x1=self.x1 * factor, y1=self.y1 * factor)

    def intersection(self, other: BBox) -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def iou(self, other: BBox) -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def coverage_by(self, other: BBox) -> float:
        """Fraction of this box that lies inside ``other``."""
        return self.intersection(other) / self.area if self.area > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def union(self, other: BBox) -> BBox:
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    @classmethod
    def from_seq(cls, seq: Any) -> BBox:
        x0, y0, x1, y1 = (float(v) for v in list(seq)[:4])
        return cls(x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1))


class SourceRef(BaseModel):
    """Pointer back into the original file, for traceability."""

    file: str
    page: int | None = None  # 1-based; None for formats without pages (DOCX)
    bbox: BBox | None = None
    locator: str | None = None  # e.g. "body/table[2]" for DOCX


class Attempt(BaseModel):
    """One method tried on a region, with the outcome of its validation."""

    method: Method
    engine: str | None = None
    confidence: float | None = None
    passed: bool
    score: float = 0.0
    issues: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0
    cached: bool = False


class Region(BaseModel):
    id: str
    page: int | None
    type: RegionType
    order: int = 0
    bbox: BBox | None = None
    layout_score: float | None = None
    level: int | None = None  # heading level (1 = document title)
    content: str = ""  # Markdown-ready text / LaTeX / table Markdown / description
    html: str | None = None  # tables: HTML with row/colspans
    data: dict[str, Any] | None = None  # structured extras (chart data, OCR lines, ...)
    method: Method = Method.NONE
    engine: str | None = None
    confidence: float | None = None
    status: ValidationStatus = ValidationStatus.UNCHECKED
    issues: list[str] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)
    source: SourceRef | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class PageResult(BaseModel):
    number: int | None
    kind: PageKind
    width: float | None = None
    height: float | None = None
    unit: Literal["pt", "px", "none"] = "pt"
    regions: list[Region] = Field(default_factory=list)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    layout_backend: str | None = None
    duration_ms: float = 0.0


class SourceInfo(BaseModel):
    filename: str
    sha256: str
    size_bytes: int
    format: Literal["pdf", "docx", "image"]
    page_count: int


class DocumentStats(BaseModel):
    pages_processed: int = 0
    regions: int = 0
    regions_by_type: dict[str, int] = Field(default_factory=dict)
    regions_by_method: dict[str, int] = Field(default_factory=dict)
    regions_by_status: dict[str, int] = Field(default_factory=dict)
    escalations: int = 0  # regions where the first method failed and another was tried
    vlm_calls: int = 0
    vlm_input_tokens: int = 0
    vlm_output_tokens: int = 0
    estimated_cost: float = 0.0
    cache_hits: int = 0
    duration_ms: float = 0.0
    ms_per_page: float = 0.0
    cost_per_page: float = 0.0


class DocumentResult(BaseModel):
    schema_version: str = "1.0"
    source: SourceInfo
    pages: list[PageResult] = Field(default_factory=list)
    stats: DocumentStats = Field(default_factory=DocumentStats)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    engines: dict[str, str | None] = Field(default_factory=dict)
    markdown: str | None = None

    def iter_regions(self):
        for page in self.pages:
            yield from page.regions

    def to_json(self, indent: int | None = 2, include_markdown: bool = False) -> str:
        exclude = None if include_markdown else {"markdown"}
        return self.model_dump_json(indent=indent, exclude=exclude)
