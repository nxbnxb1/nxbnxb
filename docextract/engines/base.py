"""Engine interfaces. Adapters (PaddleOCR, VLM, test fakes) implement these protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from ..models import BBox
from ..textutil import Line


@dataclass
class LayoutBox:
    label: str  # raw detector label, normalised later by layout.labels
    bbox: BBox  # pixels of the image passed to the detector
    score: float = 1.0


@dataclass
class OcrResult:
    text: str
    lines: list[Line] = field(default_factory=list)  # pixel coordinates within the crop

    @property
    def confidence(self) -> float | None:
        if not self.lines:
            return None
        weights = [max(1, len(ln.text)) for ln in self.lines]
        return sum(ln.score * w for ln, w in zip(self.lines, weights)) / sum(weights)


@dataclass
class TableResult:
    html: str
    confidence: float | None = None
    ocr_text: str = ""  # text the table engine read; used as evidence for cross-checks


@dataclass
class FormulaResult:
    latex: str
    confidence: float | None = None


@dataclass
class VlmRequest:
    image: Image.Image
    task: str  # see engines.vlm.PROMPTS
    hint: str | None = None  # text of the region from the text layer / OCR, to anchor exact words and numbers
    language: str | None = None  # language descriptions are written in (None = settings.vlm_language)


@dataclass
class VlmResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    data: dict[str, Any] | None = None  # parsed JSON for structured tasks (charts)


@runtime_checkable
class LayoutDetector(Protocol):
    name: str

    def detect(self, images: list[Image.Image]) -> list[list[LayoutBox]]: ...


@runtime_checkable
class TextRecognizer(Protocol):
    name: str

    def recognize(self, images: list[Image.Image]) -> list[OcrResult]: ...


@runtime_checkable
class TableRecognizer(Protocol):
    name: str

    def recognize(self, images: list[Image.Image]) -> list[TableResult]: ...


@runtime_checkable
class FormulaRecognizer(Protocol):
    name: str

    def recognize(self, images: list[Image.Image]) -> list[FormulaResult]: ...


@runtime_checkable
class OrientationClassifier(Protocol):
    name: str

    def classify(self, images: list[Image.Image]) -> list[int]:
        """Counter-clockwise rotation (0/90/180/270) that makes each page upright."""
        ...


@runtime_checkable
class VisionLanguageModel(Protocol):
    name: str

    def run(self, requests: list[VlmRequest]) -> list[VlmResult]: ...


@dataclass
class Engines:
    layout: LayoutDetector | None = None
    ocr: TextRecognizer | None = None
    table: TableRecognizer | None = None
    formula: FormulaRecognizer | None = None
    vlm: VisionLanguageModel | None = None
    orientation: OrientationClassifier | None = None

    def describe(self) -> dict[str, str | None]:
        return {
            key: getattr(getattr(self, key), "name", None)
            for key in ("layout", "ocr", "table", "formula", "vlm", "orientation")
        }
