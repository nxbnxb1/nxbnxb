"""Engine construction from settings."""

from __future__ import annotations

import logging

from ..config import Settings
from .base import (
    Engines,
    FormulaRecognizer,
    FormulaResult,
    LayoutBox,
    LayoutDetector,
    OcrResult,
    OrientationClassifier,
    TableRecognizer,
    TableResult,
    TextRecognizer,
    VisionLanguageModel,
    VlmRequest,
    VlmResult,
)

log = logging.getLogger(__name__)

__all__ = [
    "Engines",
    "FormulaRecognizer",
    "FormulaResult",
    "LayoutBox",
    "LayoutDetector",
    "OcrResult",
    "OrientationClassifier",
    "TableRecognizer",
    "TableResult",
    "TextRecognizer",
    "VisionLanguageModel",
    "VlmRequest",
    "VlmResult",
    "build_engines",
]


def _want(backend: str, available: bool, what: str) -> bool:
    if backend == "none" or backend == "heuristic":
        return False
    if backend == "paddle" and not available:
        raise RuntimeError(f"{what}: backend 'paddle' requested but paddleocr is not installed")
    return available


def build_engines(settings: Settings) -> Engines:
    """Create the engines the settings ask for; missing optional dependencies just disable a stage."""
    from .paddle import (
        PaddleFormulaRecognizer,
        PaddleLayoutDetector,
        PaddleOrientationClassifier,
        PaddleTableRecognizer,
        PaddleTextRecognizer,
        paddle_available,
    )

    has_paddle = paddle_available()
    if not has_paddle:
        log.warning("paddleocr not installed: layout falls back to PDF heuristics, OCR/table/formula disabled")
    engines = Engines()
    if _want(settings.layout_backend, has_paddle, "layout"):
        engines.layout = PaddleLayoutDetector(settings)
    if _want(settings.ocr_backend, has_paddle, "ocr"):
        engines.ocr = PaddleTextRecognizer(settings)
    if _want(settings.table_backend, has_paddle, "table"):
        engines.table = PaddleTableRecognizer(settings)
    if _want(settings.formula_backend, has_paddle, "formula"):
        engines.formula = PaddleFormulaRecognizer(settings)
    if settings.auto_orientation and has_paddle and settings.ocr_backend != "none":
        engines.orientation = PaddleOrientationClassifier(settings)
    if settings.vlm_enabled:
        from .vlm import OpenAICompatibleVLM

        engines.vlm = OpenAICompatibleVLM(settings)
    return engines
