"""docextract: PDF / Word / scanned images → Markdown + JSON, with per-region routing between
PDF parsing, PaddleOCR (layout, OCR, tables, formulas) and a VLM for visual content."""

__version__ = "0.1.0"

from .config import Settings  # noqa: E402
from .models import DocumentResult  # noqa: E402
from .pipeline import DocumentPipeline, ExtractOptions  # noqa: E402

__all__ = ["DocumentPipeline", "DocumentResult", "ExtractOptions", "Settings", "__version__"]
