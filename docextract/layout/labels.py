"""Layout candidates, detector label normalisation and box clean-up."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import BBox, RegionType

# PP-DocLayout / PP-DocLayout_plus / PP-DocLayoutV2 label names (and a few aliases).
LABEL_MAP: dict[str, RegionType] = {
    "doc_title": RegionType.TITLE,
    "title": RegionType.TITLE,
    "paragraph_title": RegionType.HEADING,
    "section_title": RegionType.HEADING,
    "text": RegionType.TEXT,
    "plain_text": RegionType.TEXT,
    "abstract": RegionType.TEXT,
    "content": RegionType.TEXT,
    "reference": RegionType.TEXT,
    "reference_content": RegionType.TEXT,
    "aside_text": RegionType.TEXT,
    "vertical_text": RegionType.TEXT,
    "vision_footnote": RegionType.FOOTNOTE,
    "list": RegionType.LIST,
    "table": RegionType.TABLE,
    "formula": RegionType.FORMULA,
    "display_formula": RegionType.FORMULA,
    "inline_formula": RegionType.FORMULA,
    "isolate_formula": RegionType.FORMULA,
    "formula_number": RegionType.TEXT,
    "image": RegionType.IMAGE,
    "figure": RegionType.IMAGE,
    "chart": RegionType.CHART,
    "figure_title": RegionType.CAPTION,
    "table_title": RegionType.CAPTION,
    "chart_title": RegionType.CAPTION,
    "figure_table_chart_title": RegionType.CAPTION,
    "figure_caption": RegionType.CAPTION,
    "table_caption": RegionType.CAPTION,
    "table_footnote": RegionType.FOOTNOTE,
    "header": RegionType.HEADER,
    "header_image": RegionType.HEADER,
    "footer": RegionType.FOOTER,
    "footer_image": RegionType.FOOTER,
    "number": RegionType.PAGE_NUMBER,
    "page_number": RegionType.PAGE_NUMBER,
    "footnote": RegionType.FOOTNOTE,
    "footnotes": RegionType.FOOTNOTE,
    "seal": RegionType.SEAL,
    "algorithm": RegionType.CODE,
    "code": RegionType.CODE,
}

# Containers whose inner text boxes are part of them (a chart's axis labels, a table's cells).
_CONTAINERS = frozenset({RegionType.TABLE, RegionType.CHART, RegionType.IMAGE, RegionType.FORMULA, RegionType.SEAL})


@dataclass
class Candidate:
    type: RegionType
    bbox: BBox  # page units
    score: float | None = None
    label: str | None = None  # raw detector label
    meta: dict[str, Any] = field(default_factory=dict)


def normalize_label(label: str) -> RegionType:
    return LABEL_MAP.get(label.strip().lower().replace(" ", "_").replace("-", "_"), RegionType.OTHER)


def clean_candidates(candidates: list[Candidate], page_area: float, min_area_ratio: float) -> list[Candidate]:
    """Drop specks, near-duplicates and text boxes that sit inside a table/figure.

    The size filter applies to model boxes only: PDF text blocks (e.g. a page number) are real.
    """
    kept = [
        c for c in candidates if c.bbox.area >= page_area * min_area_ratio or c.label in ("orphan_text", "pdf_text_block")
    ]
    kept.sort(key=lambda c: -(c.score if c.score is not None else 1.0))
    result: list[Candidate] = []
    for cand in kept:
        if any(cand.bbox.iou(other.bbox) > 0.8 for other in result):
            continue
        result.append(cand)
    containers = [c for c in result if c.type in _CONTAINERS]
    final = []
    for cand in result:
        if cand.type not in _CONTAINERS and cand.type != RegionType.CAPTION:
            if any(cand.bbox.coverage_by(box.bbox) > 0.85 for box in containers):
                continue
        final.append(cand)
    return final
