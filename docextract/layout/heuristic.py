"""Layout of a digital PDF page from its own structure (no model needed).

Used when PP-StructureV3 layout detection is not installed, and as a cheap default for
digital pages. Text blocks come from the PDF text layer, tables from vector ruling
lines, figures from embedded images and clusters of vector drawings.
"""

from __future__ import annotations

import re

import pymupdf

from ..models import BBox, RegionType
from ..preprocessing.pdf import (
    PdfLine,
    body_font_size,
    drawing_clusters,
    find_tables,
    image_boxes,
    is_math_line,
)
from ..textutil import is_list_item
from .labels import Candidate

_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.|table|chart|hình|bảng|biểu đồ|sơ đồ|đồ thị|ảnh|図|表|グラフ|写真)\s*[\dIVX０-９]+", re.IGNORECASE
)


def heuristic_layout(page: pymupdf.Page, lines: list[PdfLine], page_number: int, body_size: float | None = None) -> list[Candidate]:
    rect = page.rect
    width, height = rect.width, rect.height
    page_area = width * height
    body = body_size or body_font_size(lines) or 10.0
    largest = max((ln.size for ln in lines), default=0.0)

    candidates: list[Candidate] = []
    for bbox, table in find_tables(page, allow_text_strategy=False):
        candidates.append(Candidate(RegionType.TABLE, bbox, 0.9, "pdf_table", meta={"pdf_table": table}))
    tables = [c.bbox for c in candidates]

    for bbox in image_boxes(page):
        if bbox.area > 0.95 * page_area or bbox.area < 0.002 * page_area:
            continue  # page background / tiny decoration
        if any(bbox.coverage_by(t) > 0.5 for t in tables):
            continue
        candidates.append(Candidate(RegionType.IMAGE, bbox, 0.8, "pdf_image"))

    for bbox in drawing_clusters(page):
        if bbox.area < 0.02 * page_area or bbox.width < 40 or bbox.height < 40:
            continue
        if any(bbox.coverage_by(c.bbox) > 0.5 or c.bbox.coverage_by(bbox) > 0.5 for c in candidates):
            continue
        inside = [ln for ln in lines if bbox.contains_point(ln.cx, ln.cy)]
        if sum(len(ln.text) for ln in inside) > 400:
            continue  # a shaded text box, not a chart
        candidates.append(Candidate(RegionType.CHART, bbox, 0.6, "pdf_drawing"))

    figures = [c.bbox for c in candidates]
    blocks: dict[int, list[PdfLine]] = {}
    for line in lines:
        if any(f.contains_point(line.cx, line.cy) for f in figures):
            continue
        blocks.setdefault(line.block, []).append(line)

    for block_lines in blocks.values():
        bbox = BBox(
            x0=min(ln.x0 for ln in block_lines),
            y0=min(ln.y0 for ln in block_lines),
            x1=max(ln.x1 for ln in block_lines),
            y1=max(ln.y1 for ln in block_lines),
        )
        rtype, meta = classify_text_block(block_lines, bbox, height, body, page_number, figures, largest)
        candidates.append(Candidate(rtype, bbox, 0.7, "pdf_text_block", meta=meta))
    return candidates


def classify_text_block(
    lines: list[PdfLine],
    bbox: BBox,
    page_height: float,
    body: float,
    page_number: int,
    figures: list[BBox],
    largest: float | None = None,
) -> tuple[RegionType, dict]:
    """Type of a PDF text block from its font size, position and wording."""
    text = " ".join(ln.text for ln in lines).strip()
    size = max(ln.size for ln in lines)
    meta = {"font_size": size}
    short = len(text) <= 150 and len(lines) <= 3
    margin = 0.07 * page_height
    if short and (bbox.y1 <= margin or bbox.y0 >= page_height - margin):
        if re.fullmatch(r"[-–\s]*(?:(?:page|trang)\s*)?\d{1,4}(?:\s*(?:/|of|trên)\s*\d{1,4})?\s*(?:ページ|頁)?[-–\s]*", text, re.IGNORECASE):
            return RegionType.PAGE_NUMBER, meta
        return (RegionType.HEADER if bbox.y1 <= margin else RegionType.FOOTER), meta
    if sum(1 for ln in lines if is_math_line(ln)) > 0.5 * len(lines):
        return RegionType.FORMULA, meta
    if _CAPTION_RE.match(text) and any(_near(bbox, f) for f in figures):
        return RegionType.CAPTION, meta
    if short and size >= 1.15 * body and not text.endswith((".", ",", ";")):
        is_largest = largest is None or size >= largest - 0.5
        if page_number == 1 and is_largest and size >= 1.5 * body and bbox.y0 < 0.4 * page_height:
            return RegionType.TITLE, meta
        return RegionType.HEADING, meta
    if short and len(lines) <= 2 and all(ln.bold for ln in lines) and not text.endswith((".", ",", ";", ":")):
        return RegionType.HEADING, meta
    if len(lines) >= 1 and sum(1 for ln in lines if is_list_item(ln.text)) >= max(1, 0.5 * len(lines)):
        return RegionType.LIST, meta
    return RegionType.TEXT, meta


def _near(a: BBox, b: BBox, gap: float = 36.0) -> bool:
    horizontal = min(a.x1, b.x1) - max(a.x0, b.x0) > 0
    vertical = min(abs(a.y0 - b.y1), abs(b.y0 - a.y1)) <= gap
    return horizontal and vertical
