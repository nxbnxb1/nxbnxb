"""PDF access with PyMuPDF: page classification, rendering, text layer and vector tables.

PyMuPDF is not thread-safe, so every call that touches a document holds ``PDF_LOCK``.
Coordinates are in unrotated PDF points (the space PyMuPDF reports text in).
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, field

import pymupdf
from PIL import Image

from ..models import BBox, PageKind
from ..tables import Cell, Table
from ..textutil import Line, garbled_ratio, is_legacy_vietnamese_font, normalize

PDF_LOCK = threading.RLock()

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()  # silence the stdout hint printed by find_tables

_MATH_FONT_HINTS = ("cmmi", "cmsy", "cmex", "msam", "msbm", "math", "symbol", "stix", "euler", "mtextra")


@dataclass
class PdfLine(Line):
    block: int = -1
    fonts: tuple[str, ...] = ()

    @property
    def bbox(self) -> BBox:
        return BBox(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)


@dataclass
class PageAnalysis:
    kind: PageKind
    text_chars: int
    garbled: float
    image_coverage: float
    legacy_fonts: list[str] = field(default_factory=list)


def open_pdf(data: bytes) -> pymupdf.Document:
    with PDF_LOCK:
        doc = pymupdf.open(stream=data, filetype="pdf")
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError("PDF is password protected")
        return doc


def analyze_page(page: pymupdf.Page, min_text_chars: int, max_garbled: float, trust_ocr_layer: bool) -> PageAnalysis:
    with PDF_LOCK:
        text = page.get_text("text")
        fonts = [f[3] for f in page.get_fonts()]
        page_area = abs(page.rect)
        covered = 0.0
        for info in page.get_image_info():
            rect = pymupdf.Rect(info["bbox"]) & page.rect
            covered += abs(rect)
    chars = sum(1 for ch in text if not ch.isspace())
    garbled = garbled_ratio(text)
    coverage = min(1.0, covered / page_area) if page_area else 0.0
    legacy = [f for f in fonts if is_legacy_vietnamese_font(f)]
    if chars < min_text_chars or garbled > max_garbled:
        kind = PageKind.SCANNED
    elif coverage > 0.85 and not trust_ocr_layer:
        kind = PageKind.SCANNED  # scan with a hidden OCR text layer of unknown quality
    else:
        kind = PageKind.DIGITAL
    return PageAnalysis(kind=kind, text_chars=chars, garbled=round(garbled, 4), image_coverage=round(coverage, 3), legacy_fonts=legacy)


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    with PDF_LOCK:
        pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def pixel_matrix(page: pymupdf.Page, dpi: int) -> pymupdf.Matrix:
    """Maps unrotated page points to pixels of ``render_page(page, dpi)``."""
    with PDF_LOCK:
        return page.rotation_matrix * pymupdf.Matrix(dpi / 72, dpi / 72)


def page_lines(page: pymupdf.Page) -> list[PdfLine]:
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    with PDF_LOCK:
        data = page.get_text("dict", flags=flags)
    lines: list[PdfLine] = []
    for b_index, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text")]
            if not spans:
                continue
            text = normalize("".join(s["text"] for s in spans))
            if not text:
                continue
            sized = [s for s in spans if s["text"].strip()] or spans
            size = max(s.get("size", 0.0) for s in sized)
            bold = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in sized)
            x0, y0, x1, y1 = line["bbox"]
            lines.append(
                PdfLine(
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    size=round(size, 2),
                    bold=bool(bold),
                    block=b_index,
                    fonts=tuple(sorted({s.get("font", "") for s in sized})),
                )
            )
    return lines


def body_font_size(lines: list[PdfLine]) -> float:
    """Most common font size weighted by characters: the size of running text."""
    weights: dict[float, int] = {}
    for line in lines:
        weights[line.size] = weights.get(line.size, 0) + len(line.text)
    if not weights:
        return 0.0
    return max(weights.items(), key=lambda kv: kv[1])[0]


def is_math_line(line: PdfLine) -> bool:
    return any(hint in font.lower() for font in line.fonts for hint in _MATH_FONT_HINTS)


def lines_in(lines: list[PdfLine], bbox: BBox) -> list[PdfLine]:
    return [ln for ln in lines if bbox.contains_point(ln.cx, ln.cy)]


def find_tables(page: pymupdf.Page, clip: BBox | None = None, allow_text_strategy: bool = True) -> list[tuple[BBox, Table]]:
    """Vector tables found by PyMuPDF, with merged cells turned into row/colspans."""
    rect = pymupdf.Rect(clip.as_tuple()) if clip else None
    with PDF_LOCK:
        found = page.find_tables(clip=rect).tables
        if not found and rect is not None and allow_text_strategy:
            found = page.find_tables(clip=rect, strategy="text").tables
        tables = []
        for tab in found:
            table = _to_table(tab)
            if table is not None:
                tables.append((BBox.from_seq(tab.bbox), table))
    return tables


def _to_table(tab) -> Table | None:
    texts = tab.extract()
    grid_cells = [row.cells for row in tab.rows]
    if not texts or not grid_cells:
        return None
    col_starts = sorted({round(c[0], 1) for row in grid_cells for c in row if c is not None})
    row_starts = sorted({round(c[1], 1) for row in grid_cells for c in row if c is not None})
    rows: list[list[Cell]] = []
    header_in_table = not getattr(tab.header, "external", True)
    for r, row in enumerate(grid_cells):
        out_row = []
        for c, box in enumerate(row):
            if box is None:
                continue
            x0, y0, x1, y1 = box
            colspan = max(1, sum(1 for x in col_starts if x0 - 0.5 <= x < x1 - 0.5))
            rowspan = max(1, sum(1 for y in row_starts if y0 - 0.5 <= y < y1 - 0.5))
            value = texts[r][c] if r < len(texts) and c < len(texts[r]) else ""
            text = "\n".join(normalize(part) for part in str(value or "").split("\n")).strip()
            out_row.append(Cell(text=text, rowspan=rowspan, colspan=colspan, header=header_in_table and r == 0))
        rows.append(out_row)
    table = Table(rows)
    return table if table.n_rows and table.n_cols else None


def image_boxes(page: pymupdf.Page) -> list[BBox]:
    with PDF_LOCK:
        infos = page.get_image_info()
        page_rect = page.rect
    out = []
    for info in infos:
        rect = pymupdf.Rect(info["bbox"]) & page_rect
        if rect.is_empty:
            continue
        out.append(BBox.from_seq(rect))
    return out


def drawing_clusters(page: pymupdf.Page) -> list[BBox]:
    with PDF_LOCK:
        try:
            rects = page.cluster_drawings()
        except Exception:  # older PyMuPDF or malformed drawings
            return []
    return [BBox.from_seq(r) for r in rects if not pymupdf.Rect(r).is_empty]


def median(values: list[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default
