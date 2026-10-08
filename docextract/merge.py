"""Merging regions into the final document: heading levels and Markdown rendering."""

from __future__ import annotations

import re
import unicodedata

from .config import Settings
from .models import FURNITURE_TYPES, DocumentResult, Region, RegionType, ValidationStatus
from .tables import parse_html_table
from .textutil import is_list_item, strip_bullet

LABELS = {
    "vi": {"image": "Hình ảnh", "chart": "Biểu đồ", "seal": "Con dấu", "page": "Trang", "review": "cần kiểm tra",
           "figure_text": "Chữ trong hình", "original": "Ảnh gốc"},
    "en": {"image": "Image", "chart": "Chart", "seal": "Seal", "page": "Page", "review": "needs review",
           "figure_text": "Text in figure", "original": "Original image"},
    "ja": {"image": "画像", "chart": "グラフ", "seal": "印影", "page": "ページ", "review": "要確認",
           "figure_text": "図中の文字", "original": "元の画像"},
}

_NUMBERED_RE = re.compile(r"^\s*((?:\d+\.)*\d+)[.)]?\s+\S")
_ROMAN_RE = re.compile(r"^\s*[IVXLC]+[.)]\s+\S")
_CHAPTER_RE = re.compile(r"^\s*(chương|phần|chapter|part|mục)\b|^\s*第\s*[0-9一二三四五六七八九十百]+\s*[章部編]", re.IGNORECASE)
_SECTION_JA_RE = re.compile(r"^\s*第\s*[0-9一二三四五六七八九十百]+\s*節")


def assign_heading_levels(regions: list[Region]) -> None:
    """Fill ``level`` of headings without one: numbering depth first, then font-size rank."""
    sizes = sorted(
        {round(r.meta["font_size"], 1) for r in regions if r.type == RegionType.HEADING and r.meta.get("font_size")},
        reverse=True,
    )
    for region in regions:
        if region.type == RegionType.TITLE:
            region.level = region.level or 1
            continue
        if region.type != RegionType.HEADING or region.level is not None:
            continue
        text = unicodedata.normalize("NFKC", region.content.strip())  # full-width "１．" → "1."
        numbered = _NUMBERED_RE.match(text)
        if _CHAPTER_RE.match(text) or _ROMAN_RE.match(text):
            region.level = 2
        elif _SECTION_JA_RE.match(text):
            region.level = 3
        elif numbered:
            region.level = min(6, 1 + numbered.group(1).count(".") + 1)
        elif region.meta.get("font_size") and sizes:
            region.level = min(4, 2 + sizes.index(round(region.meta["font_size"], 1)))
        else:
            region.level = 2


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _figure_markdown(region: Region, labels: dict[str, str]) -> str:
    """Kept picture (if any) as an image link, then the description and the text read inside it."""
    label = labels[region.type.value]
    text = region.content.strip()
    quote = []
    if text:
        quote.append(f"> **[{label}]** " + text.replace("\n", "\n> "))
    figure_text = (region.data or {}).get("figure_text")
    if figure_text and region.type != RegionType.SEAL:
        quote.append(f"> {labels['figure_text']}: " + _single_line(figure_text))
    link = ""
    if region.figure:
        alt = _single_line(text.split(". ")[0])[:120].translate(str.maketrans("[]", "()")) if text else ""
        link = f"![{label}: {alt}]({region.figure})" if alt else f"![{label}]({region.figure})"
    return "\n\n".join(p for p in (link, "\n>\n".join(quote)) if p)


def region_markdown(region: Region, settings: Settings) -> str:
    labels = LABELS[settings.output_locale]
    text = region.content.strip()
    rtype = region.type
    if rtype in (RegionType.IMAGE, RegionType.CHART, RegionType.SEAL):
        return _figure_markdown(region, labels)
    body = _region_body(region, settings, text)
    if region.figure:  # structure not recovered: keep the original picture next to the text
        body = (body + "\n\n" if body else "") + f"![{labels['original']}]({region.figure})"
    return body


def _region_body(region: Region, settings: Settings, text: str) -> str:
    rtype = region.type
    if not text and rtype not in (RegionType.TABLE,):
        return ""
    if rtype in (RegionType.TITLE, RegionType.HEADING):
        level = region.level or (1 if rtype == RegionType.TITLE else 2)
        return "#" * level + " " + _single_line(text)
    if rtype == RegionType.TABLE:
        if region.html:
            table = parse_html_table(region.html)
            if table is not None:
                return table.render(settings.table_format)
        return text
    if rtype == RegionType.FORMULA:
        if region.method.value in ("pdf_text", "ocr", "docx"):
            return text  # not LaTeX
        return f"$$\n{text}\n$$"
    if rtype == RegionType.LIST:
        lines = []
        for line in text.splitlines():
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            body = line.strip()
            if re.match(r"^(?:\d+[.)]|-)\s", body):
                lines.append(" " * indent + body)
            elif is_list_item(body):
                lines.append(" " * indent + "- " + strip_bullet(body))
            else:
                lines.append(" " * indent + "- " + body)
        return "\n".join(lines)
    if rtype == RegionType.CODE:
        return f"```\n{text}\n```"
    if rtype == RegionType.CAPTION:
        return f"*{_single_line(text)}*"
    return text


def render_markdown(result: DocumentResult, settings: Settings) -> str:
    labels = LABELS[settings.output_locale]
    parts: list[str] = []
    for page in result.pages:
        if settings.markdown_page_markers and page.number is not None:
            parts.append(f"<!-- {labels['page'].lower()}: {page.number} -->")
        for region in sorted(page.regions, key=lambda r: r.order):
            if region.type in FURNITURE_TYPES and not settings.markdown_include_furniture:
                continue
            if region.status == ValidationStatus.SKIPPED and not region.content.strip() and not region.figure:
                continue
            body = region_markdown(region, settings)
            if not body:
                continue
            if settings.markdown_review_markers and region.status == ValidationStatus.NEEDS_REVIEW:
                reason = "; ".join(region.issues)[:300].replace("--", "-")
                parts.append(f"<!-- {labels['review']}: {region.id} ({region.method.value}): {reason} -->")
            parts.append(body)
    return "\n\n".join(parts).strip() + "\n"
