"""Merging regions into the final document: heading levels and Markdown rendering."""

from __future__ import annotations

import re
import unicodedata

from .config import Settings
from .models import FURNITURE_TYPES, DocumentResult, Method, Region, RegionType, ValidationStatus
from .tables import Table, parse_html_table
from .textutil import is_cjk, is_list_item, join_inline, strip_bullet

LABELS = {
    "vi": {"image": "Hình ảnh", "chart": "Biểu đồ", "seal": "Con dấu", "page": "Trang", "review": "cần kiểm tra",
           "figure_text": "Chữ trong hình", "original": "Ảnh gốc", "generated": "mô tả tự động"},
    "en": {"image": "Image", "chart": "Chart", "seal": "Seal", "page": "Page", "review": "needs review",
           "figure_text": "Text in figure", "original": "Original image", "generated": "automatic description"},
    "ja": {"image": "画像", "chart": "グラフ", "seal": "印影", "page": "ページ", "review": "要確認",
           "figure_text": "図中の文字", "original": "元の画像", "generated": "自動生成の説明"},
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
        # a VLM description is generated text, unlike the words printed in the picture below it
        tag = f"{label} · {labels['generated']}" if region.method == Method.VLM else label
        quote.append(f"> **[{tag}]** " + text.replace("\n", "\n> "))
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


_TERMINAL = tuple(".!?:;。！？」』)）…")
_OPENING = set("「『（(［[・-–—•")


def _continues(prev: str, cur: str) -> bool:
    """Does ``cur`` continue the sentence ``prev`` was cut in (column or page break)?"""
    prev, cur = prev.rstrip(), cur.lstrip()
    if not prev or not cur or prev.endswith(_TERMINAL):
        return False
    if cur[0].islower():
        return True
    return is_cjk(prev[-1]) and is_cjk(cur[0]) and cur[0] not in _OPENING


def link_continuations(result: DocumentResult) -> None:
    """Mark paragraphs/tables split across columns or pages (``meta.continues`` / ``continued_by``).

    General rules only: a paragraph continues when the previous one stops mid-sentence (no
    terminal punctuation) and the next starts in lower case (or both are CJK); a table continues
    on the next page when it is the next content block and has the same number of columns.
    """
    sequence = [
        r
        for page in result.pages
        for r in sorted(page.regions, key=lambda r: r.order)
        if r.type not in FURNITURE_TYPES and r.type != RegionType.FOOTNOTE and r.status != ValidationStatus.SKIPPED
    ]
    for prev, cur in zip(sequence, sequence[1:]):
        joined = False
        if prev.type == RegionType.TEXT and cur.type == RegionType.TEXT:
            joined = _continues(prev.content, cur.content)
        elif prev.type == RegionType.TABLE and cur.type == RegionType.TABLE and prev.page and cur.page == prev.page + 1:
            a, b = parse_html_table(prev.html or ""), parse_html_table(cur.html or "")
            joined = a is not None and b is not None and a.n_cols == b.n_cols
        if joined:
            prev.meta["continued_by"] = cur.id
            cur.meta["continues"] = prev.id


def _chain(start: Region, by_id: dict[str, Region]) -> list[Region]:
    chain = [start]
    while chain[-1].meta.get("continued_by") in by_id:
        chain.append(by_id[chain[-1].meta["continued_by"]])
    return chain


def _merged(chain: list[Region], settings: Settings) -> Region:
    """One region carrying the content of a whole continuation chain (for rendering only)."""
    first = chain[0]
    if first.type == RegionType.TABLE:
        tables = [parse_html_table(r.html or "") for r in chain]
        rows = list(tables[0].rows)
        header = [c.text for c in tables[0].rows[0]] if tables[0].rows else []
        for table in tables[1:]:
            body = table.rows[1:] if table.rows and [c.text for c in table.rows[0]] == header else table.rows
            rows.extend(body)
        merged = Table(rows)
        return first.model_copy(update={"html": merged.to_html(), "content": merged.render(settings.table_format)})
    text = ""
    for region in chain:
        part = region.content.strip()
        if text.endswith("-") and part[:1].islower():
            text = text[:-1] + part
        else:
            text = join_inline(text, part)
    return first.model_copy(update={"content": text})


def render_markdown(result: DocumentResult, settings: Settings) -> str:
    labels = LABELS[settings.output_locale]
    if settings.join_continuations:
        link_continuations(result)
    by_id = {r.id: r for r in result.iter_regions()}
    parts: list[str] = []
    for page in result.pages:
        if settings.markdown_page_markers and page.number is not None:
            parts.append(f"<!-- {labels['page'].lower()}: {page.number} -->")
        for region in sorted(page.regions, key=lambda r: r.order):
            if region.type in FURNITURE_TYPES and not settings.markdown_include_furniture:
                continue
            if region.status == ValidationStatus.SKIPPED and not region.content.strip() and not region.figure:
                continue
            if settings.join_continuations and region.meta.get("continues") in by_id:
                continue  # rendered with the region it continues
            chain = _chain(region, by_id) if settings.join_continuations else [region]
            shown = _merged(chain, settings) if len(chain) > 1 else region
            body = region_markdown(shown, settings)
            for extra in chain[1:]:
                if extra.figure:
                    body += f"\n\n![{labels['original']}]({extra.figure})"
            if not body:
                continue
            if settings.markdown_review_markers:
                for member in chain:
                    if member.status == ValidationStatus.NEEDS_REVIEW:
                        reason = "; ".join(member.issues)[:300].replace("--", "-")
                        parts.append(f"<!-- {labels['review']}: {member.id} ({member.method.value}): {reason} -->")
            parts.append(body)
    return "\n\n".join(parts).strip() + "\n"
