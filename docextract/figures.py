"""When must the original picture be kept?

Rule (no information may be lost): a region is replaced by text only when the text provably
carries everything the picture shows. Otherwise the cropped image is kept as a figure file
and linked from the Markdown, next to whatever description/text was extracted.

A picture may be replaced by text only if ALL of these hold:

1. a VLM description was obtained and parsed (kind + description + lossless flag);
2. its kind is textual (chart, diagram, flowchart, screenshot, table, text), not visual
   (photo, illustration, drawing, map, logo, signature, handwriting, other);
3. the VLM states the description is lossless;
4. the text inside the picture was read by OCR or the PDF text layer (never by the VLM),
   so labels and numbers come from a character-level source;
5. for charts: a data table was extracted, no value is marked as estimated (~), and every
   number in it appears in the text read from the picture.

Seals/stamps are always kept (their look is the evidence). Tables and formulas whose
structure could not be recovered (``needs_review``) also keep their picture.
"""

from __future__ import annotations

from .config import Settings
from .models import Region, RegionType, ValidationStatus
from .textutil import extract_numbers, multiset_recall

FIGURE_TYPES = frozenset({RegionType.IMAGE, RegionType.CHART, RegionType.SEAL})
TEXTUAL_KINDS = frozenset({"chart", "diagram", "flowchart", "screenshot", "table", "text"})


def _chart_values(chart: dict) -> list[str]:
    values = []
    for row in chart.get("rows") or []:
        cells = row.values() if isinstance(row, dict) else row if isinstance(row, list) else [row]
        values.extend(str(c) for c in cells)
    return values


def keep_figure(region: Region, settings: Settings) -> tuple[bool, str]:
    """(keep the picture?, reason)."""
    if settings.figure_policy == "always":
        return True, "figure_policy=always"
    if settings.figure_policy == "never":
        return False, "figure_policy=never"
    if region.type in (RegionType.TABLE, RegionType.FORMULA):
        if region.status == ValidationStatus.NEEDS_REVIEW:
            return True, f"{region.type.value} structure not recovered"
        return False, "structure recovered"
    if region.type == RegionType.SEAL:
        return True, "seal/stamp: the picture is the evidence"
    if region.type not in FIGURE_TYPES:
        return False, "text region"
    if region.status == ValidationStatus.SKIPPED:
        return False, "decorative image"

    data = region.data or {}
    info = data.get("figure")
    if not isinstance(info, dict) or not str(info.get("description") or "").strip():
        return True, "no structured description"
    kind = str(info.get("kind") or "other").lower()
    if kind not in TEXTUAL_KINDS:
        return True, f"visual content ({kind})"
    if info.get("lossless") is not True:
        return True, "description would lose information"
    figure_text = data.get("figure_text")
    if figure_text is None:
        return True, "text inside the picture was not read by OCR"
    if region.type == RegionType.CHART or kind == "chart":
        chart = data.get("chart") or {}
        values = _chart_values(chart)
        if not values:
            return True, "chart without extracted data"
        if any(v.strip().startswith("~") for v in values):
            return True, "chart values are estimated"
        numbers = extract_numbers(" ".join(values))
        if numbers and multiset_recall(numbers, extract_numbers(figure_text)) < settings.number_match_min:
            return True, "chart values not found in the text of the picture"
    return False, "fully described by text"
