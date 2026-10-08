"""Reading order for layout boxes: columns left-to-right, full-width blocks as separators.

Plain XY-cut splits two-column text wherever both columns happen to have a gap at the
same height. Here, boxes spanning most of the width (titles, wide tables) cut the page
into horizontal bands first; each band is then split into columns by vertical gaps.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import BBox, RegionType

_FIRST = (RegionType.HEADER,)
_LAST = (RegionType.FOOTER, RegionType.PAGE_NUMBER)


def _row_sorted(indices: list[int], boxes: Sequence[BBox]) -> list[int]:
    """Top-to-bottom; boxes whose vertical extents mostly overlap are read left-to-right."""
    ordered = sorted(indices, key=lambda i: (boxes[i].y0, boxes[i].x0))
    rows: list[list[int]] = []
    for i in ordered:
        if rows:
            row = rows[-1]
            top = min(boxes[j].y0 for j in row)
            bottom = max(boxes[j].y1 for j in row)
            overlap = min(bottom, boxes[i].y1) - max(top, boxes[i].y0)
            if overlap > 0.5 * min(boxes[i].height, bottom - top):
                row.append(i)
                continue
        rows.append([i])
    return [i for row in rows for i in sorted(row, key=lambda j: boxes[j].x0)]


def _columns(indices: list[int], boxes: Sequence[BBox], min_gap: float) -> list[list[int]]:
    spans = sorted(indices, key=lambda i: boxes[i].x0)
    groups: list[list[int]] = []
    right = None
    for i in spans:
        if right is None or boxes[i].x0 > right + min_gap:
            groups.append([i])
            right = boxes[i].x1
        else:
            groups[-1].append(i)
            right = max(right, boxes[i].x1)
    return groups


def _order(indices: list[int], boxes: Sequence[BBox], min_gap: float, depth: int) -> list[int]:
    if len(indices) <= 1 or depth > 6:
        return _row_sorted(indices, boxes)
    columns = _columns(indices, boxes, min_gap)
    if len(columns) > 1:
        return [i for col in columns for i in _order(col, boxes, min_gap, depth + 1)]
    x0 = min(boxes[i].x0 for i in indices)
    x1 = max(boxes[i].x1 for i in indices)
    width = max(1e-6, x1 - x0)
    spanners = [i for i in indices if boxes[i].width >= 0.6 * width]
    others = [i for i in indices if i not in spanners]
    if not spanners or not others:
        return _row_sorted(indices, boxes)
    spanners.sort(key=lambda i: boxes[i].y0)
    result: list[int] = []
    remaining = others
    for s in spanners:
        cy = boxes[s].center[1]
        band = [i for i in remaining if boxes[i].center[1] < cy]
        remaining = [i for i in remaining if boxes[i].center[1] >= cy]
        if band:
            result.extend(_order(band, boxes, min_gap, depth + 1))
        result.append(s)
    if remaining:
        result.extend(_order(remaining, boxes, min_gap, depth + 1))
    return result


def reading_order(
    boxes: Sequence[BBox], types: Sequence[RegionType] | None = None, page_width: float | None = None
) -> list[int]:
    """Indices of ``boxes`` in reading order; headers first, footers/page numbers last."""
    if not boxes:
        return []
    types = list(types) if types is not None else [RegionType.TEXT] * len(boxes)
    width = page_width or max(b.x1 for b in boxes)
    min_gap = max(2.0, 0.01 * width)
    first = [i for i, t in enumerate(types) if t in _FIRST]
    last = [i for i, t in enumerate(types) if t in _LAST]
    body = [i for i in range(len(boxes)) if i not in first and i not in last]
    return _row_sorted(first, boxes) + _order(body, boxes, min_gap, 0) + _row_sorted(last, boxes)
