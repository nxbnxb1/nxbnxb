"""Intelligent Router: picks an ordered chain of methods for each region.

Phase 1 is rule-based on region type, complexity and text-layer quality. Text is only
ever read from the PDF text layer or by OCR, never by the VLM: a VLM can paraphrase or
"correct" text, OCR output is traceable character by character. The VLM is reserved
for what OCR cannot do (charts, pictures) and as a fallback for table/formula
structure. Every decision
and its outcome can be logged as JSONL (``route_log_path``) to train a Neural Router
later (phase 3) that predicts the cheapest method meeting the accuracy target.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Settings
from .engines.base import Engines
from .models import Method, PageKind, RegionType

if TYPE_CHECKING:
    from .executor import RegionTask


@dataclass
class RouteFeatures:
    region_type: RegionType
    page_kind: PageKind
    area_ratio: float  # region area / page area
    aspect: float  # width / height
    layout_score: float | None
    text_layer_chars: int  # characters of usable PDF text inside the region
    known_cells: int | None = None  # table size when the layout step already parsed it
    full_page: bool = False  # no layout model: the whole page is one region

    def as_dict(self) -> dict:
        data = asdict(self)
        data["region_type"] = self.region_type.value
        data["page_kind"] = self.page_kind.value
        return data


class RuleBasedRouter:
    name = "rules-v1"

    def __init__(self, settings: Settings, engines: Engines) -> None:
        self.settings = settings
        self.engines = engines
        self._log_lock = threading.Lock()

    def available(self, method: Method) -> bool:
        e = self.engines
        return {
            Method.PDF_TEXT: True,
            Method.PDF_TABLE: True,
            Method.DOCX: True,
            Method.OCR: e.ocr is not None,
            Method.TABLE_RECOGNITION: e.table is not None,
            Method.FORMULA_RECOGNITION: e.formula is not None,
            Method.VLM: e.vlm is not None,
            Method.NONE: False,
        }[method]

    def plan(self, f: RouteFeatures) -> tuple[list[Method], str | None]:
        """Ordered methods to try, and a reason when the region is deliberately skipped."""
        s = self.settings
        has_text = f.page_kind == PageKind.DIGITAL and f.text_layer_chars > 0
        text_source = [Method.PDF_TEXT] if has_text else [Method.OCR]
        t = f.region_type

        if f.full_page:
            chain = [Method.OCR]
        elif t == RegionType.TABLE:
            chain = [Method.PDF_TABLE] if has_text else []
            complex_table = (f.known_cells or 0) > s.complex_table_cells or f.area_ratio > 0.45
            if complex_table and s.prefer_vlm_for_complex_tables and self.available(Method.VLM):
                chain += [Method.VLM, Method.TABLE_RECOGNITION]
            else:
                chain += [Method.TABLE_RECOGNITION, Method.VLM]
            chain += text_source  # last resort: keep the text, flag the region
        elif t == RegionType.FORMULA:
            chain = [Method.FORMULA_RECOGNITION, Method.VLM] + text_source
        elif t == RegionType.CHART:
            chain = [Method.VLM] + text_source
        elif t == RegionType.IMAGE:
            if not s.describe_images:
                return [], "image description disabled"
            if f.area_ratio < s.min_image_area_ratio:
                return [], "small image (logo/icon)"
            chain = [Method.VLM, Method.OCR]
        else:  # text-like regions and seals: PDF text layer when usable, otherwise OCR. Never the VLM.
            chain = text_source + [Method.OCR]

        seen: set[Method] = set()
        plan = []
        for method in chain:
            if method not in seen and self.available(method):
                seen.add(method)
                plan.append(method)
        return plan, None

    def log(self, task: RegionTask) -> None:
        path = self.settings.route_log_path
        if not path:
            return
        region = task.region
        record = {
            "router": self.name,
            "region_id": region.id,
            "features": task.features.as_dict() if task.features else None,
            "plan": [m.value for m in task.plan],
            "attempts": [
                {
                    "method": a.method.value,
                    "passed": a.passed,
                    "score": a.score,
                    "confidence": a.confidence,
                    "duration_ms": a.duration_ms,
                    "issues": a.issues,
                }
                for a in region.attempts
            ],
            "final_method": region.method.value,
            "status": region.status.value,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._log_lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
