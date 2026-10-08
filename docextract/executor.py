"""Runs the routed method chains: batched per method, validated, escalated on failure.

Execution proceeds in rounds. In each round every unfinished region contributes its
next method; regions are grouped by method so engines see whole batches, and groups
run in parallel threads (the VLM client additionally sends its requests concurrently).
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .cache import ResultCache
from .config import Settings
from .engines.base import Engines, VlmRequest
from .engines.vlm import PROMPTS
from .figures import FIGURE_TYPES, keep_figure
from .metrics import normalized_distance
from .models import Attempt, BBox, Method, PageKind, Region, RegionType, ValidationStatus
from .preprocessing import pdf as pdfmod
from .router import RouteFeatures, RuleBasedRouter
from .tables import Table, parse_html_table, parse_markdown_table, table_from_rows
from .textutil import detect_language, garbled_ratio, join_lines
from .validation import ValidationReport, validate

log = logging.getLogger(__name__)

Matrix = tuple[float, float, float, float, float, float]


def _apply(m: Matrix, bbox: BBox) -> BBox:
    a, b, c, d, e, f = m
    xs, ys = [], []
    for x, y in ((bbox.x0, bbox.y0), (bbox.x1, bbox.y0), (bbox.x0, bbox.y1), (bbox.x1, bbox.y1)):
        xs.append(x * a + y * c + e)
        ys.append(x * b + y * d + f)
    return BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))


def _invert(m: Matrix) -> Matrix:
    a, b, c, d, e, f = m
    det = a * d - b * c
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    return ia, ib, ic, id_, -(e * ia + f * ic), -(e * ib + f * id_)


@dataclass
class PageContext:
    """Everything the methods need about one page; the rendered image is loaded lazily."""

    number: int | None
    kind: PageKind
    width: float
    height: float
    unit: str = "pt"
    to_px_matrix: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    image_loader: Callable[[], Image.Image] | None = None
    lines: list[pdfmod.PdfLine] = field(default_factory=list)
    pdf_page: Any = None
    preprocessing: dict = field(default_factory=dict)
    layout_backend: str | None = None
    _image: Image.Image | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def image(self) -> Image.Image | None:
        with self._lock:
            if self._image is None and self.image_loader is not None:
                self._image = self.image_loader()
            return self._image

    def to_px(self, bbox: BBox) -> BBox:
        return _apply(self.to_px_matrix, bbox)

    def from_px(self, bbox: BBox) -> BBox:
        return _apply(_invert(self.to_px_matrix), bbox)

    def crop(self, bbox: BBox, pad: int = 4) -> Image.Image | None:
        image = self.image
        if image is None:
            return None
        px = self.to_px(bbox)
        box = (
            max(0, int(px.x0) - pad),
            max(0, int(px.y0) - pad),
            min(image.width, int(px.x1 + 0.999) + pad),
            min(image.height, int(px.y1 + 0.999) + pad),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return image.crop(box)

    def text_lines(self, bbox: BBox) -> list[pdfmod.PdfLine]:
        return pdfmod.lines_in(self.lines, bbox)

    def release(self) -> None:
        with self._lock:
            self._image = None


@dataclass
class Extraction:
    method: Method
    content: str = ""
    html: str | None = None
    confidence: float | None = None
    data: dict[str, Any] | None = None
    engine: str | None = None
    error: str | None = None
    cached: bool = False
    table: Table | None = None
    duration_ms: float = 0.0

    def to_cache(self) -> dict:
        return {"content": self.content, "html": self.html, "confidence": self.confidence, "data": self.data}

    @classmethod
    def from_cache(cls, method: Method, engine: str | None, value: dict) -> Extraction:
        ext = cls(method=method, engine=engine, cached=True, **value)
        if ext.html:
            ext.table = parse_html_table(ext.html)
        return ext


@dataclass
class RegionTask:
    region: Region
    page: PageContext
    plan: list[Method] = field(default_factory=list)
    features: RouteFeatures | None = None
    image: Image.Image | None = None  # standalone image (DOCX pictures) instead of a page crop
    pdf_table: Table | None = None  # vector table already parsed by the heuristic layout step
    evidence: dict[str, str] = field(default_factory=dict)
    results: list[tuple[Extraction, ValidationReport]] = field(default_factory=list)
    step: int = 0
    done: bool = False
    _crop: Image.Image | None = None
    _crop_hash: str | None = None

    def crop(self) -> Image.Image | None:
        if self._crop is None:
            if self.image is not None:
                self._crop = self.image
            elif self.region.bbox is not None:
                self._crop = self.page.crop(self.region.bbox)
        return self._crop

    def crop_hash(self) -> str:
        if self._crop_hash is None:
            image = self.crop()
            h = hashlib.sha1()
            if image is not None:
                h.update(f"{image.mode}{image.size}".encode())
                h.update(image.tobytes())
            self._crop_hash = h.hexdigest()
        return self._crop_hash

    @property
    def current(self) -> Method:
        return self.plan[self.step]


@dataclass
class RunStats:
    vlm_calls: int = 0
    vlm_input_tokens: int = 0
    vlm_output_tokens: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, **values: int) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, getattr(self, key) + value)


VLM_TASKS = {
    RegionType.TABLE: "table",
    RegionType.FORMULA: "formula",
    RegionType.CHART: "chart",
    RegionType.IMAGE: "image",
}
PROMPT_VERSION = hashlib.sha1("".join(PROMPTS.values()).encode()).hexdigest()[:8]


class Executor:
    def __init__(
        self,
        settings: Settings,
        engines: Engines,
        router: RuleBasedRouter,
        cache: ResultCache | None = None,
        stats: RunStats | None = None,
    ) -> None:
        self.settings = settings
        self.engines = engines
        self.router = router
        self.cache = cache
        self.stats = stats or RunStats()
        self.figures: dict[str, bytes] = {}  # region id → PNG of pictures that must be kept
        self._lang_text: list[str] = []
        self._lang_chars = 0
        self._lang_seen: set[int] = set()
        self._handlers: dict[Method, Callable[[list[RegionTask]], list[Extraction]]] = {
            Method.PDF_TEXT: self._pdf_text,
            Method.PDF_TABLE: self._pdf_table,
            Method.OCR: self._ocr,
            Method.TABLE_RECOGNITION: self._table_recognition,
            Method.FORMULA_RECOGNITION: self._formula,
            Method.VLM: self._vlm,
        }

    # --- orchestration ----------------------------------------------------------------

    def prepare(self, task: RegionTask) -> None:
        """Collect evidence and plan the method chain for a region."""
        region, page = task.region, task.page
        if page.kind == PageKind.DIGITAL and region.bbox is not None:
            layer = join_lines(page.text_lines(region.bbox))
            if layer and garbled_ratio(layer) <= self.settings.pdf_text_max_garbled_ratio:
                task.evidence["text_layer"] = layer
        bbox = region.bbox or BBox(x0=0, y0=0, x1=page.width or 1, y1=page.height or 1)
        page_area = page.area or bbox.area or 1.0
        if task.image is not None:
            area_ratio = 1.0 if min(task.image.size) >= 64 else 0.0
        else:
            area_ratio = bbox.area / page_area
        task.features = RouteFeatures(
            region_type=region.type,
            page_kind=page.kind,
            area_ratio=round(area_ratio, 5),
            aspect=round(bbox.width / bbox.height, 3) if bbox.height else 0.0,
            layout_score=region.layout_score,
            text_layer_chars=len(task.evidence.get("text_layer", "")),
            known_cells=region.meta.get("known_cells"),
            full_page=bool(region.meta.get("full_page")),
        )
        route = self.router.plan(task.features)
        task.plan = route.plan
        if not task.plan:
            region.method = Method.NONE
            if route.drop:
                region.status = ValidationStatus.SKIPPED
            elif region.type in FIGURE_TYPES:
                region.status = ValidationStatus.UNCHECKED  # the picture itself is kept
            else:
                region.status = ValidationStatus.NEEDS_REVIEW
            region.issues = [route.note or f"no engine available for {region.type.value}"]
            task.done = True

    def run(self, tasks: list[RegionTask]) -> None:
        for task in tasks:
            if not task.done and not task.plan:
                self.prepare(task)
        pending = [t for t in tasks if not t.done]
        with ThreadPoolExecutor(max_workers=self.settings.workers, thread_name_prefix="method") as pool:
            while pending:
                # The VLM works last: it is given the text of each picture read by OCR or the
                # text layer (exact labels and numbers) and the language of the document.
                ready = [t for t in pending if t.current != Method.VLM]
                if not ready:
                    ready = pending
                    self._read_figure_text([t for t in ready if t.region.type in FIGURE_TYPES])
                    self._learn_language(tasks)
                groups: dict[Method, list[RegionTask]] = defaultdict(list)
                for task in ready:
                    groups[task.current].append(task)
                list(pool.map(lambda item: self._run_group(*item), groups.items()))
                pending = [t for t in pending if not t.done]
        self._learn_language(tasks)
        self._figure_text(tasks)
        self._keep_figures(tasks)
        for task in tasks:
            self.router.log(task)

    # --- language of the document -----------------------------------------------------

    def _learn_language(self, tasks: list[RegionTask]) -> None:
        """Language of the document from the text extracted so far (OCR / text layer)."""
        for task in tasks:
            if task.done and task.region.type not in FIGURE_TYPES and id(task) not in self._lang_seen:
                self._lang_seen.add(id(task))
                if self._lang_chars < 20000 and task.region.content:
                    self._lang_text.append(task.region.content)
                    self._lang_chars += len(task.region.content)

    @property
    def language(self) -> str:
        """Language VLM descriptions are written in: fixed by settings, else the document's."""
        if self.settings.vlm_language:
            return self.settings.vlm_language
        found = detect_language(" ".join(self._lang_text), self.settings.product_info.languages)
        return found or self.settings.output_locale

    # --- pictures ---------------------------------------------------------------------

    def _read_figure_text(self, tasks: list[RegionTask]) -> None:
        """Text inside pictures comes from the PDF text layer or OCR, never from the VLM."""
        need_ocr = []
        for task in tasks:
            region = task.region
            if region.type not in FIGURE_TYPES or region.status == ValidationStatus.SKIPPED:
                continue
            if "figure_text" in task.evidence:
                continue
            if region.type == RegionType.SEAL and region.method == Method.OCR:
                task.evidence["figure_text"] = region.content
            elif task.evidence.get("text_layer"):
                task.evidence["figure_text"] = task.evidence["text_layer"]
            elif self.engines.ocr is not None:
                need_ocr.append(task)
        if need_ocr:
            try:
                results = self._ocr(need_ocr)
            except Exception as exc:  # the picture is kept anyway when its text is unknown
                log.warning("OCR of pictures failed: %s", exc)
                return
            for task, ext in zip(need_ocr, results):
                if not ext.error:
                    task.evidence["figure_text"] = ext.content

    def _figure_text(self, tasks: list[RegionTask]) -> None:
        self._read_figure_text(tasks)
        for task in tasks:
            region = task.region
            if region.type in FIGURE_TYPES and region.status != ValidationStatus.SKIPPED and "figure_text" in task.evidence:
                self._set_figure_text(region, task.evidence["figure_text"])

    @staticmethod
    def _set_figure_text(region: Region, text: str) -> None:
        region.data = {**(region.data or {}), "figure_text": text.strip()}

    def _keep_figures(self, tasks: list[RegionTask]) -> None:
        for task in tasks:
            region = task.region
            keep, reason = keep_figure(region, self.settings)
            if region.type in FIGURE_TYPES or keep:
                region.meta["figure_decision"] = reason
            if not keep:
                continue
            image = task.crop()
            if image is None:
                continue
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG", optimize=True)
            self.figures[region.id] = buf.getvalue()

    def _run_group(self, method: Method, tasks: list[RegionTask]) -> None:
        start = time.perf_counter()
        try:
            results = self._handlers[method](tasks)
        except Exception as exc:  # engine crash: every region of the batch escalates
            log.exception("%s failed on %d regions", method.value, len(tasks))
            results = [Extraction(method=method, error=f"{type(exc).__name__}: {exc}") for _ in tasks]
        elapsed = (time.perf_counter() - start) * 1000 / max(1, len(tasks))
        for task, ext in zip(tasks, results):
            if not ext.duration_ms:
                ext.duration_ms = 0.0 if ext.cached else elapsed
            self._record(task, ext)

    def _record(self, task: RegionTask, ext: Extraction) -> None:
        if ext.data and ext.data.get("trusted_text") and not task.evidence.get("trusted_text"):
            task.evidence["trusted_text"] = ext.data["trusted_text"]
        report = validate(task, ext, self.settings)
        task.results.append((ext, report))
        task.region.attempts.append(
            Attempt(
                method=ext.method,
                engine=ext.engine,
                confidence=None if ext.confidence is None else round(ext.confidence, 4),
                passed=report.passed,
                score=round(report.score, 3),
                issues=report.issues,
                duration_ms=round(ext.duration_ms, 1),
                cached=ext.cached,
            )
        )
        if ext.cached:
            self.stats.add(cache_hits=1)
        task.step += 1
        if report.passed or task.step >= len(task.plan):
            self._finalize(task)

    def _finalize(self, task: RegionTask) -> None:
        ext, report = task.results[-1]
        if not report.passed:
            ext, report = max(task.results, key=lambda r: r[1].score)
            # A picture whose description failed is kept as a figure: nothing is lost, nothing to review.
            status = ValidationStatus.UNCHECKED if task.region.type in FIGURE_TYPES else ValidationStatus.NEEDS_REVIEW
            report = ValidationReport(False, status, report.score, report.issues)
        region = task.region
        region.content = ext.content
        region.html = ext.html
        region.method = ext.method
        region.engine = ext.engine
        region.confidence = None if ext.confidence is None else round(ext.confidence, 4)
        region.status = report.status
        region.issues = list(report.issues)
        data = {k: v for k, v in (ext.data or {}).items() if k not in ("trusted_text", "line_scores")}
        if "agreement" in data:
            region.meta["ocr_agreement"] = data.pop("agreement")
        region.data = data or None
        task.done = True

    # --- caching helper ---------------------------------------------------------------

    def _batched(
        self,
        method: Method,
        engine: str,
        tasks: list[RegionTask],
        compute: Callable[[list[RegionTask]], list[Extraction]],
        extra: Callable[[RegionTask], str] = lambda t: "",
    ) -> list[Extraction]:
        results: list[Extraction | None] = [None] * len(tasks)
        keys: list[str | None] = [None] * len(tasks)
        misses: list[int] = []
        for i, task in enumerate(tasks):
            if task.crop() is None:
                results[i] = Extraction(method=method, engine=engine, error="region could not be cropped")
                continue
            if self.cache is not None:
                keys[i] = self.cache.key(method.value, engine, task.crop_hash(), extra(task))
                hit = self.cache.get(keys[i])
                if hit is not None:
                    results[i] = Extraction.from_cache(method, engine, hit)
                    continue
            misses.append(i)
        if misses:
            computed = compute([tasks[i] for i in misses])
            for i, ext in zip(misses, computed):
                results[i] = ext
                if self.cache is not None and keys[i] and not ext.error:
                    self.cache.set(keys[i], ext.to_cache())
        return [r for r in results if r is not None]

    # --- method handlers --------------------------------------------------------------

    def _pdf_text(self, tasks: list[RegionTask]) -> list[Extraction]:
        out = []
        for task in tasks:
            keep_breaks = task.region.type in (RegionType.CODE, RegionType.LIST)
            lines = task.page.text_lines(task.region.bbox) if task.region.bbox else []
            text = join_lines(lines, keep_line_breaks=keep_breaks)
            if lines and task.region.type in (RegionType.HEADING, RegionType.TITLE):
                task.region.meta.setdefault("font_size", max(ln.size for ln in lines))
            out.append(Extraction(method=Method.PDF_TEXT, content=text, engine="pymupdf"))
        return out

    def _pdf_table(self, tasks: list[RegionTask]) -> list[Extraction]:
        out = []
        for task in tasks:
            table, task.pdf_table = task.pdf_table, None
            if table is None and task.page.pdf_page is not None and task.region.bbox is not None:
                b = task.region.bbox
                clip = BBox(x0=b.x0 - 3, y0=b.y0 - 3, x1=b.x1 + 3, y1=b.y1 + 3)
                found = pdfmod.find_tables(task.page.pdf_page, clip=clip)
                if found:
                    table = max(found, key=lambda item: item[0].area)[1]
            if table is None:
                out.append(Extraction(method=Method.PDF_TABLE, engine="pymupdf", error="no vector table found"))
                continue
            out.append(self._table_extraction(Method.PDF_TABLE, "pymupdf", table))
        return out

    def _table_extraction(self, method: Method, engine: str, table: Table, **kwargs) -> Extraction:
        return Extraction(
            method=method,
            engine=engine,
            content=table.render(self.settings.table_format),
            html=table.to_html(),
            table=table,
            **kwargs,
        )

    def _ocr(self, tasks: list[RegionTask]) -> list[Extraction]:
        engine = self.engines.ocr
        assert engine is not None

        def compute(batch: list[RegionTask]) -> list[Extraction]:
            crops = [t.crop() for t in batch]
            results = engine.recognize(crops)
            # Test-time agreement: a second reading at another scale. Stable text reads the same; blur,
            # unknown glyphs or handwriting change between readings. Works for every language.
            second = engine.recognize([_rescaled(c) for c in crops]) if self.settings.ocr_agreement_check else None
            out = []
            for i, result in enumerate(results):
                trusted = " ".join(ln.text for ln in result.lines if ln.score >= 0.9)
                data = {"trusted_text": trusted, "line_scores": [round(ln.score, 4) for ln in result.lines]}
                if second is not None:
                    data["agreement"] = round(text_agreement(result.text, second[i].text), 4)
                out.append(
                    Extraction(
                        method=Method.OCR, engine=engine.name, content=result.text, confidence=result.confidence, data=data
                    )
                )
            return out

        agree = "agree" if self.settings.ocr_agreement_check else ""
        return self._batched(Method.OCR, engine.name, tasks, compute, lambda t: agree)

    def _table_recognition(self, tasks: list[RegionTask]) -> list[Extraction]:
        engine = self.engines.table
        assert engine is not None

        def compute(batch: list[RegionTask]) -> list[Extraction]:
            out = []
            for result in engine.recognize([t.crop() for t in batch]):
                table = parse_html_table(result.html) if result.html else None
                if table is None:
                    out.append(Extraction(method=Method.TABLE_RECOGNITION, engine=engine.name, content="", confidence=0.0))
                    continue
                trusted = result.ocr_text if (result.confidence or 0) >= 0.9 else ""
                out.append(
                    self._table_extraction(
                        Method.TABLE_RECOGNITION,
                        engine.name,
                        table,
                        confidence=result.confidence,
                        data={"trusted_text": trusted} if trusted else None,
                    )
                )
            return out

        return self._batched(Method.TABLE_RECOGNITION, engine.name, tasks, compute)

    def _formula(self, tasks: list[RegionTask]) -> list[Extraction]:
        engine = self.engines.formula
        assert engine is not None

        def compute(batch: list[RegionTask]) -> list[Extraction]:
            return [
                Extraction(method=Method.FORMULA_RECOGNITION, engine=engine.name, content=r.latex, confidence=r.confidence)
                for r in engine.recognize([t.crop() for t in batch])
            ]

        return self._batched(Method.FORMULA_RECOGNITION, engine.name, tasks, compute)

    def _vlm_task(self, task: RegionTask) -> str:
        return VLM_TASKS[task.region.type]

    def _vlm(self, tasks: list[RegionTask]) -> list[Extraction]:
        engine = self.engines.vlm
        assert engine is not None
        self._collect_ocr_evidence(tasks)

        language = self.language

        def hint(task: RegionTask) -> str | None:
            if task.region.type in FIGURE_TYPES:
                return task.evidence.get("figure_text") or None
            return task.evidence.get("text_layer") or None

        def compute(batch: list[RegionTask]) -> list[Extraction]:
            requests = [
                VlmRequest(image=t.crop(), task=self._vlm_task(t), hint=hint(t), language=language) for t in batch
            ]
            start = time.perf_counter()
            results = engine.run(requests)
            elapsed = (time.perf_counter() - start) * 1000
            self.stats.add(
                vlm_calls=len(results),
                vlm_input_tokens=sum(r.input_tokens for r in results),
                vlm_output_tokens=sum(r.output_tokens for r in results),
            )
            out = []
            for task, request, result in zip(batch, requests, results):
                ext = self._vlm_extraction(task, request.task, result.text, result.data, engine.name)
                ext.duration_ms = elapsed  # requests run concurrently: wall time of the batch
                out.append(ext)
            return out

        def extra(task: RegionTask) -> str:
            text = hint(task) or ""
            return f"{self._vlm_task(task)}|{PROMPT_VERSION}|{language}|{hashlib.sha1(text.encode()).hexdigest()}"

        return self._batched(Method.VLM, engine.name, tasks, compute, extra)

    def _vlm_extraction(self, task: RegionTask, kind: str, text: str, data: dict | None, engine: str) -> Extraction:
        if kind == "table":
            table = parse_html_table(text) or parse_markdown_table(text)
            if table is None:
                return Extraction(method=Method.VLM, engine=engine, content=text, error="VLM returned no table")
            return self._table_extraction(Method.VLM, engine, table)
        if kind == "formula":
            latex = text.strip().strip("$").strip()
            latex = latex.removeprefix("\\[").removesuffix("\\]").strip()
            return Extraction(method=Method.VLM, engine=engine, content=latex)
        if kind == "chart":
            info = {"kind": "chart", "description": chart_markdown(data, ""), "lossless": (data or {}).get("lossless")}
            return Extraction(
                method=Method.VLM,
                engine=engine,
                content=chart_markdown(data, text),
                data={"chart": data, "figure": info} if data else None,
            )
        if kind == "image":
            description = str((data or {}).get("description") or "").strip()
            return Extraction(
                method=Method.VLM,
                engine=engine,
                content=description or text,
                data={"figure": data} if data else None,
            )
        return Extraction(method=Method.VLM, engine=engine, content=text)

    def _collect_ocr_evidence(self, tasks: list[RegionTask]) -> None:
        """OCR scanned tables that have no evidence yet, so the VLM's numbers can be cross-checked."""
        if not (self.settings.verify_vlm_with_ocr and self.engines.ocr is not None):
            return
        need = [
            t
            for t in tasks
            if t.region.type == RegionType.TABLE
            and not t.evidence.get("text_layer")
            and not t.evidence.get("trusted_text")
        ]
        if not need:
            return
        try:
            for task, ext in zip(need, self._ocr(need)):
                if ext.data and ext.data.get("trusted_text"):
                    task.evidence["trusted_text"] = ext.data["trusted_text"]
        except Exception as exc:  # evidence is optional
            log.warning("OCR evidence collection failed: %s", exc)


def _rescaled(image: Image.Image) -> Image.Image:
    factor = 0.75 if max(image.size) > 2000 else 1.5
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.resize(size, Image.Resampling.BICUBIC)


def text_agreement(a: str, b: str) -> float:
    """Similarity of two readings, ignoring whitespace (1 = identical)."""
    a, b = "".join(a.split()), "".join(b.split())
    if not a and not b:
        return 1.0
    return 1.0 - normalized_distance(a, b)


def chart_markdown(data: dict | None, raw: str) -> str:
    """Description plus data table from the VLM's chart JSON; raw text if it was not JSON."""
    if not data:
        return raw.strip()
    parts = []
    title = str(data.get("title") or "").strip()
    chart_type = str(data.get("chart_type") or "").strip()
    if title or chart_type:
        parts.append(" — ".join(p for p in (title, chart_type) if p))
    description = str(data.get("description") or "").strip()
    if description:
        parts.append(description)
    columns = data.get("columns") or []
    rows = data.get("rows") or []
    if isinstance(columns, list) and isinstance(rows, list) and rows:
        grid: list[list[str | None]] = []
        if columns:
            grid.append([str(c) for c in columns])
        for row in rows:
            if isinstance(row, dict):
                values = [str(row.get(c, "")) for c in columns] if columns else [str(v) for v in row.values()]
            elif isinstance(row, list):
                values = [str(v) for v in row]
            else:
                values = [str(row)]
            grid.append(values)
        width = max(len(r) for r in grid)
        grid = [r + [""] * (width - len(r)) for r in grid]
        parts.append(table_from_rows(grid, header=bool(columns)).to_markdown())
    return "\n\n".join(parts) if parts else raw.strip()
