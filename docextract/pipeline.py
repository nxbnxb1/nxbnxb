"""Document pipeline: preprocessing → layout → routing → extraction/validation → merge → output."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .cache import ResultCache
from .config import Settings
from .engines import Engines, build_engines
from .executor import Executor, PageContext, RegionTask, RunStats
from .layout.heuristic import classify_text_block, heuristic_layout
from .layout.labels import Candidate, clean_candidates, normalize_label
from .layout.reading_order import reading_order
from .merge import assign_heading_levels, render_markdown
from .models import (
    BBox,
    DocumentResult,
    DocumentStats,
    Method,
    PageKind,
    PageResult,
    Region,
    RegionType,
    SourceInfo,
    SourceRef,
    ValidationStatus,
)
from .preprocessing import image as imgmod
from .preprocessing import pdf as pdfmod
from .preprocessing.docx import DocxReader
from .preprocessing.loader import detect_format
from .preprocessing.office import convert, office_target
from .router import RuleBasedRouter

log = logging.getLogger(__name__)


@dataclass
class ExtractOptions:
    """Per-request overrides."""

    pages: list[int] | None = None  # 1-based page numbers
    use_vlm: bool = True
    describe_images: bool | None = None


def parse_pages(spec: str | None) -> list[int] | None:
    """'1-3,5' → [1, 2, 3, 5]."""
    if not spec:
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return sorted(p for p in pages if p > 0)


class DocumentPipeline:
    def __init__(self, settings: Settings | None = None, engines: Engines | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.engines = engines if engines is not None else build_engines(self.settings)
        self.cache = ResultCache(self.settings.cache_dir) if self.settings.cache_dir else None

    # --- public API --------------------------------------------------------------------

    def process_file(self, path: str | Path, options: ExtractOptions | None = None) -> DocumentResult:
        path = Path(path)
        return self.process_bytes(path.read_bytes(), path.name, options)

    def process_bytes(self, data: bytes, filename: str, options: ExtractOptions | None = None) -> DocumentResult:
        options = options or ExtractOptions()
        start = time.perf_counter()
        settings = self.settings
        if options.describe_images is not None:
            settings = settings.model_copy(update={"describe_images": options.describe_images})
        engines = self.engines if options.use_vlm else dataclasses.replace(self.engines, vlm=None)
        stats = RunStats()
        router = RuleBasedRouter(settings, engines)
        executor = Executor(settings, engines, router, self.cache, stats)
        run = _Run(settings, engines, executor, filename)

        fmt = detect_format(data, filename)
        original = None
        if fmt == "office":  # one converter for every other format
            target = office_target(filename)
            data, fmt, original = convert(data, filename, target), target, Path(filename).suffix.lower()
        if fmt == "pdf":
            pages, page_count = run.pdf(data, options.pages)
        elif fmt == "image":
            pages, page_count = run.images(data, options.pages)
        else:
            pages, page_count = run.docx(data), 1

        result = DocumentResult(
            source=SourceInfo(
                filename=filename,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                format=fmt,
                page_count=page_count,
                converted_from=original,
            ),
            pages=pages,
            engines={**engines.describe(), "product": settings.product, "router": router.name, "docextract": __version__},
        )
        stem = Path(filename).stem or "document"
        for region in result.iter_regions():
            png = executor.figures.get(region.id)
            if png is not None:
                region.figure = f"{stem}_assets/{region.id}.png"
                result.figures[region.figure] = png
        assign_heading_levels(list(result.iter_regions()))
        result.markdown = render_markdown(result, settings)
        result.stats = _stats(result, stats, settings, (time.perf_counter() - start) * 1000)
        return result


class _Run:
    """State of one document run."""

    def __init__(self, settings: Settings, engines: Engines, executor: Executor, filename: str) -> None:
        self.settings = settings
        self.engines = engines
        self.executor = executor
        self.filename = filename

    # --- PDF -----------------------------------------------------------------------------

    def pdf(self, data: bytes, wanted: list[int] | None) -> tuple[list[PageResult], int]:
        doc = pdfmod.open_pdf(data)
        try:
            page_count = doc.page_count
            numbers = [n for n in (wanted or range(1, page_count + 1)) if 1 <= n <= page_count]
            if self.settings.max_pages:
                numbers = numbers[: self.settings.max_pages]
            results = []
            for chunk in _chunks(numbers, self.settings.page_batch_size):
                contexts = [self._pdf_page(doc, n) for n in chunk]
                results.extend(self._process_pages(contexts))
            return results, page_count
        finally:
            with pdfmod.PDF_LOCK:
                doc.close()

    def _pdf_page(self, doc, number: int) -> PageContext:
        s = self.settings
        page = doc[number - 1]
        analysis = pdfmod.analyze_page(page, s.min_text_chars, s.pdf_text_max_garbled_ratio, s.trust_ocr_text_layer)
        with pdfmod.PDF_LOCK:
            unrotated = page.rect * page.derotation_matrix
            rotation = page.rotation
        meta = {
            "text_chars": analysis.text_chars,
            "garbled_ratio": analysis.garbled,
            "image_coverage": analysis.image_coverage,
        }
        if analysis.legacy_fonts:
            meta["legacy_vietnamese_fonts"] = analysis.legacy_fonts[:5]
        if rotation:
            meta["pdf_rotation"] = rotation
        if analysis.kind == PageKind.DIGITAL:
            matrix = pdfmod.pixel_matrix(page, s.dpi)
            return PageContext(
                number=number,
                kind=PageKind.DIGITAL,
                width=unrotated.width,
                height=unrotated.height,
                to_px_matrix=(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f),
                image_loader=lambda: pdfmod.render_page(page, s.dpi),
                lines=pdfmod.page_lines(page),
                pdf_page=page,
                preprocessing=meta,
            )
        # Scanned page: bbox units are points of the cleaned (rotated/deskewed) page image.
        image, ops = imgmod.cleanup(pdfmod.render_page(page, s.dpi), s, self.engines.orientation)
        meta.update(ops)
        scale = s.dpi / 72
        return PageContext(
            number=number,
            kind=PageKind.SCANNED,
            width=image.width / scale,
            height=image.height / scale,
            to_px_matrix=(scale, 0.0, 0.0, scale, 0.0, 0.0),
            image_loader=lambda img=image: img,
            preprocessing=meta,
        )

    # --- images --------------------------------------------------------------------------

    def images(self, data: bytes, wanted: list[int] | None) -> tuple[list[PageResult], int]:
        frames = imgmod.load_image_pages(data)
        numbers = [n for n in (wanted or range(1, len(frames) + 1)) if 1 <= n <= len(frames)]
        if self.settings.max_pages:
            numbers = numbers[: self.settings.max_pages]
        results = []
        for chunk in _chunks(numbers, self.settings.page_batch_size):
            contexts = []
            for n in chunk:
                image, ops = imgmod.cleanup(frames[n - 1], self.settings, self.engines.orientation, resize=True)
                contexts.append(
                    PageContext(
                        number=n,
                        kind=PageKind.IMAGE,
                        width=float(image.width),
                        height=float(image.height),
                        unit="px",
                        image_loader=lambda img=image: img,
                        preprocessing=ops,
                    )
                )
            results.extend(self._process_pages(contexts))
        return results, len(frames)

    # --- shared page processing -------------------------------------------------------

    def _process_pages(self, contexts: list[PageContext]) -> list[PageResult]:
        start = time.perf_counter()
        candidates = self._layout(contexts)
        page_tasks: list[list[RegionTask]] = []
        for ctx, cands in zip(contexts, candidates):
            page_tasks.append(self._tasks_for_page(ctx, cands))
        self.executor.run([t for tasks in page_tasks for t in tasks])
        per_page_ms = (time.perf_counter() - start) * 1000 / max(1, len(contexts))
        results = []
        for ctx, tasks in zip(contexts, page_tasks):
            results.append(
                PageResult(
                    number=ctx.number,
                    kind=ctx.kind,
                    width=round(ctx.width, 2),
                    height=round(ctx.height, 2),
                    unit=ctx.unit,
                    regions=[t.region for t in tasks],
                    preprocessing=ctx.preprocessing,
                    layout_backend=ctx.layout_backend,
                    duration_ms=round(per_page_ms, 1),
                )
            )
            ctx.release()
        return results

    def _layout(self, contexts: list[PageContext]) -> list[list[Candidate]]:
        s = self.settings
        use_model = self.engines.layout is not None and s.layout_backend != "heuristic"
        out: list[list[Candidate] | None] = [None] * len(contexts)
        model_idx = [i for i, ctx in enumerate(contexts) if use_model]
        if model_idx:
            try:
                detections = self.engines.layout.detect([contexts[i].image for i in model_idx])
            except Exception:
                log.exception("layout detection failed; falling back to heuristics")
                detections = None
            if detections is not None:
                for i, boxes in zip(model_idx, detections):
                    ctx = contexts[i]
                    ctx.layout_backend = self.engines.layout.name
                    cands = [
                        Candidate(normalize_label(b.label), ctx.from_px(b.bbox), round(b.score, 4), b.label)
                        for b in boxes
                    ]
                    if ctx.kind == PageKind.DIGITAL and s.recover_orphan_text:
                        cands += self._orphan_text(ctx, cands)
                    out[i] = cands
        for i, ctx in enumerate(contexts):
            if out[i] is not None:
                continue
            if ctx.kind == PageKind.DIGITAL:
                ctx.layout_backend = "pdf-heuristic"
                out[i] = heuristic_layout(ctx.pdf_page, ctx.lines, ctx.number or 1)
            else:
                ctx.layout_backend = "full-page"
                full = BBox(x0=0, y0=0, x1=ctx.width, y1=ctx.height)
                out[i] = [Candidate(RegionType.TEXT, full, None, "full_page", meta={"full_page": True})]
        return [
            clean_candidates(cands, ctx.area, s.min_region_area_ratio) if ctx.layout_backend != "full-page" else cands
            for ctx, cands in zip(contexts, out)
        ]

    def _orphan_text(self, ctx: PageContext, cands: list[Candidate]) -> list[Candidate]:
        """PDF text no layout box covers (missed footnotes, small labels) becomes extra text regions."""
        orphans: dict[int, list[pdfmod.PdfLine]] = {}
        for line in ctx.lines:
            if not any(c.bbox.contains_point(line.cx, line.cy) for c in cands):
                orphans.setdefault(line.block, []).append(line)
        extra = []
        body = pdfmod.body_font_size(ctx.lines)
        largest = max((ln.size for ln in ctx.lines), default=0.0)
        figures = [c.bbox for c in cands if c.type in (RegionType.TABLE, RegionType.IMAGE, RegionType.CHART)]
        for lines in orphans.values():
            text = " ".join(ln.text for ln in lines).strip()
            if len(text) < 2:
                continue
            bbox = BBox(
                x0=min(ln.x0 for ln in lines),
                y0=min(ln.y0 for ln in lines),
                x1=max(ln.x1 for ln in lines),
                y1=max(ln.y1 for ln in lines),
            )
            rtype, meta = classify_text_block(lines, bbox, ctx.height, body, ctx.number or 1, figures, largest)
            meta["recovered"] = True
            extra.append(Candidate(rtype, bbox, None, "orphan_text", meta=meta))
        return extra

    def _tasks_for_page(self, ctx: PageContext, cands: list[Candidate]) -> list[RegionTask]:
        order = reading_order([c.bbox for c in cands], [c.type for c in cands], ctx.width)
        tasks = []
        for rank, idx in enumerate(order):
            cand = cands[idx]
            meta = dict(cand.meta)
            pdf_table = meta.pop("pdf_table", None)
            if pdf_table is not None:
                meta["known_cells"] = pdf_table.n_cells
            region = Region(
                id=f"p{ctx.number}-r{rank + 1}",
                page=ctx.number,
                type=cand.type,
                order=rank,
                bbox=_round_bbox(cand.bbox),
                layout_score=cand.score,
                source=SourceRef(file=self.filename, page=ctx.number, bbox=_round_bbox(cand.bbox)),
                meta=meta,
            )
            if cand.label:
                region.meta["layout_label"] = cand.label
            tasks.append(RegionTask(region=region, page=ctx, pdf_table=pdf_table))
        for task in tasks:
            self.executor.prepare(task)
        return tasks

    # --- DOCX ----------------------------------------------------------------------------

    def docx(self, data: bytes) -> list[PageResult]:
        start = time.perf_counter()
        blocks = DocxReader(data).blocks()
        ctx = PageContext(number=None, kind=PageKind.DOCX, width=0.0, height=0.0, unit="none")
        tasks: list[RegionTask] = []
        regions: list[Region] = []
        for i, block in enumerate(blocks):
            region = Region(
                id=f"d-r{i + 1}",
                page=None,
                type=block.type,
                order=i,
                level=block.level,
                source=SourceRef(file=self.filename, locator=block.locator),
            )
            regions.append(region)
            if block.type == RegionType.IMAGE and block.image is not None:
                region.content = block.content  # alt text until described
                if block.content:
                    region.meta["alt_text"] = block.content
                tasks.append(RegionTask(region=region, page=ctx, image=block.image))
                continue
            region.method = Method.DOCX
            region.engine = "python-docx"
            region.status = ValidationStatus.PASSED
            if block.table is not None:
                region.html = block.table.to_html()
                region.content = block.table.render(self.settings.table_format)
            else:
                region.content = block.content
            if block.type == RegionType.FORMULA:
                region.status = ValidationStatus.UNCHECKED
                region.issues = ["Word equation exported as linear text"]
        self.executor.run(tasks)
        for task in tasks:
            if task.region.method == Method.NONE and task.region.meta.get("alt_text"):
                task.region.content = task.region.meta["alt_text"]
        return [
            PageResult(
                number=None,
                kind=PageKind.DOCX,
                unit="none",
                regions=regions,
                layout_backend="python-docx",
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
        ]


def _round_bbox(bbox: BBox) -> BBox:
    return BBox(x0=round(bbox.x0, 2), y0=round(bbox.y0, 2), x1=round(bbox.x1, 2), y1=round(bbox.y1, 2))


def _chunks(items: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _stats(result: DocumentResult, run: RunStats, settings: Settings, duration_ms: float) -> DocumentStats:
    regions = list(result.iter_regions())
    pages = max(1, len(result.pages))
    cost = (
        run.vlm_input_tokens * settings.vlm_price_input_per_1m + run.vlm_output_tokens * settings.vlm_price_output_per_1m
    ) / 1_000_000
    return DocumentStats(
        pages_processed=len(result.pages),
        regions=len(regions),
        regions_by_type=dict(Counter(r.type.value for r in regions)),
        regions_by_method=dict(Counter(r.method.value for r in regions)),
        regions_by_status=dict(Counter(r.status.value for r in regions)),
        escalations=sum(1 for r in regions if len(r.attempts) > 1),
        vlm_calls=run.vlm_calls,
        vlm_input_tokens=run.vlm_input_tokens,
        vlm_output_tokens=run.vlm_output_tokens,
        estimated_cost=round(cost, 6),
        cache_hits=run.cache_hits,
        duration_ms=round(duration_ms, 1),
        ms_per_page=round(duration_ms / pages, 1),
        cost_per_page=round(cost / pages, 6),
    )

