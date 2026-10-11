"""Phase-2 benchmark: accuracy, structure, speed and cost on documents with ground truth.

Dataset layout: any PDF/DOCX/image ``<name>.<ext>`` next to ``<name>.gt.md`` (the expected
Markdown), in any sub-directory: the directory path is the document's category, and a first
level ``dev/`` or ``test/`` its split (tune on dev only; report test). Metrics per document,
per category and overall:

* CER / WER on plain text (Markdown markup stripped)
* word F1: bags of words, independent of reading order
* TEDS / TEDS-S for tables (HTML or pipe tables found in the Markdown)
* heading F1 (structure)
* ms/page, VLM calls/page, cost/page, share of regions flagged for review
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

from .config import Settings
from .metrics import best_match_teds, cer, cer_any_order, heading_f1, teds, wer, word_f1
from .models import DocumentStats
from .pipeline import DocumentPipeline, ExtractOptions
from .tables import Table, parse_html_tables, parse_markdown_table

_DOC_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def markdown_tables(markdown: str) -> list[Table]:
    tables = parse_html_tables(markdown)
    block: list[str] = []
    for line in markdown.splitlines() + [""]:
        if line.strip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2:
            table = parse_markdown_table("\n".join(block))
            if table is not None:
                tables.append(table)
        block = []
    return tables


def markdown_headings(markdown: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"^#{1,6}\s+(.+?)\s*$", markdown, flags=re.M)]


def document_text(markdown: str) -> str:
    """The document's own content: pictures, their generated descriptions and the text read in
    them are left out, since ground truths do not transcribe pictures."""
    blocks = [b for b in markdown.split("\n\n") if not b.lstrip().startswith(("![", "> **["))]
    return "\n\n".join(blocks)


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


SPLITS = ("dev", "test")
METRICS = ("cer", "cer_any_order", "wer", "word_f1", "teds", "teds_structure", "heading_f1", "ms_per_page",
           "vlm_calls_per_page", "cost_per_page", "needs_review_ratio")


def documents(dataset: Path, split: str | None = None):
    """(document, ground truth, category, split) for every document with a ground truth."""
    for doc in sorted(dataset.rglob("*")):
        if not doc.is_file() or doc.suffix.lower() not in _DOC_SUFFIXES:
            continue
        gt_path = doc.with_name(doc.stem + ".gt.md")
        if not gt_path.exists():
            continue
        parts = doc.parent.relative_to(dataset).parts
        doc_split = parts[0] if parts and parts[0] in SPLITS else None
        if split and doc_split != split:
            continue
        category = "/".join(parts[1:] if doc_split else parts) or "-"
        yield doc, gt_path, category, doc_split


def run_benchmark(
    dataset: Path,
    use_vlm: bool = True,
    pipeline: DocumentPipeline | None = None,
    split: str | None = None,
    warmup: bool = True,
    system=None,
    system_name: str | None = None,
) -> dict:
    """Score one conversion system (default: this pipeline) on every document with a ground truth."""
    from .baselines import Converted, DocextractSystem

    if system is None:
        system = DocextractSystem(Settings.from_env(), pipeline, use_vlm)
    docs = list(documents(dataset, split))
    if warmup:  # ms/page measures processing, not model loading
        system.warm_up()
    rows = []
    for number, (doc, gt_path, category, doc_split) in enumerate(docs, 1):
        gt = gt_path.read_text(encoding="utf-8")
        try:
            converted, error = system.convert(doc), None
        except Exception as exc:  # out of memory, a crash: this document scores as empty output
            error = f"{type(exc).__name__}: {exc}"[:300]
            converted = Converted("", DocumentStats(pages_processed=_page_count(doc)))
        pred = document_text(converted.markdown or "")
        gt_tables, pred_tables = markdown_tables(gt), markdown_tables(pred)
        table_scores = best_match_teds(gt_tables, pred_tables)
        structure_scores = [
            max((teds(ref, p, structure_only=True) for p in pred_tables), default=0.0) for ref in gt_tables
        ]
        gt_headings = markdown_headings(gt)
        s = converted.stats
        rows.append(
            {
                "document": doc.relative_to(dataset).as_posix(),
                "category": category,
                "split": doc_split,
                "pages": s.pages_processed,
                "cer": round(cer(gt, pred), 4),
                "cer_any_order": round(cer_any_order(gt, pred), 4),
                "wer": round(wer(gt, pred), 4),
                "word_f1": round(word_f1(gt, pred), 4),
                "teds": _mean(table_scores),
                "teds_structure": _mean(structure_scores),
                # only where the ground truth marks headings: some sources do not annotate them
                "heading_f1": round(heading_f1(gt_headings, markdown_headings(pred)), 4) if gt_headings else None,
                "ms_per_page": None if error else s.ms_per_page,
                "vlm_calls_per_page": round(s.vlm_calls / max(1, s.pages_processed), 3),
                "cost_per_page": s.cost_per_page,
                # only systems that flag uncertain regions report it
                "needs_review_ratio": round(s.regions_by_status.get("needs_review", 0) / s.regions, 4) if s.regions else None,
                "methods": s.regions_by_method,
                "error": error,
            }
        )
        print(f"[{number}/{len(docs)}] {rows[-1]['document']}: CER {rows[-1]['cer']}, "
              f"{rows[-1]['ms_per_page'] or '-'} ms/page" + (f", failed: {error}" if error else ""),
              file=sys.stderr, flush=True)
    summary = _summary(rows)
    categories = {c: _summary([r for r in rows if r["category"] == c]) for c in sorted({r["category"] for r in rows})}
    return {
        "system": system_name or getattr(system, "name", "system"),
        "documents": rows,
        "summary": summary,
        "categories": categories,
        "markdown": _report(rows, summary, categories),
    }


def _page_count(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    import pymupdf

    try:
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:
        return 1


def _summary(rows: list[dict]) -> dict:
    out: dict = {"documents": len(rows), "pages": sum(r["pages"] or 0 for r in rows),
                 "failed": sum(1 for r in rows if r.get("error"))}
    out.update({key: _mean([r[key] for r in rows if r[key] is not None]) for key in METRICS})
    return out


def _cell(value) -> str:
    return "-" if value is None else str(value)


def _report(rows: list[dict], summary: dict, categories: dict[str, dict]) -> str:
    cols = ["documents", "pages", "failed", "cer", "cer_any_order", "word_f1", "teds", "heading_f1", "ms_per_page",
            "vlm_calls_per_page",
            "needs_review_ratio"]
    lines = ["| category | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, values in categories.items():
        lines.append(f"| {name} | " + " | ".join(_cell(values[c]) for c in cols) + " |")
    lines.append("| **all** | " + " | ".join(_cell(summary[c]) for c in cols) + " |")
    cols = ["document", "pages", "cer", "cer_any_order", "wer", "word_f1", "teds", "heading_f1", "ms_per_page",
            "vlm_calls_per_page",
            "cost_per_page", "needs_review_ratio"]
    lines += ["", "<details><summary>Per document</summary>", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row[c]) for c in cols) + " |")
    lines += ["", "</details>"]
    failed = [r for r in rows if r.get("error")]
    if failed:
        lines += ["", "Failed (scored as empty output):", ""] + [f"- {r['document']}: {r['error']}" for r in failed]
    return "\n".join(lines) + "\n"


# --- baseline vs candidate ----------------------------------------------------------------

# (key, label, higher is better)
COMPARED = [
    ("cer", "CER", False),
    ("cer_any_order", "CER (any order)", False),
    ("word_f1", "word F1", True),
    ("teds", "TEDS", True),
    ("heading_f1", "heading F1", True),
    ("ms_per_page", "ms/page", False),
    ("needs_review_ratio", "needs review", False),
    ("failed", "failed documents", False),
]


def compare_reports(reports: list[dict], names: list[str] | None = None) -> str:
    """One table per category: every system's metrics on the same documents, best value in bold."""
    names = names or [r.get("system", f"system {i + 1}") for i, r in enumerate(reports)]
    header = "| system | " + " | ".join(label for _, label, _ in COMPARED) + " |"
    rule = "|" + "---|" * (len(COMPARED) + 1)
    lines = [
        "Lower is better for CER, ms/page, needs review and failed documents; higher for the rest. CER (any order) "
        "compares each line of the ground truth with the closest part of the output, so reading order does not count. "
        "Best value of each column in bold.",
        "",
    ]
    categories = sorted({c for r in reports for c in (r.get("categories") or {})})
    sections = [(c, [(r.get("categories") or {}).get(c, {}) for r in reports]) for c in categories]
    sections.append(("all documents", [r["summary"] for r in reports]))
    for title, values in sections:
        docs = max((v.get("documents") or 0) for v in values)
        lines += [f"**{title}** ({docs} documents)", "", header, rule]
        best = {}
        for key, _, higher in COMPARED:
            present = [v.get(key) for v in values if v.get(key) is not None]
            best[key] = (max if higher else min)(present) if present else None
        for name, v in zip(names, values):
            cells = []
            for key, _, _ in COMPARED:
                value = v.get(key)
                cells.append(f"**{value}**" if value is not None and value == best[key] and len(values) > 1 else _cell(value))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
