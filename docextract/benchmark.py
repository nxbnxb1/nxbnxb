"""Phase-2 benchmark: accuracy, structure, speed and cost on documents with ground truth.

Dataset layout: any PDF/DOCX/image ``<name>.<ext>`` next to ``<name>.gt.md`` (the expected
Markdown). Metrics per document and averaged:

* CER / WER on plain text (Markdown markup stripped)
* TEDS / TEDS-S for tables (HTML or pipe tables found in the Markdown)
* heading F1 (structure)
* ms/page, VLM calls/page, cost/page, share of regions flagged for review
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

from .config import Settings
from .metrics import best_match_teds, cer, heading_f1, teds, wer
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


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def run_benchmark(dataset: Path, use_vlm: bool = True, pipeline: DocumentPipeline | None = None) -> dict:
    pipeline = pipeline or DocumentPipeline(Settings.from_env())
    rows = []
    for doc in sorted(p for p in dataset.iterdir() if p.suffix.lower() in _DOC_SUFFIXES):
        gt_path = doc.with_name(doc.stem + ".gt.md")
        if not gt_path.exists():
            continue
        gt = gt_path.read_text(encoding="utf-8")
        result = pipeline.process_file(doc, ExtractOptions(use_vlm=use_vlm))
        pred = result.markdown or ""
        gt_tables, pred_tables = markdown_tables(gt), markdown_tables(pred)
        table_scores = best_match_teds(gt_tables, pred_tables)
        structure_scores = [
            max((teds(ref, p, structure_only=True) for p in pred_tables), default=0.0) for ref in gt_tables
        ]
        s = result.stats
        rows.append(
            {
                "document": doc.name,
                "pages": s.pages_processed,
                "cer": round(cer(gt, pred), 4),
                "wer": round(wer(gt, pred), 4),
                "teds": _mean(table_scores),
                "teds_structure": _mean(structure_scores),
                "heading_f1": round(heading_f1(markdown_headings(gt), markdown_headings(pred)), 4),
                "ms_per_page": s.ms_per_page,
                "vlm_calls_per_page": round(s.vlm_calls / max(1, s.pages_processed), 3),
                "cost_per_page": s.cost_per_page,
                "needs_review_ratio": round(s.regions_by_status.get("needs_review", 0) / max(1, s.regions), 4),
                "methods": s.regions_by_method,
            }
        )
    summary = {
        key: _mean([r[key] for r in rows if r[key] is not None])
        for key in ("cer", "wer", "teds", "teds_structure", "heading_f1", "ms_per_page", "vlm_calls_per_page", "cost_per_page", "needs_review_ratio")
    }
    return {"documents": rows, "summary": summary, "markdown": _report(rows, summary)}


def _report(rows: list[dict], summary: dict) -> str:
    cols = ["document", "pages", "cer", "wer", "teds", "heading_f1", "ms_per_page", "vlm_calls_per_page", "cost_per_page", "needs_review_ratio"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join("-" if row[c] is None else str(row[c]) for c in cols) + " |")
    lines.append("| **mean** | | " + " | ".join("-" if summary[c] is None else str(summary[c]) for c in cols[2:]) + " |")
    return "\n".join(lines) + "\n"
