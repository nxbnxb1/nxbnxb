"""Validation of extraction results: decides whether to accept a result or escalate."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import Settings
from .models import TEXT_LIKE_TYPES, Method, RegionType, ValidationStatus
from .tables import Table
from .textutil import (
    extract_numbers,
    garbled_ratio,
    multiset_recall,
    special_chars,
    vietnamese_dropout_ratio,
)

if TYPE_CHECKING:
    from .executor import Extraction, RegionTask

_REFUSAL_RE = re.compile(
    r"^(i'?m sorry|sorry,|i cannot|i can't|i am unable|as an ai|unable to (read|process))", re.IGNORECASE
)
_OCR_METHODS = frozenset({Method.OCR, Method.TABLE_RECOGNITION})


@dataclass
class ValidationReport:
    passed: bool  # stop trying further methods
    status: ValidationStatus
    score: float  # 0..1, used to pick the best result when every method fails
    issues: list[str] = field(default_factory=list)


def _fail(issues: list[str], score: float) -> ValidationReport:
    return ValidationReport(False, ValidationStatus.NEEDS_REVIEW, max(0.0, min(1.0, score)), issues)


def _accept(status: ValidationStatus = ValidationStatus.PASSED, issues: list[str] | None = None, score: float = 1.0):
    return ValidationReport(True, status, score, issues or [])


def reference_text(task: RegionTask) -> tuple[str, bool]:
    """Evidence to cross-check against, and whether it is complete (PDF text layer) or partial (trusted OCR)."""
    layer = task.evidence.get("text_layer") or ""
    if layer.strip():
        return layer, True
    return task.evidence.get("trusted_text") or "", False


def check_numbers(candidate: str, task: RegionTask, settings: Settings) -> list[str]:
    reference, complete = reference_text(task)
    ref_numbers = extract_numbers(reference)
    if not ref_numbers:
        return []
    issues = []
    cand_numbers = extract_numbers(candidate)
    recall = multiset_recall(ref_numbers, cand_numbers)
    if recall < settings.number_match_min:
        issues.append(f"numbers missing vs source ({recall:.0%} matched)")
    if complete and cand_numbers:
        precision = multiset_recall(cand_numbers, ref_numbers)
        if precision < settings.number_match_min:
            issues.append(f"numbers not present in source ({precision:.0%} supported)")
    ref_special = special_chars(reference)
    cand_special = special_chars(candidate)
    lost = [ch for ch, n in ref_special.items() if cand_special.get(ch, 0) < n]
    if complete and lost:
        issues.append("special characters lost: " + "".join(lost))
    return issues


def _text_checks(text: str, method: Method, settings: Settings, ext: Extraction) -> list[str]:
    issues = []
    garbled = garbled_ratio(text)
    if garbled > settings.pdf_text_max_garbled_ratio:
        issues.append(f"garbled characters ({garbled:.0%})")
    if method == Method.OCR:
        dropout = vietnamese_dropout_ratio(text)
        if dropout > 0.08:
            issues.append(
                f"Vietnamese letters missing ({dropout:.0%} of words without vowels): "
                "use the Vietnamese recognition model (ocr_rec_model_dir)"
            )
        agreement = (ext.data or {}).get("agreement")
        if agreement is not None and agreement < settings.ocr_min_agreement:
            issues.append(f"two OCR readings disagree ({agreement:.0%} similar)")
        if ext.confidence is not None and ext.confidence < settings.ocr_min_confidence:
            issues.append(f"low OCR confidence ({ext.confidence:.2f})")
        scores = (ext.data or {}).get("line_scores") or []
        if scores:
            low = sum(1 for s in scores if s < settings.ocr_low_line_confidence) / len(scores)
            if low > settings.ocr_max_low_lines_ratio:
                issues.append(f"{low:.0%} of lines below confidence {settings.ocr_low_line_confidence}")
    return issues


def _table_checks(table: Table | None, task: RegionTask, settings: Settings, ext: Extraction) -> list[str]:
    if table is None or table.n_cells == 0:
        return ["no table structure recognised"]
    issues = []
    if table.n_rows < 2 or table.n_cols < 2:
        issues.append(f"degenerate table ({table.n_rows}x{table.n_cols})")
    if not table.is_rectangular:
        issues.append("rows have different numbers of cells")
    if table.empty_ratio > settings.table_max_empty_ratio:
        issues.append(f"{table.empty_ratio:.0%} empty cells")
    complex_table = table.n_cells > settings.complex_table_cells or table.span_count >= 3
    min_conf = settings.table_min_confidence + (0.1 if complex_table else 0.0)
    if ext.confidence is not None and ext.confidence < min_conf:
        issues.append(f"low table confidence ({ext.confidence:.2f}{', complex table' if complex_table else ''})")
    text = table.text()
    if ext.method in _OCR_METHODS or ext.method == Method.VLM:
        dropout = vietnamese_dropout_ratio(text)
        if dropout > 0.08:
            issues.append(f"Vietnamese letters missing ({dropout:.0%} of words without vowels)")
    if ext.method != Method.TABLE_RECOGNITION:  # the table engine's own OCR is not independent evidence
        issues.extend(check_numbers(text, task, settings))
    return issues


def _formula_checks(latex: str) -> list[str]:
    issues = []
    if not latex.strip():
        return ["empty formula"]
    if latex.count("{") != latex.count("}"):
        issues.append("unbalanced braces")
    if latex.count("\\left") != latex.count("\\right"):
        issues.append("unbalanced \\left/\\right")
    if latex.count("\\begin") != latex.count("\\end"):
        issues.append("unbalanced \\begin/\\end")
    tokens = re.findall(r"\\[a-zA-Z]+|[^\s]", latex)
    run = longest = 1
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    if longest >= 10:
        issues.append("repeated tokens (decoder loop)")
    if len(latex) > 3000:
        issues.append("formula implausibly long")
    return issues


def validate(task: RegionTask, ext: Extraction, settings: Settings) -> ValidationReport:
    rtype = task.region.type
    method = ext.method
    if ext.error:
        return _fail([f"{method.value} failed: {ext.error}"], 0.0)

    # Fallback methods that cannot produce the region's real structure: keep the text, flag the region.
    if rtype in (RegionType.TABLE, RegionType.FORMULA) and method in (Method.PDF_TEXT, Method.OCR):
        if not ext.content.strip():
            return _fail([f"{method.value} found no text"], 0.0)
        what = "table structure" if rtype == RegionType.TABLE else "formula"
        return _accept(ValidationStatus.NEEDS_REVIEW, [f"{what} not recovered, plain text kept"], 0.3)
    if rtype in (RegionType.CHART, RegionType.IMAGE) and method in (Method.PDF_TEXT, Method.OCR):
        if not ext.content.strip():
            return _accept(ValidationStatus.SKIPPED, ["no VLM available and no text found"], 0.1)
        return _accept(ValidationStatus.UNCHECKED, ["no VLM description, only text inside the figure"], 0.4)

    if rtype == RegionType.TABLE:
        issues = _table_checks(ext.table, task, settings, ext)
        return _accept() if not issues else _fail(issues, 0.7 - 0.15 * len(issues))

    if rtype == RegionType.FORMULA:
        issues = _formula_checks(ext.content)
        if issues:
            return _fail(issues, 0.6 - 0.15 * len(issues))
        # Formula models give no confidence; a well-formed result is accepted but not "verified".
        return _accept(ValidationStatus.UNCHECKED)

    if rtype in (RegionType.CHART, RegionType.IMAGE):
        text = ext.content.strip()
        if not text or _REFUSAL_RE.match(text):
            return _fail(["VLM returned no usable description"], 0.1)
        return _accept(ValidationStatus.UNCHECKED)

    # Text-like regions and seals.
    text = ext.content
    if not text.strip():
        return _fail([f"{method.value} returned no text"], 0.0)
    issues = _text_checks(text, method, settings, ext)
    if issues:
        base = 0.8 if method == Method.PDF_TEXT else 0.7
        if ext.confidence is not None:
            base = min(base, ext.confidence)
        return _fail(issues, base - 0.1 * len(issues))
    if rtype not in TEXT_LIKE_TYPES and rtype != RegionType.SEAL:
        return _accept(ValidationStatus.UNCHECKED)
    return _accept()
