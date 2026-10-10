"""Quality gate before a fine-tuned OCR model is published (training/vi_ocr/gate.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "vi_ocr"))

from gate import gate  # noqa: E402


def _set(model_cer, base_cer, lines=500):
    return {"ft": {"cer": model_cer, "lines": lines}, "base": {"cer": base_cer, "lines": lines}}


def test_real_document_lines_decide():
    # better on synthetic lines, worse on real documents: not published
    results = {"eval_ja": _set(0.05, 0.06), "eval_real_ja": _set(0.13, 0.10), "eval_real_en": _set(0.02, 0.03)}
    ok, report = gate(results, "ft", "base", 0.005, 100)
    assert not ok and any("worse" in line for line in report)
    assert gate({"eval_real_vi": _set(0.05, 0.09), "eval_real_ja": _set(0.102, 0.10)}, "ft", "base", 0.005, 100)[0]


def test_synthetic_sets_without_real_ones_and_missing_results():
    assert gate({"eval_vi": _set(0.03, 0.13), "eval_long_vi": _set(0.5, 0.1)}, "ft", "base", 0.005, 100)[0]
    assert not gate({"eval_real_vi": {"ft": {"error": "x"}, "base": {"cer": 0.1}}}, "ft", "base", 0.005, 100)[0]
    assert not gate({"eval_real_vi": _set(0.01, 0.1, lines=20)}, "ft", "base", 0.005, 100)[0]  # too few lines
