"""Quality gate before a fine-tuned model is published (see .github/workflows/_train.yml).

    python gate.py report.json --model "NAME (fine-tuned)" --baseline BASE_MODEL

Passes when, on every set of lines cut from real documents of the test companies (eval_real_*),
the fine-tuned model's CER is at most the baseline's plus --tolerance. Synthetic line sets do not
decide: a model can beat its baseline on them and lose on real documents (vi_en_ja-ocr-2 did).
Without real-document sets the per-language synthetic sets (eval_<lang>) decide. Sets with fewer
than --min-lines lines are reported but do not decide. Exit status 0 = pass, 1 = fail.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def gate(results: dict, model: str, baseline: str, tolerance: float, min_lines: int) -> tuple[bool, list[str]]:
    real = sorted(name for name in results if name.startswith("eval_real"))
    sets = real or sorted(name for name in results if re.fullmatch(r"eval_[a-z]{2}", name))
    lines = ["| set | lines | fine-tuned CER | baseline CER | verdict |", "|---|---|---|---|---|"]
    ok, decided = True, 0
    for name in sets:
        mine, base = results[name].get(model, {}), results[name].get(baseline, {})
        if "cer" not in mine or "cer" not in base:
            lines.append(f"| {name} | - | {mine.get('error', '-')} | {base.get('error', '-')} | not evaluated |")
            ok = False  # a set the model could not be measured on is no evidence it is good
            continue
        if mine.get("lines", 0) < min_lines:
            verdict = "too few lines"
        elif mine["cer"] <= base["cer"] + tolerance:
            verdict, decided = "ok", decided + 1
        else:
            verdict, ok, decided = "worse", False, decided + 1
        lines.append(f"| {name} | {mine.get('lines', '-')} | {mine['cer']:.2%} | {base['cer']:.2%} | {verdict} |")
    if not decided:
        ok = False
        lines.append("")
        lines.append("No evaluation set decided (none, or all too small).")
    return ok, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="JSON written by evaluate.py (--json, or next to --report)")
    parser.add_argument("--model", required=True, help="label of the fine-tuned model in the report")
    parser.add_argument("--baseline", required=True, help="label of the baseline in the report")
    parser.add_argument("--tolerance", type=float, default=0.005, help="CER the model may lose (absolute)")
    parser.add_argument("--min-lines", type=int, default=100)
    args = parser.parse_args()
    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    ok, lines = gate(data.get("results", data), args.model, args.baseline, args.tolerance, args.min_lines)
    print(f"## Quality gate: {'passed' if ok else 'failed'}\n")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
