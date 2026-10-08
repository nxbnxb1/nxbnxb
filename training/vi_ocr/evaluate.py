"""Compare recognition models on labelled line images: CER, line accuracy, Vietnamese letters, speed.

    python evaluate.py --eval data/eval data/eval_clean \
        --model vi_en=export/ --baseline latin_PP-OCRv5_mobile_rec --baseline PP-OCRv6_medium_rec \
        --report report.md
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

_VI_EXTENDED = {chr(c) for c in range(0x1EA0, 0x1EFA)} | set("ăâđêôơưĂÂĐÊÔƠƯ")


def load_labels(directory: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in (directory / "labels.txt").read_text(encoding="utf-8").splitlines():
        rel, _, text = line.partition("\t")
        if rel and text:
            rows.append((directory / rel, unicodedata.normalize("NFC", text)))
    return rows


def recognise(model_name: str, model_dir: str | None, paths: list[Path], batch: int = 64) -> tuple[list[str], float]:
    from paddleocr import TextRecognition

    kwargs = {"model_name": model_name, "enable_mkldnn": False}
    if model_dir:
        kwargs["model_dir"] = model_dir
    model = TextRecognition(**kwargs)
    list(model.predict([str(p) for p in paths[:batch]], batch_size=batch))  # warm-up, not timed
    start = time.perf_counter()
    texts = []
    for i in range(0, len(paths), batch):
        for result in model.predict([str(p) for p in paths[i : i + batch]], batch_size=batch):
            data = result.json.get("res", result.json)
            texts.append(unicodedata.normalize("NFC", str(data.get("rec_text", ""))))
    return texts, (time.perf_counter() - start) * 1000 / max(1, len(paths))


def score(refs: list[str], hyps: list[str]) -> dict:
    edits = sum(Levenshtein.distance(r, h) for r, h in zip(refs, hyps))
    chars = sum(len(r) for r in refs)
    vi_total = vi_hit = 0
    for ref, hyp in zip(refs, hyps):
        ref_vi = [c for c in ref if c in _VI_EXTENDED]
        pool = list(hyp)
        vi_total += len(ref_vi)
        for c in ref_vi:
            if c in pool:
                pool.remove(c)
                vi_hit += 1
    return {
        "cer": round(edits / max(1, chars), 4),
        "line_accuracy": round(sum(r == h for r, h in zip(refs, hyps)) / max(1, len(refs)), 4),
        "vietnamese_letter_recall": round(vi_hit / max(1, vi_total), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", nargs="+", required=True, help="dataset directories with labels.txt")
    parser.add_argument("--model", action="append", default=[], help="label=model_dir of a fine-tuned model")
    parser.add_argument("--model-name", default="latin_PP-OCRv5_mobile_rec", help="architecture name of --model dirs")
    parser.add_argument("--baseline", action="append", default=[], help="stock PaddleOCR model name")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", default="report.md")
    args = parser.parse_args()

    models = [(name, name, None) for name in args.baseline]
    for item in args.model:
        label, _, directory = item.partition("=")
        models.append((label, args.model_name, directory))

    results: dict[str, dict[str, dict]] = {}
    examples: dict[str, list[tuple[str, str]]] = {}
    for directory in args.eval:
        rows = load_labels(Path(directory))
        if args.limit:
            rows = rows[: args.limit]
        paths, refs = [p for p, _ in rows], [t for _, t in rows]
        for label, model_name, model_dir in models:
            hyps, ms = recognise(model_name, model_dir, paths)
            metrics = score(refs, hyps)
            metrics["ms_per_line"] = round(ms, 1)
            results.setdefault(Path(directory).name, {})[label] = metrics
            examples.setdefault(label, [])
            examples[label].extend((r, h) for r, h in zip(refs[:8], hyps[:8]) if len(examples[label]) < 8)
            print(Path(directory).name, label, metrics, flush=True)

    lines = ["# Text line recognition: baseline vs fine-tuned (same runner, one after the other)", ""]
    for dataset, by_model in results.items():
        lines += [f"## {dataset}", "", "| model | CER | line accuracy | Vietnamese letter recall | ms/line |", "|---|---|---|---|---|"]
        for label, m in by_model.items():
            lines.append(f"| {label} | {m['cer']:.2%} | {m['line_accuracy']:.2%} | {m['vietnamese_letter_recall']:.2%} | {m['ms_per_line']} |")
        lines.append("")
    lines += ["## Examples", ""]
    for label, pairs in examples.items():
        lines += [f"**{label}**", "", "| reference | prediction |", "|---|---|"]
        lines += [f"| {r.replace('|', '/')} | {h.replace('|', '/')} |" for r, h in pairs]
        lines.append("")
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    Path(args.report).with_suffix(".json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
