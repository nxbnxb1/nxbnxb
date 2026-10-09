"""Compare text line recognisers on labelled line images: CER, line accuracy, Vietnamese letters, speed.

Every system reads the same line images on the same runner; models are loaded and warmed up
before timing. Systems (``--system LABEL=SPEC``, repeatable):

    paddle:<model name>                  stock PaddleOCR model (e.g. latin_PP-OCRv5_mobile_rec)
    paddle:<model name>@<model dir>      exported model with that architecture (a fine-tuned one)
    tesseract:<langs>                    Tesseract 5, tessdata_best, single text line (e.g. vie+eng)
    easyocr:<langs>                      EasyOCR recogniser (e.g. vi,en); "auto" = the language of
                                         each evaluation set (eval_vi → vi,en; eval_ja → ja,en)
    vietocr:<config>                     VietOCR (e.g. vgg_transformer)

    python evaluate.py --eval data/eval_vi data/eval_en --system "base=paddle:latin_PP-OCRv5_mobile_rec" \
        --system "tesseract=tesseract:vie+eng" --json results.json
    python evaluate.py merge results/*.json --report report.md      # one report from several runs

The older form ``--baseline NAME`` / ``--model LABEL=DIR --model-name NAME`` still works.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

_VI_EXTENDED = {chr(c) for c in range(0x1EA0, 0x1EFA)} | set("ăâđêôơưĂÂĐÊÔƠƯ")
_EASYOCR_AUTO = {"vi": ["vi", "en"], "en": ["en"], "ja": ["ja", "en"]}


def load_labels(directory: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in (directory / "labels.txt").read_text(encoding="utf-8").splitlines():
        rel, _, text = line.partition("\t")
        if rel and text:
            rows.append((directory / rel, unicodedata.normalize("NFC", text)))
    return rows


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(str(text).split()))


class Paddle:
    def __init__(self, arg: str, dataset: str) -> None:
        from paddleocr import TextRecognition

        name, _, directory = arg.partition("@")
        kwargs = {"model_name": name, "enable_mkldnn": False}
        if directory:
            kwargs["model_dir"] = directory
        self.model = TextRecognition(**kwargs)

    def read(self, paths: list[Path]) -> list[str]:
        out = []
        for result in self.model.predict([str(p) for p in paths], batch_size=len(paths)):
            data = result.json.get("res", result.json)
            out.append(_nfc(data.get("rec_text", "")))
        return out


class Tesseract:
    def __init__(self, langs: str, dataset: str) -> None:
        self.langs = langs
        self.tessdata = os.environ.get("TESSDATA_PREFIX")

    def read(self, paths: list[Path]) -> list[str]:
        cmd = ["tesseract", "{}", "stdout", "--psm", "7", "-l", self.langs]
        if self.tessdata:
            cmd += ["--tessdata-dir", self.tessdata]
        out = []
        for path in paths:
            run = subprocess.run([c.replace("{}", str(path)) for c in cmd], capture_output=True, text=True)
            out.append(_nfc(run.stdout))
        return out


class EasyOCR:
    def __init__(self, langs: str, dataset: str) -> None:
        import easyocr

        if langs == "auto":
            code = dataset.rsplit("_", 1)[-1]
            chosen = _EASYOCR_AUTO.get(code, ["en"])
        else:
            chosen = langs.split(",")
        self.reader = easyocr.Reader(chosen, gpu=False, verbose=False)

    def read(self, paths: list[Path]) -> list[str]:
        out = []
        for path in paths:
            parts = self.reader.recognize(str(path), detail=0, paragraph=True)
            out.append(_nfc(" ".join(parts)))
        return out


class VietOCR:
    def __init__(self, config: str, dataset: str) -> None:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name(config)
        cfg["device"] = "cpu"
        self.predictor = Predictor(cfg)

    def read(self, paths: list[Path]) -> list[str]:
        from PIL import Image

        return [_nfc(self.predictor.predict(Image.open(p).convert("RGB"))) for p in paths]


ENGINES = {"paddle": Paddle, "tesseract": Tesseract, "easyocr": EasyOCR, "vietocr": VietOCR}


def recognise(spec: str, dataset: str, paths: list[Path], batch: int = 64) -> tuple[list[str], float]:
    kind, _, arg = spec.partition(":")
    engine = ENGINES[kind](arg, dataset)
    engine.read(paths[: min(8, len(paths))])  # warm-up, not timed
    start = time.perf_counter()
    texts: list[str] = []
    for i in range(0, len(paths), batch):
        texts.extend(engine.read(paths[i : i + batch]))
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
        "vietnamese_letter_recall": round(vi_hit / max(1, vi_total), 4) if vi_total else None,
    }


def evaluate(datasets: list[str], systems: list[tuple[str, str]], limit: int = 0) -> dict:
    results: dict[str, dict[str, dict]] = {}
    examples: dict[str, list[tuple[str, str]]] = {}
    for directory in datasets:
        rows = load_labels(Path(directory))
        if limit:
            rows = rows[:limit]
        paths, refs = [p for p, _ in rows], [t for _, t in rows]
        name = Path(directory).name
        for label, spec in systems:
            try:
                hyps, ms = recognise(spec, name, paths)
            except Exception as exc:  # a baseline that cannot run is reported, not fatal
                results.setdefault(name, {})[label] = {"error": f"{type(exc).__name__}: {exc}"[:300], "spec": spec}
                print(name, label, "failed:", exc, flush=True)
                continue
            metrics = score(refs, hyps)
            metrics.update(ms_per_line=round(ms, 1), spec=spec, lines=len(refs))
            results.setdefault(name, {})[label] = metrics
            examples.setdefault(label, [])
            examples[label].extend((r, h) for r, h in zip(refs[:8], hyps[:8]) if len(examples[label]) < 8)
            print(name, label, metrics, flush=True)
    return {"results": results, "examples": examples}


def _pct(value) -> str:
    return "-" if value is None else f"{value:.2%}"


def report(data: dict) -> str:
    results, examples = data["results"], data.get("examples", {})
    lines = [
        "# Text line recognition: all systems on the same line images, same runner",
        "",
        "CER and line accuracy on the reference text (Unicode NFC, spaces normalised); ms/line after warm-up.",
        "",
    ]
    for dataset, by_system in results.items():
        lines += [f"## {dataset}", "", "| system | CER | line accuracy | Vietnamese letter recall | ms/line |",
                  "|---|---|---|---|---|"]
        ranked = sorted(by_system.items(), key=lambda kv: kv[1].get("cer", 9))
        for label, m in ranked:
            if "error" in m:
                lines.append(f"| {label} | could not run: {m['error'].replace('|', '/')} | | | |")
                continue
            lines.append(f"| {label} | {_pct(m['cer'])} | {_pct(m['line_accuracy'])} | "
                         f"{_pct(m['vietnamese_letter_recall'])} | {m['ms_per_line']} |")
        lines.append("")
    if examples:
        lines += ["## Examples", ""]
        for label, pairs in examples.items():
            lines += [f"**{label}**", "", "| reference | prediction |", "|---|---|"]
            lines += [f"| {r.replace('|', '/')} | {h.replace('|', '/')} |" for r, h in pairs]
            lines.append("")
    return "\n".join(lines)


def merge(paths: list[str]) -> dict:
    merged: dict = {"results": {}, "examples": {}}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for dataset, by_system in data["results"].items():
            merged["results"].setdefault(dataset, {}).update(by_system)
        merged["examples"].update(data.get("examples", {}))
    return merged


def main() -> None:
    import sys

    if sys.argv[1:2] == ["merge"]:
        parser = argparse.ArgumentParser(description="merge evaluation JSON files into one report")
        parser.add_argument("cmd")
        parser.add_argument("inputs", nargs="+")
        parser.add_argument("--report", default="report.md")
        args = parser.parse_args()
        data = merge(args.inputs)
    else:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--eval", nargs="+", required=True, help="dataset directories with labels.txt")
        parser.add_argument("--system", action="append", default=[], help="LABEL=SPEC (see above)")
        parser.add_argument("--model", action="append", default=[], help="LABEL=DIR of a fine-tuned model")
        parser.add_argument("--model-name", default="latin_PP-OCRv5_mobile_rec", help="architecture of --model dirs")
        parser.add_argument("--baseline", action="append", default=[], help="stock PaddleOCR model name")
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--json", help="write the results as JSON (for merge)")
        parser.add_argument("--report", default="report.md")
        args = parser.parse_args()
        systems = [(name, f"paddle:{name}") for name in args.baseline]
        systems += [(label, f"paddle:{args.model_name}@{d}") for label, _, d in (m.partition("=") for m in args.model)]
        systems += [(label, spec) for label, _, spec in (s.partition("=") for s in args.system)]
        data = evaluate(args.eval, systems, args.limit)
        if args.json:
            Path(args.json).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    text = report(data)
    Path(args.report).write_text(text, encoding="utf-8")
    # same format as --json (which may be this very file), so merge reads either
    Path(args.report).with_suffix(".json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
