"""Command line: ``docextract extract|serve|bench``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .preprocessing.office import OFFICE_EXTENSIONS

SUPPORTED = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"} | OFFICE_EXTENSIONS


def write_outputs(result, out_dir: Path, stem: str, formats: set[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if "md" in formats:
        path = out_dir / f"{stem}.md"
        path.write_text(result.markdown or "", encoding="utf-8")
        written.append(path)
    if "json" in formats:
        path = out_dir / f"{stem}.json"
        path.write_text(result.to_json(indent=2), encoding="utf-8")
        written.append(path)
    for rel, png in result.figures.items():  # pictures that text cannot replace, linked from the .md
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(png)
    return written


def _inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED and p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"not found: {raw}")
    return files


def cmd_extract(args: argparse.Namespace) -> int:
    from .config import Settings
    from .pipeline import DocumentPipeline, ExtractOptions, parse_pages

    overrides = {}
    if args.product:
        overrides["product"] = args.product
    if args.layout:
        overrides["layout_backend"] = args.layout
    if args.dpi:
        overrides["dpi"] = args.dpi
    settings = Settings.from_env(**overrides)
    pipeline = DocumentPipeline(settings)
    options = ExtractOptions(pages=parse_pages(args.pages), use_vlm=not args.no_vlm)
    formats = {f.strip() for f in args.format.split(",")}
    failures = 0
    files = _inputs(args.inputs)
    for path in files:
        try:
            result = pipeline.process_file(path, options)
        except Exception as exc:  # keep going with the other files
            logging.exception("failed: %s", path)
            print(f"FAILED {path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        written = write_outputs(result, Path(args.output), path.stem, formats)
        s = result.stats
        review = s.regions_by_status.get("needs_review", 0)
        print(
            f"{path} -> {', '.join(str(w) for w in written)} "
            f"({s.pages_processed} pages, {s.regions} regions, {review} need review, "
            f"{len(result.figures)} figures kept, {s.ms_per_page:.0f} ms/page)"
        )
    return 1 if failures else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("docextract.api:app", host=args.host, port=args.port, workers=1)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .benchmark import compare_reports

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    table = compare_reports(baseline, candidate, (args.baseline_name, args.candidate_name))
    if args.output:
        Path(args.output).write_text(table, encoding="utf-8")
    print(table)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    pipeline = None
    if args.product:
        from .config import Settings
        from .pipeline import DocumentPipeline

        pipeline = DocumentPipeline(Settings.from_env(product=args.product))
    report = run_benchmark(Path(args.dataset), use_vlm=not args.no_vlm, pipeline=pipeline, split=args.split)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "benchmark.md").write_text(report["markdown"], encoding="utf-8")
    print(report["markdown"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docextract", description="PDF/Word/scans → Markdown + JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="extract documents to Markdown/JSON")
    p.add_argument("inputs", nargs="+", help="files or directories")
    p.add_argument("-o", "--output", default="outputs", help="output directory")
    p.add_argument("--format", default="md,json", help="md, json or md,json")
    p.add_argument("--pages", help="page selection, e.g. 1-3,5")
    p.add_argument("--no-vlm", action="store_true", help="never call the VLM")
    p.add_argument("--layout", choices=["auto", "paddle", "heuristic"], help="layout backend")
    p.add_argument("--dpi", type=int, help="render resolution for PDF pages")
    p.add_argument("--product", choices=["vi_en", "vi_en_ja"], help="Việt + Anh or Việt + Anh + Nhật (default: env / vi_en)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("serve", help="run the FastAPI server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("bench", help="benchmark on a dataset with ground truth")
    p.add_argument("dataset", help="directory with documents and <name>.gt.md files")
    p.add_argument("-o", "--output", default="benchmark")
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--product", choices=["vi_en", "vi_en_ja"])
    p.add_argument("--split", choices=["dev", "test"], help="only documents under dev/ or test/")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("compare", help="side-by-side table of two benchmark.json files")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--baseline-name", default="baseline")
    p.add_argument("--candidate-name", default="fine-tuned")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
