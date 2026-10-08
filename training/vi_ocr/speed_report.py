"""Training speed from a PaddleOCR train.log (lines ``... avg_batch_cost: X s ... ips: Y samples/s``).

    python speed_report.py one  <train.log> --variant NAME --status N --out result.json
    python speed_report.py table <dir with result*.json> [--samples 60000 --budget-min 290]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(r"global_step: (\d+).*?avg_reader_cost: ([\d.]+) s, avg_batch_cost: ([\d.]+) s.*?ips: ([\d.]+) samples/s")


def measure(log: Path, warmup: int = 3) -> dict:
    rows = []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            m = LINE.search(line)
            if m:
                rows.append(tuple(float(g) for g in m.groups()))
    steady = rows[warmup:] or rows
    if not steady:
        return {"steps": int(rows[-1][0]) if rows else 0, "ips": None}
    return {
        "steps": int(rows[-1][0]),
        "ips": round(statistics.median(r[3] for r in steady), 2),
        "batch_s": round(statistics.median(r[2] for r in steady), 3),
        "reader_s": round(statistics.median(r[1] for r in steady), 3),
    }


def one(args: argparse.Namespace) -> None:
    result = {"variant": args.variant, "product": args.product, "status": args.status, **measure(Path(args.log))}
    Path(args.out).write_text(json.dumps(result))
    print(json.dumps(result))


def table(args: argparse.Namespace) -> None:
    results = [json.loads(p.read_text()) for p in sorted(Path(args.dir).rglob("result*.json"))]
    results.sort(key=lambda r: (r.get("product", ""), -(r.get("ips") or 0)))
    out = [
        "| product | variant | samples/s | s/batch | data s/batch | steps | exit | passes over "
        f"{args.samples} lines in {args.budget_min} min |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ips = r.get("ips")
        passes = f"{ips * 60 * args.budget_min / args.samples:.1f}" if ips else "-"
        out.append(
            f"| {r.get('product', '')} | {r['variant']} | {ips or '-'} | {r.get('batch_s', '-')} | "
            f"{r.get('reader_s', '-')} | {r['steps']} | {r['status']} | {passes} |"
        )
    print("\n".join(out))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("one")
    p.add_argument("log")
    p.add_argument("--variant", required=True)
    p.add_argument("--product", default="")
    p.add_argument("--status", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=one)
    p = sub.add_parser("table")
    p.add_argument("dir")
    p.add_argument("--samples", type=int, default=60000)
    p.add_argument("--budget-min", type=int, default=290)
    p.set_defaults(func=table)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
