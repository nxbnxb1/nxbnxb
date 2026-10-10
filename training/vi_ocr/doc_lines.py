"""Text line images cut from real enterprise documents (training split of the corpus).

    python doc_lines.py OUT --manifest corpus/manifest.jsonl --files corpus_files --dict dict.txt \\
        --langs vi,en --count 12000 --seed 1001 [--shard 3/8]

Only digital PDFs of companies in the training split (scripts/collect_corpus.py) are used, so no
test company is ever trained on. Pages are sampled with the seed; every visible line of the
PDF's own text layer becomes a labelled crop of the page rendered at a random resolution, then
degraded like the synthetic lines. Lines longer than the recogniser's limit are cut at word
boundaries with the character boxes. Lines are skipped when they contain characters outside the
dictionary, legacy-encoded or garbled text (wrong ToUnicode maps), or text drawn invisibly over a
scan (an OCR layer, whose text may be wrong).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import random
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # docextract
sys.path.insert(0, str(HERE))  # synth

MAX_LEN = 23  # default: same limit as synthetic training lines


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _usable_page(page) -> bool:
    """False for scans with an invisible OCR text layer."""
    try:
        traces = page.get_texttrace()
    except Exception:
        return True
    if not traces:
        return False
    invisible = sum(len(t.get("chars", ())) for t in traces if t.get("type") == 3)
    total = sum(len(t.get("chars", ())) for t in traces)
    return total > 0 and invisible / total < 0.2


def _chunks(chars: list[tuple[str, tuple]], max_len: int = MAX_LEN) -> list[tuple[str, tuple]]:
    """Cut a line into pieces of at most max_len characters at spaces: (text, bbox)."""
    words: list[list[tuple[str, tuple]]] = [[]]
    for ch, box in chars:
        if ch.isspace():
            if words[-1]:
                words.append([])
        else:
            words[-1].append((ch, box))
    out, current = [], []
    for word in [w for w in words if w]:
        candidate = current + ([(" ", None)] if current else []) + word
        if len(_nfc("".join(c for c, _ in candidate))) <= max_len:
            current = candidate
            continue
        if current:
            out.append(current)
        current = word if len(_nfc("".join(c for c, _ in word))) <= max_len else []
    if current:
        out.append(current)
    result = []
    for piece in out:
        boxes = [b for _, b in piece if b is not None]
        box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        result.append((_nfc("".join(c for c, _ in piece)), box))
    return result


def page_lines(path: str, page_no: int, seed: int, charset: frozenset[str], augment: bool,
               max_len: int = MAX_LEN, min_len: int = 2) -> list[tuple[bytes, str]]:
    import pymupdf
    from PIL import Image

    from docextract.textutil import garbled_ratio, legacy_encoding_ratio
    from synth import degrade

    rng = random.Random(seed)
    out = []
    with pymupdf.open(path) as doc:
        page = doc[page_no]
        if not _usable_page(page):
            return []
        dpi = rng.choice([150, 180, 200, 240, 300])
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        scale = dpi / 72
        raw = page.get_text("rawdict")
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                if abs(line.get("dir", (1, 0))[1]) > 0.01:  # vertical or rotated text
                    continue
                chars = [(c["c"], c["bbox"]) for span in line["spans"] for c in span["chars"]]
                text = _nfc("".join(c for c, _ in chars)).strip()
                if len(text) < 2 or "�" in text or "(cid:" in text:
                    continue
                if any(ch not in charset for ch in text if not ch.isspace()):
                    continue
                if garbled_ratio(text) > 0.1 or legacy_encoding_ratio(text) > 0.1:
                    continue
                for label, (x0, y0, x1, y1) in _chunks(chars, max_len):
                    if len(label) < max(2, min_len):
                        continue
                    h = (y1 - y0) * scale
                    pad_y, pad_x = h * rng.uniform(0.15, 0.35), rng.uniform(2, 8)
                    box = (int(x0 * scale - pad_x), int(y0 * scale - pad_y), int(x1 * scale + pad_x), int(y1 * scale + pad_y))
                    box = (max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3]))
                    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                        continue
                    crop = image.crop(box)
                    if augment and rng.random() < 0.7:
                        crop = degrade(crop, rng, (255, 255, 255))
                    buf = io.BytesIO()
                    crop.save(buf, format="PNG")
                    out.append((buf.getvalue(), label))
    rng.shuffle(out)
    return out[: rng.randint(8, 40)]  # a few lines per page: more pages, more variety


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--files", required=True)
    parser.add_argument("--dict", required=True)
    parser.add_argument("--langs", default="vi,en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="1/1", help="k/n: use only the k-th of n parts of the documents")
    parser.add_argument("--max-len", type=int, default=MAX_LEN, help="longer lines are cut at word boundaries")
    parser.add_argument("--min-len", type=int, default=2, help="shorter lines (or pieces) are skipped")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    langs = set(args.langs.split(","))
    k, n = (int(x) for x in args.shard.split("/"))
    charset = frozenset(line.rstrip("\n") for line in open(args.dict, encoding="utf-8") if line.rstrip("\n"))
    docs = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["split"] != args.split or item["kind"] not in ("digital", "mixed") or item["language"] not in langs:
            continue
        if int(item["sha256"][:8], 16) % n != k - 1:
            continue
        path = Path(args.files) / item["file"]
        if path.exists():
            docs.append((str(path), item["pages"]))
    if not docs:
        sys.exit(f"no {args.split} documents in {args.langs} for shard {args.shard}")
    rng = random.Random(args.seed)
    tasks = []
    for _ in range(args.count // 4 + 50):  # page draws; each gives several lines
        path, pages = rng.choice(docs)
        tasks.append((path, rng.randrange(pages), rng.randrange(2**31)))

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    labels, written = [], 0
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(page_lines, p, i, s, charset, not args.no_augment, args.max_len, args.min_len)
                   for p, i, s in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                lines = future.result()
            except Exception:
                continue
            for png, label in lines:
                if written >= args.count:
                    break
                name = f"images/{hashlib.sha1(png).hexdigest()[:16]}.png"
                (out / name).write_bytes(png)
                labels.append(f"{name}\t{label}")
                written += 1
            if written >= args.count:
                for f in futures:
                    f.cancel()
                break
    (out / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"{written} lines from {len(docs)} {args.split} documents ({args.langs}) written to {out}")


if __name__ == "__main__":
    main()
