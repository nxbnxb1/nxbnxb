"""Build a held-out evaluation set with natural ground truth (run on GitHub, see eval-set.yml).

Every document goes to ``OUT/<split>/<category>/<id>.<ext>`` with ``<id>.gt.md`` next to it.
Selection never looks at results: documents are picked by a hash of their id, and within each
language / kind a second hash puts 30% of them in ``dev/`` (allowed for tuning) and the rest in
``test/`` (report only).

Sources
  omnidocbench  real pages of 9 kinds (slides, papers, books, textbooks, exams, magazines,
                newspapers, notes, reports) with human annotations of text, tables and formulas
                (OmniDocBench, English pages; research use only, so pages are downloaded at run
                time from a pinned revision and never redistributed)
  enterprise    pages of enterprise documents (scripts/collect_corpus.py) of the companies held out
                for testing: as the digital PDF and as a scanned copy, ground truth = the
                page's own text layer
  wikipedia     featured articles in vi / en / ja (CC BY-SA): text, headings, lists and tables of
                the article are both the ground truth and the content of three renderings — a
                digital PDF and a DOCX (LibreOffice) and a scanned copy of the PDF (rasterised,
                rotated, blurred, noisy, JPEG)

    python scripts/build_eval_set.py omnidocbench OUT --per-kind 8
    python scripts/build_eval_set.py wikipedia OUT --lang vi --count 8
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import pymupdf
from lxml import html as lhtml
from PIL import Image, ImageFilter

UA = "docextract-eval/0.1 (https://github.com/nxbnxb1/nxbnxb; evaluation set builder)"
OMNIDOCBENCH = "https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/{rev}/{path}"
OMNIDOCBENCH_REV = "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
DEV_SHARE = 0.3


def digest(key: str) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)


def assign_splits(keys: list[str]) -> dict[str, str]:
    """30% of each group in dev (the lowest split hashes), the rest in test."""
    ranked = sorted(keys, key=lambda k: digest("split:" + k))
    n_dev = round(DEV_SHARE * len(keys))
    return {k: "dev" if i < n_dev else "test" for i, k in enumerate(ranked)}


def get(client: httpx.Client, url: str, **params) -> httpx.Response:
    for attempt in range(6):
        response = client.get(url, params=params or None)
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(60, 2 ** attempt * 2))
            continue
        response.raise_for_status()
        return response
    response.raise_for_status()
    return response


def write(out: Path, split: str, category: str, name: str, ext: str, data: bytes, gt: str) -> None:
    folder = out / split / category
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}{ext}").write_bytes(data)
    (folder / f"{name}.gt.md").write_text(gt.strip() + "\n", encoding="utf-8")


# --- OmniDocBench ---------------------------------------------------------------------

SKIP = {"header", "footer", "page_number", "page_footnote", "abandon", "figure", "equation_ignore", "need_mask"}


def omnidocbench_markdown(page: dict) -> str:
    blocks = [
        b
        for b in page["layout_dets"]
        if not b.get("ignore") and b["category_type"] not in SKIP and not b["category_type"].endswith("_mask")
    ]
    blocks.sort(key=lambda b: (b.get("order") is None, b.get("order") or 0))
    parts = []
    for b in blocks:
        kind = b["category_type"]
        if kind == "table":
            if b.get("html"):
                parts.append(b["html"].strip())
        elif kind == "equation_isolated":
            latex = (b.get("latex") or "").strip().strip("$").strip()
            if latex:
                parts.append(f"$$\n{latex}\n$$")
        elif kind == "title":
            text = (b.get("text") or "").strip()
            if text:
                parts.append("## " + " ".join(text.split()))
        else:
            text = (b.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def omnidocbench(args: argparse.Namespace) -> None:
    out = Path(args.out)
    with httpx.Client(headers={"User-Agent": UA}, timeout=120, follow_redirects=True) as client:
        pages = get(client, OMNIDOCBENCH.format(rev=args.revision, path="OmniDocBench.json")).json()
        by_kind: dict[str, list[dict]] = {}
        for page in pages:
            attrs = page["page_info"]["page_attribute"]
            if attrs.get("language") == "english":
                by_kind.setdefault(attrs.get("data_source") or "other", []).append(page)
        manifest = []
        for kind, items in sorted(by_kind.items()):
            items.sort(key=lambda p: digest(p["page_info"]["image_path"]))
            chosen = [p for p in items if len(omnidocbench_markdown(p)) >= 20][: args.per_kind]
            splits = assign_splits(["omnidocbench:" + p["page_info"]["image_path"] for p in chosen])
            for page in chosen:
                path = page["page_info"]["image_path"]
                gt = omnidocbench_markdown(page)
                image = get(client, OMNIDOCBENCH.format(rev=args.revision, path=f"images/{path}")).content
                name = Path(path).stem
                split = splits["omnidocbench:" + path]
                write(out, split, f"omnidocbench/{kind}", name, Path(path).suffix.lower(), image, gt)
                manifest.append({"source": "omnidocbench", "kind": kind, "image": path, "split": split})
                print(f"omnidocbench {kind}: {path} → {split}", flush=True)
    _append_manifest(out, manifest, {"omnidocbench_revision": args.revision})


# --- Wikipedia ------------------------------------------------------------------------

FEATURED = {
    "vi": "Thể loại:Bài viết chọn lọc",
    "en": "Category:Featured articles",
    "ja": "Category:秀逸な記事",
}
STOP_SECTIONS = re.compile(
    r"^(tham khảo|chú thích|liên kết ngoài|xem thêm|đọc thêm|nguồn|ghi chú|references|notes|"
    r"external links|see also|further reading|bibliography|sources|citations|脚注|出典|参考文献|"
    r"関連項目|外部リンク|注釈|参考資料)",
    re.IGNORECASE,
)
DROP = (
    "//sup[contains(@class,'reference')] | //style | //link | //figure | //*[contains(@class,'mw-editsection')]"
    " | //*[contains(@class,'infobox')] | //*[contains(@class,'navbox')] | //*[contains(@class,'hatnote')]"
    " | //*[contains(@class,'mwe-math-element')] | //*[contains(@class,'noprint')] | //*[contains(@class,'metadata')]"
    " | //*[contains(@class,'thumb')] | //*[contains(@class,'gallery')] | //*[contains(@class,'mw-empty-elt')]"
    " | //span[@typeof='mw:Nowiki'] | //table[not(contains(@class,'wikitable'))] | //ruby/rt | //ruby/rp"
)


def _text(el) -> str:
    return " ".join(el.text_content().split())


def article_blocks(page_html: str, title: str, max_chars: int) -> list[tuple[str, object]]:
    """[(kind, value)]: ("h1"|"h2"|"h3"|"p", text), ("ul"|"ol", [items]), ("table", rows)."""
    doc = lhtml.fromstring(page_html)
    for el in doc.xpath(DROP):
        el.drop_tree()
    blocks: list[tuple[str, object]] = [("h1", title)]
    size = len(title)
    stop = False
    for el in doc.xpath("//body//*[self::h2 or self::h3 or self::p or self::ul or self::ol or self::table]"):
        if stop or size >= max_chars:
            break
        if el.xpath("ancestor::table | ancestor::li | ancestor::ul | ancestor::ol"):
            continue  # nested: rendered by its container
        tag = el.tag
        if tag in ("h2", "h3"):
            text = _text(el)
            if STOP_SECTIONS.match(text):
                stop = True
                continue
            if text:
                blocks.append((tag, text))
        elif tag == "p":
            text = _text(el)
            if len(text) >= 2:
                blocks.append(("p", text))
                size += len(text)
        elif tag in ("ul", "ol"):
            items = [_text(li) for li in el.xpath("./li") if _text(li)]
            if items:
                blocks.append((tag, items))
                size += sum(len(i) for i in items)
        elif tag == "table":
            rows = []
            for tr in el.xpath(".//tr"):
                cells = []
                for cell in tr.xpath("./th|./td"):
                    cells.append(
                        {
                            "th": cell.tag == "th",
                            "text": _text(cell),
                            "colspan": int(cell.get("colspan", "1") or 1),
                            "rowspan": int(cell.get("rowspan", "1") or 1),
                        }
                    )
                if cells:
                    rows.append(cells)
            if rows and len(rows) <= 40:
                blocks.append(("table", rows))
                size += sum(len(c["text"]) for r in rows for c in r)
    # end on a complete section: drop a trailing heading without content
    while blocks and blocks[-1][0] in ("h2", "h3"):
        blocks.pop()
    return blocks


def _table_html(rows: list[list[dict]]) -> str:
    out = ["<table>"]
    for row in rows:
        out.append("<tr>")
        for c in row:
            tag = "th" if c["th"] else "td"
            span = (f' colspan="{c["colspan"]}"' if c["colspan"] > 1 else "") + (
                f' rowspan="{c["rowspan"]}"' if c["rowspan"] > 1 else ""
            )
            out.append(f"<{tag}{span}>{html.escape(c['text'])}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def blocks_html(blocks: list[tuple[str, object]], lang: str) -> str:
    body = []
    for kind, value in blocks:
        if kind in ("h1", "h2", "h3", "p"):
            body.append(f"<{kind}>{html.escape(value)}</{kind}>")
        elif kind in ("ul", "ol"):
            body.append(f"<{kind}>" + "".join(f"<li>{html.escape(i)}</li>" for i in value) + f"</{kind}>")
        elif kind == "table":
            body.append(_table_html(value).replace("<table>", '<table border="1" cellspacing="0" cellpadding="3">'))
    return (
        f'<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8"><title>{html.escape(blocks[0][1])}</title>'
        "</head><body>" + "\n".join(body) + "</body></html>"
    )


def blocks_markdown(blocks: list[tuple[str, object]]) -> str:
    parts = []
    for kind, value in blocks:
        if kind == "h1":
            parts.append("# " + value)
        elif kind == "h2":
            parts.append("## " + value)
        elif kind == "h3":
            parts.append("### " + value)
        elif kind == "p":
            parts.append(value)
        elif kind == "ul":
            parts.append("\n".join("- " + i for i in value))
        elif kind == "ol":
            parts.append("\n".join(f"{n}. {i}" for n, i in enumerate(value, 1)))
        elif kind == "table":
            parts.append(_table_html(value))
    return "\n\n".join(parts)


def soffice_convert(src: Path, target: str, outdir: Path) -> Path:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        sys.exit("LibreOffice (soffice) is required")
    filters = {"pdf": "pdf:writer_web_pdf_Export", "docx": 'docx:"MS Word 2007 XML"'}
    subprocess.run(
        [soffice, "--headless", "--norestore", "--convert-to", filters[target].replace('"', ""), "--outdir", str(outdir), str(src)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    return outdir / f"{src.stem}.{target}"


def scanned(pdf: bytes, seed: int, dpi: int = 200) -> bytes:
    """Print-and-scan look: slight rotation, blur, grey paper, noise, JPEG (varies per document)."""
    rng = np.random.default_rng(seed)
    src = pymupdf.open(stream=pdf, filetype="pdf")
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
        image = image.rotate(float(rng.uniform(-1.5, 1.5)), resample=Image.Resampling.BICUBIC, fillcolor=255)
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 0.9))))
        arr = np.asarray(image, dtype=np.float32) * rng.uniform(0.85, 0.95) + rng.uniform(8, 20)
        arr += rng.normal(0, rng.uniform(3, 9), arr.shape)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(rng.integers(55, 85)))
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    return out.tobytes(garbage=3, deflate=True)


def featured_titles(client: httpx.Client, lang: str) -> list[str]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    titles, cont = [], {}
    while True:
        data = get(
            client, api, action="query", list="categorymembers", cmtitle=FEATURED[lang], cmlimit="500",
            cmnamespace="0", format="json", **cont,
        ).json()
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        if "continue" not in data or len(titles) >= 5000:
            return titles
        cont = {"cmcontinue": data["continue"]["cmcontinue"]}


def wikipedia(args: argparse.Namespace) -> None:
    out = Path(args.out)
    lang = args.lang
    manifest = []
    with httpx.Client(headers={"User-Agent": UA}, timeout=120, follow_redirects=True) as client, tempfile.TemporaryDirectory() as tmp:
        titles = sorted(featured_titles(client, lang), key=lambda t: digest(f"wikipedia:{lang}:{t}"))
        print(f"{lang}: {len(titles)} featured articles", flush=True)
        chosen = []  # (title, revision, blocks)
        for title in titles:
            if len(chosen) >= args.count:
                break
            rest = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{quote(title.replace(' ', '_'), safe='')}"
            response = get(client, rest)
            revision = response.headers.get("etag", "").strip('W/"').split("/")[0]
            blocks = article_blocks(response.text, title, args.max_chars)
            if sum(1 for k, _ in blocks if k == "p") >= 3:
                chosen.append((title, revision, blocks))
            time.sleep(1)
        splits = assign_splits([f"wikipedia:{lang}:{title}" for title, _, _ in chosen])
        for title, revision, blocks in chosen:
            name = f"{lang}_{digest(title) % 10**8:08d}"
            split = splits[f"wikipedia:{lang}:{title}"]
            source = Path(tmp) / f"{name}.html"
            source.write_text(blocks_html(blocks, lang), encoding="utf-8")
            gt = blocks_markdown(blocks)
            pdf = soffice_convert(source, "pdf", Path(tmp)).read_bytes()
            docx = soffice_convert(source, "docx", Path(tmp)).read_bytes()
            write(out, split, f"wiki_{lang}/digital_pdf", name, ".pdf", pdf, gt)
            write(out, split, f"wiki_{lang}/docx", name, ".docx", docx, gt)
            write(out, split, f"wiki_{lang}/scan", name, ".pdf", scanned(pdf, digest(name) % 2**32), gt)
            manifest.append(
                {"source": "wikipedia", "lang": lang, "title": title, "revision": revision, "id": name, "split": split,
                 "license": "CC BY-SA 4.0", "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"}
            )
            print(f"wikipedia {lang}: {title} (rev {revision}) → {name} {split}", flush=True)
    _append_manifest(out, manifest)


# --- enterprise documents ----------------------------------------------------------------


def _visible_text_page(page) -> bool:
    try:
        traces = page.get_texttrace()
    except Exception:
        return True
    total = sum(len(t.get("chars", ())) for t in traces)
    invisible = sum(len(t.get("chars", ())) for t in traces if t.get("type") == 3)
    return total >= 200 and invisible / total < 0.2


def page_ground_truth(page) -> str:
    """Text of a digital page from its own text layer, block by block in reading order."""
    import unicodedata

    blocks = []
    for block in page.get_text("blocks", sort=True):
        if block[6] != 0:  # image block
            continue
        text = " ".join(block[4].split())
        if text:
            blocks.append(unicodedata.normalize("NFC", text))
    return "\n\n".join(blocks)


def enterprise(args: argparse.Namespace) -> None:
    """Pages of enterprise documents of the companies held out for testing (and dev).

    Each chosen digital document gives a few pages, as the PDF itself (digital_pdf) and as a
    scanned copy (scan); the ground truth is the page's own text layer. Real scans of these
    companies have no ground truth and are not scored.
    """
    import pymupdf

    out = Path(args.out)
    langs = set(args.langs.split(","))
    items = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line]
    manifest = []
    for split in ("dev", "test"):
        for lang in sorted(langs):
            docs = [i for i in items if i["split"] == split and i["language"] == lang and i["kind"] in ("digital", "mixed")
                    and (Path(args.files) / i["file"]).exists()]
            docs.sort(key=lambda i: digest("enterprise:" + i["sha256"]))
            per_company: dict[str, int] = {}
            chosen = 0
            for item in docs:
                if chosen >= args.docs_per_language or per_company.get(item["company"], 0) >= 2:
                    continue
                with pymupdf.open(Path(args.files) / item["file"]) as doc:
                    doc_pages = doc.page_count
                    order = sorted(range(doc_pages), key=lambda n: digest(f"{item['sha256']}:{n}"))
                    pages = [n for n in order if _visible_text_page(doc[n])][: args.pages_per_doc]
                    if not pages:
                        continue
                    sub = pymupdf.open()
                    gts = []
                    for n in sorted(pages):
                        sub.insert_pdf(doc, from_page=n, to_page=n)
                        gts.append(page_ground_truth(doc[n]))
                    pdf = sub.tobytes(garbage=3, deflate=True)
                gt = "\n\n".join(gts)
                name = f"{item['company'].replace(':', '_')}_{item['sha256'][:10]}"
                write(out, split, f"enterprise_{lang}/digital_pdf", name, ".pdf", pdf, gt)
                write(out, split, f"enterprise_{lang}/scan", name, ".pdf", scanned(pdf, digest(name) % 2**32), gt)
                kept = item.get("kept_pages") or list(range(doc_pages))  # page numbers of the original document
                manifest.append({"source": "enterprise", "lang": lang, "split": split, "id": name, "company": item["company"],
                                 "doc_type": item["doc_type"], "url": item["url"], "pages": [kept[p] + 1 for p in sorted(pages)]})
                per_company[item["company"]] = per_company.get(item["company"], 0) + 1
                chosen += 1
            print(f"enterprise {split} {lang}: {chosen} documents", flush=True)
    _append_manifest(out, manifest)


def _append_manifest(out: Path, items: list[dict], meta: dict | None = None) -> None:
    path = out / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"documents": []}
    data["documents"] += items
    data.update(meta or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("omnidocbench")
    p.add_argument("out")
    p.add_argument("--per-kind", type=int, default=8, help="pages per document kind")
    p.add_argument("--revision", default=OMNIDOCBENCH_REV)
    p.set_defaults(func=omnidocbench)
    p = sub.add_parser("enterprise")
    p.add_argument("out")
    p.add_argument("--manifest", default="corpus/manifest.jsonl")
    p.add_argument("--files", default="corpus_files")
    p.add_argument("--langs", default="vi,en,ja")
    p.add_argument("--docs-per-language", type=int, default=15)
    p.add_argument("--pages-per-doc", type=int, default=2)
    p.set_defaults(func=enterprise)
    p = sub.add_parser("wikipedia")
    p.add_argument("out")
    p.add_argument("--lang", choices=sorted(FEATURED), required=True)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=4000, help="article text kept (about 2 pages)")
    p.set_defaults(func=wikipedia)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
