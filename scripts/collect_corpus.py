"""Collect enterprise documents (run on GitHub, see .github/workflows/corpus.yml).

Sources — only documents companies publish themselves for the public:
  vn     Vietnamese listed companies (corpus/seeds_vn.tsv): a bounded crawl from each company's
         website along investor-relations links (robots.txt respected), collecting PDF links
         (financial statements, annual reports, AGM resolutions and minutes, disclosures, …),
         in Vietnamese and English
  tdnet  Japanese listed companies: TDnet, the Tokyo Stock Exchange's official timely
         disclosure system (daily lists of the last month)

    python scripts/collect_corpus.py crawl-vn corpus/seeds_vn.tsv candidates_vn.jsonl
    python scripts/collect_corpus.py tdnet candidates_jp.jsonl --days 30
    python scripts/collect_corpus.py download candidates_*.jsonl --dir corpus_files --manifest corpus/manifest.jsonl
    python scripts/collect_corpus.py fetch corpus/manifest.jsonl --dir corpus_files   # files of a manifest

The manifest (committed) lists URL, sha256 and metadata of every document — never the files,
which stay in the GitHub Actions cache. Documents are split by company (hash of the company id):
70% train, 10% dev, 20% test, so no company is both trained on and tested on.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # docextract (language detection)

UA = "docextract-research/0.1 (+https://github.com/nxbnxb1/nxbnxb; enterprise document OCR evaluation)"
IR_WORDS = (
    "quan-he-co-dong", "quan-he-nha-dau-tu", "nha-dau-tu", "co-dong", "investor", "shareholder", "/ir",
    "bao-cao", "tai-chinh", "thuong-nien", "cong-bo-thong-tin", "cbtt", "dai-hoi", "nghi-quyet", "financial",
    "annual", "report", "disclosure", "governance", "quan-tri", "tai-lieu", "document",
)
DOC_TYPES_VN = [
    ("financial_statement", r"b[aá]o.?c[aá]o.?t[aà]i.?ch[ií]nh|bctc|financial.?statement|fs[_-]?20"),
    ("annual_report", r"th[uư][oờ]ng.?ni[eê]n|bctn|annual.?report"),
    ("resolution", r"ngh[iị].?quy[eế]t|resolution"),
    ("minutes", r"bi[eê]n.?b[aả]n|minutes"),
    ("shareholder_meeting", r"[dđ][aạ]i.?h[oộ]i|agm|egm|t[oờ].?tr[iì]nh|tai.?lieu.?hop"),
    ("governance_report", r"qu[aả]n.?tr[iị]|governance"),
    ("charter_regulation", r"[dđ]i[eề]u.?l[eệ]|quy.?ch[eế]|charter|regulation"),
    ("explanation_letter", r"gi[aả]i.?tr[iì]nh|c[oô]ng.?v[aă]n|explanation"),
    ("announcement", r"th[oô]ng.?b[aá]o|c[oô]ng.?b[oố]|announcement|notice"),
]
DOC_TYPES_JP = [
    ("earnings_summary", "決算短信"),
    ("results_presentation", "決算説明|説明資料|プレゼン"),
    ("securities_report", "有価証券報告書|四半期報告書|半期報告書"),
    ("agm_notice", "招集"),
    ("forecast_revision", "業績予想|予想の修正"),
    ("dividend", "配当"),
    ("buyback", "自己株式"),
    ("personnel", "人事|異動|役員"),
    ("governance_report", "コーポレート・ガバナンス|ガバナンス"),
    ("financing", "借入|社債|資金"),
]
STRONG_WORDS = ("nha-dau-tu", "co-dong", "investor", "shareholder", "bao-cao", "tai-chinh", "thuong-nien",
                "cong-bo", "dai-hoi", "nghi-quyet", "financial", "annual", "disclosure", "report")
SPLITS = (("train", 70), ("dev", 10), ("test", 20))


def split_of(company: str) -> str:
    bucket = int(hashlib.sha1(f"corpus:{company}".encode()).hexdigest(), 16) % 100
    for name, share in SPLITS:
        if bucket < share:
            return name
        bucket -= share
    return "test"


def doc_type(text: str, rules) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    for name, pattern in rules:
        if re.search(pattern, text):
            return name
    return "other"


def client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": UA}, timeout=30, follow_redirects=True, verify=True)


# --- Vietnamese companies: bounded crawl ------------------------------------------------


def _same_site(url: str, root: str) -> bool:
    host, base = urlparse(url).netloc.lower(), urlparse(root).netloc.lower().removeprefix("www.")
    return host == base or host.endswith("." + base) or host.removeprefix("www.") == base


def crawl_site(ticker: str, sector: str, root: str, max_pages: int, max_pdfs: int) -> list[dict]:
    found: dict[str, dict] = {}
    with client() as c:
        robots = urllib.robotparser.RobotFileParser()
        try:
            robots.parse(c.get(urljoin(root, "/robots.txt")).text.splitlines())
        except Exception:
            robots.parse([])
        queue: list[tuple[int, int, str]] = [(0, 0, root)]  # (-relevance, depth, url)
        seen = {root}
        pages = 0
        while queue and pages < max_pages and len(found) < max_pdfs:
            queue.sort()
            _, depth, url = queue.pop(0)
            if not robots.can_fetch(UA, url):
                continue
            try:
                response = c.get(url)
            except Exception:
                continue
            pages += 1
            if "html" not in response.headers.get("content-type", ""):
                continue
            page = response.text
            for href, anchor in re.findall(r'<a[^>]+href="([^"#]+)"[^>]*>(.*?)</a>', page, re.S | re.I):
                link = urljoin(str(response.url), html.unescape(href.strip()))
                text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", anchor)).split())[:200]
                low = (link + " " + text).lower()
                if re.search(r"\.pdf($|\?)", link, re.I):
                    if link not in found and robots.can_fetch(UA, link):
                        found[link] = {"url": link, "title": text or Path(urlparse(link).path).name,
                                       "page": str(response.url)}
                elif (_same_site(link, root) and link not in seen and depth < 3
                      and any(w in low for w in IR_WORDS)):
                    seen.add(link)
                    queue.append((-sum(w in low for w in STRONG_WORDS), depth + 1, link))
            time.sleep(0.5)  # polite
    company = f"vn:{ticker}"
    return [
        {**item, "source": "vn", "company": company, "ticker": ticker, "sector": sector, "country": "VN",
         "doc_type": doc_type(item["url"] + " " + item["title"], DOC_TYPES_VN)}
        for item in found.values()
    ]


def crawl_vn(args: argparse.Namespace) -> None:
    seeds = [line.split("\t") for line in Path(args.seeds).read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    out = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(crawl_site, t, s, u.strip(), args.max_pages, args.max_pdfs): t for t, s, u in seeds}
        for future in concurrent.futures.as_completed(futures):
            items = future.result()
            print(f"{futures[future]}: {len(items)} PDF links", flush=True)
            out += items
    _write_jsonl(args.out, out)


# --- Japanese companies: TDnet ----------------------------------------------------------

TDNET = "https://www.release.tdnet.info/inbs/"


def tdnet(args: argparse.Namespace) -> None:
    out = []
    today = dt.date.today()
    with client() as c:
        for back in range(args.days):
            day = (today - dt.timedelta(days=back)).strftime("%Y%m%d")
            for page in range(1, 50):
                response = c.get(f"{TDNET}I_list_{page:03d}_{day}.html")
                if response.status_code != 200:
                    break
                rows = re.findall(r"<tr>(.*?)</tr>", response.content.decode("utf-8", "replace"), re.S)
                added = 0
                for row in rows:
                    cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                             for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
                    links = re.findall(r'href="([^"]+\.pdf)"', row)
                    if len(cells) < 4 or not links:
                        continue
                    code, name, title = cells[1], cells[2], cells[3]
                    out.append({"url": TDNET + links[0], "title": title, "company": f"jp:{code[:4]}", "ticker": code,
                                "company_name": name, "date": day, "source": "tdnet", "country": "JP",
                                "doc_type": doc_type(title, DOC_TYPES_JP)})
                    added += 1
                if not added:
                    break
                time.sleep(0.5)
            print(f"TDnet {day}: {len(out)} disclosures so far", flush=True)
    _write_jsonl(args.out, out)


# --- selection, download, inspection ------------------------------------------------------


def select(candidates: list[dict], per_company: int, per_source: dict[str, int]) -> list[dict]:
    """A few documents per company, spread over document types; chosen by hash, not by content."""
    by_company: dict[str, list[dict]] = collections.defaultdict(list)
    for item in candidates:
        by_company[item["company"]].append(item)
    chosen = []
    for company, items in by_company.items():
        items.sort(key=lambda i: hashlib.sha1(i["url"].encode()).hexdigest())
        by_type: dict[str, list[dict]] = collections.defaultdict(list)
        for item in items:
            by_type[item["doc_type"]].append(item)
        picked: list[dict] = []
        while len(picked) < per_company and any(by_type.values()):
            for kind in sorted(by_type):
                if by_type[kind] and len(picked) < per_company:
                    picked.append(by_type[kind].pop(0))
        chosen += picked
    limited, counts = [], collections.Counter()
    for item in sorted(chosen, key=lambda i: hashlib.sha1(i["company"].encode()).hexdigest()):
        if counts[item["source"]] < per_source.get(item["source"], 10**9):
            counts[item["source"]] += 1
            limited.append(item)
    return limited


def inspect_pdf(path: Path) -> dict:
    import pymupdf

    from docextract.textutil import detect_language

    with pymupdf.open(path) as doc:
        pages = doc.page_count
        sample = [doc[i] for i in range(min(pages, 8))]
        chars = [len(p.get_text("text").strip()) for p in sample]
        text = " ".join(p.get_text("text") for p in sample)[:20000]
    digital_pages = sum(1 for n in chars if n >= 200)
    kind = "digital" if digital_pages >= max(1, len(chars) * 0.6) else ("scan" if digital_pages == 0 else "mixed")
    return {"pages": pages, "kind": kind, "language": detect_language(text, ("vi", "en", "ja")) or "unknown"}


def download(args: argparse.Namespace) -> None:
    candidates = [json.loads(line) for path in args.candidates for line in Path(path).read_text().splitlines() if line]
    chosen = select(candidates, args.per_company, {"vn": args.max_vn, "tdnet": args.max_jp})
    print(f"{len(candidates)} candidates → {len(chosen)} selected", flush=True)
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []

    def one(item: dict) -> dict | None:
        with client() as c:
            try:
                with c.stream("GET", item["url"]) as response:
                    if response.status_code != 200:
                        return None
                    data = b""
                    for chunk in response.iter_bytes():
                        data += chunk
                        if len(data) > args.max_mb * 2**20:
                            return None
            except Exception:
                return None
        if not data.startswith(b"%PDF"):
            return None
        sha = hashlib.sha256(data).hexdigest()
        path = directory / f"{sha[:16]}.pdf"
        path.write_bytes(data)
        try:
            info = inspect_pdf(path)
        except Exception:
            path.unlink(missing_ok=True)
            return None
        return {**item, **info, "sha256": sha, "file": path.name, "bytes": len(data), "split": split_of(item["company"])}

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(one, chosen):
            if result:
                manifest.append(result)
    seen, unique = set(), []
    for item in manifest:  # the same file linked from several pages
        if item["sha256"] not in seen:
            seen.add(item["sha256"])
            unique.append(item)
    _write_jsonl(args.manifest, unique)
    print(stats(unique))


def fetch(args: argparse.Namespace) -> None:
    """Download the files of a manifest that are not in the directory yet (checksums verified)."""
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line]
    if args.split:
        items = [i for i in items if i["split"] in args.split.split(",")]
    missing = [i for i in items if not (directory / i["file"]).exists()]
    print(f"{len(items)} documents, {len(missing)} to download", flush=True)

    def one(item: dict) -> bool:
        with client() as c:
            try:
                data = c.get(item["url"]).content
            except Exception:
                return False
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            return False  # changed or removed at the source: skipped, not replaced
        (directory / item["file"]).write_bytes(data)
        return True

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        ok = sum(pool.map(one, missing))
    print(f"downloaded {ok} of {len(missing)}")


def stats(items: list[dict]) -> str:
    lines = ["| split | language | kind | documents | pages |", "|---|---|---|---|---|"]
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for i in items:
        groups[(i["split"], i["language"], i["kind"])].append(i)
    for key in sorted(groups):
        docs = groups[key]
        lines.append(f"| {' | '.join(key)} | {len(docs)} | {sum(d['pages'] for d in docs)} |")
    types = collections.Counter(i["doc_type"] for i in items)
    companies = len({i["company"] for i in items})
    lines += ["", f"{len(items)} documents from {companies} companies; types: "
              + ", ".join(f"{k} {v}" for k, v in types.most_common())]
    return "\n".join(lines)


def _write_jsonl(path: str, items: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("crawl-vn")
    p.add_argument("seeds")
    p.add_argument("out")
    p.add_argument("--max-pages", type=int, default=60)
    p.add_argument("--max-pdfs", type=int, default=80)
    p.add_argument("--workers", type=int, default=12)
    p.set_defaults(func=crawl_vn)
    p = sub.add_parser("tdnet")
    p.add_argument("out")
    p.add_argument("--days", type=int, default=30)
    p.set_defaults(func=tdnet)
    p = sub.add_parser("download")
    p.add_argument("candidates", nargs="+")
    p.add_argument("--dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--per-company", type=int, default=6)
    p.add_argument("--max-vn", type=int, default=500)
    p.add_argument("--max-jp", type=int, default=400)
    p.add_argument("--max-mb", type=int, default=30)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=download)
    p = sub.add_parser("fetch")
    p.add_argument("manifest")
    p.add_argument("--dir", required=True)
    p.add_argument("--split", help="only these splits, e.g. train or dev,test")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=fetch)
    p = sub.add_parser("stats")
    p.add_argument("manifest")
    p.set_defaults(func=lambda a: print(stats([json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x])))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
