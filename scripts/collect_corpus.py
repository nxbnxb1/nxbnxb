"""Collect enterprise documents (run on GitHub, see .github/workflows/corpus.yml).

Only documents companies publish themselves on their own websites (investor relations,
disclosures): financial statements, annual and integrated reports, AGM notices, resolutions and
minutes, governance reports, results presentations, … Seeds (company id <TAB> sector <TAB> website):
  corpus/seeds_vn.tsv           Vietnamese listed companies (curated)
  corpus/seeds_vn_wikidata.tsv  businesses in Vietnam with an official website on Wikidata
  corpus/seeds_jp_wikidata.tsv  companies listed on the Tokyo Stock Exchange, from Wikidata
From each website a bounded crawl follows investor-relations links and collects links to PDFs.
Pages are read as HTML first; sites that build their pages with JavaScript, where that finds few
documents, are crawled again in a headless browser (--render). robots.txt is honoured for every
host, with the * and $ patterns of RFC 9309, at crawl time and again at every download.

    python scripts/collect_corpus.py seeds-wikidata VN corpus/seeds_vn_wikidata.tsv --exclude corpus/seeds_vn.tsv
    python scripts/collect_corpus.py crawl VN corpus/seeds_vn.tsv corpus/seeds_vn_wikidata.tsv --out vn.jsonl [--shard 2/8] [--render]
    python scripts/collect_corpus.py download vn.jsonl jp.jsonl --dir corpus_files --manifest corpus/manifest.jsonl
    python scripts/collect_corpus.py fetch corpus/manifest.jsonl --dir corpus_files   # files of a manifest

The manifest (committed) lists URL, sha256 and metadata of every document — never the files,
which stay in the GitHub Actions cache. Every run adds to it. A long document is stored as at most
MAX_PAGES of its pages (SCAN_PAGES for scans), chosen from its checksum, so fetch stores the same
pages again. Documents are split by company (hash of the company id): 70% train, 10% dev, 20%
test, so no company is both trained on and tested on.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # docextract (language detection)

AGENT = "docextract-research"
UA = f"{AGENT}/0.2 (+https://github.com/nxbnxb1/nxbnxb; enterprise document OCR evaluation)"
MAX_PAGES = 20  # pages stored per document
SCAN_PAGES = 5  # scans have no text layer, hence no ground truth: a few pages are enough
SPLITS = (("train", 70), ("dev", 10), ("test", 20))

# Words of links worth following (matched on folded text: lower case, no Vietnamese accents, "-")
IR_WORDS = {
    "VN": ("quan-he-co-dong", "quan-he-nha-dau-tu", "nha-dau-tu", "co-dong", "investor", "shareholder", "/ir",
           "bao-cao", "tai-chinh", "thuong-nien", "cong-bo-thong-tin", "cbtt", "dai-hoi", "nghi-quyet", "financial",
           "annual", "report", "disclosure", "governance", "quan-tri", "tai-lieu", "document"),
    "JP": ("/ir", "ir-", "-ir", "ir情報", "investor", "shareholder", "株主", "投資家", "決算", "有価証券", "ガバナンス",
           "ライブラリ", "library", "financial", "results", "report", "disclosure", "適時開示", "統合報告", "integrated",
           "presentation", "説明会", "governance", "kabunushi"),
}
STRONG_WORDS = {
    "VN": ("nha-dau-tu", "co-dong", "investor", "shareholder", "bao-cao", "tai-chinh", "thuong-nien", "cong-bo",
           "dai-hoi", "nghi-quyet", "financial", "annual", "disclosure", "report"),
    "JP": ("investor", "/ir", "ir情報", "株主", "投資家", "決算", "有価証券", "ライブラリ", "library", "financial",
           "results", "適時開示", "統合報告", "report"),
}
# Links that may serve a file without ".pdf" in the URL: checked with a GET of the headers only
DOWNLOAD_HINTS = ("download", "tai-ve", "tai-xuong", "attachment", "getfile", "viewfile", "/files/", "/file/",
                  "/media/", "/documents/", "ダウンロード")
DOC_TYPES = {
    "VN": [
        ("financial_statement", r"b[aá]o.?c[aá]o.?t[aà]i.?ch[ií]nh|bctc|financial.?statement|fs[_-]?20"),
        ("annual_report", r"th[uư][oờ]ng.?ni[eê]n|bctn|annual.?report"),
        ("resolution", r"ngh[iị].?quy[eế]t|resolution"),
        ("minutes", r"bi[eê]n.?b[aả]n|minutes"),
        ("shareholder_meeting", r"[dđ][aạ]i.?h[oộ]i|agm|egm|t[oờ].?tr[iì]nh|tai.?lieu.?hop"),
        ("governance_report", r"qu[aả]n.?tr[iị]|governance"),
        ("charter_regulation", r"[dđ]i[eề]u.?l[eệ]|quy.?ch[eế]|charter|regulation"),
        ("explanation_letter", r"gi[aả]i.?tr[iì]nh|c[oô]ng.?v[aă]n|explanation"),
        ("announcement", r"th[oô]ng.?b[aá]o|c[oô]ng.?b[oố]|announcement|notice"),
    ],
    "JP": [
        ("earnings_summary", r"決算短信|tanshin|financial.?results|earnings"),
        ("securities_report", r"有価証券報告書|四半期報告書|半期報告書|securities.?report|yuho"),
        ("annual_report", r"統合報告|アニュアル|annual.?report|integrated.?report"),
        ("results_presentation", r"決算説明|説明資料|説明会|プレゼン|presentation"),
        ("agm_notice", r"招集|株主総会|notice.?of.*meeting|convocation|agm"),
        ("governance_report", r"ガバナンス|governance"),
        ("forecast_revision", r"業績予想|予想の修正|forecast"),
        ("dividend", r"配当|dividend"),
        ("buyback", r"自己株式|treasury|buyback|repurchase"),
        ("sustainability_report", r"サステナビリティ|esg|sustainability|csr"),
        ("fact_book", r"ファクトブック|fact.?book|データブック|data.?book"),
    ],
}
# schools, universities and public bodies (second-level domains) are not enterprises
PUBLIC_DOMAINS = re.compile(r"(^|\.)(edu|gov|ac|go|ed|lg|mil|int)\.[a-z]{2}$|\.(edu|gov|mil|int)$")
SKIP_HOSTS = ("facebook.com", "fb.com", "google.com", "linkedin.com", "youtube.com", "twitter.com", "x.com",
              "instagram.com", "tiktok.com", "wordpress.com", "blogspot.com", "wixsite.com", "wikipedia.org")
WIKIDATA = "https://query.wikidata.org/sparql"
WIKIDATA_QUERIES = {
    # businesses (any subclass) in Vietnam with an official website
    "VN": """SELECT ?item ?site (SAMPLE(?code) AS ?ticker) (SAMPLE(?ind) AS ?industry) WHERE {
  ?item wdt:P17 wd:Q881 ; wdt:P856 ?site ; wdt:P31 ?cls . ?cls wdt:P279* wd:Q4830453 .
  MINUS { ?item wdt:P31/wdt:P279* wd:Q2385804 }  # educational institutions
  MINUS { ?item wdt:P31/wdt:P279* wd:Q327333 }   # government agencies
  OPTIONAL { ?item p:P414 ?st . ?st pq:P249 ?code }
  OPTIONAL { ?item wdt:P452 ?i . ?i rdfs:label ?ind . FILTER(LANG(?ind) = "en") }
} GROUP BY ?item ?site""",
    # companies listed on the Tokyo Stock Exchange with an official website
    "JP": """SELECT ?item ?site (SAMPLE(?code) AS ?ticker) (SAMPLE(?ind) AS ?industry) WHERE {
  ?item p:P414 ?st ; wdt:P856 ?site . ?st ps:P414 wd:Q217475 .
  OPTIONAL { ?st pq:P249 ?code }
  OPTIONAL { ?item wdt:P452 ?i . ?i rdfs:label ?ind . FILTER(LANG(?ind) = "en") }
} GROUP BY ?item ?site""",
}
PDF_URL = re.compile(r"\.pdf($|[?#])", re.I)
PDF_IN_TEXT = re.compile(r"""(?:https?://|/)[^\s"'<>()\\]{1,300}?\.pdf(?![a-z0-9])(?:\?[^\s"'<>()\\]{0,200})?""", re.I)


def split_of(company: str) -> str:
    bucket = int(hashlib.sha1(f"corpus:{company}".encode()).hexdigest(), 16) % 100
    for name, share in SPLITS:
        if bucket < share:
            return name
        bucket -= share
    return "test"


def fold(text: str) -> str:
    """Lower case, accents of Latin letters removed, đ → d, runs of other characters → "-"."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", text.lower()):
        if unicodedata.combining(ch) and out and ord(out[-1]) < 0x250:
            continue  # Vietnamese accents; Japanese voicing marks stay
        out.append(ch)
    text = unicodedata.normalize("NFC", "".join(out)).replace("đ", "d")
    return re.sub(r"[^\w/]+|_", "-", text).strip("-")


def doc_type(text: str, country: str) -> str:
    text = unicodedata.normalize("NFC", unquote(text).lower())
    for name, pattern in DOC_TYPES[country]:
        if re.search(pattern, text):
            return name
    return "other"


def domain(url: str) -> str:
    return (urlparse(url if "//" in url else "//" + url).hostname or "").removeprefix("www.")


def client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": UA}, timeout=30, follow_redirects=True, verify=True)


def read_body(response: httpx.Response, max_bytes: int, seconds: float) -> bytes | None:
    """The body of a streamed response, or None when it is larger than max_bytes or takes longer
    than seconds (a server trickling bytes would otherwise hold a worker for hours)."""
    deadline = time.monotonic() + seconds
    data = bytearray()
    for chunk in response.iter_bytes():
        data += chunk
        if len(data) > max_bytes or time.monotonic() > deadline:
            return None
    return bytes(data)


# --- robots.txt ---------------------------------------------------------------------------


class Robots:
    """The rules of a robots.txt that apply to this crawler (RFC 9309: user-agent groups, longest
    match wins, allow wins a tie, * and $ patterns). None: robots.txt unreachable → nothing allowed."""

    def __init__(self, text: str | None) -> None:
        self.rules: list[tuple[int, bool, re.Pattern]] = []
        self.delay = 0.0
        self.blocked = text is None
        groups: list[tuple[set[str], list[tuple[bool, str]], list[float]]] = []
        in_agents = False
        for raw in (text or "").splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if not in_agents:
                    groups.append((set(), [], []))
                groups[-1][0].add(value.lower())
                in_agents = True
            elif key in ("allow", "disallow") and groups:
                in_agents = False
                if value:  # an empty Disallow allows everything
                    groups[-1][1].append((key == "allow", value))
            elif key == "crawl-delay" and groups:
                in_agents = False
                try:
                    groups[-1][2].append(float(value))
                except ValueError:
                    pass
        mine = [g for g in groups if AGENT in g[0]] or [g for g in groups if "*" in g[0]]
        for _, rules, delays in mine:
            for allow, path in rules:
                body = re.escape(path.removesuffix("$")).replace(r"\*", ".*")
                self.rules.append((len(path), allow, re.compile(body + ("$" if path.endswith("$") else ""))))
            self.delay = max([self.delay, *delays])

    def allowed(self, url: str) -> bool:
        if self.blocked:
            return False
        parts = urlparse(url)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        if target == "/robots.txt":
            return True
        best = (-1, True)
        for length, allow, pattern in self.rules:
            if (length, allow) > best and pattern.match(target):
                best = (length, allow)
        return best[1]


class RobotsCache:
    """robots.txt of every host a URL is on, fetched once (thread-safe)."""

    def __init__(self) -> None:
        self._robots: dict[str, Robots] = {}
        self._lock = threading.Lock()

    def get(self, url: str) -> Robots:
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with self._lock:
            if origin in self._robots:
                return self._robots[origin]
        text = None  # 5xx or no answer: nothing allowed (RFC 9309 2.3.1)
        for attempt in range(3):
            try:
                with client() as c, c.stream("GET", origin + "/robots.txt") as response:
                    status = response.status_code
                    body = read_body(response, 512 * 1024, 30) if status == 200 else b""
            except httpx.HTTPError:
                time.sleep(2 + 4 * attempt)
                continue
            if status < 500:  # 4xx: no robots.txt, everything allowed
                text = (body or b"").decode("utf-8", "replace") if status == 200 else ""
                break
            time.sleep(2 + 4 * attempt)
        robots = Robots(text)
        with self._lock:
            self._robots[origin] = robots
        return robots

    def allowed(self, url: str) -> bool:
        return self.get(url).allowed(url)


# --- crawling -------------------------------------------------------------------------------


class _Links(HTMLParser):
    """(href, anchor text) of the <a> of a page, and PDF URLs in other attributes."""

    ATTRS = ("src", "data", "data-href", "data-url", "data-src", "data-file", "data-link", "onclick", "value")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[list[str]] = []
        self.files: list[str] = []
        self._anchor: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        values = {k: v or "" for k, v in attrs}
        if tag == "a" and values.get("href"):
            self._anchor = [values["href"], values.get("title", "")]
            self.links.append(self._anchor)
        for name in self.ATTRS:
            self.files += [m.group(0) for m in PDF_IN_TEXT.finditer(values.get(name, ""))]

    def handle_endtag(self, tag):
        if tag == "a":
            self._anchor = None

    def handle_data(self, data):
        if self._anchor is not None:
            self._anchor[1] += " " + data


def page_links(base: str, page: str) -> list[tuple[str, str]]:
    """Absolute links of an HTML page with their text, including PDF URLs written in scripts
    (pages that fill their document lists from embedded JSON)."""
    parser = _Links()
    try:
        parser.feed(page)
    except Exception:
        pass
    out = [(href, text) for href, text in parser.links] + [(f, "") for f in parser.files]
    out += [(m.group(0), "") for m in PDF_IN_TEXT.finditer(page.replace("\\/", "/"))]
    links, seen = [], set()
    for href, text in out:
        href = html.unescape(href.strip())
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        link = urljoin(base, href).split("#", 1)[0]
        if link.startswith("http") and link not in seen:
            seen.add(link)
            links.append((link, " ".join(text.split())[:200]))
    return links


def _same_site(url: str, bases: set[str]) -> bool:
    host = domain(url)
    return any(host == base or host.endswith("." + base) for base in bases)


class StaticFetcher:
    """Pages as served (no JavaScript). A link that turns out to be a PDF is reported as such."""

    def __init__(self) -> None:
        self.c = client()

    def close(self) -> None:
        self.c.close()

    def __call__(self, url: str):
        try:
            with self.c.stream("GET", url) as response:
                if response.status_code != 200:
                    return None
                ctype = response.headers.get("content-type", "").lower()
                disposition = response.headers.get("content-disposition", "").lower()
                if "pdf" in ctype or ".pdf" in disposition:
                    return ("pdf", str(response.url))
                if "html" not in ctype:
                    return None
                body = read_body(response, 8 * 2**20, 60)
                if body is None:
                    return None
                return ("html", [(str(response.url), body.decode(response.encoding or "utf-8", "replace"))])
        except Exception:
            return None


class BrowserFetcher:
    """Pages rendered by Chromium (one browser per fetcher; pictures, fonts and media not loaded)."""

    def __init__(self) -> None:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(executable_path=os.environ.get("CORPUS_CHROMIUM") or None)
        self._context = self._browser.new_context(user_agent=UA, accept_downloads=False)
        self._context.route("**/*", lambda route: route.abort()
                            if route.request.resource_type in ("image", "media", "font") else route.continue_())
        self._page = self._context.new_page()

    def close(self) -> None:
        for close in (self._context.close, self._browser.close, self._pw.stop):
            try:
                close()
            except Exception:
                pass

    def __call__(self, url: str):
        if PDF_URL.search(url):
            return None
        try:
            response = self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            if response is None or response.status >= 400 or "html" not in response.headers.get("content-type", ""):
                return None
            try:
                self._page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            pages = []
            for frame in self._page.frames:  # IR pages are often widgets of an IR provider in a frame
                try:
                    pages.append((frame.url, frame.content()))
                except Exception:
                    pass
            return ("html", pages)
        except Exception as exc:
            if "closed" in str(exc):  # the page crashed: a new one for the next URL
                self._page = self._context.new_page()
            return None


def crawl_site(seed: dict, fetch, robots: RobotsCache, args: argparse.Namespace) -> dict[str, dict]:
    """PDF links of one website, following the most relevant investor-relations links first."""
    root, country = seed["website"], seed["country"]
    ir_words, strong = IR_WORDS[country], STRONG_WORDS[country]
    found: dict[str, dict] = {}
    maybe: dict[str, dict] = {}
    queue_: list[tuple[int, int, str]] = [(0, 0, root)]  # (-relevance, depth, url)
    seen, pages = {root}, 0
    bases = {domain(root)}
    deadline = time.monotonic() + args.site_seconds
    # a seed marked "ticker" counts only if the website shows its ticker (in a page, or in the name
    # of a document): a wrong or re-registered domain then yields nothing, not another company's files
    ticker = None
    if seed.get("check") == "ticker":
        ticker = re.compile(rf"(?<![A-Za-z0-9]){seed['id']}(?![A-Za-z0-9])", re.IGNORECASE)
    confirmed = ticker is None
    while queue_ and pages < args.max_pages and len(found) < args.max_pdfs and time.monotonic() < deadline:
        queue_.sort()
        _, depth, url = queue_.pop(0)
        if not robots.allowed(url):
            continue
        got = fetch(url)
        pages += 1
        if got is None:
            continue
        if got[0] == "pdf":
            found.setdefault(got[1], {"url": got[1], "title": Path(urlparse(url).path).name, "page": root})
            continue
        if pages == 1 and got[1]:
            bases.add(domain(got[1][0][0]))  # the website moved to another domain
        for page_url, page in got[1]:
            confirmed = confirmed or bool(re.search(rf"\b{seed['id']}\b", page))  # upper case in pages
            for link, text in page_links(page_url, page):
                if time.monotonic() > deadline:  # robots.txt of many other hosts can take long
                    break
                low = fold(unquote(link)) + " " + fold(text)
                if PDF_URL.search(link):
                    if link not in found and robots.allowed(link):
                        found[link] = {"url": link, "title": text or Path(urlparse(link).path).name, "page": page_url}
                elif _same_site(link, bases) and link not in seen:
                    if depth < 3 and any(w in low for w in ir_words):
                        seen.add(link)
                        queue_.append((-sum(w in low for w in strong), depth + 1, link))
                    elif any(h in low for h in DOWNLOAD_HINTS) and len(maybe) < 40:
                        maybe[link] = {"url": link, "title": text, "page": page_url}
        time.sleep(min(10.0, max(0.5, robots.get(root).delay)))  # polite
    if len(found) < args.max_pdfs and maybe:
        probe = StaticFetcher()
        try:
            for link, item in maybe.items():
                if time.monotonic() > deadline or len(found) >= args.max_pdfs:
                    break
                if robots.allowed(link) and (got := probe(link)) and got[0] == "pdf":
                    found.setdefault(got[1], {**item, "url": got[1], "title": item["title"] or Path(urlparse(got[1]).path).name})
                time.sleep(0.5)
        finally:
            probe.close()
    if not confirmed and not any(ticker.search(unquote(i["url"]) + " " + i["title"]) for i in found.values()):
        if found:
            print(f"{seed['id']} {root}: ticker not seen on the website, {len(found)} links dropped", flush=True)
        return {}
    company = f"{country.lower()}:{seed['id']}"
    return {
        url: {**item, "source": country.lower(), "company": company, "ticker": seed["id"], "sector": seed["sector"],
              "country": country, "doc_type": doc_type(item["url"] + " " + item["title"], country)}
        for url, item in found.items()
    }


def read_seeds(paths: list[str], country: str) -> list[dict]:
    seeds, domains = [], set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            cid, sector, website, *rest = (part.strip() for part in line.split("\t"))
            if domain(website) in domains:  # the same website under two ids
                continue
            domains.add(domain(website))
            seeds.append({"id": cid, "sector": sector, "website": website, "country": country,
                          "check": rest[0] if rest else ""})
    return seeds


def _render_child(seed: dict, args: argparse.Namespace, conn) -> None:
    """One website in a browser, in a process of its own (see crawl)."""
    os.setpgrp()  # its own process group: killing it also ends the browser
    fetch = BrowserFetcher()
    try:
        conn.send(crawl_site(seed, fetch, RobotsCache(), args))
    finally:
        fetch.close()


def _render(seed: dict, args: argparse.Namespace) -> dict[str, dict]:
    """crawl_site in a browser with a hard time limit: a page can hang Chromium for good, so the
    website gets a process that is killed when its time is up."""
    import multiprocessing
    import signal

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_render_child, args=(seed, args, child), daemon=True)
    proc.start()
    child.close()
    items: dict[str, dict] = {}
    try:
        if parent.poll(args.site_seconds + 60):
            items = parent.recv()
    except (EOFError, OSError):
        pass
    finally:
        # SIGTERM first: the Playwright driver closes the browser on it; then nothing is left alive
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
            if not proc.is_alive():
                break
            try:
                os.killpg(proc.pid, sig)
            except OSError:
                proc.kill()
            proc.join(wait)
        parent.close()
    return items


def crawl(args: argparse.Namespace) -> None:
    """PDF links of the websites of a shard, written to --out as each website is done (JSON lines),
    so a run stopped from outside keeps what it found. No website is started after --budget-minutes."""
    seeds = read_seeds(args.seeds, args.country)
    k, n = (int(x) for x in args.shard.split("/"))
    seeds = [s for s in seeds if int(hashlib.sha1(s["id"].encode()).hexdigest(), 16) % n == k - 1]
    print(f"{len(seeds)} {args.country} websites (shard {args.shard})", flush=True)
    stop_at = time.monotonic() + args.budget_minutes * 60
    robots = RobotsCache()
    found: dict[str, dict[str, dict]] = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    lock = threading.Lock()

    def record(seed: dict, items: dict[str, dict], how: str) -> None:
        with lock:
            site = found.setdefault(seed["id"], {})
            new = {url: item for url, item in items.items() if url not in site}
            site.update(new)
            with out.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(i, ensure_ascii=False) + "\n" for i in new.values())
            print(f"{seed['id']} {seed['website']}: {len(found[seed['id']])} PDF links ({how})", flush=True)

    def static(seed: dict) -> None:
        if time.monotonic() > stop_at:
            return
        fetch = StaticFetcher()
        try:
            items = crawl_site(seed, fetch, robots, args)
        except Exception as exc:
            print(f"{seed['id']}: {type(exc).__name__}: {exc}", flush=True)
            items = {}
        finally:
            fetch.close()
        record(seed, items, "HTML")

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(static, seeds))

    sparse = [s for s in seeds if s["id"] in found and len(found[s["id"]]) < args.render_below
              and robots.allowed(s["website"])]
    if args.render and sparse:
        print(f"rendering {len(sparse)} websites with few documents in a browser", flush=True)

        def render(seed: dict) -> None:
            if time.monotonic() <= stop_at:
                record(seed, _render(seed, args), "browser")

        with concurrent.futures.ThreadPoolExecutor(args.render_workers) as pool:
            list(pool.map(render, sparse))
    skipped = len(seeds) - len(found)
    total = sum(len(v) for v in found.values())
    print(f"{total} PDF links from {sum(1 for v in found.values() if v)} of {len(seeds)} websites"
          + (f"; {skipped} not reached within the time budget" if skipped else ""), flush=True)


def seeds_wikidata(args: argparse.Namespace) -> None:
    """Seed list from Wikidata (one query; the public endpoint allows few requests per minute)."""
    request = urllib.request.Request(
        f"{WIKIDATA}?{urllib.parse.urlencode({'query': WIKIDATA_QUERIES[args.country]})}",
        headers={"User-Agent": f"{UA} python-urllib", "Accept": "application/sparql-results+json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                rows = json.load(response)["results"]["bindings"]
            break
        except Exception as exc:  # rate limit or a busy endpoint
            print(f"Wikidata: {exc}; retrying in 90 s", flush=True)
            time.sleep(90)
    else:
        sys.exit("Wikidata did not answer")
    exclude = {domain(s["website"]) for path in args.exclude for s in read_seeds([path], args.country)}
    tld = {"VN": ".vn", "JP": ".jp"}[args.country]
    sites: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)  # company → [(website, industry)]
    for row in rows:
        site = row["site"]["value"].strip()
        host = domain(site) if site.startswith("http") else ""
        if (not host or host in exclude or PUBLIC_DOMAINS.search(host)
                or any(host == h or host.endswith("." + h) for h in SKIP_HOSTS)):
            continue
        qid = row["item"]["value"].rsplit("/", 1)[1]
        cid = re.sub(r"\W", "", row.get("ticker", {}).get("value", "")) or qid
        sites[cid].append((site, re.sub(r"\s+", " ", row.get("industry", {}).get("value", "other"))))
    seeds, hosts = [], set()
    for cid, options in sorted(sites.items()):
        # one website per company: the one under the country's domain, else the shortest address
        site, sector = min(options, key=lambda o: (not domain(o[0]).endswith(tld), len(o[0]), o[0]))
        if domain(site) not in hosts:
            hosts.add(domain(site))
            seeds.append((cid, sector, site))
    lines = [f"# {args.country} companies from Wikidata ({WIKIDATA}): id <TAB> industry <TAB> official website.",
             "# Regenerate: python scripts/collect_corpus.py seeds-wikidata " + args.country + " <this file>"]
    lines += ["\t".join(v) for v in seeds]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(seeds)} websites written to {args.out}")


# --- selection, download, storage ------------------------------------------------------------


def select(candidates: list[dict], per_company: int, per_source: dict[str, int], have: dict[str, int]) -> list[dict]:
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
        room = per_company - have.get(company, 0)
        while len(picked) < room and any(by_type.values()):
            for kind in sorted(by_type):
                if by_type[kind] and len(picked) < room:
                    picked.append(by_type[kind].pop(0))
        chosen += picked
    limited, counts = [], collections.Counter()
    for item in sorted(chosen, key=lambda i: hashlib.sha1((i["company"] + i["url"]).encode()).hexdigest()):
        if counts[item["source"]] < per_source.get(item["source"], 10**9):
            counts[item["source"]] += 1
            limited.append(item)
    return limited


def keep_pages(sha: str, pages: int, limit: int) -> list[int]:
    """The pages a long document is stored with: chosen by its checksum, in document order."""
    if pages <= limit:
        return list(range(pages))
    return sorted(sorted(range(pages), key=lambda i: hashlib.sha1(f"{sha}:{i}".encode()).hexdigest())[:limit])


INSPECTION = 3  # version of the description below; earlier entries are described again

# letters only Vietnamese uses (French or Spanish accents are not among them)
_VI_ONLY = set("ăắằẳẵặơớờởỡợưứừửữựđạảấầẩẫậẹẻẽếềểễệỉịọỏốồổỗộụủỳỵỷỹ")
_EN_WORDS = frozenset("the and of to in for is on with by as at from that this are be was our we or an it which".split())


def confirmed_language(text: str) -> str:
    """vi, en or ja when the text clearly is that language; "other" for anything else (French,
    Spanish, Chinese, ...), which a choice among vi / en / ja alone would file under the nearest."""
    from docextract.textutil import detect_language

    text = unicodedata.normalize("NFC", text)
    language = detect_language(text, ("vi", "en", "ja"))
    if language is None:
        return "unknown"
    latin = [ch for ch in text.lower() if ch.isalpha() and ord(ch) < 0x2E80]
    if language == "vi":
        ok = sum(ch in _VI_ONLY for ch in latin) >= 0.03 * len(latin)
    elif language == "en":
        words = re.findall(r"[a-z]+", text.lower())
        ok = bool(words) and sum(w in _EN_WORDS for w in words) >= 0.05 * len(words)
    else:  # kana: Japanese; kanji alone: Chinese
        cjk = sum(1 for ch in text if 0x3040 <= ord(ch) <= 0x9FFF)
        kana = sum(1 for ch in text if 0x3040 <= ord(ch) <= 0x30FF)
        ok = kana >= 0.1 * cjk
    return language if ok else "other"


def language_of(pages) -> str:
    """Language of the pages' text layers, from text that is not broken or legacy-encoded only:
    a garbled text layer (wrong ToUnicode map, TCVN3/VNI fonts) can look like any language."""
    from docextract.textutil import garbled_ratio

    texts = [t for t in (p.get_text("text") for p in pages) if t.strip() and garbled_ratio(t) <= 0.02]
    return confirmed_language(" ".join(texts)[:20000])


def store(data: bytes, sha: str, path: Path) -> dict:
    """Write the document (at most MAX_PAGES / SCAN_PAGES pages of it) and describe it."""
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("encrypted")
        total = doc.page_count
        sample = [doc[i] for i in range(min(total, 8))]
        chars = [len(p.get_text("text").strip()) for p in sample]
        language = language_of(sample)
        digital = sum(1 for n in chars if n >= 200)
        kind = "digital" if digital >= max(1, len(chars) * 0.6) else ("scan" if digital == 0 else "mixed")
        keep = keep_pages(sha, total, SCAN_PAGES if kind == "scan" else MAX_PAGES)
        if len(keep) < total:
            doc.select(keep)
            path.write_bytes(doc.tobytes(garbage=3, deflate=True))
        else:
            path.write_bytes(data)
    info = {"pages": len(keep), "source_pages": total, "kind": kind, "language": language, "inspected": INSPECTION}
    if len(keep) < total:
        info["kept_pages"] = keep
    return info


def _get(url: str, max_mb: int) -> bytes | None:
    try:
        with client() as c, c.stream("GET", url) as response:
            if response.status_code != 200:
                return None
            return read_body(response, max_mb * 2**20, 120 + 10 * max_mb)
    except Exception:
        return None


def _read_manifest(path: str) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line] if p.exists() else []


def download(args: argparse.Namespace) -> None:
    """Add documents of the candidate lists to the manifest (earlier entries are kept while
    robots.txt still allows them; files no entry refers to are deleted)."""
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    robots = RobotsCache()
    previous = _read_manifest(args.manifest)

    def still_allowed(item: dict) -> bool:  # an unreachable robots.txt is no reason to drop a document
        rules = robots.get(item["url"])
        return rules.blocked or rules.allowed(item["url"])

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        allowed = list(pool.map(still_allowed, previous))
    manifest = [i for i, ok in zip(previous, allowed) if ok]
    if len(manifest) < len(previous):
        print(f"{len(previous) - len(manifest)} earlier documents removed: robots.txt disallows them", flush=True)
    import pymupdf

    for item in manifest:  # entries of earlier versions
        path = directory / item["file"]
        if not path.exists():
            continue
        if "source_pages" not in item:  # stored whole
            item.update(store(path.read_bytes(), item["sha256"], path))
        elif item.get("inspected", 1) < INSPECTION:
            with pymupdf.open(path) as doc:
                item.update(language=language_of([doc[i] for i in range(min(doc.page_count, 8))]), inspected=INSPECTION)

    candidates = [json.loads(line) for path in args.candidates if Path(path).exists()
                  for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    known = {i["url"] for i in manifest}
    have = collections.Counter(i["company"] for i in manifest)
    chosen = select([c for c in candidates if c["url"] not in known], args.per_company,
                    {"vn": args.max_vn, "jp": args.max_jp}, have)
    print(f"{len(candidates)} candidates → {len(chosen)} new selected", flush=True)
    budget = args.max_gb * 2**30 - sum(p.stat().st_size for p in directory.glob("*.pdf"))
    lock = threading.Lock()
    hashes = {i["sha256"] for i in manifest}

    def one(item: dict) -> dict | None:
        nonlocal budget
        if budget <= 0 or not robots.allowed(item["url"]):
            return None
        data = _get(item["url"], args.max_mb)
        if not data or not data.startswith(b"%PDF"):
            return None
        sha = hashlib.sha256(data).hexdigest()
        with lock:
            if sha in hashes:  # the same file under another URL
                return None
            hashes.add(sha)
        path = directory / f"{sha[:16]}.pdf"
        try:
            info = store(data, sha, path)
        except Exception:
            path.unlink(missing_ok=True)
            return None
        with lock:
            budget -= path.stat().st_size
        return {**item, **info, "sha256": sha, "file": path.name, "bytes": len(data), "split": split_of(item["company"])}

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        added = [r for r in pool.map(one, chosen) if r]
    manifest += added
    referenced = {i["file"] for i in manifest}
    for path in directory.glob("*.pdf"):
        if path.name not in referenced:
            path.unlink()
    _write_jsonl(args.manifest, manifest)
    print(f"{len(added)} documents added")
    print(stats(manifest))


def fetch(args: argparse.Namespace) -> None:
    """Download the files of a manifest that are not in the directory yet (checksums verified)."""
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    items = _read_manifest(args.manifest)
    if args.split:
        items = [i for i in items if i["split"] in args.split.split(",")]
    missing = [i for i in items if not (directory / i["file"]).exists()]
    print(f"{len(items)} documents, {len(missing)} to download", flush=True)
    robots = RobotsCache()

    def one(item: dict) -> bool:
        if not robots.allowed(item["url"]):
            return False
        data = _get(item["url"], 200)
        if not data or hashlib.sha256(data).hexdigest() != item["sha256"]:
            return False  # changed or removed at the source: skipped, not replaced
        store(data, item["sha256"], directory / item["file"])
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
    lines += ["", "| country | companies | documents | pages stored |", "|---|---|---|---|"]
    for country in sorted({i["country"] for i in items}):
        docs = [i for i in items if i["country"] == country]
        lines.append(f"| {country} | {len({i['company'] for i in docs})} | {len(docs)} | {sum(d['pages'] for d in docs)} |")
    types = collections.Counter(i["doc_type"] for i in items)
    lines += ["", f"{len(items)} documents from {len({i['company'] for i in items})} companies; types: "
              + ", ".join(f"{k} {v}" for k, v in types.most_common())]
    return "\n".join(lines)


def _write_jsonl(path: str, items: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("seeds-wikidata")
    p.add_argument("country", choices=sorted(WIKIDATA_QUERIES))
    p.add_argument("out")
    p.add_argument("--exclude", nargs="*", default=[], help="seed files whose websites are left out")
    p.set_defaults(func=seeds_wikidata)
    p = sub.add_parser("crawl")
    p.add_argument("country", choices=sorted(IR_WORDS))
    p.add_argument("seeds", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", default="1/1", help="k/n: only the k-th of n parts of the websites")
    p.add_argument("--max-pages", type=int, default=60, help="pages read per website")
    p.add_argument("--max-pdfs", type=int, default=80, help="PDF links collected per website")
    p.add_argument("--site-seconds", type=int, default=600, help="time spent per website")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--render", action="store_true", help="crawl websites with few documents again in a browser")
    p.add_argument("--render-below", type=int, default=10)
    p.add_argument("--render-workers", type=int, default=4)
    p.add_argument("--budget-minutes", type=float, default=150, help="no website is started after this")
    p.set_defaults(func=crawl)
    p = sub.add_parser("download")
    p.add_argument("candidates", nargs="+")
    p.add_argument("--dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--per-company", type=int, default=10)
    p.add_argument("--max-vn", type=int, default=2500, help="new Vietnamese documents per run")
    p.add_argument("--max-jp", type=int, default=2000, help="new Japanese documents per run")
    p.add_argument("--max-mb", type=int, default=60, help="largest file downloaded")
    p.add_argument("--max-gb", type=float, default=5, help="size of all stored files")
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
    p.set_defaults(func=lambda a: print(stats(_read_manifest(a.manifest))))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
