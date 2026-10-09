"""Enterprise corpus collection (scripts/collect_corpus.py): the parts that need no network."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import collect_corpus as cc  # noqa: E402


def test_robots_wildcards_and_groups():
    robots = cc.Robots(
        "User-agent: Googlebot\nDisallow: /\n\n"
        "User-agent: *\nDisallow: /*.pdf$\nDisallow: /api/\nAllow: /api/public/\nCrawl-delay: 2\n"
    )
    assert not robots.allowed("https://x.vn/files/report.pdf")
    assert robots.allowed("https://x.vn/files/report.pdf?v=2")  # "$" anchors the end
    assert not robots.allowed("https://x.vn/api/media/a")
    assert robots.allowed("https://x.vn/api/public/a")  # the longer rule wins
    assert robots.allowed("https://x.vn/quan-he-co-dong/")
    assert robots.delay == 2


def test_robots_everything_disallowed_and_unreachable():
    assert not cc.Robots("User-agent: *\nDisallow: /\n").allowed("https://x.jp/ir/")
    assert cc.Robots("User-agent: *\nDisallow:\n").allowed("https://x.jp/ir/")
    assert cc.Robots("").allowed("https://x.jp/ir/")  # no robots.txt
    assert not cc.Robots(None).allowed("https://x.jp/ir/")  # robots.txt unreachable
    own = cc.Robots(f"User-agent: *\nDisallow: /\n\nUser-agent: {cc.AGENT}\nAllow: /\n")
    assert own.allowed("https://x.jp/ir/")  # a group naming this crawler replaces "*"


def test_fold_keeps_japanese_and_drops_vietnamese_accents():
    assert cc.fold("Quan hệ Nhà đầu tư") == "quan-he-nha-dau-tu"
    assert cc.fold("IR情報・ガバナンス") == "ir情報-ガバナンス"


def test_page_links_finds_pdfs_in_attributes_and_scripts():
    page = """<a href='/vi/nha-dau-tu'>Quan hệ <b>cổ đông</b></a>
    <a href="docs/bctc-2025.pdf" title="BCTC">Tải về</a>
    <div data-file="/files/nghi-quyet.pdf"></div>
    <script>var docs = [{"url": "https:\\/\\/cdn.x.vn\\/bao-cao-thuong-nien.pdf"}];</script>"""
    links = dict(cc.page_links("https://x.vn/ir/", page))
    assert links["https://x.vn/vi/nha-dau-tu"] == "Quan hệ cổ đông"
    assert "https://x.vn/ir/docs/bctc-2025.pdf" in links
    assert "https://x.vn/files/nghi-quyet.pdf" in links
    assert "https://cdn.x.vn/bao-cao-thuong-nien.pdf" in links


def test_kept_pages_are_reproducible():
    keep = cc.keep_pages("ab" * 32, 200, 20)
    assert keep == cc.keep_pages("ab" * 32, 200, 20) == sorted(keep)
    assert len(set(keep)) == 20 and max(keep) < 200
    assert cc.keep_pages("ab" * 32, 7, 20) == list(range(7))


def test_doc_types():
    assert cc.doc_type("https://x.vn/Bao%20cao%20tai%20chinh%202025.pdf", "VN") == "financial_statement"
    assert cc.doc_type("2026年3月期 決算短信〔日本基準〕(連結)", "JP") == "earnings_summary"
    assert cc.doc_type("第46回定時株主総会招集ご通知", "JP") == "agm_notice"


def test_split_is_by_company():
    assert cc.split_of("vn:ACB") == cc.split_of("vn:ACB")
    shares = [cc.split_of(f"jp:{i}") for i in range(2000)]
    assert 0.6 < shares.count("train") / 2000 < 0.8
