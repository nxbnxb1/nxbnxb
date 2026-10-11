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


def test_seeds_with_ticker_check(tmp_path):
    seeds = tmp_path / "seeds.tsv"
    seeds.write_text("# comment\nVNM\tconsumer\thttps://www.vinamilk.com.vn\nHPG\tsteel\thttps://hoaphat.com.vn\tticker\n"
                     "XYZ\tother\thttps://vinamilk.com.vn/\n", encoding="utf-8")
    read = cc.read_seeds([str(seeds)], "VN")
    assert [(s["id"], s["check"]) for s in read] == [("VNM", ""), ("HPG", "ticker")]  # same website once


def test_languages_outside_the_products_are_not_filed_under_the_nearest():
    assert cc.confirmed_language("Cộng hòa xã hội chủ nghĩa Việt Nam. Báo cáo tài chính hợp nhất năm 2025 đã được kiểm toán.") == "vi"
    assert cc.confirmed_language("The Board of Directors approved the annual report and the dividend for the year.") == "en"
    assert cc.confirmed_language("2026年3月期の決算短信について、当社は以下のとおりお知らせいたします。") == "ja"
    assert cc.confirmed_language("Société Générale publie son rapport financier annuel. Les résultats sont présentés à l'assemblée.") == "other"
    assert cc.confirmed_language("Banco Santander presenta los resultados del año. La junta general de accionistas aprobó el dividendo.") == "other"
    assert cc.confirmed_language("河内嘉佩乐酒店概况介绍，酒店位于市中心，提供豪华客房和餐饮服务以及会议设施。") == "other"
