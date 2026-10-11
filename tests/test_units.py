from docextract.layout.labels import Candidate, clean_candidates, normalize_label
from docextract.layout.reading_order import reading_order
from docextract.metrics import cer, cer_any_order, heading_f1, teds
from docextract.models import BBox, RegionType
from docextract.tables import parse_html_table, parse_markdown_table
from docextract.textutil import (
    extract_numbers,
    garbled_ratio,
    join_lines,
    Line,
    normalize,
    vietnamese_dropout_ratio,
)


def test_normalize_composes_vietnamese():
    decomposed = "Việt Nam"
    assert normalize(decomposed) == "Việt Nam"


def test_garbled_detects_legacy_tcvn3_and_cid():
    assert garbled_ratio("Céng hßa x· héi chñ nghÜa ViÖt Nam §éc lËp - Tù do - H¹nh phóc") > 0.05
    assert garbled_ratio("(cid:12)(cid:13)(cid:14) abc") > 0.5
    assert garbled_ratio("Cộng hòa xã hội chủ nghĩa Việt Nam - Độc lập - Tự do - Hạnh phúc") == 0.0


def test_numbers_are_locale_independent():
    assert extract_numbers("1.234,5 tỷ và 1,234.5 USD, tăng 12,3%") == ["12345", "12345", "123"]


def test_vietnamese_dropout_signal():
    assert vietnamese_dropout_ratio("Cng hòa xã hi ch nghĩa Vit Nam") > 0.08
    assert vietnamese_dropout_ratio("Cộng hòa xã hội chủ nghĩa Việt Nam") == 0.0
    assert vietnamese_dropout_ratio("UBND TP HCM ban hành quyết định số 15/QĐ-UBND") == 0.0
    assert vietnamese_dropout_ratio("BÁO CÁO KT QU KINH DOANH NM 2025") > 0.08
    assert vietnamese_dropout_ratio("BÁO CÁO KẾT QUẢ KINH DOANH NĂM 2025 CỦA CTCP ABC") == 0.0
    assert vietnamese_dropout_ratio("The quick brown fox jumps over Mr Smith") == 0.0


def test_join_lines_paragraphs_and_hyphens():
    lines = [
        Line("Hello wor-", 0, 0, 100, 10),
        Line("ld again", 0, 12, 100, 22),
        Line("New paragraph", 0, 40, 100, 50),
    ]
    assert join_lines(lines) == "Hello world again\n\nNew paragraph"


def test_table_spans_and_rendering():
    html = (
        "<table><tr><th rowspan='2'>A</th><th colspan='2'>B</th></tr>"
        "<tr><td>b1</td><td>b2</td></tr><tr><td>x</td><td>1</td><td>2</td></tr></table>"
    )
    table = parse_html_table(html)
    assert (table.n_rows, table.n_cols) == (3, 3)
    assert table.is_rectangular and table.has_spans
    assert table.render("auto").startswith("<table>")
    simple = parse_markdown_table("| a | b |\n|---|---|\n| 1 | 2 |")
    assert simple.to_markdown() == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_teds():
    a = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    b = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>3</td></tr></table>"
    c = "<table><tr><td>a</td></tr></table>"
    assert teds(a, a) == 1.0
    assert 0.8 < teds(a, b) < 1.0
    assert teds(a, b, structure_only=True) == 1.0
    assert teds(a, c) < 0.5


def test_cer_and_headings():
    assert cer("# Xin chào", "Xin chào") == 0.0
    assert 0 < cer("Việt Nam", "Vit Nam") < 0.2
    assert heading_f1(["A", "B"], ["A", "B"]) == 1.0
    assert heading_f1(["A", "B"], ["A"]) == 2 / 3


def test_cer_any_order_ignores_reading_order_only():
    gt = "Doanh thu năm 2025 tăng 12%\nLợi nhuận sau thuế 1.234 tỷ đồng\nTiền cuối kỳ 567 tỷ"
    swapped = "Tiền cuối kỳ 567 tỷ\nDoanh thu năm 2025 tăng 12%\nLợi nhuận sau thuế 1.234 tỷ đồng"
    assert cer(gt, swapped) > 0.4 and cer_any_order(gt, swapped) == 0.0
    assert 0 < cer_any_order(gt, swapped.replace("567", "561")) < 0.05  # a wrong digit still counts
    assert cer_any_order(gt, "Doanh thu năm 2025 tăng 12%") > 0.5  # missing lines count
    assert cer_any_order(gt, gt + "\nchữ thừa không có trong đáp án") > 0.3  # so does extra text
    assert cer_any_order("Item 2024 2025\nRevenue 10 12", "| Item | 2024 | 2025 |\n|---|---|---|\n| Revenue | 10 | 12 |") == 0


def test_reading_order_two_columns_with_spanning_title():
    boxes = [
        BBox(x0=300, y0=100, x1=500, y1=200),  # right column, top
        BBox(x0=50, y0=100, x1=250, y1=200),  # left column, top
        BBox(x0=50, y0=20, x1=500, y1=60),  # title spanning both columns
        BBox(x0=50, y0=220, x1=250, y1=300),  # left column, bottom
        BBox(x0=300, y0=220, x1=500, y1=300),  # right column, bottom
        BBox(x0=250, y0=780, x1=300, y1=790),  # page number
    ]
    types = [RegionType.TEXT] * 5 + [RegionType.PAGE_NUMBER]
    assert reading_order(boxes, types, 550) == [2, 1, 3, 0, 4, 5]


def test_labels_and_cleanup():
    assert normalize_label("doc_title") == RegionType.TITLE
    assert normalize_label("paragraph_title") == RegionType.HEADING
    assert normalize_label("chart") == RegionType.CHART
    table = Candidate(RegionType.TABLE, BBox(x0=0, y0=0, x1=100, y1=100), 0.9)
    inner = Candidate(RegionType.TEXT, BBox(x0=10, y0=10, x1=50, y1=50), 0.8)
    dup = Candidate(RegionType.TEXT, BBox(x0=200, y0=0, x1=300, y1=50), 0.7)
    dup2 = Candidate(RegionType.TEXT, BBox(x0=201, y0=0, x1=300, y1=50), 0.6)
    kept = clean_candidates([table, inner, dup, dup2], page_area=1e6, min_area_ratio=0.0)
    assert table in kept and dup in kept
    assert inner not in kept and dup2 not in kept
