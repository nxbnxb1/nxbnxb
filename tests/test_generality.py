"""General mechanisms: continuation across columns/pages, OCR agreement, office formats."""

import pytest

from docextract import DocumentPipeline
from docextract.config import Settings
from docextract.engines.base import Engines, OcrResult
from docextract.merge import render_markdown
from docextract.models import (
    DocumentResult,
    Method,
    PageKind,
    PageResult,
    Region,
    RegionType,
    SourceInfo,
    ValidationStatus,
)
from docextract.preprocessing import office
from docextract.preprocessing.loader import detect_format
from docextract.textutil import Line

from .conftest import FakeLayout, FakeOCR, make_scanned_pdf


def _doc(pages: list[list[Region]]) -> DocumentResult:
    return DocumentResult(
        source=SourceInfo(filename="x.pdf", sha256="0", size_bytes=1, format="pdf", page_count=len(pages)),
        pages=[PageResult(number=i + 1, kind=PageKind.DIGITAL, regions=regs) for i, regs in enumerate(pages)],
    )


def _text(rid, page, order, content):
    return Region(id=rid, page=page, type=RegionType.TEXT, order=order, content=content,
                  method=Method.PDF_TEXT, status=ValidationStatus.PASSED)


def _table(rid, page, order, rows):
    html = "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"
    return Region(id=rid, page=page, type=RegionType.TABLE, order=order, html=html, method=Method.PDF_TABLE,
                  status=ValidationStatus.PASSED)


def test_paragraph_split_by_a_page_break_is_joined():
    doc = _doc([[_text("a", 1, 0, "Doanh thu năm nay tăng mạnh nhờ")], [_text("b", 2, 0, "mở rộng thị trường.")]])
    md = render_markdown(doc, Settings(markdown_page_markers=False))
    assert md.strip() == "Doanh thu năm nay tăng mạnh nhờ mở rộng thị trường."
    assert doc.pages[1].regions[0].meta["continues"] == "a"


def test_finished_paragraphs_stay_separate():
    doc = _doc([[_text("a", 1, 0, "Câu đã kết thúc."), _text("b", 1, 1, "câu mới bắt đầu.")],
                [_text("c", 2, 0, "Không nối")], ])
    md = render_markdown(doc, Settings(markdown_page_markers=False))
    assert md.count("\n\n") == 2


def test_japanese_paragraph_continues_without_space():
    doc = _doc([[_text("a", 1, 0, "当期の売上高は増加し")], [_text("b", 2, 0, "ました。")]])
    assert render_markdown(doc, Settings(markdown_page_markers=False)).strip() == "当期の売上高は増加しました。"


def test_table_continued_on_next_page_is_merged_without_repeated_header():
    doc = _doc([[_table("t1", 1, 0, [["A", "B"], ["1", "2"]])], [_table("t2", 2, 0, [["A", "B"], ["3", "4"]])]])
    md = render_markdown(doc, Settings(markdown_page_markers=False))
    assert md.strip() == "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"


def test_tables_with_different_columns_are_not_merged():
    doc = _doc([[_table("t1", 1, 0, [["A", "B"], ["1", "2"]])], [_table("t2", 2, 0, [["X", "Y", "Z"]])]])
    assert "| 1 | 2 |\n\n| X | Y | Z |" in render_markdown(doc, Settings(markdown_page_markers=False))


class UnstableOCR(FakeOCR):
    """Reads differently when the picture is rescaled: an uncertain region."""

    def recognize(self, images):
        out = []
        for image in images:
            text = "Doanh thu 1.234" if image.width < 700 else "Doamh thv 7.Z34"
            out.append(OcrResult(text=text, lines=[Line(text, 0, 0, 10, 10, score=0.95)]))
        return out


def test_disagreeing_ocr_readings_are_flagged_and_original_kept(settings):
    engines = Engines(layout=FakeLayout(), ocr=UnstableOCR())
    result = DocumentPipeline(settings, engines).process_bytes(make_scanned_pdf(), "scan.pdf")
    text = next(r for r in result.iter_regions() if r.type == RegionType.TEXT)
    assert text.status == ValidationStatus.NEEDS_REVIEW
    assert any("disagree" in issue for issue in text.issues)
    assert text.figure and f"]({text.figure})" in result.markdown  # original picture kept for review


def test_stable_ocr_passes(settings):
    result = DocumentPipeline(settings, Engines(layout=FakeLayout(), ocr=FakeOCR())).process_bytes(
        make_scanned_pdf(), "scan.pdf"
    )
    text = next(r for r in result.iter_regions() if r.type == RegionType.TEXT)
    assert text.status == ValidationStatus.PASSED and text.meta["ocr_agreement"] == 1.0 and text.figure is None


@pytest.mark.parametrize("name", ["slides.pptx", "sheet.xlsx", "page.html", "notes.txt", "old.doc", "a.rtf", "b.odt"])
def test_other_formats_go_through_libreoffice(name):
    assert detect_format(b"anything", name) == "office"


def test_missing_libreoffice_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(office.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="LibreOffice"):
        office.convert(b"x", "slides.pptx", "pdf")
