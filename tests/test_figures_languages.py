"""General rules: when a picture may be replaced by text, and Japanese text handling."""

import pytest

from docextract.config import Settings
from docextract.figures import keep_figure
from docextract.models import Method, Region, RegionType, ValidationStatus
from docextract.textutil import Line, extract_numbers, join_lines


def _picture(rtype=RegionType.IMAGE, figure=None, figure_text="", chart=None, status=ValidationStatus.UNCHECKED):
    data = {"figure": figure} if figure is not None else {}
    if figure_text is not None:
        data["figure_text"] = figure_text
    if chart is not None:
        data["chart"] = chart
    return Region(id="p1-r1", page=1, type=rtype, method=Method.VLM, status=status, data=data)


LOSSLESS_DIAGRAM = {"kind": "diagram", "description": "A → B", "lossless": True}


@pytest.mark.parametrize(
    "region, keep",
    [
        (_picture(figure=LOSSLESS_DIAGRAM, figure_text="A B"), False),
        (_picture(figure={"kind": "photo", "description": "Ảnh nhà máy", "lossless": True}), True),
        (_picture(figure={"kind": "diagram", "description": "A → B", "lossless": False}), True),
        (_picture(figure=LOSSLESS_DIAGRAM, figure_text=None), True),  # labels not read by OCR
        (_picture(figure=None), True),  # no structured description
        (_picture(RegionType.SEAL, figure=LOSSLESS_DIAGRAM, figure_text="CÔNG TY"), True),
    ],
)
def test_keep_figure_rules(region, keep):
    assert keep_figure(region, Settings())[0] is keep


def test_chart_kept_unless_every_value_is_printed():
    info = {"kind": "chart", "description": "Doanh thu", "lossless": True}
    printed = {"rows": [["2024", "1.100"], ["2025", "1.234,5"]]}
    assert not keep_figure(_picture(RegionType.CHART, info, "2024 2025 1.100 1.234,5", printed), Settings())[0]
    estimated = {"rows": [["2024", "~1.100"]]}
    assert keep_figure(_picture(RegionType.CHART, info, "2024", estimated), Settings())[0]
    unseen = {"rows": [["2024", "1.100"], ["2025", "9.999"]]}
    assert keep_figure(_picture(RegionType.CHART, info, "2024 2025 1.100", unseen), Settings())[0]
    assert keep_figure(_picture(RegionType.CHART, info, "2024", {"rows": []}), Settings())[0]


def test_unrecovered_table_keeps_its_picture_and_policy_overrides():
    table = Region(id="t", page=1, type=RegionType.TABLE, status=ValidationStatus.NEEDS_REVIEW)
    assert keep_figure(table, Settings())[0]
    table.status = ValidationStatus.PASSED
    assert not keep_figure(table, Settings())[0]
    assert keep_figure(table, Settings(figure_policy="always"))[0]
    assert not keep_figure(_picture(figure=None), Settings(figure_policy="never"))[0]


def test_japanese_lines_join_without_spaces():
    lines = [Line("日本語の文書を", 0, 0, 100, 10), Line("処理します。", 0, 12, 100, 22), Line("Next English line", 0, 24, 100, 34)]
    assert join_lines(lines) == "日本語の文書を処理します。 Next English line"


def test_full_width_numbers_match_ascii():
    assert extract_numbers("売上高は１，２３４．５億円") == extract_numbers("1,234.5")
