from docextract.config import Settings
from docextract.engines.base import Engines
from docextract.executor import Extraction, PageContext, RegionTask
from docextract.models import BBox, Method, PageKind, Region, RegionType, ValidationStatus
from docextract.router import RouteFeatures, RuleBasedRouter
from docextract.tables import parse_html_table
from docextract.validation import validate

from .conftest import FakeFormula, FakeOCR, FakeTable, FakeVLM


def _router(**overrides) -> RuleBasedRouter:
    engines = Engines(ocr=FakeOCR(), table=FakeTable(), formula=FakeFormula(), vlm=FakeVLM())
    return RuleBasedRouter(Settings(**overrides), engines)


def _features(rtype: RegionType, kind: PageKind = PageKind.SCANNED, area: float = 0.1, chars: int = 0, **kw):
    return RouteFeatures(rtype, kind, area, 2.0, 0.9, chars, **kw)


def test_text_is_never_sent_to_the_vlm():
    router = _router()
    for rtype in (RegionType.TEXT, RegionType.TITLE, RegionType.HEADING, RegionType.LIST, RegionType.SEAL, RegionType.FOOTER):
        assert router.plan(_features(rtype))[0] == [Method.OCR]
        assert router.plan(_features(rtype, PageKind.DIGITAL, chars=200))[0] == [Method.PDF_TEXT, Method.OCR]
    assert router.plan(_features(RegionType.TEXT, full_page=True))[0] == [Method.OCR]


def test_table_formula_chart_image_plans():
    router = _router()
    assert router.plan(_features(RegionType.TABLE, PageKind.DIGITAL, chars=50))[0] == [
        Method.PDF_TABLE,
        Method.TABLE_RECOGNITION,
        Method.VLM,
        Method.PDF_TEXT,
    ]
    assert router.plan(_features(RegionType.TABLE, area=0.6))[0][:2] == [Method.VLM, Method.TABLE_RECOGNITION]
    assert router.plan(_features(RegionType.FORMULA))[0][:2] == [Method.FORMULA_RECOGNITION, Method.VLM]
    assert router.plan(_features(RegionType.CHART))[0] == [Method.VLM, Method.OCR]
    plan, reason = router.plan(_features(RegionType.IMAGE, area=0.001))
    assert plan == [] and "small" in reason


def test_unavailable_engines_are_dropped():
    router = RuleBasedRouter(Settings(), Engines())
    assert router.plan(_features(RegionType.TEXT))[0] == []
    assert router.plan(_features(RegionType.TABLE, PageKind.DIGITAL, chars=10))[0] == [Method.PDF_TABLE, Method.PDF_TEXT]


def _task(rtype: RegionType) -> RegionTask:
    page = PageContext(number=1, kind=PageKind.SCANNED, width=100, height=100)
    region = Region(id="p1-r1", page=1, type=rtype, bbox=BBox(x0=0, y0=0, x1=50, y1=50))
    return RegionTask(region=region, page=page)


def test_ocr_with_dropped_vietnamese_letters_needs_review():
    task = _task(RegionType.TEXT)
    bad = Extraction(Method.OCR, content="Cng hòa xã hi ch nghĩa Vit Nam", confidence=0.98)
    report = validate(task, bad, Settings())
    assert not report.passed and report.status == ValidationStatus.NEEDS_REVIEW
    assert any("Vietnamese" in issue for issue in report.issues)
    good = Extraction(Method.OCR, content="Cộng hòa xã hội chủ nghĩa Việt Nam", confidence=0.98)
    assert validate(task, good, Settings()).passed


def test_low_confidence_ocr_fails():
    task = _task(RegionType.TEXT)
    ext = Extraction(Method.OCR, content="some text", confidence=0.5, data={"line_scores": [0.5]})
    assert not validate(task, ext, Settings()).passed


def test_table_checks():
    task = _task(RegionType.TABLE)
    ragged = parse_html_table("<table><tr><td>a</td><td>b</td></tr><tr><td>1</td></tr></table>")
    ext = Extraction(Method.TABLE_RECOGNITION, table=ragged, confidence=0.95)
    report = validate(task, ext, Settings())
    assert not report.passed and "rows have different numbers of cells" in report.issues

    task.evidence["text_layer"] = "a b 1 2 3"
    good = parse_html_table("<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>")
    report = validate(task, Extraction(Method.VLM, table=good), Settings())
    assert not report.passed and any("numbers missing" in i for i in report.issues)


def test_formula_checks():
    task = _task(RegionType.FORMULA)
    assert validate(task, Extraction(Method.FORMULA_RECOGNITION, content="\\frac{a}{b}"), Settings()).passed
    assert not validate(task, Extraction(Method.FORMULA_RECOGNITION, content="\\frac{a}{b"), Settings()).passed
    assert not validate(task, Extraction(Method.FORMULA_RECOGNITION, content="x " * 20), Settings()).passed
