import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from docextract import DocumentPipeline
from docextract.api import app, set_pipeline
from docextract.benchmark import compare_reports, run_benchmark
from docextract.cli import main as cli_main
from docextract.config import Settings
from docextract.engines.base import Engines, VlmRequest
from docextract.engines.vlm import OpenAICompatibleVLM, prompt_for
from docextract.preprocessing.image import estimate_skew

from .conftest import make_digital_pdf


def test_api_extract_and_jobs(settings):
    set_pipeline(DocumentPipeline(settings, Engines()))
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    files = {"file": ("report.pdf", make_digital_pdf(), "application/pdf")}
    body = client.post("/v1/extract", files=files, data={"use_vlm": "false"}).json()
    assert "# ACME Annual Report" in body["markdown"]
    assert body["document"]["source"]["filename"] == "report.pdf"
    junk = {"file": ("x.bin", b"\x00\x01 not a document", "application/octet-stream")}
    assert client.post("/v1/extract", files=junk).status_code == 415

    job = client.post("/v1/jobs", files=files).json()
    for _ in range(100):
        status = client.get(f"/v1/jobs/{job['id']}").json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "done"
    assert client.get(f"/v1/jobs/{job['id']}/markdown").text.startswith("<!-- trang: 1 -->")


def test_vlm_client_retries_and_parses_chart(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "busy"})
        payload = json.loads(request.content)
        assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        content = '```json\n{"chart_type": "bar", "description": "Tăng", "columns": ["Năm", "Giá trị"], "rows": [["2024", "1"]]}\n```'
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 50, "completion_tokens": 9}}
        )

    settings = Settings(vlm_base_url="http://vlm.test/v1", vlm_max_retries=2)
    client = httpx.Client(base_url="http://vlm.test/v1", transport=httpx.MockTransport(handler))
    vlm = OpenAICompatibleVLM(settings, client=client)
    monkeypatch.setattr("docextract.engines.vlm.time.sleep", lambda seconds: None)
    (result,) = vlm.run([VlmRequest(image=Image.new("RGB", (50, 50), "white"), task="chart")])
    assert calls["n"] == 2
    assert result.data["chart_type"] == "bar" and result.input_tokens == 50


def test_prompts_follow_the_product_languages():
    assert "(Vietnamese or English)" in prompt_for("image", None, None)
    assert "(Vietnamese, English or Japanese)" in prompt_for("image", None, None, languages=("vi", "en", "ja"))
    assert "in English." in prompt_for("chart", None, "en")
    assert "in Japanese." in prompt_for("image", None, "ja")
    assert "lossless" in prompt_for("image", None, None) and "lossless" in prompt_for("chart", None, None)
    assert "1.234,5" in prompt_for("table", "Doanh thu 1.234,5", None)


def test_products_have_their_own_baseline_and_languages():
    import pytest

    vi_en = Settings(product="vi_en")
    vi_en_ja = Settings(product="vi_en_ja", output_locale="ja")
    assert vi_en.product_info.baseline_model == "latin_PP-OCRv5_mobile_rec"
    assert vi_en_ja.product_info.baseline_model == "PP-OCRv5_mobile_rec"
    assert vi_en_ja.product_info.languages == ("vi", "en", "ja")
    with pytest.raises(ValueError):
        Settings(product="vi_en", output_locale="ja")  # Japanese is not part of the vi_en product


def test_deskew_estimate():
    font = ImageFont.load_default()
    image = Image.new("RGB", (1000, 1300), "white")
    draw = ImageDraw.Draw(image)
    for i in range(30):
        draw.rectangle((80, 80 + i * 38, 900, 92 + i * 38), fill="black")
        draw.text((80, 95 + i * 38), "x", font=font, fill="black")
    rotated = image.rotate(2.0, fillcolor="white")
    assert abs(estimate_skew(rotated) + 2.0) < 0.3
    assert estimate_skew(image) == 0.0


def test_benchmark_and_cli(tmp_path, settings, monkeypatch):
    (tmp_path / "report.pdf").write_bytes(make_digital_pdf())
    (tmp_path / "report.gt.md").write_text(
        "# ACME Annual Report\n\n## 1. Introduction\n\n"
        + " ".join(f"Revenue in 2025 reached 1,234.{i} million USD, up 12.{i}% year on year." for i in range(4))
        + "\n\n| Item | Q1 | Q2 |\n|---|---|---|\n| 10.5 | 11.5 | 12.5 |\n| 20.5 | 21.5 | 22.5 |\n| 30.5 | 31.5 | 32.5 |\n\n"
        "## 2. Outlook\n\nWe expect growth to continue.\n",
        encoding="utf-8",
    )
    report = run_benchmark(tmp_path, pipeline=DocumentPipeline(settings, Engines()))
    summary = report["summary"]
    assert summary["cer"] < 0.02 and summary["teds"] == 1.0 and summary["heading_f1"] == 1.0
    assert summary["word_f1"] > 0.98

    # sub-directories are categories; a first-level dev/ or test/ is the split
    for split in ("dev", "test"):
        folder = tmp_path / "eval" / split / "reports"
        folder.mkdir(parents=True)
        for suffix in (".pdf", ".gt.md"):
            (folder / f"report{suffix}").write_bytes((tmp_path / f"report{suffix}").read_bytes())
    report = run_benchmark(tmp_path / "eval", pipeline=DocumentPipeline(settings, Engines()), split="test")
    assert [r["document"] for r in report["documents"]] == ["test/reports/report.pdf"]
    assert report["documents"][0]["category"] == "reports" and list(report["categories"]) == ["reports"]

    # several systems on the same documents: one table per category and overall, best in bold
    worse = json.loads(json.dumps(report))
    worse["summary"]["cer"] = report["summary"]["cer"] + 0.1
    worse["system"] = "other"
    table = compare_reports([report, worse])
    assert "**reports** (1 documents)" in table and "**all documents**" in table
    assert f"| docextract | **{report['summary']['cer']}** |" in table

    # a baseline without OCR: the text layer of the PDF
    from docextract.baselines import TextLayerSystem

    plain = run_benchmark(tmp_path, system=TextLayerSystem(), system_name="text layer")
    assert plain["system"] == "text layer" and plain["summary"]["word_f1"] > 0.9

    # shards of the documents (several runners) merge into the report of the whole set
    from docextract.benchmark import merge_reports

    whole = run_benchmark(tmp_path / "eval", system=TextLayerSystem(), system_name="text layer")
    parts = [run_benchmark(tmp_path / "eval", system=TextLayerSystem(), system_name="text layer", shard=f"{k}/3")
             for k in (1, 2, 3)]
    assert sum(len(part["documents"]) for part in parts) == len(whole["documents"]) == 2
    merged = merge_reports(parts)
    assert [r["document"] for r in merged["documents"]] == [r["document"] for r in whole["documents"]]
    assert merged["summary"]["word_f1"] == whole["summary"]["word_f1"]

    # a document a system cannot convert scores as empty output instead of stopping the run
    class Crashing(TextLayerSystem):
        def convert(self, path):
            raise MemoryError("out of memory")

    crashed = run_benchmark(tmp_path / "eval", split="test", system=Crashing(), system_name="crashing")
    assert crashed["summary"]["failed"] == 1 and crashed["summary"]["cer"] == 1.0
    assert crashed["documents"][0]["ms_per_page"] is None and "MemoryError" in crashed["markdown"]

    monkeypatch.setenv("DOCEXTRACT_LAYOUT_BACKEND", "heuristic")
    monkeypatch.setenv("DOCEXTRACT_OCR_BACKEND", "none")
    monkeypatch.setenv("DOCEXTRACT_TABLE_BACKEND", "none")
    monkeypatch.setenv("DOCEXTRACT_FORMULA_BACKEND", "none")
    out = tmp_path / "out"
    assert cli_main(["extract", str(tmp_path / "report.pdf"), "-o", str(out)]) == 0
    assert (out / "report.md").read_text(encoding="utf-8").count("# ACME Annual Report") == 1
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["source"]["filename"] == "report.pdf"


def test_settings_from_env():
    s = Settings.from_env({"DOCEXTRACT_DPI": "150", "DOCEXTRACT_VLM_BASE_URL": "none", "DOCEXTRACT_DESKEW": "false"})
    assert s.dpi == 150 and s.vlm_base_url is None and s.deskew is False


def test_isolated_system_survives_memory_exhaustion(tmp_path, monkeypatch):
    from docextract import baselines
    from docextract.baselines import IsolatedSystem

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(make_digital_pdf())
    system = IsolatedSystem("text_layer", timeout=120, max_memory=2**30)
    try:
        assert "ACME Annual Report" in system.convert(pdf).markdown
        # a worker above its memory limit is replaced; a finished document keeps its result
        monkeypatch.setattr(baselines, "_rss", lambda pid: 10**12)
        assert "ACME Annual Report" in system.convert(pdf).markdown
        assert system._proc is None
        # one that cannot even hold its models within the limit is an error the caller records
        with pytest.raises(MemoryError):
            system.convert(pdf)
    finally:
        system.close()
