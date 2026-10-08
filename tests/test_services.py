import json
import time

import httpx
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from docextract import DocumentPipeline
from docextract.api import app, set_pipeline
from docextract.benchmark import run_benchmark
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
    assert client.post("/v1/extract", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415

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


def test_prompts_languages():
    assert "Vietnamese, English or Japanese" in prompt_for("image", None, None)
    assert "in English." in prompt_for("chart", None, "en")
    assert "in Japanese." in prompt_for("image", None, "ja")
    assert "lossless" in prompt_for("image", None, None) and "lossless" in prompt_for("chart", None, None)
    assert "1.234,5" in prompt_for("table", "Doanh thu 1.234,5", None)


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
