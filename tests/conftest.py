"""Shared fixtures: synthetic documents and fake engines (no PaddleOCR / VLM needed)."""

from __future__ import annotations

import io
import json

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw

from docextract.config import Settings
from docextract.engines.base import FormulaResult, LayoutBox, OcrResult, TableResult, VlmResult
from docextract.models import BBox
from docextract.textutil import Line


def make_digital_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 70), "ACME Annual Report", fontsize=24)
    page.insert_text((72, 110), "1. Introduction", fontsize=15)
    y = 135
    for i in range(4):
        page.insert_text((72, y), f"Revenue in 2025 reached 1,234.{i} million USD, up 12.{i}% year on year.", fontsize=10)
        y += 14
    for r in range(4):
        for c in range(3):
            rect = pymupdf.Rect(72 + c * 120, 220 + r * 22, 72 + (c + 1) * 120, 220 + (r + 1) * 22)
            page.draw_rect(rect, width=0.7)
            label = ["Item", "Q1", "Q2"][c] if r == 0 else f"{r * 10 + c}.5"
            page.insert_text((rect.x0 + 4, rect.y0 + 15), label, fontsize=10)
    page.insert_text((72, 330), "2. Outlook", fontsize=15)
    page.insert_text((72, 352), "We expect growth to continue.", fontsize=10)
    page.insert_text((290, 825), "1", fontsize=9)
    return doc.tobytes()


# Scanned page: flat grey blocks whose shade tells the fake engines what they are looking at.
SCAN_SIZE = (600, 800)
SCAN_BLOCKS = {
    "title": (BBox(x0=60, y0=40, x1=540, y1=90), 60),
    "text": (BBox(x0=60, y0=120, x1=540, y1=260), 110),
    "table": (BBox(x0=60, y0=300, x1=540, y1=480), 160),
    "image": (BBox(x0=60, y0=520, x1=400, y1=740), 210),
}


def make_scan_image() -> Image.Image:
    image = Image.new("RGB", SCAN_SIZE, "white")
    draw = ImageDraw.Draw(image)
    for bbox, shade in SCAN_BLOCKS.values():
        draw.rectangle(bbox.as_tuple(), fill=(shade, shade, shade))
    return image


def make_scanned_pdf() -> bytes:
    buf = io.BytesIO()
    make_scan_image().save(buf, format="PNG")
    doc = pymupdf.open()
    page = doc.new_page(width=SCAN_SIZE[0], height=SCAN_SIZE[1])
    page.insert_image(page.rect, stream=buf.getvalue())
    return doc.tobytes()


def shade_of(image: Image.Image) -> int:
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    inner = arr[6:-6, 6:-6] if arr.shape[0] > 12 and arr.shape[1] > 12 else arr
    return int(round(float(np.median(inner)) / 10) * 10)


class FakeLayout:
    name = "fake-layout"

    def detect(self, images):
        labels = {"title": "doc_title", "text": "text", "table": "table", "image": "image"}
        return [
            [LayoutBox(label=labels[key], bbox=bbox, score=0.95) for key, (bbox, _) in SCAN_BLOCKS.items()]
            for _ in images
        ]


class FakeOCR:
    """Reads the block shade; the text block comes back with Vietnamese letters dropped."""

    name = "fake-ocr"

    def __init__(self, texts: dict[int, tuple[str, float]] | None = None) -> None:
        self.calls = 0
        self.texts = texts or {
            60: ("BÁO CÁO TÀI CHÍNH NĂM 2025", 0.99),
            110: ("Doanh thu năm 2025 đạt 1.234,5 tỷ đồng, tăng 12,3% so với năm trước.", 0.97),
            160: ("Chỉ tiêu Năm 2024 Năm 2025 Doanh thu 1.100 1.234,5", 0.95),
            210: ("", 0.0),
        }

    def recognize(self, images):
        self.calls += len(images)
        out = []
        for image in images:
            text, score = self.texts.get(shade_of(image), ("", 0.0))
            lines = [Line(text=text, x0=0, y0=0, x1=100, y1=20, score=score)] if text else []
            out.append(OcrResult(text=text, lines=lines))
        return out


class FakeTable:
    name = "fake-table"

    def __init__(self, html: str | None = None, confidence: float = 0.95) -> None:
        self.html = html or (
            "<table><tr><td>Chỉ tiêu</td><td>Năm 2024</td><td>Năm 2025</td></tr>"
            "<tr><td>Doanh thu</td><td>1.100</td><td>1.234,5</td></tr></table>"
        )
        self.confidence = confidence

    def recognize(self, images):
        return [TableResult(html=self.html, confidence=self.confidence, ocr_text="") for _ in images]


class FakeFormula:
    name = "fake-formula"

    def recognize(self, images):
        return [FormulaResult(latex="E = mc^{2}") for _ in images]


class FakeVLM:
    name = "fake-vlm"

    def __init__(self, image_reply: dict | None = None) -> None:
        self.requests = []
        self.image_reply = image_reply or {
            "kind": "logo",
            "description": "Biểu tượng công ty ACME màu xám.",
            "lossless": False,
        }

    def run(self, requests):
        self.requests.extend(requests)
        out = []
        for request in requests:
            data = None
            if request.task == "table":
                text = (
                    "<table><tr><th>Chỉ tiêu</th><th>Năm 2024</th><th>Năm 2025</th></tr>"
                    "<tr><td>Doanh thu</td><td>1.100</td><td>1.234,5</td></tr></table>"
                )
            elif request.task == "image":
                data = dict(self.image_reply)
                text = json.dumps(data, ensure_ascii=False)
            elif request.task == "formula":
                text = "E = mc^{2}"
            else:
                data = {"chart_type": "bar", "title": "Doanh thu", "description": "Tăng đều.", "columns": ["Năm", "Doanh thu"],
                        "rows": [["2024", "1.100"], ["2025", "1.234,5"]], "lossless": True}
                text = json.dumps(data, ensure_ascii=False)
            out.append(VlmResult(text=text, input_tokens=100, output_tokens=20, data=data))
        return out


@pytest.fixture
def settings() -> Settings:
    return Settings(dpi=72, deskew=False, enhance_contrast=False, auto_orientation=False)
