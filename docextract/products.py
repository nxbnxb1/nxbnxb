"""The two products built from this code base. Each one is self-contained: its own languages,
its own fine-tuned recognition model, and its own baseline (the stock PaddleOCR model that
fine-tuning starts from, used when no fine-tuned model is installed and in benchmarks)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    key: str
    name: str
    languages: tuple[str, ...]
    model: str  # fine-tuned recognition model (GitHub Release asset <model>.tar.gz)
    baseline_model: str  # stock PaddleOCR recognition model = baseline of this product
    paddle_lang: str  # PaddleOCR language code used with the baseline
    tesseract_langs: str  # Tesseract languages of the product (benchmark baseline)
    release: str  # GitHub Release of the fine-tuned model the product is pinned to
    latin_product: str | None = None  # product whose model reads lines without kana/kanji


PRODUCTS: dict[str, Product] = {
    "vi_en": Product(
        key="vi_en",
        name="Việt + Anh",
        languages=("vi", "en"),
        model="vi_en_PP-OCRv5_mobile_rec",
        baseline_model="latin_PP-OCRv5_mobile_rec",
        paddle_lang="vi",
        tesseract_langs="vie+eng",
        release="vi_en-ocr-7",
    ),
    "vi_en_ja": Product(
        key="vi_en_ja",
        name="Việt + Anh + Nhật",
        languages=("vi", "en", "ja"),
        model="vi_en_ja_PP-OCRv5_mobile_rec",
        baseline_model="PP-OCRv5_mobile_rec",
        paddle_lang="japan",
        tesseract_langs="vie+eng+jpn",
        release="vi_en_ja-ocr-4",
        # Vietnamese and English lines are read better by the vi_en model (scanned documents:
        # English 93.9% vs 86.4% word F1, Vietnamese 88.8% vs 80.3%)
        latin_product="vi_en",
    ),
}

LANGUAGE_NAMES = {"vi": "Vietnamese", "en": "English", "ja": "Japanese"}
