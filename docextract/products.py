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


PRODUCTS: dict[str, Product] = {
    "vi_en": Product(
        key="vi_en",
        name="Việt + Anh",
        languages=("vi", "en"),
        model="vi_en_PP-OCRv5_mobile_rec",
        baseline_model="latin_PP-OCRv5_mobile_rec",
        paddle_lang="vi",
    ),
    "vi_en_ja": Product(
        key="vi_en_ja",
        name="Việt + Anh + Nhật",
        languages=("vi", "en", "ja"),
        model="vi_en_ja_PP-OCRv5_mobile_rec",
        baseline_model="PP-OCRv5_mobile_rec",
        paddle_lang="japan",
    ),
}

LANGUAGE_NAMES = {"vi": "Vietnamese", "en": "English", "ja": "Japanese"}
