"""Pipeline settings. Every field can be set from the environment as ``DOCEXTRACT_<FIELD>``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ENV_PREFIX = "DOCEXTRACT_"

Backend = Literal["auto", "paddle", "none"]


class Settings(BaseModel):
    # --- Input / preprocessing -------------------------------------------------
    dpi: int = Field(200, ge=72, le=600, description="Render resolution for PDF pages")
    max_pages: int | None = Field(None, description="Process at most this many pages")
    page_batch_size: int = Field(8, ge=1, description="Pages held in memory at once")
    min_text_chars: int = Field(20, description="Text-layer chars needed to treat a PDF page as digital")
    trust_ocr_text_layer: bool = Field(
        False, description="Trust the hidden text layer of scanned-and-OCRed PDFs instead of re-running OCR"
    )
    deskew: bool = True
    auto_orientation: bool = True
    enhance_contrast: bool = True
    min_image_side: int = Field(1200, description="Upscale image inputs whose long side is shorter")
    max_image_side: int = Field(4000, description="Downscale image inputs whose long side is longer")

    # --- Layout ----------------------------------------------------------------
    layout_backend: Literal["auto", "paddle", "heuristic"] = "auto"
    layout_model: str = "PP-DocLayout_plus-L"
    layout_threshold: float = 0.5
    min_region_area_ratio: float = Field(0.0002, description="Drop layout boxes smaller than this page fraction")

    # --- OCR / table / formula (PaddleOCR) -------------------------------------
    ocr_backend: Backend = "auto"
    ocr_lang: Literal["vi", "en"] = Field("vi", description="Documents are Vietnamese and/or English")
    ocr_version: str | None = Field("PP-OCRv5", description="PP-OCRv5, PP-OCRv6, ...; None = PaddleOCR default")
    ocr_det_model: str | None = Field(None, description="Override text detection model name")
    ocr_rec_model: str | None = Field(None, description="Override text recognition model name")
    ocr_rec_model_dir: str | None = Field(None, description="Custom (e.g. fine-tuned Vietnamese) recognition model")
    ocr_det_unclip_ratio: float | None = Field(
        None, description="Grow detected text boxes; larger keeps stacked Vietnamese diacritics inside the box"
    )
    ocr_det_box_thresh: float | None = None
    ocr_det_limit_side_len: int | None = None
    table_backend: Backend = "auto"
    formula_backend: Backend = "auto"
    formula_model: str = "PP-FormulaNet_plus-M"
    orientation_model: str = "PP-LCNet_x1_0_doc_ori"
    device: str | None = Field(None, description="Paddle device, e.g. 'cpu' or 'gpu:0'")
    paddle_enable_mkldnn: bool = Field(
        False, description="oneDNN on CPU; off by default because PaddlePaddle 3.3 fails with it on some models"
    )

    # --- VLM (OpenAI-compatible endpoint, e.g. vLLM serving Qwen-VL) ------------
    vlm_base_url: str | None = Field(None, description="e.g. http://localhost:8000/v1; VLM is off when unset")
    vlm_api_key: str | None = None
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_max_tokens: int = 2048
    vlm_temperature: float = 0.0
    vlm_timeout: float = 120.0
    vlm_max_concurrency: int = 4
    vlm_max_retries: int = 2
    vlm_max_image_side: int = Field(1600, description="Crops sent to the VLM are downscaled to this")
    vlm_price_input_per_1m: float = Field(0.0, description="Cost per 1M input tokens, for cost/page")
    vlm_price_output_per_1m: float = 0.0
    vlm_language: Literal["vi", "en"] | None = Field(
        None, description="Language of image/chart descriptions; None = Vietnamese if the image has Vietnamese text"
    )

    # --- Router / validation ---------------------------------------------------
    describe_images: bool = True
    min_image_area_ratio: float = Field(0.01, description="Smaller images (logos, icons) are not described")
    prefer_vlm_for_complex_tables: bool = True
    complex_table_cells: int = Field(120, description="Tables with more cells count as complex")
    verify_vlm_with_ocr: bool = Field(True, description="Cross-check VLM numbers against OCR tokens")
    ocr_min_confidence: float = 0.85
    ocr_low_line_confidence: float = 0.6
    ocr_max_low_lines_ratio: float = 0.25
    table_min_confidence: float = 0.80
    table_max_empty_ratio: float = 0.6
    pdf_text_max_garbled_ratio: float = 0.02
    number_match_min: float = Field(0.9, description="Share of reference numbers a result must contain")
    recover_orphan_text: bool = Field(True, description="Keep PDF text that no layout box covered")

    # --- Output ----------------------------------------------------------------
    output_locale: Literal["vi", "en"] = "vi"
    table_format: Literal["auto", "markdown", "html"] = "auto"
    markdown_page_markers: bool = True
    markdown_include_furniture: bool = False
    markdown_review_markers: bool = True

    # --- Infrastructure --------------------------------------------------------
    cache_dir: str | None = Field(None, description="Directory for the result cache; None disables it")
    route_log_path: str | None = Field(None, description="JSONL log of router decisions (Neural Router data)")
    workers: int = Field(4, ge=1, description="Threads for running method groups in parallel")

    @property
    def vlm_enabled(self) -> bool:
        return bool(self.vlm_base_url)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **overrides) -> Settings:
        """Build settings from ``DOCEXTRACT_*`` variables (and ``DOCEXTRACT_CONFIG`` JSON file)."""
        env = dict(os.environ if env is None else env)
        data: dict[str, object] = {}
        config_file = env.get(f"{ENV_PREFIX}CONFIG")
        if config_file:
            data.update(json.loads(Path(config_file).read_text(encoding="utf-8")))
        for name, field in cls.model_fields.items():
            raw = env.get(f"{ENV_PREFIX}{name.upper()}")
            if raw is None:
                continue
            allows_none = field.default is None or "None" in repr(field.annotation)
            data[name] = None if allows_none and raw.strip().lower() in ("", "none", "null") else raw
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(data)

    def public_dict(self) -> dict:
        """Settings snapshot without secrets."""
        return self.model_dump(exclude={"vlm_api_key"})
