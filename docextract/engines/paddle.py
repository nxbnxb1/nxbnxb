"""PaddleOCR 3.x adapters: PP-StructureV3 layout module, PP-OCR, table and formula recognition.

Only the individual modules are used (not the full PP-StructureV3 pipeline) so that no
region is processed twice. Models load lazily on first use; predictors are not
thread-safe, so every call holds the adapter's lock.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..config import Settings
from ..models import BBox
from ..textutil import Line, join_lines, normalize
from .base import FormulaResult, LayoutBox, OcrResult, TableResult

log = logging.getLogger(__name__)


def paddle_available() -> bool:
    return importlib.util.find_spec("paddleocr") is not None


def to_bgr(image: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])


def _res(result: Any) -> dict:
    """PaddleX result objects expose their payload as ``.json['res']``."""
    data = getattr(result, "json", None)
    if callable(data):
        data = data()
    if isinstance(data, dict):
        return data.get("res", data)
    return dict(result)


_PADDLE_LANG = {"vi": "vi", "en": "en", "ja": "japan"}


def _rec_model(settings: Settings) -> dict[str, str]:
    """Recognition model overrides; a custom model dir carries its architecture name in inference.yml."""
    kwargs: dict[str, str] = {}
    name = settings.ocr_rec_model
    if settings.ocr_rec_model_dir:
        kwargs["text_recognition_model_dir"] = settings.ocr_rec_model_dir
        if not name:
            config = Path(settings.ocr_rec_model_dir) / "inference.yml"
            match = re.search(r"^\s*model_name:\s*([\w.-]+)", config.read_text(encoding="utf-8"), re.M) if config.exists() else None
            name = match.group(1) if match else None
    if name:
        kwargs["text_recognition_model_name"] = name
    return kwargs


def _prepare_crop(image: Image.Image, min_height: int = 48, pad: int = 12) -> Image.Image:
    """Upscale tiny crops and add a white margin: the text detector misses text touching the border."""
    image = image.convert("RGB")
    if image.height < min_height:
        factor = min_height / max(1, image.height)
        image = image.resize((max(1, round(image.width * factor)), min_height), Image.Resampling.BICUBIC)
    return ImageOps.expand(image, border=pad, fill="white")


class _PaddleAdapter:
    name = "paddle"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _common(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"enable_mkldnn": self.settings.paddle_enable_mkldnn}
        if self.settings.device:
            kwargs["device"] = self.settings.device
        return kwargs

    def _load(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _predict(self, *args, **kwargs) -> list:
        with self._lock:
            if self._model is None:
                log.info("loading %s", self.name)
                self._model = self._load()
            return list(self._model.predict(*args, **kwargs))


class PaddleLayoutDetector(_PaddleAdapter):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.name = f"paddle:{settings.layout_model}"

    def _load(self):
        from paddleocr import LayoutDetection

        return LayoutDetection(model_name=self.settings.layout_model, **self._common())

    def detect(self, images: list[Image.Image]) -> list[list[LayoutBox]]:
        if not images:
            return []
        results = self._predict(
            [to_bgr(im) for im in images],
            batch_size=min(len(images), 4),
            threshold=self.settings.layout_threshold,
            layout_nms=True,
        )
        out = []
        for result in results:
            boxes = []
            for box in _res(result).get("boxes", []):
                boxes.append(
                    LayoutBox(label=str(box["label"]), bbox=BBox.from_seq(box["coordinate"]), score=float(box["score"]))
                )
            out.append(boxes)
        return out


class PaddleTextRecognizer(_PaddleAdapter):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        model = settings.ocr_rec_model or settings.ocr_version or "default"
        if settings.ocr_rec_model_dir:
            model = f"{model}@{Path(settings.ocr_rec_model_dir).name}"
        self.name = f"paddle:ocr:{model}:{settings.ocr_lang}"

    def _load(self):
        from paddleocr import PaddleOCR

        s = self.settings
        kwargs = dict(
            lang=_PADDLE_LANG[s.ocr_lang],
            ocr_version=s.ocr_version,
            use_doc_orientation_classify=False,  # done once per page in preprocessing
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **self._common(),
        )
        if s.ocr_det_model:
            kwargs["text_detection_model_name"] = s.ocr_det_model
        kwargs.update(_rec_model(s))
        if s.ocr_det_unclip_ratio is not None:
            kwargs["text_det_unclip_ratio"] = s.ocr_det_unclip_ratio
        if s.ocr_det_box_thresh is not None:
            kwargs["text_det_box_thresh"] = s.ocr_det_box_thresh
        if s.ocr_det_limit_side_len is not None:
            kwargs["text_det_limit_side_len"] = s.ocr_det_limit_side_len
        return PaddleOCR(**kwargs)

    def recognize(self, images: list[Image.Image]) -> list[OcrResult]:
        if not images:
            return []
        results = self._predict([to_bgr(_prepare_crop(im)) for im in images])
        return [self._to_ocr_result(_res(r)) for r in results]

    @staticmethod
    def _to_ocr_result(res: dict) -> OcrResult:
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        boxes = res.get("rec_boxes")
        if boxes is None or len(boxes) != len(texts):
            boxes = [_poly_to_box(p) for p in (res.get("rec_polys") or [])]
        lines = []
        for text, score, box in zip(texts, scores, boxes):
            x0, y0, x1, y1 = (float(v) for v in list(box)[:4])
            lines.append(Line(text=normalize(str(text)), x0=x0, y0=y0, x1=x1, y1=y1, score=float(score)))
        return OcrResult(text=join_lines(lines), lines=lines)


def _poly_to_box(poly) -> list[float]:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]


class PaddleTableRecognizer(_PaddleAdapter):
    """TableRecognitionPipelineV2 run on the cropped table only (its own layout step is off)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.name = "paddle:table:TableRecognitionPipelineV2"

    def _load(self):
        from paddleocr import TableRecognitionPipelineV2

        s = self.settings
        kwargs = dict(use_layout_detection=False, use_doc_orientation_classify=False, use_doc_unwarping=False)
        if s.ocr_det_model:
            kwargs["text_detection_model_name"] = s.ocr_det_model
        kwargs.update(_rec_model(s))  # same (Vietnamese) recogniser for table cells
        return TableRecognitionPipelineV2(**kwargs, **self._common())

    def recognize(self, images: list[Image.Image]) -> list[TableResult]:
        out = []
        for image in images:  # the pipeline handles one table image per call
            results = self._predict(
                to_bgr(_prepare_crop(image, pad=8)),
                use_layout_detection=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_table_orientation_classify=False,
            )
            tables = _res(results[0]).get("table_res_list") or [] if results else []
            if not tables:
                out.append(TableResult(html="", confidence=0.0))
                continue
            table = tables[0]
            html = re.sub(r"</?(html|body)>", "", str(table.get("pred_html", "")))
            ocr = table.get("table_ocr_pred") or {}
            texts = [normalize(str(t)) for t in ocr.get("rec_texts", [])]
            scores = [float(v) for v in ocr.get("rec_scores", [])]
            confidence = sum(scores) / len(scores) if scores else None
            out.append(TableResult(html=html, confidence=confidence, ocr_text=" ".join(texts)))
        return out


class PaddleFormulaRecognizer(_PaddleAdapter):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.name = f"paddle:{settings.formula_model}"

    def _load(self):
        from paddleocr import FormulaRecognition

        return FormulaRecognition(model_name=self.settings.formula_model, **self._common())

    def recognize(self, images: list[Image.Image]) -> list[FormulaResult]:
        if not images:
            return []
        results = self._predict([to_bgr(_prepare_crop(im, pad=6)) for im in images], batch_size=min(4, len(images)))
        return [FormulaResult(latex=str(_res(r).get("rec_formula", "")).strip()) for r in results]


class PaddleOrientationClassifier(_PaddleAdapter):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.name = f"paddle:{settings.orientation_model}"

    def _load(self):
        from paddleocr import DocImgOrientationClassification

        return DocImgOrientationClassification(model_name=self.settings.orientation_model, **self._common())

    def classify(self, images: list[Image.Image]) -> list[int]:
        if not images:
            return []
        out = []
        for result in self._predict([to_bgr(im) for im in images]):
            res = _res(result)
            labels = res.get("label_names") or ["0"]
            scores = res.get("scores") or [1.0]
            angle = int(labels[0]) if float(scores[0]) >= 0.6 else 0
            out.append(angle % 360)
        return out
