"""Image cleanup for scans and photos: orientation, resolution, deskew, contrast."""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image, ImageOps, ImageSequence, ImageStat

from ..config import Settings
from ..engines.base import OrientationClassifier

log = logging.getLogger(__name__)


def load_image_pages(data: bytes) -> list[Image.Image]:
    """All frames of an image file (multi-page TIFF gives several pages), EXIF-rotated, RGB."""
    image = Image.open(io.BytesIO(data))
    pages = []
    for frame in ImageSequence.Iterator(image):
        frame = ImageOps.exif_transpose(frame.copy())
        if frame.mode in ("RGBA", "LA", "P"):
            frame = frame.convert("RGBA")
            background = Image.new("RGB", frame.size, "white")
            background.paste(frame, mask=frame.getchannel("A"))
            frame = background
        pages.append(frame.convert("RGB"))
    return pages


def normalize_resolution(image: Image.Image, min_side: int, max_side: int) -> tuple[Image.Image, float]:
    """Resize so the long side is within [min_side, max_side]; returns the image and the factor used."""
    long_side = max(image.size)
    if long_side < min_side:
        factor = min_side / long_side
    elif long_side > max_side:
        factor = max_side / long_side
    else:
        return image, 1.0
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.resize(size, Image.Resampling.LANCZOS), factor


def _ink_mask(image: Image.Image, width: int = 1000) -> np.ndarray | None:
    gray = image.convert("L")
    if gray.width > width:
        gray = gray.resize((width, max(1, round(gray.height * width / gray.width))), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    threshold = min(160.0, arr.mean() - arr.std() * 0.5)
    ink = arr < threshold
    ratio = ink.mean()
    if ratio < 0.002 or ratio > 0.4:  # blank page or photo: projection profiles are meaningless
        return None
    return ink


def _profile_score(ink_img: Image.Image, angle: float) -> float:
    rotated = ink_img.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
    rows = np.asarray(rotated, dtype=np.float32).sum(axis=1)
    return float(np.square(np.diff(rows)).sum())


def estimate_skew(image: Image.Image, max_angle: float = 5.0) -> float:
    """Angle (degrees, counter-clockwise) that makes text lines horizontal; 0 when unsure."""
    ink = _ink_mask(image)
    if ink is None:
        return 0.0
    ink_img = Image.fromarray((ink * 255).astype(np.uint8))
    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.5)
    scores = [_profile_score(ink_img, a) for a in coarse]
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1)
    fine_scores = [_profile_score(ink_img, a) for a in fine]
    best = float(fine[int(np.argmax(fine_scores))])
    baseline = _profile_score(ink_img, 0.0)
    # Require a clear improvement over no rotation to avoid "correcting" noise.
    if baseline > 0 and max(fine_scores) < baseline * 1.05:
        return 0.0
    return round(best, 2)


def enhance(image: Image.Image) -> tuple[Image.Image, bool]:
    """Stretch contrast of faded scans; leaves good images untouched (models dislike binarisation)."""
    stat = ImageStat.Stat(image.convert("L"))
    if stat.stddev[0] >= 45:
        return image, False
    return ImageOps.autocontrast(image, cutoff=1), True


def cleanup(
    image: Image.Image,
    settings: Settings,
    orientation: OrientationClassifier | None = None,
    resize: bool = False,
) -> tuple[Image.Image, dict]:
    """Apply the enabled preprocessing steps; returns the image and a record of what was done."""
    ops: dict = {}
    if resize:
        image, factor = normalize_resolution(image, settings.min_image_side, settings.max_image_side)
        if factor != 1.0:
            ops["resize_factor"] = round(factor, 4)
    if settings.auto_orientation and orientation is not None:
        try:
            angle = orientation.classify([image])[0]
        except Exception as exc:  # orientation is best effort
            log.warning("orientation classification failed: %s", exc)
            angle = 0
        if angle:
            image = image.rotate(angle, expand=True, fillcolor="white")
            ops["rotation"] = angle
    if settings.deskew:
        skew = estimate_skew(image)
        if abs(skew) >= 0.3:
            image = image.rotate(skew, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
            ops["deskew"] = skew
    if settings.enhance_contrast:
        image, changed = enhance(image)
        if changed:
            ops["autocontrast"] = True
    return image, ops
