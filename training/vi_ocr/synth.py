"""Synthetic text-line images for fine-tuning the PaddleOCR recogniser on Vietnamese + English.

Text comes from word-frequency lists (``wordfreq``) for both languages, document-style
patterns (amounts in đồng, dates, decision numbers, percentages, codes) and a coverage
mode that forces rare letters (ẵ, ỹ, ỵ, ...) to appear often enough to be learnt.
Lines are rendered with every installed font that covers Vietnamese, then degraded
like scans/photos (blur, low resolution, JPEG, noise, skew, ink spread).

Output follows PaddleOCR's SimpleDataSet format: ``<out>/images/*.jpg`` and
``<out>/labels.txt`` with ``relative/path<TAB>text`` per line.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from charset import build_charset, vietnamese_letters

CHARSET = set(build_charset()) | {" "}
_ONSETS = ["", "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh", "l", "m", "n", "ng", "ngh", "nh", "p", "ph", "qu", "r", "s", "t", "th", "tr", "v", "x"]
_CODAS = ["", "", "", "c", "ch", "m", "n", "ng", "nh", "p", "t", "i", "o", "u", "y"]
_UNITS = ["đồng", "VNĐ", "tỷ đồng", "triệu đồng", "USD", "%", "kg", "km", "m²", "tấn", "người", "hộ", "ha"]
_EN_UNITS = ["USD", "million", "billion", "%", "kg", "km", "units", "items"]
_ORGS = ["UBND", "HĐND", "TTg", "BTC", "NĐ-CP", "QĐ-UBND", "TT-BTC", "CT-TTg", "KH-UBND", "BC-UBND", "CV"]


def _zipf_weights(n: int, offset: float = 8.0) -> list[float]:
    return [1.0 / (i + offset) for i in range(n)]


class TextSource:
    def __init__(self, rng: random.Random) -> None:
        from wordfreq import top_n_list

        self.rng = rng
        self.vi = [w for w in top_n_list("vi", 12000) if self._ok(w)]
        self.en = [w for w in top_n_list("en", 25000) if self._ok(w) and w.isascii()]
        self.vi_w = _zipf_weights(len(self.vi))
        self.en_w = _zipf_weights(len(self.en), 30.0)
        self.letters = vietnamese_letters()

    @staticmethod
    def _ok(word: str) -> bool:
        return bool(word) and all(ch in CHARSET for ch in word) and any(ch.isalpha() for ch in word)

    def _words(self, pool: list[str], weights: list[float], n: int) -> list[str]:
        return self.rng.choices(pool, weights=weights, k=n)

    def syllable(self) -> str:
        """Random phonotactically plausible syllable; forces rare letters into the data."""
        vowel = self.rng.choice([ch for ch in self.letters if ch.islower()])
        onset = self.rng.choice(_ONSETS)
        coda = self.rng.choice(_CODAS)
        return onset + vowel + coda

    def number(self, vi: bool = True) -> str:
        r = self.rng.random()
        if r < 0.3:
            n = self.rng.randint(1000, 999_999_999)
            sep = "." if vi else ","
            return f"{n:,}".replace(",", sep)
        if r < 0.55:
            whole, frac = self.rng.randint(0, 999), self.rng.randint(0, 99)
            return f"{whole}{',' if vi else '.'}{frac}"
        if r < 0.75:
            return str(self.rng.randint(0, 2100))
        if r < 0.9:
            return f"{self.rng.randint(1, 28):02d}/{self.rng.randint(1, 12):02d}/{self.rng.randint(1975, 2030)}"
        return f"{self.rng.randint(1, 9999)}/{self.rng.choice(_ORGS)}"

    def vi_line(self) -> str:
        words = self._words(self.vi, self.vi_w, self.rng.randint(2, 7))
        if self.rng.random() < 0.35:
            words.insert(self.rng.randint(0, len(words)), self.number())
        if self.rng.random() < 0.2:
            words.append(self.rng.choice(_UNITS))
        return " ".join(words)

    def en_line(self) -> str:
        words = self._words(self.en, self.en_w, self.rng.randint(2, 6))
        if self.rng.random() < 0.3:
            words.insert(self.rng.randint(0, len(words)), self.number(vi=False))
        if self.rng.random() < 0.15:
            words.append(self.rng.choice(_EN_UNITS))
        return " ".join(words)

    def coverage_line(self) -> str:
        return " ".join(self.syllable() for _ in range(self.rng.randint(2, 6)))

    def misc_line(self) -> str:
        r = self.rng.random()
        if r < 0.25:
            return f"Số: {self.rng.randint(1, 9999)}/{self.rng.choice(_ORGS)}"
        if r < 0.45:
            return f"ngày {self.rng.randint(1, 31)} tháng {self.rng.randint(1, 12)} năm {self.rng.randint(1990, 2030)}"
        if r < 0.6:
            return f"{self.number()} {self.rng.choice(_UNITS)}"
        if r < 0.75:
            user = "".join(self.rng.choices("abcdefghijklmnopqrstuvwxyz0123456789._", k=self.rng.randint(4, 10)))
            return f"{user.strip('.')}@{self.rng.choice(['gmail.com', 'gov.vn', 'company.vn', 'mail.com'])}"
        if r < 0.85:
            return f"Tel: 0{self.rng.randint(20, 99)} {self.rng.randint(1000, 9999)} {self.rng.randint(1000, 9999)}"
        return f"({self.rng.choice('abcdefgh')}) {self.number()}; {self.number()} - {self.number()}"

    def line(self, max_len: int) -> str:
        r = self.rng.random()
        if r < 0.5:
            text = self.vi_line()
        elif r < 0.68:
            text = self.en_line()
        elif r < 0.83:
            text = self.coverage_line()
        else:
            text = self.misc_line()
        if self.rng.random() < 0.12:
            text += self.rng.choice([".", ",", ":", ";", "?", "!", " -", "…"])
        r = self.rng.random()
        if r < 0.15:
            text = text.upper()
        elif r < 0.27:
            text = " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
        elif r < 0.6:
            text = text[:1].upper() + text[1:]
        text = unicodedata.normalize("NFC", " ".join(text.split()))
        if len(text) > max_len:  # cut at a word boundary
            cut = text.rfind(" ", 0, max_len + 1)
            text = text[: cut if cut > 3 else max_len].strip()
        return text


def find_fonts(font_dirs: list[str]) -> list[str]:
    """Installed fonts that contain every Vietnamese letter."""
    from fontTools.ttLib import TTCollection, TTFont

    needed = {ord(ch) for ch in vietnamese_letters()} | {ord(ch) for ch in "0123456789AZaz%₫"}
    fonts = []
    for directory in font_dirs:
        for path in sorted(Path(directory).rglob("*")):
            if path.suffix.lower() not in (".ttf", ".otf"):
                continue
            try:
                font = TTFont(str(path), lazy=True, fontNumber=0)
                cmap = font.getBestCmap() or {}
            except Exception:
                continue
            name = path.name.lower()
            if "unifont" in name or "emoji" in name:
                continue  # bitmap / symbol fonts
            if needed <= set(cmap) and _draws_diacritics(str(path)):
                fonts.append(str(path))
    return fonts


_PAIRS = [("e", "ê"), ("a", "ã"), ("u", "ủ"), ("o", "ộ"), ("a", "ạ"), ("y", "ỹ"), ("A", "Ấ"), ("U", "Ự"), ("d", "đ"), ("i", "ị")]


def _draws_diacritics(font_path: str) -> bool:
    """Some fonts map Vietnamese code points to glyphs without the marks; their labels would be wrong."""
    try:
        font = ImageFont.truetype(font_path, 40)
    except OSError:
        return False

    def ink(ch: str) -> np.ndarray:
        image = Image.new("L", (80, 90), 255)
        ImageDraw.Draw(image).text((10, 10), ch, font=font, fill=0)
        return np.asarray(image) < 128

    for base, marked in _PAIRS:
        a, b = ink(base), ink(marked)
        if b.sum() <= a.sum() * 1.03 or np.array_equal(a, b):
            return False
    return True


def font_families(fonts: list[str]) -> list[list[str]]:
    """Group font files by family so families with many weights (Inter) do not dominate."""
    groups: dict[str, list[str]] = {}
    for path in fonts:
        stem = Path(path).stem
        family = stem.split("-")[0]
        for suffix in ("BoldOblique", "BoldItalic", "Bold", "Oblique", "Italic"):
            family = family.removesuffix(suffix)
        groups.setdefault(family, []).append(path)
    return list(groups.values())


def render(text: str, font_path: str, rng: random.Random, augment: bool = True) -> Image.Image:
    size = rng.randint(20, 44)
    font = ImageFont.truetype(font_path, size)
    left, top, right, bottom = font.getbbox(text)
    ascent, descent = font.getmetrics()
    top = min(top, 0)
    bottom = max(bottom, ascent + descent)
    pad_x, pad_y = rng.randint(2, 14), rng.randint(2, 9)
    width = right - left + 2 * pad_x
    height = bottom - top + 2 * pad_y
    bg = rng.randint(190, 255)
    tint = [max(0, min(255, bg + rng.randint(-12, 6))) for _ in range(3)]
    image = Image.new("RGB", (width, height), tuple(tint))
    ink = rng.randint(0, 90)
    ImageDraw.Draw(image).text((pad_x - left, pad_y - top), text, font=font, fill=(ink, ink, ink + rng.randint(0, 30)))
    return degrade(image, rng, tuple(tint)) if augment else image


def degrade(image: Image.Image, rng: random.Random, bg: tuple[int, int, int]) -> Image.Image:
    if rng.random() < 0.3:
        image = image.rotate(rng.uniform(-1.5, 1.5), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=bg)
    if rng.random() < 0.15:
        image = image.filter(ImageFilter.MinFilter(3) if rng.random() < 0.5 else ImageFilter.MaxFilter(3))
    if rng.random() < 0.35:
        factor = rng.uniform(0.35, 0.8)
        small = image.resize((max(8, round(image.width * factor)), max(8, round(image.height * factor))), Image.Resampling.BILINEAR)
        image = small.resize(image.size, Image.Resampling.BICUBIC)
    if rng.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    if rng.random() < 0.3:
        arr = np.asarray(image, dtype=np.float32)
        arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 14), arr.shape)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.2:
        arr = np.asarray(image, dtype=np.float32)
        lo = rng.uniform(40, 110)
        arr = lo + arr * (255 - lo) / 255
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return image


def _work(args: tuple) -> list[tuple[str, str]]:
    start, count, seed, out_dir, fonts, max_len, augment = args
    rng = random.Random(seed + start)
    source = TextSource(rng)
    rows = []
    for i in range(start, start + count):
        text = ""
        while not text:
            text = source.line(max_len)
        font = rng.choice(rng.choice(fonts))
        try:
            image = render(text, font, rng, augment)
        except Exception:
            continue
        rel = f"images/{i:07d}.jpg"
        quality = rng.randint(30, 95) if augment and rng.random() < 0.4 else 95
        image.save(os.path.join(out_dir, rel), quality=quality)
        rows.append((rel, text))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", help="output directory")
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=25, help="max characters per line (model max_text_length)")
    parser.add_argument("--no-augment", action="store_true", help="clean renderings (for evaluation)")
    parser.add_argument("--font-dir", action="append", default=None, help="font directory (repeatable)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args()

    fonts = find_fonts(args.font_dir or ["/usr/share/fonts", str(Path.home() / ".fonts")])
    if not fonts:
        raise SystemExit("no font with full Vietnamese coverage found; pass --font-dir")
    families = font_families(fonts)
    print(f"{len(fonts)} fonts in {len(families)} families render Vietnamese correctly")
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    chunk = max(1, math.ceil(args.count / (args.workers * 8)))
    jobs = [
        (s, min(chunk, args.count - s), args.seed * 10_000_000, args.out, families, args.max_len, not args.no_augment)
        for s in range(0, args.count, chunk)
    ]
    rows: list[tuple[str, str]] = []
    with Pool(args.workers) as pool:
        for part in pool.imap(_work, jobs):
            rows.extend(part)
    with open(os.path.join(args.out, "labels.txt"), "w", encoding="utf-8") as fh:
        for rel, text in rows:
            fh.write(f"{rel}\t{text}\n")
    print(f"{len(rows)} lines written to {args.out}")


if __name__ == "__main__":
    main()
