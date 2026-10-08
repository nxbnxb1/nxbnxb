"""Synthetic text-line images for fine-tuning the PaddleOCR recogniser (Vietnamese, English, Japanese).

Text comes from word-frequency lists (``wordfreq``), document-style patterns (amounts in
đồng/円, dates, decision numbers, percentages, codes) and a coverage mode that forces rare
Vietnamese letters (ẵ, ỹ, ỵ, ...) to appear often enough to be learnt. Each line is
rendered with a font that covers all of its characters (Vietnamese lines only with fonts
that really draw the diacritics), then degraded like scans/photos (blur, low resolution,
JPEG, noise, skew, ink spread).

Output follows PaddleOCR's SimpleDataSet format: ``<out>/images/*.jpg`` and
``<out>/labels.txt`` with ``relative/path<TAB>text`` per line.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import unicodedata
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from charset import build_charset, read_dict, vietnamese_letters

_ONSETS = ["", "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh", "l", "m", "n", "ng", "ngh", "nh", "p", "ph", "qu", "r", "s", "t", "th", "tr", "v", "x"]
_CODAS = ["", "", "", "c", "ch", "m", "n", "ng", "nh", "p", "t", "i", "o", "u", "y"]
_UNITS = ["đồng", "VNĐ", "tỷ đồng", "triệu đồng", "USD", "%", "kg", "km", "m²", "tấn", "người", "hộ", "ha"]
_EN_UNITS = ["USD", "million", "billion", "%", "kg", "km", "units", "items"]
_ORGS = ["UBND", "HĐND", "TTg", "BTC", "NĐ-CP", "QĐ-UBND", "TT-BTC", "CT-TTg", "KH-UBND", "BC-UBND", "CV"]
_JA_UNITS = ["円", "万円", "億円", "%", "人", "件", "台", "個", "kg", "km", "年", "か月"]
_JA_PUNCT = ["、", "。", "・", "：", "（", "）", "「", "」"]
_VI_SET = set(vietnamese_letters()) - set("aeiouyAEIOUY")
_KANA = "".join(chr(c) for c in range(0x3041, 0x3097)) + "".join(chr(c) for c in range(0x30A1, 0x30FB))

# Share of each generator per language; normalised over the languages that are enabled.
_MIX = {"vi": {"vi": 0.38, "coverage": 0.14, "misc_vi": 0.08}, "en": {"en": 0.17, "misc_en": 0.03}, "ja": {"ja": 0.27, "misc_ja": 0.06}}


def _zipf_weights(n: int, offset: float = 8.0) -> list[float]:
    return [1.0 / (i + offset) for i in range(n)]


class TextSource:
    def __init__(self, rng: random.Random, langs: tuple[str, ...], charset: set[str]) -> None:
        from wordfreq import top_n_list

        self.rng = rng
        self.charset = charset | {" "}
        self.lists: dict[str, tuple[list[str], list[float]]] = {}
        for lang, n, offset in (("vi", 12000, 8.0), ("en", 25000, 30.0), ("ja", 30000, 30.0)):
            if lang in langs:
                words = [w for w in top_n_list(lang, n) if self._ok(w)]
                self.lists[lang] = (words, _zipf_weights(len(words), offset))
        self.letters = [ch for ch in vietnamese_letters() if ch.islower()]
        mix = {k: v for lang in langs for k, v in _MIX[lang].items()}
        self.kinds = list(mix)
        self.weights = [mix[k] for k in self.kinds]

    def _ok(self, word: str) -> bool:
        return bool(word) and all(ch in self.charset for ch in word) and any(ch.isalpha() for ch in word)

    def _words(self, lang: str, n: int) -> list[str]:
        words, weights = self.lists[lang]
        return self.rng.choices(words, weights=weights, k=n)

    def syllable(self) -> str:
        """Random phonotactically plausible syllable; forces rare letters into the data."""
        return self.rng.choice(_ONSETS) + self.rng.choice(self.letters) + self.rng.choice(_CODAS)

    def number(self, style: str = "vi") -> str:
        r = self.rng.random()
        if r < 0.3:
            n = self.rng.randint(1000, 999_999_999)
            return f"{n:,}".replace(",", "." if style == "vi" else ",")
        if r < 0.55:
            return f"{self.rng.randint(0, 999)}{',' if style == 'vi' else '.'}{self.rng.randint(0, 99)}"
        if r < 0.75:
            return str(self.rng.randint(0, 2100))
        if r < 0.9:
            d, m, y = self.rng.randint(1, 28), self.rng.randint(1, 12), self.rng.randint(1975, 2030)
            return f"{y}/{m:02d}/{d:02d}" if style == "ja" else f"{d:02d}/{m:02d}/{y}"
        return f"{self.rng.randint(1, 9999)}/{self.rng.choice(_ORGS)}" if style == "vi" else str(self.rng.randint(1, 99))

    def make(self, kind: str) -> str:
        rng = self.rng
        if kind == "vi":
            words = self._words("vi", rng.randint(2, 7))
            if rng.random() < 0.35:
                words.insert(rng.randint(0, len(words)), self.number())
            if rng.random() < 0.2:
                words.append(rng.choice(_UNITS))
            return " ".join(words)
        if kind == "en":
            words = self._words("en", rng.randint(2, 6))
            if rng.random() < 0.3:
                words.insert(rng.randint(0, len(words)), self.number("en"))
            if rng.random() < 0.15:
                words.append(rng.choice(_EN_UNITS))
            return " ".join(words)
        if kind == "ja":
            words = self._words("ja", rng.randint(2, 5))
            if rng.random() < 0.3:
                words.insert(rng.randint(0, len(words)), self.number("ja") + rng.choice(_JA_UNITS))
            if rng.random() < 0.25:
                words.insert(rng.randint(1, len(words)), rng.choice(_JA_PUNCT))
            if rng.random() < 0.1:
                words.append("".join(rng.choices(_KANA, k=rng.randint(2, 5))))
            return "".join(words)
        if kind == "coverage":
            return " ".join(self.syllable() for _ in range(rng.randint(2, 6)))
        if kind == "misc_vi":
            r = rng.random()
            if r < 0.3:
                return f"Số: {rng.randint(1, 9999)}/{rng.choice(_ORGS)}"
            if r < 0.55:
                return f"ngày {rng.randint(1, 31)} tháng {rng.randint(1, 12)} năm {rng.randint(1990, 2030)}"
            if r < 0.8:
                return f"{self.number()} {rng.choice(_UNITS)}"
            return f"Tel: 0{rng.randint(20, 99)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}"
        if kind == "misc_en":
            user = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789._", k=rng.randint(4, 10))).strip(".") or "info"
            return rng.choice([f"{user}@{rng.choice(['gmail.com', 'company.com', 'mail.vn'])}", f"Page {rng.randint(1, 300)} of {rng.randint(300, 999)}", f"(a) {self.number('en')}; {self.number('en')}"])
        # misc_ja
        r = rng.random()
        if r < 0.35:
            return f"{rng.randint(1990, 2030)}年{rng.randint(1, 12)}月{rng.randint(1, 28)}日"
        if r < 0.6:
            return f"第{rng.randint(1, 30)}条{rng.choice(['', '第' + str(rng.randint(1, 9)) + '項'])}"
        if r < 0.85:
            return f"{self.number('en')}{rng.choice(_JA_UNITS)}"
        return f"〒{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"

    def line(self, max_len: int) -> str:
        kind = self.rng.choices(self.kinds, weights=self.weights)[0]
        text = self.make(kind)
        if kind in ("ja", "misc_ja"):
            max_len = min(max_len, 14)  # CJK glyphs are square: keep the line's aspect ratio sane
        elif self.rng.random() < 0.12:
            text += self.rng.choice([".", ",", ":", ";", "?", "!", " -", "…"])
        r = self.rng.random()
        if kind not in ("ja", "misc_ja"):
            if r < 0.15:
                text = text.upper()
            elif r < 0.27:
                text = " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
            elif r < 0.6:
                text = text[:1].upper() + text[1:]
        text = unicodedata.normalize("NFC", " ".join(text.split()))
        if len(text) > max_len:  # cut at a word boundary when there is one
            cut = text.rfind(" ", 0, max_len + 1)
            text = text[: cut if cut > 3 else max_len].strip()
        if any(ch not in self.charset for ch in text):
            return ""
        return text


@dataclass
class FontInfo:
    path: str
    family: str
    codepoints: frozenset[int]
    draws_vietnamese: bool


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


def _family(path: Path) -> str:
    family = path.stem.split("-")[0]
    for suffix in ("BoldOblique", "BoldItalic", "Bold", "Oblique", "Italic"):
        family = family.removesuffix(suffix)
    return family


# fonts-noto-core ships one family per script; keep only the general-purpose ones so they do not
# swamp the Latin font mix.
_NOTO_KEEP = {"NotoSans", "NotoSerif", "NotoSansMono", "NotoSansDisplay", "NotoSerifDisplay", "NotoSansCJK", "NotoSerifCJK", "NotoSansJP", "NotoSerifJP"}


def find_fonts(font_dirs: list[str]) -> list[FontInfo]:
    """Usable fonts (index 0 of collections) with the code points they cover."""
    from fontTools.ttLib import TTFont

    basics = {ord(ch) for ch in "0123456789AZaz%"}
    fonts = []
    for directory in font_dirs:
        for path in sorted(Path(directory).rglob("*")):
            if path.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                continue
            name = path.name.lower()
            if any(skip in name for skip in ("unifont", "emoji", "symbol", "dingbat", "math")):
                continue
            family = _family(path)
            if family.startswith("Noto") and family not in _NOTO_KEEP:
                continue
            try:
                font = TTFont(str(path), lazy=True, fontNumber=0)
                cmap = frozenset((font.getBestCmap() or {}).keys())
            except Exception:
                continue
            if not basics <= cmap:
                continue
            vi = {ord(ch) for ch in vietnamese_letters()} <= cmap and _draws_diacritics(str(path))
            fonts.append(FontInfo(str(path), family, cmap, vi))
    return fonts


def pick_font(text: str, fonts: list[FontInfo], rng: random.Random) -> FontInfo | None:
    needed = {ord(ch) for ch in text if not ch.isspace()}
    vietnamese = any(ch in _VI_SET for ch in text)
    usable = [f for f in fonts if needed <= f.codepoints and (f.draws_vietnamese or not vietnamese)]
    if not usable:
        return None
    families: dict[str, list[FontInfo]] = {}
    for font in usable:
        families.setdefault(font.family, []).append(font)
    return rng.choice(rng.choice(list(families.values())))


def render(text: str, font_path: str, rng: random.Random, augment: bool = True) -> Image.Image:
    size = rng.randint(20, 44)
    font = ImageFont.truetype(font_path, size, index=0)
    left, top, right, bottom = font.getbbox(text)
    ascent, descent = font.getmetrics()
    top = min(top, 0)
    bottom = max(bottom, ascent + descent)
    pad_x, pad_y = rng.randint(2, 14), rng.randint(2, 9)
    width = right - left + 2 * pad_x
    height = bottom - top + 2 * pad_y
    bg = rng.randint(190, 255)
    tint = tuple(max(0, min(255, bg + rng.randint(-12, 6))) for _ in range(3))
    image = Image.new("RGB", (width, height), tint)
    ink = rng.randint(0, 90)
    ImageDraw.Draw(image).text((pad_x - left, pad_y - top), text, font=font, fill=(ink, ink, ink + rng.randint(0, 30)))
    return degrade(image, rng, tint) if augment else image


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


_WORKER: dict = {}


def _init_worker(font_dirs: list[str], langs: tuple[str, ...], charset: set[str]) -> None:
    _WORKER["fonts"] = find_fonts(font_dirs)
    _WORKER["langs"] = langs
    _WORKER["charset"] = charset


def _work(args: tuple) -> list[tuple[str, str]]:
    start, count, seed, out_dir, max_len, augment = args
    rng = random.Random(seed + start)
    source = TextSource(rng, _WORKER["langs"], _WORKER["charset"])
    rows = []
    i = start
    attempts = 0
    while i < start + count and attempts < count * 20:
        attempts += 1
        text = source.line(max_len)
        font = pick_font(text, _WORKER["fonts"], rng) if text else None
        if font is None:
            continue
        try:
            image = render(text, font.path, rng, augment)
        except Exception:
            continue
        rel = f"images/{i:07d}.jpg"
        quality = rng.randint(30, 95) if augment and rng.random() < 0.4 else 95
        image.save(os.path.join(out_dir, rel), quality=quality)
        rows.append((rel, text))
        i += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", help="output directory")
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--langs", default="vi,en", help="languages to generate: any of vi,en,ja")
    parser.add_argument("--dict", help="model dictionary; lines with other characters are skipped")
    parser.add_argument("--max-len", type=int, default=23, help="max characters per line (NRTR drops labels >= max_text_length - 1)")
    parser.add_argument("--no-augment", action="store_true", help="clean renderings (for evaluation)")
    parser.add_argument("--font-dir", action="append", default=None, help="font directory (repeatable)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args()

    langs = tuple(args.langs.split(","))
    charset = set(read_dict(args.dict)) if args.dict else set(build_charset(langs))
    font_dirs = args.font_dir or ["/usr/share/fonts", str(Path.home() / ".fonts")]
    fonts = find_fonts(font_dirs)
    families = {f.family for f in fonts}
    vi_families = {f.family for f in fonts if f.draws_vietnamese}
    ja_families = {f.family for f in fonts if ord("あ") in f.codepoints and ord("日") in f.codepoints}
    print(f"{len(fonts)} fonts / {len(families)} families; Vietnamese: {len(vi_families)}, Japanese: {len(ja_families)}")
    if "vi" in langs and not vi_families:
        raise SystemExit("no font draws Vietnamese correctly; pass --font-dir")
    if "ja" in langs and not ja_families:
        raise SystemExit("no Japanese font found (install fonts-noto-cjk); pass --font-dir")

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    chunk = max(1, math.ceil(args.count / (args.workers * 8)))
    jobs = [
        (s, min(chunk, args.count - s), args.seed * 10_000_000, args.out, args.max_len, not args.no_augment)
        for s in range(0, args.count, chunk)
    ]
    rows: list[tuple[str, str]] = []
    with Pool(args.workers, initializer=_init_worker, initargs=(font_dirs, langs, charset)) as pool:
        for part in pool.imap(_work, jobs):
            rows.extend(part)
    with open(os.path.join(args.out, "labels.txt"), "w", encoding="utf-8") as fh:
        for rel, text in rows:
            fh.write(f"{rel}\t{text}\n")
    print(f"{len(rows)} lines written to {args.out}")


if __name__ == "__main__":
    main()
