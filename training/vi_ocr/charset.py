"""Character sets for the recognition models.

* ``vi,en``: compact set for Vietnamese + English (281 characters), trained from
  ``latin_PP-OCRv5_mobile_rec``.
* ``vi,en,ja``: the characters of the ``PP-OCRv5_mobile_rec`` dictionary that belong to the
  standard Japanese character set (encodable in CP932 = JIS X 0208 + common vendor extensions:
  kana, 6 221 kanji, full-width forms, symbols), in their original order, with the Vietnamese
  letters and symbols appended: about 7 000 classes instead of 18 383. Chinese-only characters
  are dropped: they never occur in Japanese text, and every output class costs training time
  on a CPU runner. Every kept class keeps its pretrained weights.

The stock PaddleOCR latin/PP-OCRv5/PP-OCRv6 dictionaries miss most Vietnamese letters with
stacked diacritics (ộ, ủ, ệ, ạ, ...), so those models silently drop them.
"""

from __future__ import annotations

import argparse
import string
import unicodedata

_VI_BASE = "aăâeêioôơuưy"
_TONES = ["", "̀", "́", "̉", "̃", "̣"]  # ngang, huyền, sắc, hỏi, ngã, nặng


def vietnamese_letters() -> list[str]:
    letters = []
    for base in _VI_BASE:
        for tone in _TONES:
            ch = unicodedata.normalize("NFC", base + tone)
            letters.append(ch)
            letters.append(ch.upper())
    letters += ["đ", "Đ"]
    return letters


SYMBOLS = list("₫€£¥°±×÷≤≥≠≈→←↑↓↔…“”‘’–—•·§©®™‰½¼¾²³µπΩαβγδλσΣ∑√∞∆№«»")


def _unique(chars) -> list[str]:
    seen: set[str] = set()
    out = []
    for ch in chars:
        if ch and ch not in seen and not ch.isspace():
            seen.add(ch)
            out.append(ch)
    return out


def build_charset(langs: tuple[str, ...] = ("vi", "en"), base_dict: list[str] | None = None) -> list[str]:
    latin = list(string.digits) + list(string.ascii_letters) + list(string.punctuation)
    vi_en = _unique(latin + vietnamese_letters() + SYMBOLS)
    if "ja" not in langs:
        return vi_en
    if not base_dict:
        raise ValueError("a Japanese-capable base dictionary (ppocrv5_dict.txt) is required for 'ja'")
    return _unique([ch for ch in base_dict if is_japanese_charset(ch)] + vi_en)


def is_japanese_charset(ch: str) -> bool:
    try:
        ch.encode("cp932")
    except UnicodeEncodeError:
        return False
    return True


def read_dict(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n").rstrip("\r") for line in fh if line.rstrip("\n\r")]


def write_dict(path: str, langs: tuple[str, ...] = ("vi", "en"), base_dict_path: str | None = None) -> list[str]:
    chars = build_charset(langs, read_dict(base_dict_path) if base_dict_path else None)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(chars) + "\n")
    return chars


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default="vi_en_dict.txt")
    parser.add_argument("--langs", default="vi,en", help="vi,en or vi,en,ja")
    parser.add_argument("--base-dict", help="ppocrv5_dict.txt (required with ja)")
    args = parser.parse_args()
    chars = write_dict(args.out, tuple(args.langs.split(",")), args.base_dict)
    print(f"{len(chars)} characters written to {args.out}")
