"""Character set for the Vietnamese + English recognition model.

The stock PaddleOCR latin/PP-OCRv6 dictionaries miss most Vietnamese letters with
stacked diacritics (ộ, ủ, ệ, ạ, ...), so the model silently drops them. This set
covers every Vietnamese letter (NFC, both cases), ASCII and the symbols common in
Vietnamese/English administrative, financial and technical documents.
"""

from __future__ import annotations

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


def build_charset() -> list[str]:
    chars: list[str] = []
    for group in (string.digits, string.ascii_letters, string.punctuation):
        chars.extend(group)
    chars.extend(vietnamese_letters())
    chars.extend(SYMBOLS)
    seen: set[str] = set()
    unique = []
    for ch in chars:
        if ch not in seen and not ch.isspace():
            seen.add(ch)
            unique.append(ch)
    return unique


def write_dict(path: str) -> list[str]:
    chars = build_charset()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(chars) + "\n")
    return chars


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "vi_en_dict.txt"
    print(f"{len(write_dict(out))} characters written to {out}")
