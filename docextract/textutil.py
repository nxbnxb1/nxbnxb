"""Text helpers: normalisation, quality signals, number/symbol extraction, line joining."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

# Vietnamese letters that live in Latin-1; anything else from U+00A1..U+00FF in a
# Vietnamese text layer is a sign of a legacy TCVN3/VNI font without ToUnicode map.
_VI_LATIN1 = set("àáâãèéêìíòóôõùúýÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝ")
_COMMON_LATIN1_SYMBOLS = set("°±×÷©®§«»µ¼½¾²³¹¢£¥¤¦¬¯´·¸¿¡ªº")
_LEGACY_FONT_PREFIXES = (".vn", "vni-", "vni_", "vn-", "abc")
_CID_RE = re.compile(r"\(cid:\d+\)")
_NUMBER_RE = re.compile(r"(?<![0-9A-Za-z])[-+−]?\d+(?:[.,  ]\d+)*%?")
_BULLET_RE = re.compile(r"^\s*(?:[-•●▪◦‣∙*+]|\d{1,3}[.)]|[a-zA-Z][.)]|[ivxIVX]{1,5}[.)])\s+")

SPECIAL_CHARS = set("%$€£¥₫円±≤≥≠≈×÷°§©®™→←↑↓↔⇒√∞∑∏∫∂∆∇µπσΩαβγδλ‰′″※〒")


def normalize(text: str) -> str:
    """NFC (Vietnamese diacritics are often decomposed in PDFs) + unify whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[ \t  -​ 　]+", " ", text)
    return text.strip()


def garbled_ratio(text: str) -> float:
    """Share of characters that indicate a broken text layer."""
    if not text:
        return 0.0
    cid_chars = sum(len(m) for m in _CID_RE.findall(text))
    bad = 0
    letters = 0
    for ch in _CID_RE.sub("", text):
        code = ord(ch)
        if ch == "�" or 0xE000 <= code <= 0xF8FF:
            bad += 1
        elif unicodedata.category(ch) == "Cc" and ch not in "\n\r\t":
            bad += 1
        if ch.isalpha():
            letters += 1
    legacy = legacy_encoding_ratio(text)
    total = max(1, len(text))
    return min(1.0, (bad + cid_chars) / total + legacy)


def legacy_encoding_ratio(text: str) -> float:
    """Share of letters that look like TCVN3/VNI mojibake (e.g. 'Céng hßa x· héi')."""
    letters = [ch for ch in text if ch.isalpha() or 0xA1 <= ord(ch) <= 0xFF]
    if len(letters) < 20:
        return 0.0
    suspicious = sum(
        1 for ch in letters if 0xA1 <= ord(ch) <= 0xFF and ch not in _VI_LATIN1 and ch not in _COMMON_LATIN1_SYMBOLS
    )
    # Latin-1 symbols that TCVN3 uses for Vietnamese letters (¸ µ ¶ · ¹ ¨ ...) when mixed into words.
    in_word = len(re.findall(r"[a-zA-Z][¸µ¶·¹¨©ª«¬®¯°±²³´»¼½¾][a-zA-Z]", text))
    ratio = (suspicious + 2 * in_word) / len(letters)
    return ratio if ratio >= 0.08 else 0.0


def is_legacy_vietnamese_font(font_name: str) -> bool:
    name = font_name.lower().split("+")[-1]
    return name.startswith(_LEGACY_FONT_PREFIXES)


_VI_MARKS = set("ăâêôơưđĂÂÊÔƠƯĐ")
VI_ABBREVIATIONS = frozenset(
    """UBND HĐND MTTQ TNHH CTCP TP HCM QĐ NĐ CP TT TTG BTC BCT BYT BGDĐT BXD BKHĐT NHNN NSNN VN VNĐ VND USD
    TW KT XH KH BC CV SXKD GTGT TNDN TNCN BHXH BHYT BHTN CBCNV CNTT THPT THCS ĐH PGS TS THS GS STT ĐVT HĐQT
    ĐHĐCĐ BKS MST CMND CCCD QLDA XDCB HĐ KHCN PCCC ATTP""".split()
)
_VI_TONES = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}  # grave, acute, tilde, hook, dot below


def looks_vietnamese(text: str) -> bool:
    decomposed = unicodedata.normalize("NFD", text)
    marks = sum(1 for ch in decomposed if ch in _VI_TONES) + sum(1 for ch in text if ch in _VI_MARKS)
    return marks >= 2


def detect_language(text: str, languages: Sequence[str] = ("vi", "en", "ja"), min_letters: int = 12) -> str | None:
    """Language of a text among ``languages`` from its script, or None if there is too little text.

    Kana/kanji → ja; Latin text with Vietnamese letters or tone marks on at least 2% of its
    letters (Vietnamese has them on most syllables, English on none) → vi; other Latin → en.
    """
    text = unicodedata.normalize("NFC", text)
    cjk = sum(1 for ch in text if is_cjk(ch) and ch.isalpha())
    latin = [ch for ch in text if ch.isalpha() and not is_cjk(ch)]
    if 2 * cjk + len(latin) < min_letters:
        return None
    scores: dict[str, float] = {}
    if "ja" in languages:
        scores["ja"] = cjk * 2.0  # one CJK character carries about a word
    if latin:
        decomposed = unicodedata.normalize("NFD", "".join(latin))
        marked = sum(1 for ch in decomposed if ch in _VI_TONES) + sum(1 for ch in latin if ch in _VI_MARKS)
        vi = "vi" in languages and marked >= max(2, 0.02 * len(latin))
        scores["vi" if vi or "en" not in languages else "en"] = float(len(latin))
    best = max(scores, key=scores.get) if scores else None
    return best if best in languages else None


def _has_vowel(token: str) -> bool:
    base = "".join(ch for ch in unicodedata.normalize("NFD", token.lower()) if not unicodedata.combining(ch))
    return any(ch in "aeiouy" for ch in base)


def vietnamese_dropout_ratio(text: str) -> float:
    """Signal that an OCR model silently dropped Vietnamese letters.

    Stock PaddleOCR dictionaries lack most letters with stacked diacritics (ộ, ủ, ệ, ...):
    "Cộng hòa" comes back as "Cng hòa" with high confidence. Vietnamese syllables always
    contain a vowel, so lowercase/titlecase tokens without one are near-certain dropouts;
    long Vietnamese text without any U+1EA0..U+1EF9 letter is also implausible.
    """
    if not looks_vietnamese(text):
        return 0.0
    tokens = re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", text))
    if len(tokens) < 3:
        return 0.0
    bad = sum(1 for t in tokens if not t.isupper() and not _has_vowel(t))
    # ALL-CAPS words without a vowel are usually abbreviations (UBND, TNHH), but in a capitalised
    # title ("BÁO CÁO KT QU KINH DOANH NM 2025") they are dropped letters.
    upper = [t for t in tokens if t.isupper() and len(t) >= 2]
    if sum(1 for t in upper if _has_vowel(t)) >= 3:
        bad += sum(1 for t in upper if not _has_vowel(t) and t not in VI_ABBREVIATIONS)
    ratio = bad / len(tokens)
    if len(tokens) >= 20 and not any("\u1ea0" <= ch <= "\u1ef9" for ch in text):
        ratio = max(ratio, 0.5)
    return ratio


def extract_numbers(text: str) -> list[str]:
    """Numbers reduced to their digits, so '1.234,5', '1,234.5' and full-width '１，２３４．５' compare equal."""
    out = []
    for match in _NUMBER_RE.findall(unicodedata.normalize("NFKC", text)):
        digits = re.sub(r"\D", "", match)
        if digits:
            out.append(digits)
    return out


def special_chars(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        if ch in SPECIAL_CHARS:
            counts[ch] = counts.get(ch, 0) + 1
    return counts


def multiset_recall(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """Share of reference items (with multiplicity) found in the candidate."""
    if not reference:
        return 1.0
    pool: dict[str, int] = {}
    for item in candidate:
        pool[item] = pool.get(item, 0) + 1
    hit = 0
    for item in reference:
        if pool.get(item, 0) > 0:
            pool[item] -= 1
            hit += 1
    return hit / len(reference)


def is_list_item(line: str) -> bool:
    return bool(_BULLET_RE.match(line))


def strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line, count=1)


@dataclass
class Line:
    """A visual text line with its box (any unit) and optional font info."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float = 0.0
    bold: bool = False
    score: float = 1.0

    @property
    def height(self) -> float:
        return max(1e-6, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def is_cjk(ch: str) -> bool:
    """Japanese/CJK characters, which are written without spaces between words."""
    code = ord(ch)
    return (
        0x3000 <= code <= 0x30FF  # CJK punctuation, hiragana, katakana
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF  # full-width forms
    )


def join_inline(left: str, right: str) -> str:
    """Concatenate two pieces of one line/paragraph: no space between CJK characters."""
    if not left:
        return right
    if not right:
        return left
    if is_cjk(left[-1]) and is_cjk(right[0]):
        return left + right
    return left + " " + right


def group_rows(lines: Sequence[Line]) -> list[list[Line]]:
    """Group fragments that sit on the same visual row, rows top-to-bottom, fragments left-to-right."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda ln: (ln.cy, ln.x0)):
        if rows:
            row = rows[-1]
            ref_cy = sum(ln.cy for ln in row) / len(row)
            ref_h = sum(ln.height for ln in row) / len(row)
            if abs(line.cy - ref_cy) < 0.5 * min(ref_h, line.height):
                row.append(line)
                continue
        rows.append([line])
    return [sorted(row, key=lambda ln: ln.x0) for row in rows]


def join_lines(lines: Sequence[Line], keep_line_breaks: bool = False) -> str:
    """Rebuild paragraphs from visual lines.

    Lines of one paragraph are joined with spaces (hyphenated words are glued back),
    a vertical gap larger than ~0.8 line height starts a new paragraph, and list items
    always start on a new line.
    """
    rows = group_rows([ln for ln in lines if ln.text.strip()])
    if not rows:
        return ""
    paragraphs: list[list[str]] = [[]]
    prev_bottom = None
    prev_height = None
    for row in rows:
        text = ""
        for ln in row:
            text = join_inline(text, normalize(ln.text))
        text = normalize(text)
        if not text:
            continue
        top = min(ln.y0 for ln in row)
        height = max(ln.height for ln in row)
        new_para = False
        if prev_bottom is not None:
            gap = top - prev_bottom
            if gap > 0.8 * max(height, prev_height or height):
                new_para = True
        if new_para and paragraphs[-1]:
            paragraphs.append([])
        if (keep_line_breaks or is_list_item(text)) and paragraphs[-1]:
            paragraphs[-1].append("\n" + text)
        else:
            paragraphs[-1].append(text)
        prev_bottom = max(ln.y1 for ln in row)
        prev_height = height
    return "\n\n".join(_glue(parts) for parts in paragraphs if parts)


def _glue(parts: list[str]) -> str:
    out = ""
    for part in parts:
        if not out:
            out = part.lstrip("\n")
        elif part.startswith("\n"):
            out += part
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and part[:1].islower():
            out = out[:-1] + part
        else:
            out = join_inline(out, part)
    return out


def markdown_to_plain(text: str) -> str:
    """Rough Markdown/HTML stripping used by metrics (CER on content, not on markup)."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"</t[dh]>", " ", text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)
    text = re.sub(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", " ", text, flags=re.M)
    text = text.replace("|", " ")
    text = re.sub(r"(\*\*|__|`{1,3}|\$\$)", "", text)
    text = re.sub(r"(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)", "", text)
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()
