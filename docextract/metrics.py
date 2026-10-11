"""Evaluation metrics: CER/WER, order-independent word F1, TEDS for tables, heading structure F1."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .tables import Table, parse_html_table
from .textutil import is_cjk, markdown_to_plain, normalize

try:  # rapidfuzz is optional but makes CER on whole documents fast
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein
    from rapidfuzz.fuzz import partial_ratio_alignment as _partial_alignment
except ImportError:  # pragma: no cover - exercised only without rapidfuzz
    _RFLevenshtein = None
    _partial_alignment = None


def levenshtein(a: Sequence, b: Sequence) -> int:
    if _RFLevenshtein is not None:
        return _RFLevenshtein.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalized_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def cer(reference: str, hypothesis: str, plain: bool = True) -> float:
    """Character Error Rate = edit distance / reference length (on plain text by default)."""
    if plain:
        reference, hypothesis = markdown_to_plain(reference), markdown_to_plain(hypothesis)
    else:
        reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def _best_window(line: str, text: str) -> tuple[int, int] | None:
    """Span of ``text`` most similar to ``line`` (None if nothing is close)."""
    if _partial_alignment is not None:
        found = _partial_alignment(line, text, score_cutoff=50)
        return (found.dest_start, found.dest_end) if found else None
    import difflib  # pragma: no cover - only without rapidfuzz

    blocks = [b for b in difflib.SequenceMatcher(None, line, text, autojunk=False).get_matching_blocks() if b.size]
    if not blocks or sum(b.size for b in blocks) < len(line) / 2:
        return None
    start = max(0, blocks[0].b - blocks[0].a)
    return start, min(len(text), start + len(line))


def cer_any_order(reference: str, hypothesis: str) -> float:
    """CER that does not depend on reading order: each reference line is compared with the
    stretch of the output closest to it, wherever it is; output text no reference line accounts
    for counts as insertions. For ground truths whose order is arbitrary (text layers of
    multi-column pages, tables), where the plain CER mostly measures order."""
    lines = [plain for plain in (markdown_to_plain(line) for line in reference.splitlines()) if plain]
    total = sum(len(line) for line in lines)
    remaining = markdown_to_plain(hypothesis)
    if not total:
        return 0.0 if not remaining else 1.0
    errors = 0
    for line in sorted(lines, key=len, reverse=True):
        window = _best_window(line, remaining) if remaining else None
        if window is None:
            errors += len(line)
            continue
        start, end = window
        errors += levenshtein(line, remaining[start:end])
        remaining = remaining[:start] + "\x00" + remaining[end:]
    errors += sum(1 for ch in remaining if ch not in " \x00")
    return errors / total


def wer(reference: str, hypothesis: str, plain: bool = True) -> float:
    if plain:
        reference, hypothesis = markdown_to_plain(reference), markdown_to_plain(hypothesis)
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def tokens(text: str) -> list[str]:
    """Words (lower case, without punctuation); each CJK character counts as one token."""
    out = []
    for word in re.findall(r"\w+", normalize(text).lower()):
        if any(is_cjk(ch) for ch in word):
            out.extend(re.findall(r"[^\W\d_]|\d+", word) if not word.isascii() else [word])
        else:
            out.append(word)
    return out


def word_f1(reference: str, hypothesis: str, plain: bool = True) -> float:
    """F1 of the bags of words: ignores reading order, which ground truths of complex layouts
    (columns, sidebars, forms) often define differently from any reasonable reading."""
    if plain:
        reference, hypothesis = markdown_to_plain(reference), markdown_to_plain(hypothesis)
    ref, hyp = Counter(tokens(reference)), Counter(tokens(hypothesis))
    if not ref and not hyp:
        return 1.0
    common = sum((ref & hyp).values())
    if not common:
        return 0.0
    precision, recall = common / sum(hyp.values()), common / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


# --- TEDS (Tree-Edit-Distance-based Similarity, Zhong et al. 2020) -------------------


@dataclass
class _Node:
    tag: str
    text: str = ""
    colspan: int = 1
    rowspan: int = 1
    children: list[_Node] = field(default_factory=list)


def _table_tree(table: Table) -> _Node:
    root = _Node("table")
    for row in table.rows:
        tr = _Node("tr")
        for cell in row:
            tr.children.append(_Node("td", normalize(cell.text), cell.colspan, cell.rowspan))
        root.children.append(tr)
    return root


def _postorder(root: _Node) -> tuple[list[_Node], list[int]]:
    nodes: list[_Node] = []
    leftmost: list[int] = []

    def visit(node: _Node) -> int:
        first_leaf = None
        for child in node.children:
            lm = visit(child)
            if first_leaf is None:
                first_leaf = lm
        nodes.append(node)
        index = len(nodes) - 1
        leftmost.append(index if first_leaf is None else first_leaf)
        return leftmost[index]

    visit(root)
    return nodes, leftmost


def _keyroots(leftmost: list[int]) -> list[int]:
    last: dict[int, int] = {}
    for i, lm in enumerate(leftmost):
        last[lm] = i
    return sorted(last.values())


def tree_edit_distance(a: _Node, b: _Node, rename: Callable[[_Node, _Node], float]) -> float:
    """Zhang-Shasha tree edit distance with unit insert/delete cost."""
    na, la = _postorder(a)
    nb, lb = _postorder(b)
    td = [[0.0] * len(nb) for _ in range(len(na))]
    for i in _keyroots(la):
        for j in _keyroots(lb):
            li, lj = la[i], lb[j]
            rows, cols = i - li + 2, j - lj + 2
            fd = [[0.0] * cols for _ in range(rows)]
            for x in range(1, rows):
                fd[x][0] = fd[x - 1][0] + 1
            for y in range(1, cols):
                fd[0][y] = fd[0][y - 1] + 1
            for x in range(1, rows):
                ai = li + x - 1
                for y in range(1, cols):
                    bj = lj + y - 1
                    if la[ai] == li and lb[bj] == lj:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1, fd[x - 1][y - 1] + rename(na[ai], nb[bj]))
                        td[ai][bj] = fd[x][y]
                    else:
                        p, q = la[ai] - li, lb[bj] - lj
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1, fd[p][q] + td[ai][bj])
    return td[-1][-1]


def teds(reference: Table | str, prediction: Table | str | None, structure_only: bool = False) -> float:
    """TEDS in [0, 1]; 1 = identical. Accepts Table objects or HTML strings."""
    ref = parse_html_table(reference) if isinstance(reference, str) else reference
    pred = parse_html_table(prediction) if isinstance(prediction, str) else prediction
    if ref is None:
        return 1.0 if pred is None else 0.0
    if pred is None:
        return 0.0

    def rename(x: _Node, y: _Node) -> float:
        if x.tag != y.tag:
            return 1.0
        if x.tag != "td":
            return 0.0
        if x.colspan != y.colspan or x.rowspan != y.rowspan:
            return 1.0
        return 0.0 if structure_only else normalized_distance(x.text, y.text)

    ta, tb = _table_tree(ref), _table_tree(pred)
    size_a = len(_postorder(ta)[0])
    size_b = len(_postorder(tb)[0])
    return 1.0 - tree_edit_distance(ta, tb, rename) / max(size_a, size_b)


def best_match_teds(references: list[Table], predictions: list[Table]) -> list[float]:
    """Greedy one-to-one matching of predicted tables to reference tables by TEDS."""
    scores = []
    remaining = list(predictions)
    for ref in references:
        if not remaining:
            scores.append(0.0)
            continue
        values = [teds(ref, pred) for pred in remaining]
        best = max(range(len(values)), key=values.__getitem__)
        scores.append(values[best])
        remaining.pop(best)
    return scores


def heading_f1(reference: Sequence[str], prediction: Sequence[str]) -> float:
    ref = [normalize(h).lower() for h in reference]
    pred = [normalize(h).lower() for h in prediction]
    if not ref and not pred:
        return 1.0
    pool = list(pred)
    tp = 0
    for heading in ref:
        if heading in pool:
            pool.remove(heading)
            tp += 1
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    return 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)
