"""Evaluation metrics: CER/WER, TEDS for tables, heading structure F1."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .tables import Table, parse_html_table
from .textutil import markdown_to_plain, normalize

try:  # rapidfuzz is optional but makes CER on whole documents fast
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein
except ImportError:  # pragma: no cover - exercised only without rapidfuzz
    _RFLevenshtein = None


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


def wer(reference: str, hypothesis: str, plain: bool = True) -> float:
    if plain:
        reference, hypothesis = markdown_to_plain(reference), markdown_to_plain(hypothesis)
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


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
