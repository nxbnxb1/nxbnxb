"""Table model: HTML parsing with row/colspans, quality signals and Markdown/HTML rendering."""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from .textutil import normalize


@dataclass
class Cell:
    text: str
    row: int = 0
    col: int = 0
    rowspan: int = 1
    colspan: int = 1
    header: bool = False


@dataclass
class Table:
    rows: list[list[Cell]] = field(default_factory=list)  # cells as written, row by row

    def __post_init__(self) -> None:
        self._layout()

    def _layout(self) -> None:
        """Assign grid positions, honouring cells pushed right by rowspans from above."""
        occupied: set[tuple[int, int]] = set()
        self.row_widths: list[int] = []
        for r, row in enumerate(self.rows):
            c = 0
            for cell in row:
                while (r, c) in occupied:
                    c += 1
                cell.row, cell.col = r, c
                for dr in range(cell.rowspan):
                    for dc in range(cell.colspan):
                        occupied.add((r + dr, c + dc))
                c += cell.colspan
        self.n_rows = max((r for r, _ in occupied), default=-1) + 1
        self.n_cols = max((c for _, c in occupied), default=-1) + 1
        self.row_widths = [sum(1 for (rr, _) in occupied if rr == r) for r in range(self.n_rows)]

    @property
    def cells(self) -> list[Cell]:
        return [cell for row in self.rows for cell in row]

    @property
    def n_cells(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def has_spans(self) -> bool:
        return any(c.rowspan > 1 or c.colspan > 1 for c in self.cells)

    @property
    def span_count(self) -> int:
        return sum(1 for c in self.cells if c.rowspan > 1 or c.colspan > 1)

    @property
    def is_rectangular(self) -> bool:
        return bool(self.row_widths) and all(w == self.n_cols for w in self.row_widths)

    @property
    def empty_ratio(self) -> float:
        cells = self.cells
        if not cells:
            return 1.0
        return sum(1 for c in cells if not c.text.strip()) / len(cells)

    def grid(self) -> list[list[str]]:
        """Expanded grid; positions covered by a span are empty strings."""
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for cell in self.cells:
            if cell.row < self.n_rows and cell.col < self.n_cols:
                grid[cell.row][cell.col] = cell.text
        return grid

    def text(self) -> str:
        return " ".join(c.text for c in self.cells if c.text)

    def to_html(self) -> str:
        parts = ["<table>"]
        for row in self.rows:
            parts.append("<tr>")
            for cell in row:
                tag = "th" if cell.header else "td"
                attrs = ""
                if cell.rowspan > 1:
                    attrs += f' rowspan="{cell.rowspan}"'
                if cell.colspan > 1:
                    attrs += f' colspan="{cell.colspan}"'
                body = "<br>".join(html_lib.escape(part) for part in cell.text.split("\n"))
                parts.append(f"<{tag}{attrs}>{body}</{tag}>")
            parts.append("</tr>")
        parts.append("</table>")
        return "".join(parts)

    def to_markdown(self) -> str:
        if self.n_rows == 0 or self.n_cols == 0:
            return ""
        grid = self.grid()

        def fmt(text: str) -> str:
            return text.replace("|", "\\|").replace("\n", "<br>").strip()

        header = grid[0]
        lines = ["| " + " | ".join(fmt(t) for t in header) + " |", "|" + "---|" * self.n_cols]
        for row in grid[1:]:
            lines.append("| " + " | ".join(fmt(t) for t in row) + " |")
        return "\n".join(lines)

    def render(self, fmt: str = "auto") -> str:
        """Markdown when it can represent the table faithfully, otherwise HTML."""
        if fmt == "html" or (fmt == "auto" and self.has_spans):
            return self.to_html()
        return self.to_markdown()


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[Cell]]] = []
        self._depth = 0  # nested tables are flattened into the outer cell
        self._row: list[Cell] | None = None
        self._cell: Cell | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self.tables.append([])
        elif self._depth != 1:
            if tag == "br" and self._cell is not None:
                self._buf.append("\n")
            return
        elif tag == "tr":
            self._close_row()
            self._row = []
        elif tag in ("td", "th"):
            self._close_cell()
            if self._row is None:
                self._row = []
            a = dict(attrs)
            self._cell = Cell(
                text="", rowspan=_span(a.get("rowspan")), colspan=_span(a.get("colspan")), header=tag == "th"
            )
            self._buf = []
        elif tag == "br" and self._cell is not None:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "table":
            if self._depth == 1:
                self._close_row()
            self._depth = max(0, self._depth - 1)
        elif self._depth != 1:
            return
        elif tag in ("td", "th"):
            self._close_cell()
        elif tag == "tr":
            self._close_row()
        elif tag in ("p", "div", "li") and self._cell is not None:
            self._buf.append("\n")

    def handle_data(self, data):
        if self._cell is not None:
            self._buf.append(data)

    def _close_cell(self):
        if self._cell is not None and self._row is not None:
            text = "\n".join(normalize(t) for t in "".join(self._buf).split("\n"))
            self._cell.text = re.sub(r"\n{2,}", "\n", text).strip("\n")
            self._row.append(self._cell)
        self._cell = None
        self._buf = []

    def _close_row(self):
        self._close_cell()
        if self._row is not None and self.tables:
            if self._row:
                self.tables[-1].append(self._row)
        self._row = None


def _span(value: str | None) -> int:
    try:
        return max(1, min(1000, int(str(value).strip())))
    except (TypeError, ValueError):
        return 1


def parse_html_tables(markup: str) -> list[Table]:
    parser = _TableParser()
    parser.feed(markup)
    parser.close()
    parser._close_row()
    return [Table(rows) for rows in parser.tables if rows]


def parse_html_table(markup: str) -> Table | None:
    tables = parse_html_tables(markup)
    if not tables:
        return None
    # Prefer the largest table if a model returned several fragments.
    return max(tables, key=lambda t: t.n_cells)


def table_from_rows(rows: list[list[str | None]], header: bool = True) -> Table:
    """Build a table from a plain grid (``None`` = covered by the cell to the left)."""
    out: list[list[Cell]] = []
    for r, row in enumerate(rows):
        cells: list[Cell] = []
        for value in row:
            if value is None and cells:
                cells[-1].colspan += 1
                continue
            cells.append(Cell(text=normalize(str(value or "")), header=header and r == 0))
        out.append(cells)
    return Table(out)


def parse_markdown_table(markdown: str) -> Table | None:
    lines = [ln.strip() for ln in markdown.strip().splitlines() if ln.strip().startswith("|")]
    rows = []
    for ln in lines:
        if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?", ln):
            continue
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
        rows.append(cells)
    return table_from_rows(rows) if rows else None
