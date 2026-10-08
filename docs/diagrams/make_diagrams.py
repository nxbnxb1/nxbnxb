"""Generate the pipeline diagrams in docs/images/*.svg (Vietnamese labels).

    python docs/diagrams/make_diagrams.py

Plain SVG with a white background so the figures read the same in GitHub's light and dark themes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "images"
FONT = "Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
INK = "#1f2937"
MUTED = "#475569"


@dataclass(frozen=True)
class Theme:
    fill: str
    stroke: str
    title: str


T = {
    "input": Theme("#f1f5f9", "#64748b", "#334155"),
    "pre": Theme("#e0f2fe", "#0284c7", "#075985"),
    "layout": Theme("#ccfbf1", "#0d9488", "#115e59"),
    "router": Theme("#fef3c7", "#d97706", "#92400e"),
    "text": Theme("#dbeafe", "#2563eb", "#1e40af"),
    "ocr": Theme("#dbeafe", "#2563eb", "#1e40af"),
    "table": Theme("#e0e7ff", "#4f46e5", "#3730a3"),
    "formula": Theme("#fce7f3", "#db2777", "#9d174d"),
    "vlm": Theme("#f3e8ff", "#9333ea", "#6b21a8"),
    "check": Theme("#dcfce7", "#16a34a", "#166534"),
    "out": Theme("#ede9fe", "#7c3aed", "#5b21b6"),
    "warn": Theme("#fee2e2", "#dc2626", "#991b1b"),
    "note": Theme("#f8fafc", "#94a3b8", "#334155"),
    "gh": Theme("#f8fafc", "#334155", "#0f172a"),
    "plain": Theme("#ffffff", "#cbd5e1", "#334155"),
}


class Svg:
    def __init__(self, width: int, height: int, title: str, subtitle: str = "") -> None:
        self.w, self.h = width, height
        self.parts: list[str] = [
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        ]
        self.text(width / 2, 40, [title], size=24, weight=700, color="#0f172a")
        if subtitle:
            self.text(width / 2, 68, [subtitle], size=15, color=MUTED)

    # --- primitives ----------------------------------------------------------------------

    def rect(self, x, y, w, h, theme: Theme, rx=12, dash=False, width=1.6):
        dash_attr = ' stroke-dasharray="7 5"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{theme.fill}" '
            f'stroke="{theme.stroke}" stroke-width="{width}"{dash_attr}/>'
        )

    def text(self, x, y, lines, size=14, weight=400, color=INK, anchor="middle", line_height=None, italic=False):
        line_height = line_height or round(size * 1.38)
        style = ' font-style="italic"' if italic else ""
        spans = []
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else line_height
            spans.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}"{style}>{"".join(spans)}</text>'
        )

    def box(self, x, y, w, h, title, lines=(), theme="plain", size=13.5, align="middle", title_size=15.5, dash=False):
        th = T[theme]
        self.rect(x, y, w, h, th, dash=dash)
        tx = x + w / 2 if align == "middle" else x + 16
        top = y + 26
        if title:
            self.text(tx, top, [title], size=title_size, weight=700, color=th.title, anchor="middle" if align == "middle" else "start")
            top += round(title_size * 1.5)
        if lines:
            self.text(tx, top, list(lines), size=size, anchor="middle" if align == "middle" else "start")

    def section(self, x, y, w, h, label, theme):
        """Container with a tab label in the top-left corner."""
        th = T[theme]
        self.rect(x, y, w, h, Theme("none", th.stroke, th.title), rx=16, dash=True, width=1.4)
        tab_w = 14 + len(label) * 8.6
        self.parts.append(
            f'<rect x="{x + 18}" y="{y - 14}" width="{tab_w}" height="28" rx="14" fill="{th.stroke}"/>'
        )
        self.text(x + 18 + tab_w / 2, y + 5, [label], size=14, weight=700, color="#ffffff")

    def arrow(self, points, color="#475569", label=None, label_at=None, dashed=False, width=2.2, label_anchor="middle"):
        d = "M " + " L ".join(f"{px} {py}" for px, py in points)
        marker = self._marker(color)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash} '
            f'marker-end="url(#{marker})"/>'
        )
        if label:
            lx, ly = label_at or (
                (points[0][0] + points[-1][0]) / 2,
                (points[0][1] + points[-1][1]) / 2,
            )
            lines = label if isinstance(label, list) else [label]
            width_px = max(len(s) for s in lines) * 7.1 + 14
            height_px = 20 * len(lines) + 4
            bx = lx - width_px / 2 if label_anchor == "middle" else lx - 7
            self.parts.append(
                f'<rect x="{bx}" y="{ly - 15}" width="{width_px}" height="{height_px}" rx="6" fill="#ffffff" '
                f'stroke="{color}" stroke-width="1"/>'
            )
            self.text(lx if label_anchor == "middle" else lx, ly, lines, size=12.5, color=color, weight=600, anchor=label_anchor, line_height=20)

    _markers: dict[str, str] = {}

    def _marker(self, color: str) -> str:
        key = "m" + color.strip("#")
        if key not in self._markers:
            self._markers[key] = (
                f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
            )
        return key

    def chip(self, x, y, w, h, label, theme, size=13, bold=True, lines=None):
        th = T[theme]
        self.rect(x, y, w, h, th, rx=10)
        content = lines or [label]
        offset = (len(content) - 1) * size * 1.3 / 2
        self.text(x + w / 2, y + h / 2 + size * 0.35 - offset, content, size=size, weight=700 if bold else 500, color=th.title, line_height=round(size * 1.3))

    def diamond(self, cx, cy, w, h, label, theme):
        th = T[theme]
        pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
        self.parts.append(f'<polygon points="{pts}" fill="{th.fill}" stroke="{th.stroke}" stroke-width="1.6"/>')
        self.text(cx, cy + 5, [label], size=14, weight=700, color=th.title)

    def save(self, name: str) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        defs = "<defs>" + "".join(self._markers.values()) + "</defs>"
        body = "\n".join(self.parts)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" role="img">\n{defs}\n{body}\n</svg>\n'
        )
        path = OUT / name
        path.write_text(svg, encoding="utf-8")
        self._markers.clear()
        return path


def node(s: Svg, x, y, w, h, title, sub=None, theme="plain"):
    """Box with a short bold title and at most one short sub-line."""
    th = T[theme]
    s.rect(x, y, w, h, th, rx=14)
    if sub:
        s.text(x + w / 2, y + h / 2 - 4, [title], size=18, weight=700, color=th.title)
        s.text(x + w / 2, y + h / 2 + 18, [sub], size=14.5, color=MUTED)
    else:
        s.text(x + w / 2, y + h / 2 + 6, [title], size=18, weight=700, color=th.title)


GRAY = "#475569"


# --- Hình 1: tổng quan ---------------------------------------------------------------------


def overview() -> Path:
    s = Svg(1220, 1010, "Hình 1. Tổng quan pipeline", "2 sản phẩm: Việt + Anh  ·  Việt + Anh + Nhật")
    cx = 610
    node(s, 410, 100, 400, 60, "PDF · Word · Ảnh scan", None, "input")
    s.arrow([(cx, 160), (cx, 192)], GRAY)
    node(s, 410, 194, 400, 72, "① Tiền xử lý", "đọc text PDF · làm sạch ảnh scan", "pre")
    s.arrow([(cx, 266), (cx, 298)], GRAY)
    node(s, 410, 300, 400, 72, "② Nhận diện bố cục", "PP-StructureV3 · thứ tự đọc", "layout")
    s.arrow([(cx, 372), (cx, 404)], GRAY)
    node(s, 410, 406, 400, 72, "③ Router", "chọn cách xử lý cho từng vùng", "router")
    engines = [
        (150, "Văn bản", "text layer PDF → OCR", "text"),
        (470, "Bảng · Công thức", "PaddleOCR → VLM dự phòng", "table"),
        (790, "Hình ảnh · Biểu đồ", "VLM mô tả · OCR đọc chữ", "vlm"),
    ]
    for x, title, sub, theme in engines:
        s.arrow([(cx, 478), (cx, 508), (x + 140, 508), (x + 140, 538)], "#d97706", width=2)
        node(s, x, 540, 280, 76, title, sub, theme)
        s.arrow([(x + 140, 616), (x + 140, 646), (cx, 646), (cx, 676)], "#16a34a", width=2)
    node(s, 410, 678, 400, 72, "④ Kiểm tra & hợp nhất", "đối chiếu nguồn · ghép theo thứ tự đọc", "check")
    s.arrow([(410, 714), (80, 714), (80, 442), (408, 442)], "#dc2626", dashed=True,
            label=["chưa đạt:", "cách khác"], label_at=(80, 590))
    s.arrow([(810, 230), (1150, 230), (1150, 714), (812, 714)], "#0284c7", dashed=True,
            label=["Word:", "cấu trúc có sẵn"], label_at=(1150, 470))
    s.arrow([(cx, 750), (cx, 782)], GRAY)
    node(s, 360, 784, 500, 76, "⑤ Đầu ra", "Markdown · JSON · hình giữ lại", "out")
    s.arrow([(cx, 860), (cx, 892)], GRAY)
    node(s, 460, 894, 300, 60, "AI · RAG · LLM", None, "input")
    s.text(cx, 990, ["Văn bản chỉ lấy từ text layer PDF hoặc OCR, không dùng VLM"], size=14.5, color="#991b1b", weight=600)
    return s.save("pipeline_1_overview.svg")


# --- Hình 2: router và kiểm tra -------------------------------------------------------------------


def router() -> Path:
    s = Svg(1240, 720, "Hình 2. Router và kiểm tra", "Mỗi vùng thử lần lượt từng cách cho đến khi đạt")
    s.text(40, 118, ["Chuỗi phương pháp theo loại vùng"], size=16, weight=700, color=INK, anchor="start")
    rows = [
        ("Văn bản", "text", [("Text layer PDF", "text"), ("OCR", "ocr")]),
        ("Bảng", "table", [("Bảng PDF", "text"), ("PaddleOCR", "table"), ("VLM", "vlm")]),
        ("Công thức", "formula", [("PaddleOCR", "formula"), ("VLM", "vlm")]),
        ("Hình ảnh", "vlm", [("VLM mô tả", "vlm"), ("Giữ hình", "out")]),
    ]
    y = 140
    for label, theme, chain in rows:
        s.chip(40, y, 150, 54, label, theme, size=16)
        x = 230
        s.arrow([(192, y + 27), (228, y + 27)], GRAY, width=2)
        for i, (name, th) in enumerate(chain):
            if i:
                s.arrow([(x - 34, y + 27), (x - 4, y + 27)], GRAY, width=2)
            s.chip(x, y, 150, 54, name, th, size=15, bold=False)
            x += 186
        y += 84

    # vòng thử - kiểm tra
    node(s, 800, 120, 320, 58, "Thử cách tiếp theo", None, "router")
    s.arrow([(960, 178), (960, 206)], GRAY)
    node(s, 800, 208, 320, 58, "Kiểm tra kết quả", None, "check")
    s.arrow([(960, 266), (960, 290)], GRAY)
    s.diamond(960, 330, 170, 78, "Đạt?", "check")
    s.arrow([(960, 369), (960, 404)], "#16a34a", label="đạt", label_at=(990, 390))
    node(s, 840, 406, 240, 58, "Nhận kết quả", None, "check")
    s.arrow([(1045, 330), (1165, 330), (1165, 149), (1122, 149)], "#dc2626", label="chưa", label_at=(1165, 245))
    s.arrow([(1165, 330), (1165, 517), (1122, 517)], "#dc2626", dashed=True, label="hết cách", label_at=(1165, 440))
    node(s, 800, 488, 320, 58, "Đánh dấu cần kiểm tra", None, "warn")

    s.text(40, 618, ["Kiểm tra gì?"], size=16, weight=700, color=INK, anchor="start")
    checks = ["Độ tin cậy OCR", "Mất dấu tiếng Việt", "Cấu trúc bảng", "Đối chiếu số liệu"]
    x = 40
    for name in checks:
        s.chip(x, 636, 266, 50, name, "check", size=15, bold=False)
        x += 290
    return s.save("pipeline_2_router.svg")


# --- Hình 3: hình ảnh -----------------------------------------------------------------------------


def figures() -> Path:
    s = Svg(1200, 560, "Hình 3. Hình ảnh: thay bằng chữ hay giữ hình?", "Nguyên tắc: không được mất thông tin")
    node(s, 40, 250, 210, 76, "Hình ảnh", "ảnh · biểu đồ · sơ đồ", "input")
    s.arrow([(250, 288), (298, 288)], GRAY)
    node(s, 300, 250, 250, 76, "Mô tả + đọc chữ", "VLM mô tả · OCR đọc chữ", "vlm")
    s.arrow([(550, 288), (588, 288)], GRAY)
    s.diamond(710, 288, 240, 104, "Chữ thay được hết?", "router")
    s.arrow([(710, 236), (710, 175), (858, 175)], "#16a34a", label="có", label_at=(740, 205))
    node(s, 860, 138, 300, 76, "Thay bằng chữ", "mô tả + số liệu", "check")
    s.arrow([(710, 340), (710, 401), (858, 401)], "#7c3aed", label="không", label_at=(752, 372))
    node(s, 860, 364, 300, 76, "Giữ hình", "kèm mô tả và chữ trong hình", "out")
    s.text(1010, 240, ["VD: lưu đồ, biểu đồ có in số"], size=14, color=MUTED)
    s.text(1010, 466, ["VD: ảnh chụp, bản đồ, chữ ký, logo"], size=14, color=MUTED)
    s.text(1010, 488, ["chưa có VLM → luôn giữ hình"], size=14, color=MUTED)
    s.text(300, 410, ["Thay khi đủ cả 3:"], size=15, weight=700, color=INK, anchor="start")
    s.text(300, 438, ["① hình mang tính văn bản", "② mô tả không mất thông tin", "③ chữ và số đọc được bằng OCR"],
           size=14.5, color=INK, anchor="start", line_height=24)
    return s.save("pipeline_3_figures.svg")


# --- Hình 4: huấn luyện OCR -------------------------------------------------------------------------


def training() -> Path:
    s = Svg(1200, 530, "Hình 4. Huấn luyện model OCR cho từng sản phẩm", "Chạy trên GitHub Actions")
    s.chip(250, 92, 700, 44, "Model gốc thiếu chữ có dấu chồng:  “Cộng hòa” → “Cng hòa”", "warn", size=15)
    top = [("Model gốc", "baseline của sản phẩm", "input"), ("Thêm chữ tiếng Việt", "ộ ← ô ← o", "pre"),
           ("Dữ liệu tổng hợp", "60–80 nghìn dòng chữ", "layout")]
    bottom = [("Fine-tune", "tối đa ~5 giờ", "text"), ("Đánh giá", "CER so với baseline", "check"),
              ("Release", "pipeline tự tải về", "out")]
    for row, items in ((170, top), (320, bottom)):
        x = 60
        for i, (title, sub, theme) in enumerate(items):
            node(s, x, row, 300, 76, title, sub, theme)
            if i:
                s.arrow([(x - 60, row + 38), (x - 2, row + 38)], GRAY)
            x += 390
    s.arrow([(990, 246), (990, 283), (210, 283), (210, 318)], GRAY)
    s.chip(60, 450, 520, 50, "Việt + Anh: từ latin_PP-OCRv5_mobile_rec", "plain", size=15, bold=False)
    s.chip(620, 450, 520, 50, "Việt + Anh + Nhật: từ PP-OCRv5_mobile_rec", "plain", size=15, bold=False)
    return s.save("pipeline_4_ocr_training.svg")


if __name__ == "__main__":
    for path in (overview(), router(), figures(), training()):
        print(path)
