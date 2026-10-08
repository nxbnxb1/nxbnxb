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


# --- Hình 1: tổng quan ---------------------------------------------------------------------


def overview() -> Path:
    s = Svg(1400, 1335, "Hình 1. Tổng quan pipeline trích xuất tài liệu (OCR + VLM)",
            "2 sản phẩm: Việt + Anh · Việt + Anh + Nhật — OCR/Parser xử lý phần dễ, VLM chỉ xử lý hình ảnh khó — GitHub Actions")
    gray = "#475569"

    # Đầu vào
    s.text(30, 128, ["ĐẦU VÀO"], size=13, weight=700, color=MUTED, anchor="start")
    s.box(190, 95, 300, 72, "PDF", ["có text layer hoặc bản scan"], "input")
    s.box(550, 95, 300, 72, "Ảnh scan", ["PNG · JPG · TIFF nhiều trang"], "input")
    s.box(910, 95, 300, 72, "Word", [".docx  (.doc tự chuyển sang .docx)"], "input")

    # ① Tiền xử lý
    s.section(150, 205, 1100, 200, "① TIỀN XỬ LÝ", "pre")
    s.box(170, 230, 340, 158, "PDF có text layer", [
        "PyMuPDF đọc chữ, font, vị trí",
        "Không OCR lại chữ đã đúng",
        "Phát hiện lỗi mã hóa: TCVN3/VNI,",
        "(cid:…) → xử lý như bản scan",
    ], "pre", size=13)
    s.box(530, 230, 340, 158, "PDF scan · ảnh", [
        "Render trang 200 DPI",
        "Xoay trang (PP-LCNet doc_ori)",
        "Chống nghiêng · tăng tương phản",
        "Chuẩn hóa độ phân giải",
    ], "pre", size=13)
    s.box(890, 230, 340, 158, "Word", [
        "python-docx đọc cấu trúc gốc:",
        "tiêu đề, danh sách, bảng ô gộp,",
        "ảnh nhúng",
        "→ không cần layout/OCR",
    ], "pre", size=13)
    s.arrow([(340, 167), (340, 228)], gray)
    s.arrow([(440, 167), (610, 228)], gray, label="trang scan", label_at=(520, 193))
    s.arrow([(700, 167), (700, 228)], gray)
    s.arrow([(1060, 167), (1060, 228)], gray)

    # ② Layout
    s.section(150, 445, 820, 150, "② NHẬN DIỆN BỐ CỤC", "layout")
    s.box(170, 468, 780, 112, "PP-DocLayout_plus-L  (module Layout của PP-StructureV3)", [
        "Vùng: tiêu đề · văn bản · danh sách · bảng · công thức · biểu đồ · hình ảnh · header/footer",
        "Thứ tự đọc: tách cột, khối tràn cột; header đầu trang, footer/số trang cuối trang",
        "Chưa cài PaddleOCR: bố cục suy ra từ chính file PDF · giữ cả chữ PDF nằm ngoài mọi vùng",
    ], "layout", size=13)
    s.arrow([(340, 405), (340, 466)], gray)
    s.arrow([(700, 405), (700, 466)], gray)

    # Tối ưu
    s.box(1010, 445, 240, 270, "Tối ưu tốc độ", [
        "• Crop đúng vùng,",
        "  không gửi cả trang",
        "• Gom batch theo",
        "  phương pháp",
        "• Chạy song song các nhóm",
        "• Gọi VLM đồng thời",
        "• Cache SQLite theo",
        "  điểm ảnh của vùng",
        "• VLM chỉ khi thật cần",
    ], "note", size=13, align="start")

    # ③ Router
    s.section(150, 635, 820, 100, "③ ROUTER THÔNG MINH (luật)", "router")
    s.box(170, 655, 780, 66, "", [
        "Mỗi vùng → một chuỗi phương pháp, chọn theo loại vùng, độ phức tạp,",
        "chất lượng text layer và engine sẵn có  (chi tiết: Hình 2)",
    ], "router", size=13.5)
    s.arrow([(560, 595), (560, 653)], gray)

    # ④ Bộ xử lý (arrows first, so the section tab is drawn over them)
    for xc in (268, 484, 700, 916):
        s.arrow([(xc, 721), (xc, 795)], "#d97706", width=1.6)
    s.arrow([(950, 688), (1132, 688), (1132, 795)], "#d97706", width=1.6)
    engines = [
        ("Text layer PDF", ["PyMuPDF", "văn bản, tiêu đề,", "bảng vector"], "text"),
        ("PaddleOCR", ["PP-OCRv5 fine-tune", "theo từng sản phẩm", "(Hình 4)"], "ocr"),
        ("Nhận dạng bảng", ["PaddleOCR Table", "→ HTML (rowspan,", "colspan)"], "table"),
        ("Công thức", ["PP-FormulaNet_plus-M", "→ LaTeX"], "formula"),
        ("VLM (Qwen-VL)", ["mô tả hình, biểu đồ ·", "dự phòng bảng/công thức"], "vlm"),
    ]
    x = 168
    for title, lines, theme in engines:
        s.box(x, 797, 200, 138, title, lines, theme, size=13)
        x += 216
    s.text(1132, 912, ["KHÔNG đọc văn bản"], size=13, weight=700, color="#dc2626")
    s.section(150, 775, 1100, 175, "④ BỘ XỬ LÝ THEO VÙNG", "text")

    # ⑤ Kiểm tra & hợp nhất
    s.section(150, 990, 820, 150, "⑤ KIỂM TRA & HỢP NHẤT", "check")
    s.box(170, 1012, 780, 112, "", [
        "Kiểm tra: độ tin cậy OCR · mất chữ tiếng Việt · cấu trúc bảng · đối chiếu số liệu, ký tự đặc biệt",
        "Không đạt → thử phương pháp kế tiếp; hết cách → đánh dấu “cần kiểm tra” (Hình 2)",
        "Hình không thay được bằng chữ mà không mất thông tin → giữ file ảnh (Hình 3)",
        "Ghép các vùng theo thứ tự đọc, gán cấp tiêu đề (#, ##, ###)",
    ], "check", size=13.5)
    for xc in (268, 484, 700, 916):
        s.arrow([(xc, 935), (min(xc, 900), 1010)], "#16a34a", width=1.6)
    s.arrow([(1132, 935), (1132, 1068), (952, 1068)], "#16a34a", width=1.6)

    # vòng lặp không đạt → router
    s.arrow([(170, 1068), (112, 1068), (112, 688), (168, 688)], "#dc2626", dashed=True,
            label=["không đạt →", "cách kế tiếp"], label_at=(112, 860))
    # Word đi thẳng
    s.arrow([(1230, 309), (1300, 309), (1300, 1100), (952, 1100)], "#0284c7", dashed=True,
            label=["Word: cấu trúc", "có sẵn"], label_at=(1300, 990))

    # ⑥ Đầu ra
    s.section(150, 1180, 860, 120, "⑥ ĐẦU RA", "out")
    s.box(170, 1200, 290, 86, "Markdown (.md)", ["tiêu đề, đoạn, bảng, $$công thức$$,", "mô tả hình, ![hình giữ lại](…)"], "out", size=13)
    s.box(476, 1200, 300, 86, "JSON", ["trang · tọa độ · loại vùng · phương pháp", "· độ tin cậy · trạng thái · nguồn"], "out", size=13)
    s.box(792, 1200, 200, 86, "Ảnh giữ lại", ["<tên>_assets/", "p3-r5.png"], "out", size=13)
    s.box(1050, 1200, 200, 86, "AI / RAG / LLM", ["tìm kiếm, hỏi đáp,", "phân tích"], "input", size=13)
    s.arrow([(560, 1140), (560, 1198)], gray)
    s.arrow([(1010, 1243), (1048, 1243)], gray)
    return s.save("pipeline_1_overview.svg")


# --- Hình 2: router, thực thi, kiểm tra ----------------------------------------------------------


def router() -> Path:
    s = Svg(1400, 1210, "Hình 2. Router, thực thi theo vòng và kiểm tra kết quả",
            "Mỗi vùng có một chuỗi phương pháp; thử lần lượt đến khi kết quả đạt kiểm tra")
    gray = "#475569"
    s.section(30, 100, 930, 610, "CHUỖI PHƯƠNG PHÁP THEO LOẠI VÙNG", "router")

    rows = [
        ("Văn bản", ["tiêu đề · đoạn · danh sách", "header/footer · con dấu"], "text",
         [("Text layer PDF *", "text"), ("PaddleOCR", "ocr")],
         ("✕ Không dùng VLM cho văn bản", "warn")),
        ("Bảng", ["bảng biểu"], "table",
         [("Bảng vector PDF *", "text"), ("PaddleOCR Table", "table"), ("VLM", "vlm"), ("Chữ + ảnh gốc, cần kiểm tra", "warn")],
         ("Bảng lớn (>45% trang hoặc >120 ô): VLM trước PaddleOCR Table", "note")),
        ("Công thức", ["toán học"], "formula",
         [("PP-FormulaNet", "formula"), ("VLM", "vlm"), ("Chữ + ảnh gốc, cần kiểm tra", "warn")], None),
        ("Biểu đồ", ["sơ đồ, đồ thị"], "vlm",
         [("VLM: mô tả + số liệu", "vlm"), ("Giữ hình (Hình 3)", "out")],
         ("Chữ và số in trong hình luôn do OCR / text layer đọc", "note")),
        ("Hình ảnh", ["ảnh, logo, chữ ký…"], "vlm",
         [("VLM: loại + mô tả", "vlm"), ("Giữ hình (Hình 3)", "out")],
         ("Nhỏ hơn 1% trang: giữ hình, không mô tả · nhỏ hơn 0,2%: trang trí, bỏ", "note")),
    ]
    y = 128
    for name, sub, theme, chain, note in rows:
        s.box(50, y, 190, 84, name, sub, theme, size=12.5)
        x = 268
        for i, (label, th) in enumerate(chain):
            w = max(84, round(len(label) * 7.0 + 26))
            s.chip(x, y + 18, w, 48, label, th, size=13)
            if i:
                s.arrow([(x - 26, y + 42), (x - 2, y + 42)], gray, width=1.8)
            x += w + 28
        s.arrow([(242, y + 42), (266, y + 42)], gray, width=1.8)
        if note:
            text, th = note
            color = T[th].title if th != "warn" else "#dc2626"
            s.text(268, y + 88, [text], size=12.5, weight=600 if th == "warn" else 500, color=color, anchor="start")
        y += 112
    s.text(50, 690, ["* chỉ khi trang PDF có text layer dùng được. Engine chưa cấu hình (VD: chưa có VLM) được bỏ khỏi chuỗi."],
           size=12.5, color=MUTED, anchor="start", italic=True)

    # vòng thực thi
    s.section(1000, 100, 370, 610, "VÒNG THỰC THI", "check")
    s.box(1025, 128, 320, 64, "1. Gom vùng theo phương pháp", ["kế tiếp trong chuỗi của từng vùng"], "router", size=12.5, title_size=14)
    s.box(1025, 222, 320, 64, "2. Chạy theo batch", ["nhóm chạy song song · cache"], "text", size=12.5, title_size=14)
    s.box(1025, 316, 320, 64, "3. Kiểm tra kết quả", ["các phép kiểm tra bên dưới"], "check", size=12.5, title_size=14)
    s.diamond(1185, 432, 150, 70, "Đạt?", "check")
    s.box(1025, 500, 140, 70, "Chốt kết quả", ["passed / unchecked"], "check", size=12, title_size=13.5)
    s.diamond(1270, 535, 150, 70, "Còn cách?", "router")
    s.box(1060, 612, 290, 84, "Hết cách", ["lấy kết quả điểm cao nhất,", "needs_review + lý do"], "warn", size=12.5, title_size=14)
    s.arrow([(1185, 192), (1185, 220)], gray)
    s.arrow([(1185, 286), (1185, 314)], gray)
    s.arrow([(1185, 380), (1185, 396)], gray)
    s.arrow([(1140, 450), (1095, 498)], "#16a34a", label="có", label_at=(1102, 470))
    s.arrow([(1230, 450), (1262, 499)], "#dc2626", label="không", label_at=(1262, 470))
    s.arrow([(1345, 535), (1362, 535), (1362, 160), (1347, 160)], "#d97706", dashed=True, label="còn", label_at=(1362, 360))
    s.arrow([(1270, 571), (1270, 610)], "#dc2626", label="hết", label_at=(1300, 592))

    # các phép kiểm tra
    s.section(30, 750, 1340, 330, "CÁC PHÉP KIỂM TRA", "check")
    cards = [
        ("Văn bản OCR", ["Độ tin cậy trung bình ≥ 0,85", "Dòng tin cậy < 0,6: ≤ 25%", "Ký tự lỗi: �, vùng riêng, (cid:)",
                         "Mất chữ tiếng Việt: từ không", "có nguyên âm (“Cng”, “Vit”)"], "ocr"),
        ("Text layer PDF", ["Font cũ TCVN3/VNI", "(“Céng hßa x· héi”)", "(cid:…), ký tự vùng riêng", "Không đạt → OCR"], "text"),
        ("Bảng", ["Mỗi hàng đủ số ô (tính", "rowspan/colspan) · ô trống ≤ 60%", "Độ tin cậy ≥ 0,8 (+0,1 nếu phức tạp)",
                  "Đối chiếu số liệu ≥ 90% và", "ký tự đặc biệt % ₫ € ± ≤ ≥"], "table"),
        ("Công thức", ["Cân ngoặc { }", "\\left/\\right, \\begin/\\end", "Không lặp token", "(lỗi giải mã)"], "formula"),
        ("Hình · biểu đồ", ["Có mô tả, không bị từ chối", "Thay bằng chữ chỉ khi", "không mất thông tin;", "ngược lại giữ hình (Hình 3)"], "vlm"),
    ]
    x = 50
    for title, lines, theme in cards:
        s.box(x, 778, 252, 180, title, lines, theme, size=12.5, align="start")
        x += 264

    # chú giải trạng thái + log
    s.text(50, 990, ["Trạng thái mỗi vùng trong JSON:"], size=13.5, weight=700, color=INK, anchor="start")
    legend = [("passed", "đạt kiểm tra", "check", 270), ("unchecked", "không kiểm tra tự động được", "note", 520),
              ("needs_review", "cần người kiểm tra", "warn", 860), ("skipped", "bỏ qua (trang trí)", "input", 1130)]
    for key, desc, theme, x in legend:
        s.chip(x, 972, 104, 28, key, theme, size=12)
        s.text(x + 112, 991, [desc], size=12.5, color=MUTED, anchor="start")
    s.box(50, 1020, 1300, 46, "", ["Mỗi quyết định của Router và kết quả kiểm tra được ghi vào route_log.jsonl → dữ liệu huấn luyện Neural Router (giai đoạn 3)"],
          "router", size=13.5)
    s.text(700, 1120, ["Ví dụ: bảng trong bản scan → PaddleOCR Table cho hàng thiếu ô → không đạt → VLM dựng lại bảng HTML"], size=13.5, color=MUTED)
    s.text(700, 1146, ["→ số liệu của VLM được đối chiếu với token OCR tin cậy cao → đạt → status = passed, attempts = [table_recognition, vlm]"], size=13.5, color=MUTED)
    s.text(700, 1180, ["Văn bản OCR bị mất chữ tiếng Việt → không chuyển sang VLM → giữ kết quả OCR, needs_review, <!-- cần kiểm tra --> trong Markdown"],
           size=13.5, color="#991b1b", weight=600)
    return s.save("pipeline_2_router.svg")


# --- Hình 3: hình ảnh — thay bằng chữ hay giữ hình -------------------------------------------------


def figures() -> Path:
    s = Svg(1400, 990, "Hình 3. Xử lý hình ảnh: thay bằng chữ hay giữ lại hình?",
            "Nguyên tắc: không được mất thông tin — chỉ thay hình bằng chữ khi chữ mang được toàn bộ nội dung của hình")
    gray = "#475569"
    s.box(150, 98, 420, 56, "Vùng hình ảnh · biểu đồ · con dấu", [], "input")
    s.arrow([(360, 154), (360, 178)], gray)
    s.diamond(360, 215, 270, 72, "Nhỏ hơn 0,2% trang?", "note")
    s.box(565, 188, 215, 56, "Bỏ qua", ["trang trí: đường kẻ, chấm"], "input", size=12.5, title_size=14)
    s.arrow([(495, 215), (563, 215)], gray, label="có", label_at=(528, 206))
    s.arrow([(360, 251), (360, 286)], gray, label="không", label_at=(398, 272))
    s.diamond(360, 320, 210, 66, "Con dấu?", "note")
    s.box(565, 292, 215, 56, "Giữ hình", ["+ OCR chữ trên con dấu"], "out", size=12.5, title_size=14)
    s.arrow([(465, 320), (563, 320)], gray, label="có", label_at=(514, 311))
    s.arrow([(360, 353), (360, 376), (225, 376), (225, 398)], gray)
    s.arrow([(360, 376), (575, 376), (575, 398)], gray)
    s.text(395, 368, ["không"], size=12.5, color=gray, weight=600, anchor="start")
    s.box(60, 400, 330, 98, "VLM", ["phân loại hình · mô tả ·", "“lossless”: mô tả có mất", "thông tin không? (JSON)"], "vlm", size=13)
    s.box(410, 400, 330, 98, "OCR / text layer PDF", ["đọc chữ và số in trong hình", "(chữ không bao giờ lấy", "từ VLM)"], "ocr", size=13)
    s.arrow([(225, 498), (225, 528)], gray)
    s.arrow([(575, 498), (575, 528)], gray)
    s.box(60, 530, 680, 178, "Điều kiện để thay hình bằng chữ (phải đủ cả 5)", [
        "1. Có mô tả hợp lệ (VLM trả về JSON đọc được)",
        "2. Loại hình mang tính văn bản: biểu đồ, sơ đồ, lưu đồ, ảnh chụp màn hình, bảng, chữ",
        "3. VLM xác nhận mô tả không làm mất thông tin (lossless = true)",
        "4. Chữ trong hình đã được OCR / text layer đọc",
        "5. Biểu đồ: có bảng số liệu, không giá trị ước lượng (~), mọi số đều in trong hình",
    ], "check", size=13.5, align="start")
    s.arrow([(400, 708), (400, 724)], gray)
    s.diamond(400, 762, 290, 76, "Đủ cả 5 điều kiện?", "check")
    s.box(60, 838, 330, 118, "Thay bằng chữ", [
        "> **[Biểu đồ]** mô tả",
        "+ bảng số liệu",
        "> Chữ trong hình: …",
        "không lưu file ảnh",
    ], "check", size=13, align="start", title_size=15)
    s.box(420, 838, 340, 118, "Giữ hình", [
        "![Hình ảnh: mô tả](<tên>_assets/p3-r5.png)",
        "+ mô tả + chữ trong hình",
        "JSON: figure, figure_decision (lý do)",
    ], "out", size=12.5, align="start", title_size=15)
    s.arrow([(255, 762), (225, 762), (225, 836)], "#16a34a", label="có", label_at=(205, 800))
    s.arrow([(545, 762), (590, 762), (590, 836)], "#7c3aed", label="không", label_at=(625, 800))

    s.section(800, 98, 570, 760, "VÍ DỤ", "out")
    examples = [
        ("Ảnh chụp nhà máy, con người", "Giữ hình", "out"),
        ("Bản đồ, bản vẽ kỹ thuật", "Giữ hình", "out"),
        ("Logo, chữ ký, chữ viết tay", "Giữ hình", "out"),
        ("Lưu đồ có nhãn rõ (A → B → C)", "Thay bằng chữ", "check"),
        ("Ảnh chụp màn hình toàn chữ", "Thay bằng chữ", "check"),
        ("Biểu đồ cột có in số trên cột", "Thay: mô tả + số liệu", "check"),
        ("Biểu đồ phải ước lượng giá trị (~)", "Giữ hình + mô tả", "out"),
        ("Chưa cấu hình VLM / VLM lỗi", "Giữ hình", "out"),
        ("Bảng, công thức không khôi phục được", "Giữ ảnh gốc cạnh chữ", "out"),
    ]
    y = 128
    for case, result, theme in examples:
        s.chip(820, y, 300, 56, case, "plain", size=13, bold=False)
        s.arrow([(1122, y + 28), (1146, y + 28)], gray, width=1.8)
        s.chip(1148, y, 202, 56, result, theme, size=13)
        y += 80
    s.box(800, 880, 570, 76, "figure_policy", [
        "auto (mặc định): theo các điều kiện bên trái",
        "always: luôn giữ hình · never: không lưu hình",
    ], "note", size=13)
    return s.save("pipeline_3_figures.svg")


# --- Hình 4: model OCR Việt · Anh · Nhật ---------------------------------------------------------------


def training() -> Path:
    s = Svg(1400, 1010, "Hình 4. Cải thiện PaddleOCR: fine-tune model nhận dạng cho từng sản phẩm",
            "Việt + Anh (từ latin_PP-OCRv5_mobile_rec) · Việt + Anh + Nhật (từ PP-OCRv5_mobile_rec) — workflow “… · Train OCR model”")
    gray = "#475569"
    s.box(150, 95, 1100, 96, "Vấn đề: model gốc (latin/PP-OCRv5, PP-OCRv6) thiếu ~90 chữ tiếng Việt có dấu chồng (ạ ả ấ ầ … ộ … ỹ)", [
        "Ảnh “Cộng hòa xã hội chủ nghĩa Việt Nam”  →  OCR “Cng hòa xã hi ch nghĩa Vit Nam” với độ tin cậy 0,98",
        "Bộ giải mã CTC không thể sinh ra chữ không có trong từ điển, nên bỏ mất cả chữ chứ không chỉ mất dấu",
    ], "warn", size=13.5, title_size=15)

    steps_top = [
        ("① Bộ ký tự", ["Việt + Anh: 281 ký tự gọn", "(ASCII, 134 chữ Việt, ₫ € ° ±…)", "Việt + Anh + Nhật: từ điển PP-OCRv5",
                        "(18.383 ký tự: kana, kanji, Latin)", "+ 118 chữ Việt & ký hiệu còn thiếu"], "pre"),
        ("② Khởi tạo trọng số", ["Từ model baseline của sản phẩm:", "giữ backbone PPLCNetV3 + SVTR;", "lớp CTC & NRTR: chữ cũ chép",
                                 "nguyên, chữ mới lấy từ chữ gần", "nhất: ộ ← ô ← o,  ẵ ← ă ← a"], "router"),
        ("③ Dữ liệu tổng hợp", ["Từ vựng Việt/Anh/Nhật theo tần suất", "Mẫu văn bản: số tiền đồng/円,", "ngày tháng, số hiệu, 第N条",
                                "Âm tiết phủ chữ Việt hiếm, IN HOA", "Font phủ đủ ký tự của từng dòng", "(loại font vẽ thiếu dấu tiếng Việt)",
                                "Làm xấu như scan: mờ, nhiễu, JPEG…"], "layout"),
    ]
    x = 150
    for title, lines, theme in steps_top:
        s.box(x, 240, 340, 232, title, lines, theme, size=13, align="start")
        x += 380
    s.arrow([(490, 356), (528, 356)], gray)
    s.arrow([(870, 356), (908, 356)], gray)
    s.text(1066, 494, ["60–80 nghìn dòng huấn luyện + 1.000 dòng đánh giá mỗi ngôn ngữ"], size=13, color=MUTED, anchor="end")

    steps_bottom = [
        ("④ Fine-tune", ["CPU runner, giới hạn ~5 giờ", "CTC + NRTR (đa đầu ra)", "không random crop", "(tránh cắt mất dấu)"], "text"),
        ("⑤ Export", ["Định dạng PaddleOCR 3.x", "inference.json / .pdiparams", "/ .yml (kèm từ điển)"], "pre"),
        ("⑥ Đánh giá", ["CER, độ chính xác dòng theo", "từng ngôn ngữ của sản phẩm;", "so với baseline của chính nó"], "check"),
        ("⑦ GitHub Release", ["<sản phẩm>-ocr-N:", "model + checkpoint", "+ báo cáo đánh giá"], "out"),
    ]
    x = 150
    for title, lines, theme in steps_bottom:
        s.box(x, 540, 255, 160, title, lines, theme, size=13, align="start")
        x += 281
    for x0 in (405, 686, 967):
        s.arrow([(x0, 620), (x0 + 24, 620)], gray)
    s.arrow([(1080, 472), (1080, 502), (277, 502), (277, 538)], gray)

    # dùng model
    s.box(150, 760, 600, 112, "Dùng trong pipeline", [
        "Extract / Benchmark của mỗi sản phẩm tự tải Release mới nhất",
        "của sản phẩm đó → PaddleOCR (văn bản, ô bảng, chữ trong",
        "hình); chưa có Release thì chạy bằng baseline",
    ], "ocr", size=13, align="start")
    s.box(800, 760, 450, 112, "Kết quả mong muốn", [
        "“Cộng hòa xã hội chủ nghĩa Việt Nam”, “売上高”",
        "giữ nguyên dấu và chữ Nhật; số liệu",
        "1.234,5 tỷ đồng, 12,3% … đúng từng ký tự",
    ], "check", size=13, align="start")
    s.arrow([(1124, 700), (1124, 730), (450, 730), (450, 758)], gray)
    s.arrow([(750, 816), (798, 816)], gray)
    # huấn luyện tiếp
    s.arrow([(1250, 640), (1300, 640), (1300, 520), (200, 520), (200, 538)], "#d97706", dashed=True,
            label=["huấn luyện tiếp", "(resume_tag)"], label_at=(1300, 585))

    s.box(150, 905, 1100, 80, "Muốn chính xác hơn trên tài liệu thật", [
        "Thêm dòng chữ cắt từ tài liệu thật (đường_dẫn<TAB>nhãn) vào dữ liệu, tăng số dòng/epoch, chạy tiếp từ checkpoint (resume_tag)",
    ], "note", size=13.5)
    return s.save("pipeline_4_ocr_training.svg")


if __name__ == "__main__":
    for path in (overview(), router(), figures(), training()):
        print(path)
