"""Generate demo documents (Vietnamese + English) and their ground truth:

    samples/bao_cao_digital.pdf   PDF with a text layer
    samples/bao_cao_scan.pdf      the same page as a skewed, noisy scan (image only)
    samples/bao_cao.docx          Word version with headings, list and a merged-cell table
    samples/houkoku_digital.pdf   Japanese report with a text layer (built-in CJK font of PyMuPDF)
    samples/houkoku_scan.pdf      the same Japanese page as a scan
    samples/hinh_anh.pdf          a page with a bar chart and a photo-like picture (no ground truth:
                                  exercises the VLM and the keep-or-replace rule for pictures)
    samples/<name>.gt.md          expected Markdown, for `docextract bench samples`
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pymupdf
from docx import Document
from PIL import Image, ImageFilter

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
TITLE = "BÁO CÁO KẾT QUẢ KINH DOANH NĂM 2025"
PARAGRAPHS = [
    "Năm 2025, doanh thu thuần của Công ty đạt 1.234,5 tỷ đồng, tăng 12,3% so với năm 2024. "
    "Lợi nhuận sau thuế đạt 156,8 tỷ đồng, vượt 4,2% kế hoạch được Đại hội đồng cổ đông thông qua.",
    "The Board of Directors proposes a cash dividend of 15% for the 2025 financial year.",
]
BULLETS = ["Mở rộng hệ thống phân phối tại 12 tỉnh, thành phố.", "Đầu tư dây chuyền sản xuất mới trị giá 85 tỷ đồng."]
TABLE = [["Chỉ tiêu", "Năm 2024", "Năm 2025", "Tăng trưởng"],
         ["Doanh thu thuần", "1.099,2", "1.234,5", "12,3%"],
         ["Lợi nhuận gộp", "312,4", "355,0", "13,6%"],
         ["Lợi nhuận sau thuế", "140,1", "156,8", "11,9%"]]


JA_TITLE = "2025年度 事業報告書"
JA_HEADINGS = ["1. 業績の概要", "2. 今後の取り組み"]
JA_PARAGRAPH = ["当期の売上高は1,234億円となり、前期比12.3%の増加と", "なりました。営業利益は156億円でした。"]
JA_TABLE = [["項目", "2024年度", "2025年度", "増減率"], ["売上高", "1,099", "1,234", "12.3%"], ["営業利益", "140", "156", "11.4%"]]
JA_BULLETS = ["ベトナム市場での販売網を拡大します。", "新工場に85億円を投資します。"]


def font_path() -> str:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    raise SystemExit("no Vietnamese-capable font found (install fonts-dejavu-core)")


def digital_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="vn", fontfile=font_path())
    page.insert_text((60, 70), TITLE, fontsize=17, fontname="vn")
    page.insert_text((60, 110), "1. Kết quả hoạt động", fontsize=13, fontname="vn")
    box = pymupdf.Rect(60, 122, 535, 230)
    page.insert_textbox(box, "\n".join(PARAGRAPHS), fontsize=10.5, fontname="vn")
    y = 245
    for row_index, row in enumerate(TABLE):
        for col, value in enumerate(row):
            cell = pymupdf.Rect(60 + col * 118, y, 60 + (col + 1) * 118, y + 22)
            page.draw_rect(cell, width=0.6)
            page.insert_text((cell.x0 + 5, cell.y0 + 15), value, fontsize=9.5, fontname="vn")
        y += 22
    page.insert_text((60, y + 35), "2. Kế hoạch năm 2026", fontsize=13, fontname="vn")
    for i, bullet in enumerate(BULLETS):
        page.insert_text((70, y + 58 + i * 16), f"- {bullet}", fontsize=10.5, fontname="vn")
    page.insert_text((290, 815), "1", fontsize=9, fontname="vn")
    return doc.tobytes(garbage=3, deflate=True)


def japanese_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    jp = "japan"  # PyMuPDF's built-in CJK font, no system font needed
    page.insert_text((60, 70), JA_TITLE, fontsize=17, fontname=jp)
    page.insert_text((60, 110), JA_HEADINGS[0], fontsize=13, fontname=jp)
    for i, line in enumerate(JA_PARAGRAPH):
        page.insert_text((60, 138 + i * 17), line, fontsize=10.5, fontname=jp)
    y = 185
    for row in JA_TABLE:
        for col, value in enumerate(row):
            cell = pymupdf.Rect(60 + col * 118, y, 60 + (col + 1) * 118, y + 22)
            page.draw_rect(cell, width=0.6)
            page.insert_text((cell.x0 + 5, cell.y0 + 15), value, fontsize=9.5, fontname=jp)
        y += 22
    page.insert_text((60, y + 35), JA_HEADINGS[1], fontsize=13, fontname=jp)
    for i, bullet in enumerate(JA_BULLETS):
        page.insert_text((70, y + 58 + i * 16), f"- {bullet}", fontsize=10.5, fontname=jp)
    page.insert_text((290, 815), "1", fontsize=9, fontname=jp)
    return doc.tobytes(garbage=3, deflate=True)


def bar_chart() -> bytes:
    from PIL import ImageDraw, ImageFont

    font = ImageFont.truetype(font_path(), 22)
    image = Image.new("RGB", (900, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 20), "Doanh thu thuần (tỷ đồng)", font=font, fill="black")
    draw.line((90, 480, 860, 480), fill="black", width=2)
    draw.line((90, 80, 90, 480), fill="black", width=2)
    for i, (year, value) in enumerate([("2022", 905.3), ("2023", 1012.8), ("2024", 1099.2), ("2025", 1234.5)]):
        x = 150 + i * 180
        top = 480 - int(value / 1300 * 380)
        draw.rectangle((x, top, x + 100, 480), fill=(37, 99, 235))
        draw.text((x + 2, top - 32), f"{value:,.1f}".replace(",", " ").replace(".", ",").replace(" ", "."), font=font, fill="black")
        draw.text((x + 22, 490), year, font=font, fill="black")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def photo() -> bytes:
    """A photo-like picture: smooth colour gradients and noise, nothing a text could replace."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:420, 0:640]
    arr = np.stack([(x / 640 * 180 + 40), (y / 420 * 150 + 60), ((x + y) / 1060 * 120 + 80)], axis=-1)
    arr += rng.normal(0, 18, arr.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def pictures_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="vn", fontfile=font_path())
    page.insert_text((60, 70), "Phụ lục: hình ảnh và biểu đồ", fontsize=16, fontname="vn")
    page.insert_image(pymupdf.Rect(60, 100, 535, 395), stream=bar_chart())
    page.insert_text((60, 415), "Hình 1. Doanh thu thuần giai đoạn 2022–2025", fontsize=10, fontname="vn")
    page.insert_image(pymupdf.Rect(60, 450, 420, 686), stream=photo())
    page.insert_text((60, 705), "Hình 2. Toàn cảnh nhà máy mới", fontsize=10, fontname="vn")
    return doc.tobytes(garbage=3, deflate=True)


def scanned_pdf(digital: bytes) -> bytes:
    src = pymupdf.open(stream=digital, filetype="pdf")
    pix = src[0].get_pixmap(dpi=200, alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    image = image.rotate(1.2, resample=Image.Resampling.BICUBIC, fillcolor="white").filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(image, dtype=np.float32) * 0.92 + 12
    arr += np.random.default_rng(0).normal(0, 6, arr.shape)
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    out = pymupdf.open()
    page = out.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buf.getvalue())
    return out.tobytes(garbage=3, deflate=True)


def docx() -> bytes:
    doc = Document()
    doc.add_heading(TITLE, level=0)
    doc.add_heading("1. Kết quả hoạt động", level=1)
    for paragraph in PARAGRAPHS:
        doc.add_paragraph(paragraph)
    table = doc.add_table(rows=len(TABLE) + 1, cols=4)
    table.style = "Table Grid"
    table.cell(0, 0).merge(table.cell(1, 0)).text = "Chỉ tiêu (tỷ đồng)"
    table.cell(0, 1).merge(table.cell(0, 2)).text = "Giá trị"
    table.cell(0, 3).merge(table.cell(1, 3)).text = "Tăng trưởng"
    table.cell(1, 1).text, table.cell(1, 2).text = "Năm 2024", "Năm 2025"
    for r, row in enumerate(TABLE[1:], start=2):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    doc.add_heading("2. Kế hoạch năm 2026", level=1)
    for bullet in BULLETS:
        doc.add_paragraph(bullet, style="List Bullet")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def ground_truth(table: str) -> str:
    bullets = "\n".join(f"- {b}" for b in BULLETS)
    return (
        f"# {TITLE}\n\n## 1. Kết quả hoạt động\n\n" + "\n\n".join(PARAGRAPHS) + f"\n\n{table}\n\n"
        f"## 2. Kế hoạch năm 2026\n\n{bullets}\n"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default="samples")
    parser.add_argument("--product", choices=["vi_en", "vi_en_ja"], default="vi_en",
                        help="vi_en: Vietnamese/English documents; vi_en_ja: plus Japanese documents")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digital = digital_pdf()
    (out / "bao_cao_digital.pdf").write_bytes(digital)
    (out / "bao_cao_scan.pdf").write_bytes(scanned_pdf(digital))
    (out / "bao_cao.docx").write_bytes(docx())
    (out / "hinh_anh.pdf").write_bytes(pictures_pdf())
    pipe_table = "\n".join(
        ["| " + " | ".join(TABLE[0]) + " |", "|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in TABLE[1:]]
    )
    rows = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in TABLE[1:])
    html_table = (
        '<table><tr><th rowspan="2">Chỉ tiêu (tỷ đồng)</th><th colspan="2">Giá trị</th>'
        '<th rowspan="2">Tăng trưởng</th></tr><tr><th>Năm 2024</th><th>Năm 2025</th></tr>' + rows + "</table>"
    )
    for name in ("bao_cao_digital", "bao_cao_scan"):
        (out / f"{name}.gt.md").write_text(ground_truth(pipe_table), encoding="utf-8")
    (out / "bao_cao.gt.md").write_text(ground_truth(html_table), encoding="utf-8")

    if args.product != "vi_en_ja":
        print("\n".join(str(p) for p in sorted(out.iterdir())))
        return
    japanese = japanese_pdf()
    (out / "houkoku_digital.pdf").write_bytes(japanese)
    (out / "houkoku_scan.pdf").write_bytes(scanned_pdf(japanese))
    ja_table = "\n".join(
        ["| " + " | ".join(JA_TABLE[0]) + " |", "|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in JA_TABLE[1:]]
    )
    ja_bullets = "\n".join(f"- {b}" for b in JA_BULLETS)
    ja_gt = (
        f"# {JA_TITLE}\n\n## {JA_HEADINGS[0]}\n\n{''.join(JA_PARAGRAPH)}\n\n{ja_table}\n\n"
        f"## {JA_HEADINGS[1]}\n\n{ja_bullets}\n"
    )
    for name in ("houkoku_digital", "houkoku_scan"):
        (out / f"{name}.gt.md").write_text(ja_gt, encoding="utf-8")
    print("\n".join(str(p) for p in sorted(out.iterdir())))


if __name__ == "__main__":
    main()
