# docextract — Trích xuất tài liệu bằng OCR + VLM (tiếng Việt & tiếng Anh)

Chuyển **PDF, Word (.docx/.doc) và ảnh scan** thành **Markdown + JSON** có cấu trúc, giữ nguyên nội dung,
số liệu, bảng biểu và thứ tự đọc, kèm metadata để truy vết về tài liệu gốc. Phục vụ AI / RAG / LLM.

Mọi thứ chạy trên **GitHub Actions** — không cần cài đặt gì trên máy cá nhân.

## Cách dùng (trên GitHub)

| Việc cần làm | Cách làm |
|---|---|
| Trích xuất tài liệu | Đẩy file vào thư mục `inputs/` → workflow **Extract documents** tự chạy, sinh `outputs/<tên>.md` và `outputs/<tên>.json`, commit lại vào nhánh và đính kèm artifact. Hoặc vào tab *Actions → Extract documents → Run workflow* (chọn file/thư mục, trang). |
| Huấn luyện model OCR tiếng Việt | *Actions → Train Vietnamese OCR → Run workflow*. Model được kiểm thử rồi xuất bản thành Release `vi-ocr-<số>`; workflow Extract tự tải bản mới nhất. |
| Kiểm thử mã nguồn | Workflow **CI** chạy `pytest` ở mỗi lần push / pull request. |

Cấu hình tùy chọn (*Settings → Secrets and variables → Actions*):

- `DOCEXTRACT_VLM_BASE_URL`, `DOCEXTRACT_VLM_API_KEY` (secrets) và `DOCEXTRACT_VLM_MODEL` (variable):
  endpoint tương thích OpenAI của một VLM (ví dụ Qwen-VL qua vLLM hoặc DashScope). VLM chỉ dùng cho biểu đồ,
  hình ảnh và làm phương án dự phòng cho bảng/công thức. Không cấu hình VLM thì mọi phần khác vẫn chạy.

## Kiến trúc

```text
PDF / Word / Image
      │
      ▼
Tiền xử lý ── PDF có text: PyMuPDF đọc trực tiếp (không OCR lại)
      │      PDF scan / ảnh: xoay trang (PP-LCNet doc_ori), chống nghiêng, chuẩn hóa độ phân giải, tăng tương phản
      │      Word: python-docx đọc cấu trúc gốc (tiêu đề, danh sách, bảng có ô gộp, ảnh)
      ▼
Layout ────── PP-StructureV3: chỉ dùng module Layout Detection (PP-DocLayout_plus-L)
      │      PDF có text khi chưa cài PaddleOCR: layout heuristic từ chính file PDF
      │      Thứ tự đọc: tách cột + khối tràn cột
      ▼
Router ────── chọn chuỗi phương pháp cho từng vùng (theo loại vùng, độ phức tạp, chất lượng text layer)
      ▼
Thực thi theo vòng: gom vùng theo phương pháp → batch → kiểm tra → vùng chưa đạt chuyển phương pháp kế tiếp
      ▼
Hợp nhất theo thứ tự đọc → Markdown + JSON (metadata, trạng thái kiểm chứng, tham chiếu nguồn)
```

### Router (giai đoạn 1: luật)

| Loại vùng | Chuỗi phương pháp (thử lần lượt đến khi đạt kiểm tra) |
|---|---|
| Văn bản, tiêu đề, danh sách, chú thích, header/footer, con dấu | PDF text layer → PaddleOCR. **Không bao giờ dùng VLM cho văn bản.** |
| Bảng | PyMuPDF (bảng vector) → nhận dạng bảng PaddleOCR → VLM → giữ text (đánh dấu cần kiểm tra). Bảng lớn/phức tạp: VLM trước nhận dạng bảng |
| Công thức | Formula Recognition (PP-FormulaNet_plus-M) → VLM → giữ text |
| Biểu đồ | VLM (mô tả + bảng số liệu JSON) → chỉ lấy chữ trong biểu đồ |
| Hình ảnh | VLM (mô tả) → OCR chữ trong ảnh; ảnh nhỏ (logo/icon) bỏ qua |

Mỗi quyết định và kết quả có thể ghi ra JSONL (`DOCEXTRACT_ROUTE_LOG_PATH`) làm dữ liệu huấn luyện
Neural Router ở giai đoạn 3.

### Kiểm tra & hợp nhất

- **Văn bản OCR:** độ tin cậy trung bình, tỷ lệ dòng tin cậy thấp, ký tự lỗi, và **phát hiện mất chữ tiếng Việt**
  (từ không có nguyên âm như "Cng", "Vit": dấu hiệu model OCR không có ký tự ộ, ệ, ủ…).
- **Text layer PDF:** phát hiện font cũ TCVN3/VNI (`Céng hßa x· héi`), `(cid:..)`, ký tự vùng riêng → chuyển sang OCR.
- **Bảng:** số cột mỗi hàng nhất quán (tính cả rowspan/colspan), tỷ lệ ô trống, độ tin cậy;
  **đối chiếu số liệu và ký tự đặc biệt** với text layer PDF hoặc token OCR tin cậy cao.
- **Công thức:** cân ngoặc, `\left/\right`, `\begin/\end`, lặp token.
- Vùng không đạt sau mọi phương pháp: giữ kết quả tốt nhất, `status = needs_review`, ghi lý do,
  và chèn `<!-- cần kiểm tra: ... -->` vào Markdown.

### Model OCR tiếng Việt

Model nhận dạng gốc của PaddleOCR (`latin_PP-OCRv5_mobile_rec`, `PP-OCRv6_medium_rec`) **thiếu hầu hết chữ
tiếng Việt có dấu chồng** (ộ, ủ, ệ, ạ, ỹ…): "Cộng hòa xã hội chủ nghĩa" bị đọc thành "Cng hòa xã hi ch nghĩa"
mà độ tin cậy vẫn ~0.98. Thư mục [`training/vi_ocr`](training/vi_ocr) cải thiện chính PaddleOCR:

1. Bộ ký tự Việt + Anh (281 ký tự: ASCII, toàn bộ 134 chữ tiếng Việt có dấu, ký hiệu ₫ € % ° ± ≤ ≥ …).
2. Chuyển trọng số PP-OCRv5: giữ nguyên backbone, mở rộng lớp đầu ra; mỗi chữ mới khởi tạo từ chữ gần nhất
   (`ộ ← ô ← o`) nên chỉ cần học thêm dấu.
3. Sinh dữ liệu tổng hợp: từ vựng tiếng Việt/Anh theo tần suất (wordfreq), mẫu văn bản hành chính – tài chính
   (số tiền, ngày tháng, số hiệu văn bản), chế độ phủ chữ hiếm; nhiều font (đã loại font hiển thị thiếu dấu);
   làm xấu như ảnh scan (mờ, nhiễu, JPEG, nghiêng, độ phân giải thấp).
4. Fine-tune (CTC + NRTR), export sang định dạng PaddleOCR 3.x, đánh giá CER so với model gốc, xuất bản Release.

## Đầu ra

`outputs/<tên>.md`: Markdown theo thứ tự đọc (tiêu đề `#`, bảng Markdown hoặc HTML khi có ô gộp, công thức `$$…$$`,
mô tả hình/biểu đồ, `<!-- trang: N -->`).

`outputs/<tên>.json` (rút gọn):

```json
{
  "source": {"filename": "baocao.pdf", "sha256": "…", "format": "pdf", "page_count": 12},
  "pages": [{
    "number": 1, "kind": "digital", "width": 595.0, "height": 842.0, "unit": "pt",
    "layout_backend": "paddle:PP-DocLayout_plus-L",
    "regions": [{
      "id": "p1-r4", "type": "table", "order": 3,
      "bbox": {"x0": 72.0, "y0": 220.0, "x1": 432.0, "y1": 308.0},
      "content": "| Chỉ tiêu | 2024 | 2025 |…", "html": "<table>…</table>",
      "method": "pdf_table", "confidence": null, "status": "passed", "issues": [],
      "attempts": [{"method": "pdf_table", "passed": true, "score": 1.0, "duration_ms": 3.1}],
      "source": {"file": "baocao.pdf", "page": 1, "bbox": {"x0": 72.0, "y0": 220.0, "x1": 432.0, "y1": 308.0}}
    }]
  }],
  "stats": {"ms_per_page": 850.0, "vlm_calls": 2, "cost_per_page": 0.0004, "escalations": 1,
            "regions_by_method": {"pdf_text": 30, "pdf_table": 2, "vlm": 1}, "regions_by_status": {"passed": 33}}
}
```

`type`: `title, heading, text, list, table, formula, image, chart, caption, header, footer, page_number, footnote, seal, code`.
`method`: `pdf_text, pdf_table, docx, ocr, table_recognition, formula_recognition, vlm`.
`status`: `passed` (đã kiểm tra), `unchecked` (không có cách kiểm tra tự động, ví dụ mô tả ảnh), `needs_review`, `skipped`.
Tọa độ: điểm PDF (pt) với PDF có text; với PDF scan là điểm trên trang đã chuẩn hóa (xoay/chống nghiêng ghi trong
`preprocessing`); với ảnh là pixel; Word dùng `source.locator` (ví dụ `body/tbl[2]`).

## Cấu hình

Mọi tham số trong [`docextract/config.py`](docextract/config.py) đặt được bằng biến môi trường `DOCEXTRACT_<TÊN>`
(trong workflow: mục `env`). Hay dùng:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DOCEXTRACT_OCR_REC_MODEL_DIR` | (Release mới nhất) | Model nhận dạng tiếng Việt đã fine-tune |
| `DOCEXTRACT_OCR_LANG` | `vi` | `vi` hoặc `en` |
| `DOCEXTRACT_LAYOUT_BACKEND` | `auto` | `paddle`, `heuristic` (nhanh, chỉ PDF có text) |
| `DOCEXTRACT_DPI` | `200` | Độ phân giải render trang PDF |
| `DOCEXTRACT_TABLE_FORMAT` | `auto` | `markdown`, `html`, `auto` (HTML khi có ô gộp) |
| `DOCEXTRACT_OCR_MIN_CONFIDENCE` | `0.85` | Ngưỡng tin cậy OCR |
| `DOCEXTRACT_CACHE_DIR` | — | Cache kết quả theo nội dung vùng cắt (SQLite) |
| `DOCEXTRACT_ROUTE_LOG_PATH` | — | Log quyết định của Router (JSONL) |
| `DOCEXTRACT_OUTPUT_LOCALE` | `vi` | Nhãn trong Markdown (`vi`/`en`) |

## Thành phần khác

- **API FastAPI** (`docextract/api.py`): `POST /v1/extract`, `POST /v1/jobs`, `GET /v1/jobs/{id}`,
  `GET /v1/jobs/{id}/markdown`, `GET /health` — dùng khi muốn triển khai thành dịch vụ (`docextract serve`).
- **Benchmark giai đoạn 2** (`docextract bench <thư mục>`): mỗi tài liệu kèm `<tên>.gt.md`; tính CER, WER,
  TEDS/TEDS-S cho bảng, F1 tiêu đề, ms/trang, số lần gọi VLM/trang, chi phí/trang, tỷ lệ vùng cần kiểm tra.
- **Tối ưu:** không OCR lại text đã có trong PDF; chỉ crop vùng cần xử lý; batch theo phương pháp; các nhóm phương
  pháp chạy song song; gọi VLM đồng thời; cache theo pixel của vùng cắt.

## Lộ trình

- [x] Giai đoạn 1: Parser + OCR + Layout Detection + Router luật + kiểm tra/hợp nhất + VLM cho nội dung hình ảnh.
- [x] Model OCR tiếng Việt + Anh (workflow huấn luyện).
- [ ] Giai đoạn 2: benchmark trên tài liệu thực tế (`docextract bench`), tinh chỉnh ngưỡng Router.
- [ ] Giai đoạn 3: Neural Router học từ log `route_log.jsonl`.

Tài liệu thiết kế gốc: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
