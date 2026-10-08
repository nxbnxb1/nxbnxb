# HƯỚNG DẪN XÂY DỰNG HỆ THỐNG TRÍCH XUẤT TÀI LIỆU BẰNG OCR + VLM

> Tài liệu thiết kế gốc. Các quyết định bổ sung khi triển khai:
>
> - Văn bản **bắt buộc** lấy từ text layer PDF hoặc OCR; **không dùng VLM cho văn bản**. VLM chỉ dùng cho biểu đồ,
>   hình ảnh và làm dự phòng cấu trúc cho bảng/công thức.
> - Chỉ phục vụ **tiếng Việt và tiếng Anh**. Model nhận dạng gốc của PaddleOCR thiếu chữ tiếng Việt có dấu chồng,
>   nên dự án fine-tune PP-OCRv5 với bộ ký tự Việt + Anh (`training/vi_ocr`).
> - Toàn bộ chạy trên **GitHub Actions** (CI, huấn luyện model OCR, trích xuất tài liệu).

## 1. Mục tiêu

Xây dựng hệ thống tự động chuyển đổi tài liệu PDF, Word, ảnh scan thành dữ liệu có cấu trúc (Markdown/JSON), giúp AI dễ dàng đọc hiểu, tìm kiếm và phân tích.

**Yêu cầu chính:**
- **Độ chính xác cao:** Giữ nguyên nội dung, số liệu, bảng biểu và cấu trúc tài liệu.
- **Tốc độ nhanh:** Ưu tiên thuật toán nhẹ, chỉ sử dụng VLM khi cần thiết.
- **Tự động hóa:** Hệ thống tự nhận diện nội dung và lựa chọn phương pháp xử lý phù hợp.

## 2. Kiến trúc hệ thống

```text
         PDF / Word / Image
                 |
                 v
       Document Preprocessing
       (PDF Parser, Image Cleanup)
                 |
                 v
       PP-StructureV3 Layout
       (Phân loại vùng tài liệu)
                 |
                 v
          Intelligent Router
                 |
       +---------+----------+
       |         |          |
       v         v          v
    OCR/Text   Table/     VLM
    Parser     Formula    (Complex)
       |         |          |
       +---------+----------+
                 |
                 v
       Validation & Merging
                 |
                 v
         Markdown + JSON
                 |
                 v
           AI / RAG / LLM
```

## 3. Quy trình xử lý

**Bước 1 — Tiền xử lý tài liệu**

- PDF có text: trích xuất trực tiếp bằng PyMuPDF.
- PDF scan hoặc ảnh: chuẩn hóa độ phân giải, xoay trang, cải thiện chất lượng ảnh.
- Word: đọc nội dung và cấu trúc trực tiếp.

**Bước 2 — Nhận diện bố cục**

Sử dụng module Layout Detection thuộc PaddleOCR PP-StructureV3 để xác định các vùng: văn bản, tiêu đề, bảng, hình ảnh, biểu đồ, công thức và thứ tự đọc.

Lưu ý: PP-StructureV3 là pipeline gồm nhiều module; trong kiến trúc này chỉ sử dụng module cần thiết để tránh xử lý lặp.

**Bước 3 — Tự động lựa chọn thuật toán**

| Loại nội dung | Thuật toán ưu tiên |
|---|---|
| Văn bản, tiêu đề | PDF Parser / PaddleOCR |
| Bảng đơn giản | Table Recognition |
| Công thức toán học | Formula Recognition |
| Biểu đồ, sơ đồ | VLM |
| Hình ảnh cần mô tả | VLM |
| Bảng phức tạp, OCR lỗi | VLM dự phòng |

Router sử dụng loại vùng, độ phức tạp và điểm tin cậy để lựa chọn thuật toán. Giai đoạn đầu sử dụng quy tắc (*rule-based*), chưa cần huấn luyện Neural Network riêng.

**Bước 4 — Kiểm tra và hợp nhất**

- Kiểm tra kết quả với tài liệu gốc.
- Đối chiếu số liệu, ký tự đặc biệt và cấu trúc bảng.
- Chuyển sang VLM khi phương pháp ban đầu không đạt yêu cầu.
- Đánh dấu các vùng không chắc chắn để kiểm tra thủ công.
- Ghép kết quả theo đúng thứ tự đọc.

**Bước 5 — Chuẩn hóa đầu ra**

Xuất tài liệu thành Markdown và JSON, kèm metadata:

- Nội dung văn bản, bảng, công thức và mô tả hình ảnh.
- Số trang, tọa độ vùng, loại nội dung.
- Phương pháp trích xuất, trạng thái kiểm chứng.
- Tham chiếu đến tài liệu gốc để truy vết.

## 4. Công nghệ đề xuất

| Thành phần | Công nghệ |
|---|---|
| Document Parser | PyMuPDF, python-docx |
| Layout Detection | PaddleOCR PP-StructureV3 |
| OCR | PP-OCRv5 |
| Table Recognition | Module nhận dạng bảng của PaddleOCR |
| Formula Recognition | Module công thức của PaddleOCR |
| VLM | Qwen-VL hoặc VLM chuyên dụng cho tài liệu |
| Backend | Python, FastAPI |
| Output | Markdown, JSON |

## 5. Chiến lược tối ưu

**Nguyên tắc: OCR/Parser xử lý phần dễ, VLM xử lý phần khó.**

1. Không OCR lại văn bản đã trích xuất chính xác từ PDF gốc.
2. Chỉ gọi VLM cho những vùng thực sự cần hiểu hình ảnh hoặc cấu trúc phức tạp.
3. Crop vùng cần xử lý thay vì gửi toàn bộ trang.
4. Xử lý song song, batching và cache kết quả.
5. Đánh giá bằng Character Error Rate (CER), độ chính xác bảng, chất lượng cấu trúc, thời gian/trang và chi phí/trang.

## 6. Lộ trình triển khai

**Giai đoạn 1:** Xây dựng pipeline Parser + OCR + Layout Detection + VLM fallback.

**Giai đoạn 2:** Benchmark trên tài liệu thực tế, tối ưu Router và kiểm tra chất lượng.

**Giai đoạn 3:** Khi có đủ dữ liệu, huấn luyện Neural Router để tự chọn phương pháp có chi phí thấp nhất nhưng vẫn đạt yêu cầu độ chính xác.

**Kết quả mong muốn:** Một hệ thống trích xuất tài liệu thông minh, tự động lựa chọn thuật toán theo từng vùng nội dung, tối ưu tốc độ xử lý và tạo dữ liệu có cấu trúc phục vụ AI/RAG/LLM.
