# Kết quả docextract 1.0.0

Hai sản phẩm **độc lập**. Mỗi sản phẩm chỉ chạy bằng model OCR của chính nó, và được so với cùng một bộ hệ thống khác
trên cùng tài liệu. Mọi số đo đều chạy trên GitHub Actions (máy 4 nhân CPU, không GPU), không dùng VLM.

| Sản phẩm | Ngôn ngữ | Model OCR (Release) | Nên dùng cho |
|---|---|---|---|
| **Việt + Anh** (`vi_en`) | Việt, Anh | `vi_en_PP-OCRv5_mobile_rec` (`vi_en-ocr-7`) | tài liệu chỉ có tiếng Việt và/hoặc tiếng Anh |
| **Việt + Anh + Nhật** (`vi_en_ja`) | Việt, Anh, Nhật | `vi_en_ja_PP-OCRv5_mobile_rec` (`vi_en_ja-ocr-4`) | tài liệu có tiếng Nhật |

## Các hệ thống được so

| Tên trong bảng | Là gì |
|---|---|
| **docextract + model fine-tune (sản phẩm)** | công cụ này, với model OCR đã huấn luyện thêm |
| docextract + model gốc | cùng công cụ, model OCR gốc của PaddleOCR (điểm xuất phát của việc huấn luyện): `latin_PP-OCRv5_mobile_rec` / `PP-OCRv5_mobile_rec` |
| docextract + PP-OCRv6_medium_rec | cùng công cụ, model OCR mới nhất và lớn hơn của PaddleOCR |
| PP-StructureV3 | công cụ đọc tài liệu chính thức của PaddleOCR (`lang=vi` / `lang=japan`) |
| Tesseract 5 | OCR mã nguồn mở phổ biến nhất (`tessdata_best`, `vie+eng` / `vie+eng+jpn`) |
| Chỉ lấy chữ có sẵn trong PDF | không OCR: mốc thấp nhất (scan không có chữ nào) |

Bảng đọc từng dòng chữ có thêm PP-OCRv5_server_rec, VietOCR và EasyOCR.

## Cách đọc số

- **Độ đúng** (F1 theo từ): % số từ đọc đúng so với văn bản chuẩn; cao là tốt. 100% là hoàn hảo.
- **Nhận đúng tiêu đề** (F1 tiêu đề): % tiêu đề được nhận ra đúng chữ và đúng là tiêu đề. Tesseract và
  "chỉ lấy chữ có sẵn" không đánh dấu tiêu đề.
- **Ký tự sai** (CER): % số ký tự đọc sai trên dòng chữ cắt từ tài liệu thật của các công ty dùng để test; thấp là tốt.
- **Giây/trang**, **thời gian/dòng**: mọi hệ thống chạy trên cùng một máy, cùng tài liệu, model đã nạp sẵn.
  Đây là mẫu cố định nhỏ, và máy GitHub mỗi lần có thể khác nhau, nên chỉ nên so các hệ thống với nhau trong cùng bảng.
- **Tài liệu kiểm tra:** báo cáo tài chính, báo cáo thường niên, nghị quyết… của các công ty không dùng để
  huấn luyện (PDF số, và bản scan của chính chúng), trang Wikipedia (PDF, Word, scan) và trang của bộ chuẩn
  OmniDocBench. Số in đậm là tốt nhất trong cột.

## Việt + Anh (`vi_en`)

### Đọc tài liệu

| Hệ thống | Độ đúng: tất cả (131 tài liệu) | Scan tiếng Việt | Scan tiếng Anh | PDF số tiếng Việt | Nhận đúng tiêu đề | Giây/trang: PDF số | Giây/trang: scan |
|---|---|---|---|---|---|---|---|
| **docextract + model fine-tune (sản phẩm)** | **94,0%** | **88,9%** | 93,9% | 92,9% | **83,0%** | 8,7 s | 26,4 s |
| docextract + model gốc | 87,9% | 52,4% | 94,5% | 93,0% | 78,0% | 8,8 s | 21,9 s |
| docextract + PP-OCRv6_medium_rec | 87,9% | 51,7% | 93,7% | 93,0% | 77,0% | 10,3 s | 51,3 s |
| PP-StructureV3 | 78,0% | 49,3% | 91,3% | 52,0% | 55,0% | 30,6 s | 18,8 s |
| Tesseract 5 | 90,7% | 87,9% | **96,4%** | 88,5% | – | 2,6 s | 5,8 s |
| Chỉ lấy chữ có sẵn trong PDF | 44,1% | 0,0% | 0,0% | **98,8%** | – | **0,0 s** | **0,0 s** |

### Đọc từng dòng chữ

| Hệ thống | Tiếng Việt | Tiếng Anh | Dòng dài | Thời gian/dòng |
|---|---|---|---|---|
| **Model fine-tune (sản phẩm)** | 4,0% | 3,4% | 6,3% | 32,1 ms |
| Model gốc | 15,1% | 4,8% | 14,7% | **31,1 ms** |
| PP-OCRv6_medium_rec | 13,2% | **3,2%** | 9,2% | 161,7 ms |
| PP-OCRv5_server_rec | 19,7% | 4,9% | 43,5% | 135,9 ms |
| Tesseract 5 | 6,7% | 6,6% | 6,5% | 227,8 ms |
| VietOCR | **3,9%** | 5,7% | **6,0%** | 385,5 ms |
| EasyOCR | 8,5% | 8,6% | 9,1% | 39,2 ms |

## Việt + Anh + Nhật (`vi_en_ja`)

### Đọc tài liệu

| Hệ thống | Độ đúng: tất cả (182 tài liệu) | Scan tiếng Việt | Scan tiếng Anh | Scan tiếng Nhật | PDF số tiếng Nhật | Nhận đúng tiêu đề | Giây/trang: PDF số | Giây/trang: scan |
|---|---|---|---|---|---|---|---|---|
| **docextract + model fine-tune (sản phẩm)** | **92,5%** | 80,9% | 85,2% | 93,4% | 98,3% | 74,0% | 6,6 s | 24,0 s |
| docextract + model gốc | 89,0% | 41,2% | 92,5% | 94,6% | 98,1% | **83,0%** | 6,6 s | 23,0 s |
| docextract + PP-OCRv6_medium_rec | 90,7% | 51,7% | 93,7% | **95,8%** | 98,1% | 82,0% | 7,3 s | 46,9 s |
| PP-StructureV3 | 72,7% | 36,7% | 73,4% | 93,0% | 94,7% | 53,0% | 30,6 s | 27,5 s |
| Tesseract 5 | 87,6% | **87,7%** | **95,8%** | 89,4% | 91,7% | – | 4,5 s | 7,7 s |
| Chỉ lấy chữ có sẵn trong PDF | 47,6% | 0,0% | 0,0% | 0,0% | **100,0%** | – | **0,0 s** | **0,0 s** |

### Đọc từng dòng chữ

| Hệ thống | Tiếng Việt | Tiếng Anh | Tiếng Nhật | Dòng dài | Thời gian/dòng |
|---|---|---|---|---|---|
| **Model fine-tune (sản phẩm)** | 5,4% | 5,7% | **9,1%** | **9,4%** | **37,5 ms** |
| Model gốc | 20,2% | 5,2% | 12,9% | 16,9% | 39,9 ms |
| PP-OCRv6_medium_rec | 13,2% | **3,2%** | 11,1% | 12,6% | 169,6 ms |
| PP-OCRv5_server_rec | 19,7% | 4,5% | 10,9% | 38,5% | 148,8 ms |
| Tesseract 5 | 6,6% | 6,8% | 56,4% | 13,8% | 261,4 ms |
| VietOCR | **3,9%** | 6,1% | 90,2% | 18,2% | 299,2 ms |
| EasyOCR | 8,4% | 9,0% | 24,2% | 25,8% | 39,0 ms |

## Giới hạn đã biết

1. **Bản Việt + Anh + Nhật đọc scan tiếng Anh kém hơn model gốc** (85,2% so với 92,5%).
   Bản Việt + Anh đọc scan tiếng Anh 93,9%. Tài liệu chỉ có tiếng Anh hoặc tiếng Việt nên dùng bản **Việt + Anh**.
   Nó cũng nhận đúng tiêu đề ít hơn model gốc (74% so với 83%).
2. **Chậm hơn Tesseract:** khoảng 26 giây cho một trang scan so với 6 giây của Tesseract
   (Việt + Anh), vì docextract còn tìm bố cục, bảng, tiêu đề và đọc từng vùng.
3. **Cấu trúc bảng:** trên các tài liệu có bảng, TEDS của docextract là 0,47 (Việt + Anh) và 0,43 (Việt + Anh + Nhật),
   của PP-StructureV3 là 0,70 và 0,69. Chênh lệch lớn nhất ở bảng trên trang scan.
4. **PDF số:** văn bản chuẩn lấy từ chính lớp chữ của PDF, nên "chỉ lấy chữ có sẵn" gần 100%. Cột này chủ yếu cho thấy
   docextract không làm hỏng chữ có sẵn.
5. **Phần mô tả hình và biểu đồ bằng VLM chưa được chấm** (benchmark chạy không có VLM).
6. **Tập kiểm tra còn nhỏ:** 15 tài liệu doanh nghiệp cho mỗi loại. Chênh lệch vài điểm % giữa hai hệ thống chưa đủ
   để kết luận; chênh lệch lớn (như scan tiếng Việt 89% so với 52%) nhất quán giữa hai sản phẩm và với kết quả từng dòng.

## Nguồn số liệu

Benchmark trên GitHub Actions: tài liệu và tốc độ [Việt + Anh](https://github.com/nxbnxb1/nxbnxb/actions/runs/39008185159), [Việt + Anh + Nhật](https://github.com/nxbnxb1/nxbnxb/actions/runs/39008351908); dòng chữ [Việt + Anh](https://github.com/nxbnxb1/nxbnxb/actions/runs/39016193063), [Việt + Anh + Nhật](https://github.com/nxbnxb1/nxbnxb/actions/runs/39016194824).
