# Model nhận dạng PaddleOCR cho tiếng Việt + tiếng Anh

Chạy bằng workflow **Train Vietnamese OCR** (`.github/workflows/train-vi-ocr.yml`) trên GitHub Actions.

## Vì sao cần

Model nhận dạng gốc (`latin_PP-OCRv5_mobile_rec`, `PP-OCRv6_medium_rec`) không có trong từ điển phần lớn chữ thuộc
khối Unicode U+1EA0–U+1EF9 (ạ ả ấ ầ ẩ ẫ ậ … ộ … ỹ). Bộ giải mã CTC không thể sinh ra các chữ đó nên **bỏ mất cả chữ**:
"Cộng hòa xã hội chủ nghĩa Việt Nam" → "Cng hòa xã hi ch nghĩa Vit Nam", độ tin cậy vẫn ~0.98.

## Các bước (`run_training.sh`)

| Bước | File | Nội dung |
|---|---|---|
| Bộ ký tự | `charset.py` | 281 ký tự: số, chữ & dấu câu ASCII, 134 chữ tiếng Việt (NFC, hoa/thường), ký hiệu ₫ € ° ± × ≤ ≥ … |
| Khởi tạo trọng số | `init_weights.py` | Từ `latin_PP-OCRv5_mobile_rec` pretrained: chép hàng của ký tự có sẵn; chữ mới lấy từ chữ gần nhất (`ộ ← ô ← o`) cho lớp CTC và NRTR |
| Dữ liệu tổng hợp | `synth.py` | Từ vựng Việt/Anh theo tần suất (wordfreq), số tiền/ngày tháng/số hiệu văn bản, âm tiết phủ chữ hiếm, chữ hoa; mọi font có đủ chữ Việt và **hiển thị đúng dấu** (font lỗi bị loại); làm xấu như ảnh scan |
| Huấn luyện | `vi_en_PP-OCRv5_mobile_rec.yml` | PP-OCRv5 mobile (PPLCNetV3 + SVTR, CTC + NRTR), CPU, giới hạn thời gian; không dùng random crop (cắt mất dấu) |
| Export | `tools/export_model.py` của PaddleOCR | Định dạng PaddleOCR 3.x (`inference.json/.pdiparams/.yml`) |
| Đánh giá | `evaluate.py` | CER, độ chính xác dòng, tỷ lệ giữ chữ tiếng Việt; so với model gốc |

Kết quả: Release `vi-ocr-<số>` gồm `vi_en_PP-OCRv5_mobile_rec.tar.gz` (model), `checkpoint.tar.gz` (để huấn luyện
tiếp với tham số `resume_tag`), `report.md`.

## Tham số workflow

- `samples`: số dòng tổng hợp (mặc định 60 000).
- `epochs`: số epoch (mặc định 4).
- `time_budget_minutes`: dừng huấn luyện sau N phút (job GitHub tối đa 6 giờ); checkpoint tốt nhất vẫn được export.
- `resume_tag`: huấn luyện tiếp từ checkpoint của một Release trước.

Muốn chính xác hơn trên tài liệu thật: thêm dòng chữ cắt từ tài liệu thật (định dạng `đường_dẫn<TAB>nhãn`) vào
`labels.txt` trước bước huấn luyện, hoặc tăng `samples`/`epochs` và chạy tiếp với `resume_tag`.
