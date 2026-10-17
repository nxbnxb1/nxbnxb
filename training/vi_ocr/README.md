# Fine-tune model nhận dạng PaddleOCR cho hai sản phẩm

Toàn bộ quá trình chạy trên GitHub Actions, dùng runner CPU miễn phí. Hai sản phẩm được triển khai theo cùng một quy
trình:

| Sản phẩm | Workflow | Model gốc (baseline) | Từ điển | Release |
|---|---|---|---|---|
| Việt + Anh (`vi_en`) | *Việt + Anh · Train OCR model* | `latin_PP-OCRv5_mobile_rec` | 281 ký tự | `vi_en-ocr-<số>` |
| Việt + Anh + Nhật (`vi_en_ja`) | *Việt + Anh + Nhật · Train OCR model* | `PP-OCRv5_mobile_rec` | 6.973 ký tự | `vi_en_ja-ocr-<số>` |

## Vấn đề

Từ điển của model gốc PaddleOCR thiếu phần lớn chữ tiếng Việt có dấu chồng (ạ ả ấ ầ … ộ … ỹ). Bộ giải mã CTC không
thể sinh ra các chữ này, nên **bỏ mất cả chữ** trong khi độ tin cậy vẫn cao. Ví dụ: "Số: 4770/BC-UBND" thành
"S: 4770/BC-UBND", "Cộng hòa" thành "Cng hòa".

## Các bước

| Bước | File | Nội dung |
|---|---|---|
| Bộ ký tự | `charset.py` | `vi_en`: chữ số, chữ cái và dấu câu ASCII, 134 chữ Việt, ký hiệu. `vi_en_ja`: các ký tự trong từ điển PP-OCRv5 thuộc bộ chữ chuẩn tiếng Nhật CP932 (kana, 6.221 kanji, ký tự toàn độ rộng), thêm chữ Việt; bỏ chữ Hán chỉ dùng trong tiếng Trung (18.383 → 6.973 lớp) |
| Khởi tạo | `init_weights.py` | Ký tự có sẵn giữ trọng số; chữ mới lấy trọng số của chữ gần nhất (`ộ ← ô ← o`) |
| Dòng tổng hợp | `synth.py` | Dòng chữ theo tần suất từ vựng Việt, Anh, Nhật, kèm số liệu, ngày tháng, số hiệu văn bản. Font được chọn sao cho có đủ ký tự và **vẽ đúng dấu**. Ảnh được làm xấu như ảnh scan. Dữ liệu được sinh mới ở mỗi vòng |
| Dòng thật | `doc_lines.py` | Một nửa số dòng huấn luyện, cắt từ PDF số của các công ty thuộc phần train của kho tài liệu. Nhãn là lớp chữ của PDF. Mỗi ngôn ngữ được lấy một phần bằng nhau |
| Huấn luyện | `run_training.sh`, `train_cpu.py` | Song song trên nhiều runner, theo vòng (xem dưới) |
| Export | `train_cpu.py export` | Định dạng PaddleOCR 3.x (`inference.json/.pdiparams/.yml`), cùng kiến trúc với model gốc |
| Đánh giá | `evaluate.py` | CER, độ chính xác dòng, tỷ lệ giữ chữ có dấu, ms/dòng. Model mới và model gốc chạy trên cùng runner, cùng dữ liệu |
| Cổng chất lượng | `gate.py` | Điều kiện để phát hành Release (xem dưới) |

## Huấn luyện song song (local SGD)

Mỗi vòng diễn ra như sau:

1. N runner bắt đầu từ cùng một bộ trọng số.
2. Mỗi runner huấn luyện trên dữ liệu riêng trong `round_minutes` phút.
3. Trọng số của các runner được lấy trung bình. Vòng sau bắt đầu từ trọng số trung bình này.

Cấu hình mặc định là 8 runner × 4 vòng × 50 phút. Learning rate không đổi trong một vòng và giảm theo hàm cosine
qua các vòng. Tham số `resume_tag` cho phép huấn luyện tiếp từ trọng số của một Release.

Model huấn luyện ở dạng dùng khi suy luận: các nhánh backbone được gộp và chỉ giữ nhánh CTC. Nhánh NRTR chỉ dùng
khi huấn luyện. Bảng dưới là tốc độ huấn luyện, tính bằng mẫu/giây, trên runner 4 nhân với lô 32:

| Cấu hình | vi_en | vi_en_ja |
|---|---|---|
| Cấu hình huấn luyện gốc của PaddleOCR (CTC + NRTR, backbone nhiều nhánh) | 1,0 | 0,8 |
| Gộp nhánh backbone | 2,4 | 1,6 |
| Gộp nhánh backbone, chỉ nhánh CTC (**mặc định**) | **8,8** | **7,1** |

## Đánh giá và cổng chất lượng

Model của vòng cuối được export, rồi đánh giá cùng model gốc trên các bộ dòng sau:

- dòng tổng hợp theo từng ngôn ngữ;
- dòng dài, 30–60 ký tự;
- dòng cắt từ tài liệu thật của các công ty dùng để test: 1.000 dòng cho mỗi ngôn ngữ và 500 dòng dài.

Model chỉ được phát hành thành Release khi qua cổng chất lượng: trên mọi bộ dòng của tài liệu thật, CER không cao
hơn model gốc quá 0,5 điểm %. Nếu không qua, model chỉ được lưu làm artifact. Workflow *Publish OCR model* cho phép
phát hành một model như vậy, kèm lý do được ghi vào Release.

Kết quả đánh giá khi phát hành hai model của v1.0.0 (CER trên dòng của tài liệu thật):

| Release | Bộ dòng | Model gốc | Model fine-tune |
|---|---|---|---|
| `vi_en-ocr-7` | tiếng Việt | 15,25% | **5,30%** |
| | tiếng Anh | 6,71% | **5,74%** |
| | dòng dài | 12,98% | **3,39%** |
| `vi_en_ja-ocr-4` | tiếng Việt | 20,24% | **5,44%** |
| | tiếng Anh | **5,22%** | 5,66% |
| | tiếng Nhật | 13,06% | **9,36%** |
| | dòng dài | 16,91% | **9,41%** |

Kết quả so với các hệ thống OCR khác: [docs/RESULTS.md](../../docs/RESULTS.md).
