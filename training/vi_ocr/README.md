# Fine-tune model nhận dạng PaddleOCR cho hai sản phẩm

Toàn bộ chạy trên GitHub Actions (runner CPU miễn phí), hai sản phẩm triển khai giống hệt nhau:

| Sản phẩm | Workflow | Model gốc (baseline) | Từ điển | Release |
|---|---|---|---|---|
| Việt + Anh (`vi_en`) | *Việt + Anh · Train OCR model* | `latin_PP-OCRv5_mobile_rec` | 281 ký tự | `vi_en-ocr-<số>` |
| Việt + Anh + Nhật (`vi_en_ja`) | *Việt + Anh + Nhật · Train OCR model* | `PP-OCRv5_mobile_rec` | 6.973 ký tự | `vi_en_ja-ocr-<số>` |

## Vì sao cần

Model gốc của PaddleOCR thiếu trong từ điển phần lớn chữ tiếng Việt có dấu chồng (ạ ả ấ ầ … ộ … ỹ). Bộ giải mã CTC không
thể sinh ra các chữ đó nên **bỏ mất cả chữ** mà độ tin cậy vẫn cao: "Số: 4770/BC-UBND" → "S: 4770/BC-UBND",
"Cộng hòa" → "Cng hòa".

## Các bước

| Bước | File | Nội dung |
|---|---|---|
| Bộ ký tự | `charset.py` | `vi_en`: số, chữ & dấu câu ASCII, 134 chữ Việt, ký hiệu. `vi_en_ja`: các ký tự của từ điển PP-OCRv5 thuộc bộ chữ chuẩn tiếng Nhật (CP932: kana, 6.221 kanji, ký tự toàn độ rộng) + chữ Việt; bỏ chữ Hán chỉ dùng trong tiếng Trung (18.383 → 6.973 lớp) |
| Khởi tạo | `init_weights.py` | Chép trọng số của ký tự có sẵn; chữ mới lấy từ chữ gần nhất (`ộ ← ô ← o`) |
| Dữ liệu | `synth.py` | Dòng chữ tổng hợp theo tần suất từ vựng Việt/Anh/Nhật, số liệu, ngày tháng, số hiệu văn bản; mọi font đủ ký tự và **vẽ đúng dấu**; làm xấu như ảnh scan. Mỗi runner, mỗi vòng sinh dòng mới (không lặp lại) |
| Huấn luyện | `run_training.sh`, `train_cpu.py` | Nhiều runner cùng lúc, theo vòng (xem dưới) |
| Export | `train_cpu.py export` | Định dạng PaddleOCR 3.x (`inference.json/.pdiparams/.yml`), cùng kiến trúc với model gốc |
| Đánh giá | `evaluate.py` | CER, độ chính xác dòng, tỷ lệ giữ chữ có dấu, ms/dòng — **baseline và model mới trên cùng runner, cùng dữ liệu** |

## Huấn luyện trên nhiều runner (local SGD)

Một runner GitHub có 4 nhân CPU, không GPU. Mỗi **vòng**: N runner bắt đầu từ cùng trọng số, mỗi runner huấn luyện trên
dòng tổng hợp riêng trong `round_minutes` phút, rồi trọng số được **lấy trung bình**; vòng sau bắt đầu từ trung bình đó.
Sau mỗi vòng có kiểm tra trên dòng validation (trong *Summary* của workflow). Vòng cuối được export, đánh giá với baseline
và xuất bản Release. Mặc định: 8 runner × 4 vòng × 50 phút. Learning rate không đổi trong vòng, giảm theo cosine qua các
vòng. Runner nào lỗi chỉ làm bớt một phần của trung bình.

Tham số workflow: `runners`, `rounds` (1–6), `round_minutes`, `lines` (dòng mới mỗi runner mỗi vòng), `resume_tag`
(huấn luyện tiếp từ trọng số của một Release), `publish`.

## Tốc độ trên CPU (đo bằng workflow *OCR training · speed probe*)

Mẫu/giây trên runner 4 nhân, lô 32:

| Cấu hình | vi_en | vi_en_ja |
|---|---|---|
| Như PaddleOCR huấn luyện (nhánh NRTR + backbone nhiều nhánh) | 1,0 | 0,8 |
| Gộp nhánh backbone thành 1 conv như lúc suy luận (`FUSE=1`) | 2,4 | 1,6 |
| Gộp + chỉ nhánh CTC (`GTC=0`) — **mặc định** | **8,8** | **7,1** |
| Gộp + chỉ CTC + đóng băng tầng 1–4 | 10,1 | 5,9 |

Nhánh NRTR chỉ dùng khi huấn luyện (model suy luận chỉ có nhánh CTC) nhưng chiếm phần lớn thời gian trên CPU. Đóng băng
tầng đầu lợi không đáng kể nên toàn bộ backbone vẫn được huấn luyện. oneDNN và static graph không cải thiện / không chạy được.

## Thử nhanh

Bản thử 2 runner × 2 vòng × 8 phút (sản phẩm `vi_en`), trên dòng đánh giá tiếng Việt: CER 12,75% → 9,33%, độ chính xác
dòng 15,7% → 27,8%, tỷ lệ giữ chữ có dấu 24% → 43%; tiếng Anh giữ nguyên (1,88% → 1,77%); tốc độ như nhau (~180 ms/dòng).
