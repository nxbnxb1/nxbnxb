#!/usr/bin/env bash
# The evaluate.py --system argument (LABEL=SPEC) of one text-line system of the benchmark.
#
#   scripts/line_system_spec.sh SYSTEM   (fine-tuned, stock, server, ppocrv6, tesseract, easyocr, vietocr)
#
# Env: PRODUCT; DOCEXTRACT_OCR_REC_MODEL_DIR / OCR_MODEL_TAG for "fine-tuned" (scripts/fetch_ocr_model.sh).
# Prints nothing for "fine-tuned" without a fine-tuned model.
set -euo pipefail
eval "$(python - <<'PY'
from docextract.products import PRODUCTS
import os
p = PRODUCTS[os.environ["PRODUCT"]]
print(f"BASE='{p.baseline_model}' TESS='{p.tesseract_langs}' EASY='{'auto' if 'ja' in p.languages else 'vi,en'}'")
PY
)"
case "$1" in
  fine-tuned)
    [ -n "${DOCEXTRACT_OCR_REC_MODEL_DIR:-}" ] || exit 0
    echo "PROPOSED: fine-tuned (${OCR_MODEL_TAG:-?})=paddle:$BASE@$DOCEXTRACT_OCR_REC_MODEL_DIR" ;;
  stock) echo "$BASE (stock, fine-tuning source)=paddle:$BASE" ;;
  server) echo "PP-OCRv5_server_rec (stock)=paddle:PP-OCRv5_server_rec" ;;
  ppocrv6) echo "PP-OCRv6_medium_rec (stock)=paddle:PP-OCRv6_medium_rec" ;;
  tesseract) echo "Tesseract 5 ($TESS, tessdata_best)=tesseract:$TESS" ;;
  easyocr) echo "EasyOCR ($EASY)=easyocr:$EASY" ;;
  vietocr) echo "VietOCR (vgg_transformer)=vietocr:vgg_transformer" ;;
  *) echo "unknown system $1" >&2; exit 2 ;;
esac
