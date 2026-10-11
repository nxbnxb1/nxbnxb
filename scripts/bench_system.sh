#!/usr/bin/env bash
# Score one system of the benchmark (see .github/workflows/_benchmark.yml).
#
#   scripts/bench_system.sh SYSTEM DATASET OUT [docextract bench options]
#     SYSTEM: fine-tuned (the proposed system: docextract + the product's fine-tuned model),
#             stock, ppocrv6, ppstructure, tesseract, text_layer
#
# Env: PRODUCT; the fine-tuned model from scripts/fetch_ocr_model.sh (DOCEXTRACT_OCR_REC_MODEL_DIR,
# OCR_MODEL_TAG), used by "fine-tuned" only, so one job can score several systems in a row.
set -euo pipefail
system="$1" dataset="$2" out="$3"
shift 3
eval "$(python - <<'EOF'
from docextract.products import PRODUCTS
import os
p = PRODUCTS[os.environ["PRODUCT"]]
print(f"BASE='{p.baseline_model}' PADDLE_LANG='{p.paddle_lang}' TESS='{p.tesseract_langs}'")
EOF
)"
fine_tuned="${DOCEXTRACT_OCR_REC_MODEL_DIR:-}"
unset DOCEXTRACT_OCR_REC_MODEL_DIR DOCEXTRACT_OCR_REC_MODEL
case "$system" in
  fine-tuned)
    if [ -z "$fine_tuned" ]; then echo "no fine-tuned model yet"; exit 0; fi
    export DOCEXTRACT_OCR_REC_MODEL_DIR="$fine_tuned"
    args=(--system docextract --name "PROPOSED: docextract + fine-tuned model (${OCR_MODEL_TAG:-?})") ;;
  stock) args=(--system docextract --name "docextract + $BASE (stock, fine-tuning source)") ;;
  ppocrv6)
    export DOCEXTRACT_OCR_REC_MODEL=PP-OCRv6_medium_rec
    args=(--system docextract --name "docextract + PP-OCRv6_medium_rec (stock)") ;;
  ppstructure) args=(--system ppstructure --name "PP-StructureV3 (PaddleOCR, lang=$PADDLE_LANG)") ;;
  tesseract) args=(--system tesseract --name "Tesseract 5 ($TESS, tessdata_best)") ;;
  text_layer) args=(--system text_layer --name "text layer only, no OCR") ;;
  *) echo "unknown system $system" >&2; exit 2 ;;
esac
python -m docextract bench "$dataset" "${args[@]}" "$@" -o "$out"
