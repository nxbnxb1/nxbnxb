#!/usr/bin/env bash
# Download the Vietnamese OCR model from a GitHub Release (newest vi-ocr-* when no tag is given)
# and export DOCEXTRACT_OCR_REC_MODEL_DIR for the following workflow steps.
set -euo pipefail
tag="${1:-}"
if [ -z "$tag" ]; then
  tag=$(gh release list --limit 50 --json tagName,createdAt \
    --jq '[.[] | select(.tagName | startswith("vi-ocr-"))] | sort_by(.createdAt) | last | .tagName // ""')
fi
if [ -z "$tag" ]; then
  echo "::warning::No vi-ocr-* release yet: using the stock PaddleOCR model, which drops many Vietnamese letters. Run the 'Train Vietnamese OCR' workflow first."
  exit 0
fi
mkdir -p models
gh release download "$tag" -p "vi_en_PP-OCRv5_mobile_rec.tar.gz" -D models --clobber
tar -xzf models/vi_en_PP-OCRv5_mobile_rec.tar.gz -C models
echo "DOCEXTRACT_OCR_REC_MODEL_DIR=$PWD/models/vi_en_PP-OCRv5_mobile_rec" >> "$GITHUB_ENV"
echo "Vietnamese OCR model: $tag" | tee -a "$GITHUB_STEP_SUMMARY"
