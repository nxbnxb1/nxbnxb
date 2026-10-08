#!/usr/bin/env bash
# Download the fine-tuned recognition model of a product from the GitHub Releases and export
# DOCEXTRACT_OCR_REC_MODEL_DIR for the following workflow steps.
#
#   scripts/fetch_ocr_model.sh <product> [tag]
#     product: vi_en (Việt + Anh) or vi_en_ja (Việt + Anh + Nhật)
#     tag:     release to use; empty = newest release that contains this product's model
# Without a release the product runs on its baseline (stock PaddleOCR model).
set -euo pipefail
product="${1:?product: vi_en or vi_en_ja}"
tag="${2:-}"
asset="${product}_PP-OCRv5_mobile_rec.tar.gz"
if [ -z "$tag" ]; then
  for candidate in $(gh release list --limit 50 --json tagName,createdAt --jq 'sort_by(.createdAt) | reverse | .[].tagName'); do
    if gh release view "$candidate" --json assets --jq '.assets[].name' | grep -qx "$asset"; then
      tag="$candidate"
      break
    fi
  done
fi
if [ -z "$tag" ]; then
  echo "::warning::No release with $asset yet: product $product runs on its baseline (stock PaddleOCR model). Run its 'Train OCR' workflow first."
  exit 0
fi
mkdir -p models
gh release download "$tag" -p "$asset" -D models --clobber
tar -xzf "models/$asset" -C models
echo "DOCEXTRACT_OCR_REC_MODEL_DIR=$PWD/models/${product}_PP-OCRv5_mobile_rec" >> "$GITHUB_ENV"
echo "OCR model of $product: release $tag" | tee -a "$GITHUB_STEP_SUMMARY"
