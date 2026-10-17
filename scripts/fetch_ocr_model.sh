#!/usr/bin/env bash
# Download the fine-tuned recognition model of a product from the GitHub Releases and export
# DOCEXTRACT_OCR_REC_MODEL_DIR / OCR_MODEL_TAG for the following workflow steps.
#
#   scripts/fetch_ocr_model.sh <product> [tag]
#     product: vi_en (Việt + Anh) or vi_en_ja (Việt + Anh + Nhật)
#     tag:     release to use; empty = the release the product is pinned to (release in
#              docextract/products.py), "latest" = newest release that contains the product's model
# A product with a latin_product (vi_en_ja) also gets that product's pinned model, exported as
# DOCEXTRACT_OCR_LATIN_REC_MODEL_DIR / OCR_LATIN_MODEL_TAG: lines without kana/kanji are read with it.
# Without a release the product runs on its baseline (stock PaddleOCR model).
set -euo pipefail
product="${1:?product: vi_en or vi_en_ja}"
tag="${2:-}"

info() {  # pinned release, model name and latin product of a product
  python - "$1" <<'EOF'
import sys
from docextract.products import PRODUCTS
p = PRODUCTS[sys.argv[1]]
print(p.release, p.model, p.latin_product or "")
EOF
}

fetch() {  # product tag var_dir var_tag
  local product="$1" tag="$2" release model latin asset
  read -r release model latin < <(info "$product")
  asset="$model.tar.gz"
  [ -n "$tag" ] || tag="$release"
  if [ "$tag" = latest ]; then
    tag=""
    for candidate in $(gh release list --limit 50 --json tagName,createdAt --jq 'sort_by(.createdAt) | reverse | .[].tagName'); do
      if gh release view "$candidate" --json assets --jq '.assets[].name' | grep -qx "$asset"; then
        tag="$candidate"
        break
      fi
    done
  fi
  if [ -z "$tag" ] || ! gh release view "$tag" --json assets --jq '.assets[].name' | grep -qx "$asset"; then
    echo "::warning::No release with $asset (${tag:-none}): product $product runs on its baseline (stock PaddleOCR model)."
    return 0
  fi
  mkdir -p models
  gh release download "$tag" -p "$asset" -D models --clobber
  tar -xzf "models/$asset" -C models
  echo "$3=$PWD/models/$model" >> "$GITHUB_ENV"
  echo "$4=$tag" >> "$GITHUB_ENV"
  echo "OCR model of $product: release $tag" | tee -a "$GITHUB_STEP_SUMMARY"
}

fetch "$product" "$tag" DOCEXTRACT_OCR_REC_MODEL_DIR OCR_MODEL_TAG
read -r _ _ latin < <(info "$product")
if [ -n "$latin" ]; then
  fetch "$latin" "" DOCEXTRACT_OCR_LATIN_REC_MODEL_DIR OCR_LATIN_MODEL_TAG
fi
