#!/usr/bin/env bash
# Tesseract 5 with the most accurate public models (tessdata_best) for the given languages, and
# TESSDATA_PREFIX for the following workflow steps.
#
#   scripts/fetch_tessdata.sh vie+eng[+jpn]
set -euo pipefail
langs="${1:?languages, e.g. vie+eng}"
sudo apt-get install -y -qq tesseract-ocr >/dev/null
dir="$PWD/tessdata"
mkdir -p "$dir"
for lang in ${langs//+/ }; do
  curl -fsSL --retry 5 --retry-all-errors -o "$dir/$lang.traineddata" \
    "https://github.com/tesseract-ocr/tessdata_best/raw/main/$lang.traineddata"
done
echo "TESSDATA_PREFIX=$dir" >> "$GITHUB_ENV"
tesseract --version | head -1
