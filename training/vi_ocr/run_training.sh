#!/usr/bin/env bash
# End-to-end fine-tuning of the PP-OCRv5 recogniser for Vietnamese + English.
# Used by .github/workflows/train-vi-ocr.yml; every step writes into $WORK.
#
# Env: WORK (work dir), PADDLEOCR_DIR (PaddleOCR repo checkout), SAMPLES, EPOCHS,
#      BATCH, TIME_BUDGET_MIN, RESUME_CHECKPOINT (optional path prefix, e.g. .../latest)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$PWD/vi_ocr_work}"
PADDLEOCR_DIR="${PADDLEOCR_DIR:?checkout of github.com/PaddlePaddle/PaddleOCR}"
SAMPLES="${SAMPLES:-60000}"
EPOCHS="${EPOCHS:-4}"
BATCH="${BATCH:-64}"
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-300}"
WORKERS="$(nproc)"
PRETRAINED_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams"
CONFIG="$HERE/vi_en_PP-OCRv5_mobile_rec.yml"
mkdir -p "$WORK/pretrained" "$WORK/output"

echo "::group::dictionary + pretrained weights"
python "$HERE/charset.py" "$WORK/vi_en_dict.txt"
if [ ! -s "$WORK/pretrained/latin.pdparams" ]; then
  curl -fsSL --retry 6 --retry-delay 10 --retry-all-errors -o "$WORK/pretrained/latin.pdparams" "$PRETRAINED_URL"
fi
python "$HERE/init_weights.py" \
  --pretrained "$WORK/pretrained/latin.pdparams" \
  --old-dict "$PADDLEOCR_DIR/ppocr/utils/dict/ppocrv5_latin_dict.txt" \
  --new-dict "$WORK/vi_en_dict.txt" \
  --out "$WORK/pretrained/vi_en_init.pdparams"
echo "::endgroup::"

echo "::group::synthetic data"
cd "$HERE"
[ -s "$WORK/data/train/labels.txt" ] || python synth.py "$WORK/data/train" --count "$SAMPLES" --seed 1 --workers "$WORKERS"
[ -s "$WORK/data/eval/labels.txt" ] || python synth.py "$WORK/data/eval" --count 2000 --seed 2 --workers "$WORKERS"
[ -s "$WORK/data/eval_clean/labels.txt" ] || python synth.py "$WORK/data/eval_clean" --count 1000 --seed 3 --workers "$WORKERS" --no-augment
echo "::endgroup::"

OVERRIDES=(
  "Global.epoch_num=$EPOCHS"
  "Global.character_dict_path=$WORK/vi_en_dict.txt"
  "Global.save_model_dir=$WORK/output"
  "Global.save_res_path=$WORK/output/predicts.txt"
  "Train.dataset.data_dir=$WORK/data/train/"
  "Train.dataset.label_file_list=[$WORK/data/train/labels.txt]"
  "Train.loader.batch_size_per_card=$BATCH"
  "Train.loader.num_workers=2"
  "Eval.dataset.data_dir=$WORK/data/eval/"
  "Eval.dataset.label_file_list=[$WORK/data/eval/labels.txt]"
)
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  OVERRIDES+=("Global.checkpoints=$RESUME_CHECKPOINT" "Global.pretrained_model=")
else
  OVERRIDES+=("Global.pretrained_model=$WORK/pretrained/vi_en_init.pdparams")
fi

echo "::group::training (budget ${TIME_BUDGET_MIN} min)"
cd "$PADDLEOCR_DIR"
set +e
timeout --signal=INT "${TIME_BUDGET_MIN}m" python tools/train.py -c "$CONFIG" -o "${OVERRIDES[@]}"
status=$?
set -e
echo "train.py exit status: $status (124/130 = stopped by the time budget)"
echo "::endgroup::"

BEST="$WORK/output/best_accuracy"
[ -f "$BEST.pdparams" ] || BEST="$WORK/output/latest"
[ -f "$BEST.pdparams" ] || { echo "no checkpoint was written"; exit 1; }

echo "::group::export inference model from $BEST"
rm -rf "$WORK/export"
python tools/export_model.py -c "$CONFIG" -o \
  "Global.pretrained_model=$BEST" \
  "Global.character_dict_path=$WORK/vi_en_dict.txt" \
  "Global.save_inference_dir=$WORK/export"
ls -la "$WORK/export"
echo "::endgroup::"
