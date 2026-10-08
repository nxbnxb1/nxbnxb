#!/usr/bin/env bash
# End-to-end fine-tuning of the PP-OCRv5 mobile recogniser.
# Used by .github/workflows/train-vi-ocr.yml; every step writes into $WORK.
#
# LANGS=vi,en     → from latin_PP-OCRv5_mobile_rec, compact Vietnamese + English dictionary
# LANGS=vi,en,ja  → from PP-OCRv5_mobile_rec (kana/kanji/Latin), Vietnamese letters appended
#
# Env: WORK, PADDLEOCR_DIR (PaddleOCR repo checkout), LANGS, SAMPLES, EPOCHS, BATCH,
#      TIME_BUDGET_MIN, RESUME_CHECKPOINT (optional path prefix, e.g. .../latest)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$PWD/vi_ocr_work}"
PADDLEOCR_DIR="${PADDLEOCR_DIR:?checkout of github.com/PaddlePaddle/PaddleOCR}"
LANGS="${LANGS:-vi,en}"
SAMPLES="${SAMPLES:-60000}"
EPOCHS="${EPOCHS:-4}"
BATCH="${BATCH:-64}"
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-300}"
WORKERS="$(nproc)"
CONFIG="$HERE/vi_en_PP-OCRv5_mobile_rec.yml"
DICTS="$PADDLEOCR_DIR/ppocr/utils/dict"

if [[ ",$LANGS," == *",ja,"* ]]; then
  BASE_MODEL="PP-OCRv5_mobile_rec"
  OLD_DICT="$DICTS/ppocrv5_dict.txt"
  NAME="vi_en_ja_PP-OCRv5_mobile_rec"
  LR="0.0002"  # gentler: keep what the model knows about kanji/kana that the data does not cover
  DICT_ARGS=(--langs "$LANGS" --base-dict "$OLD_DICT")
else
  BASE_MODEL="latin_PP-OCRv5_mobile_rec"
  OLD_DICT="$DICTS/ppocrv5_latin_dict.txt"
  NAME="vi_en_PP-OCRv5_mobile_rec"
  LR="0.0003"
  DICT_ARGS=(--langs "$LANGS")
fi
PRETRAINED_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/${BASE_MODEL}_pretrained.pdparams"
DICT="$WORK/dict.txt"
mkdir -p "$WORK/pretrained" "$WORK/output"
echo "NAME=$NAME" > "$WORK/model.env"
echo "BASE_MODEL=$BASE_MODEL" >> "$WORK/model.env"

echo "::group::dictionary + pretrained weights ($BASE_MODEL)"
python "$HERE/charset.py" "$DICT" "${DICT_ARGS[@]}"
if [ ! -s "$WORK/pretrained/base.pdparams" ]; then
  curl -fsSL --retry 6 --retry-delay 10 --retry-all-errors -o "$WORK/pretrained/base.pdparams" "$PRETRAINED_URL"
fi
python "$HERE/init_weights.py" --pretrained "$WORK/pretrained/base.pdparams" --old-dict "$OLD_DICT" \
  --new-dict "$DICT" --out "$WORK/pretrained/init.pdparams"
echo "::endgroup::"

echo "::group::synthetic data ($LANGS)"
cd "$HERE"
synth() { python synth.py "$@" --dict "$DICT" --workers "$WORKERS"; }
[ -s "$WORK/data/train/labels.txt" ] || synth "$WORK/data/train" --count "$SAMPLES" --seed 1 --langs "$LANGS"
IFS=',' read -ra LANG_LIST <<< "$LANGS"
seed=10
for lang in "${LANG_LIST[@]}"; do
  [ -s "$WORK/data/eval_$lang/labels.txt" ] || synth "$WORK/data/eval_$lang" --count 1000 --seed "$seed" --langs "$lang"
  seed=$((seed + 1))
done
[ -s "$WORK/data/eval_clean/labels.txt" ] || synth "$WORK/data/eval_clean" --count 600 --seed 3 --langs "$LANGS" --no-augment
# validation during training: a mix of every language
mkdir -p "$WORK/data/eval"
: > "$WORK/data/eval/labels.txt"
for lang in "${LANG_LIST[@]}"; do sed "s#^#../eval_$lang/#" "$WORK/data/eval_$lang/labels.txt" >> "$WORK/data/eval/labels.txt"; done
echo "::endgroup::"

OVERRIDES=(
  "Global.model_name=$BASE_MODEL"
  "Global.epoch_num=$EPOCHS"
  "Global.character_dict_path=$DICT"
  "Global.save_model_dir=$WORK/output"
  "Global.save_res_path=$WORK/output/predicts.txt"
  "Optimizer.lr.learning_rate=$LR"
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
  OVERRIDES+=("Global.pretrained_model=$WORK/pretrained/init.pdparams")
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
  "Global.model_name=$BASE_MODEL" \
  "Global.pretrained_model=$BEST" \
  "Global.character_dict_path=$DICT" \
  "Global.save_inference_dir=$WORK/export"
ls -la "$WORK/export"
echo "::endgroup::"
