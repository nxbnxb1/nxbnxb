#!/usr/bin/env bash
# End-to-end fine-tuning of the PP-OCRv5 mobile recogniser on a CPU runner.
# Used by .github/workflows/_train.yml and ocr-train-speed.yml; every step writes into $WORK.
#
# LANGS=vi,en     → from latin_PP-OCRv5_mobile_rec, compact Vietnamese + English dictionary
# LANGS=vi,en,ja  → from PP-OCRv5_mobile_rec (kana/kanji/Latin), Vietnamese letters appended
#
# Env: WORK, PADDLEOCR_DIR (PaddleOCR repo checkout), LANGS,
#      SAMPLES       synthetic training lines
#      EPOCHS        passes over the training lines
#      CHUNK         lines per PaddleOCR "epoch": each epoch is a fresh random CHUNK of the data,
#                    followed by validation and a checkpoint, so a run stopped by the time
#                    budget always leaves a recent checkpoint
#      BATCH, TIME_BUDGET_MIN, RESUME_CHECKPOINT (optional path prefix, e.g. .../latest)
#      VAL_COUNT     validation lines per language (separate from the evaluation sets)
#      EVAL_COUNT    evaluation lines per language (evaluate.py, after training)
#      PRINT_STEP    log every N iterations
#      CPU speed:    THREADS (default: all cores) and FUSE / FREEZE / GTC (see train_cpu.py);
#                    measured by the "OCR training · speed probe" workflow
#      PROBE=1       speed measurement only: no export
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$PWD/vi_ocr_work}"
PADDLEOCR_DIR="${PADDLEOCR_DIR:?checkout of github.com/PaddlePaddle/PaddleOCR}"
LANGS="${LANGS:-vi,en}"
SAMPLES="${SAMPLES:-60000}"
EPOCHS="${EPOCHS:-4}"
CHUNK="${CHUNK:-6000}"
BATCH="${BATCH:-64}"
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-300}"
VAL_COUNT="${VAL_COUNT:-300}"
EVAL_COUNT="${EVAL_COUNT:-1000}"
PRINT_STEP="${PRINT_STEP:-10}"
THREADS="${THREADS:-$(nproc)}"
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
echo "FUSE=${FUSE:-0}" >> "$WORK/model.env"  # checkpoints of a fused model resume/export fused

echo "::group::dictionary + pretrained weights ($BASE_MODEL)"
if [ -n "${RESUME_DICT:-}" ]; then
  cp "$RESUME_DICT" "$DICT"  # a resumed checkpoint keeps the dictionary it was trained with
else
  python "$HERE/charset.py" "$DICT" "${DICT_ARGS[@]}"
fi
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
  # evaluation sets (evaluate.py) and validation sets (best checkpoint) use different seeds
  [ -s "$WORK/data/eval_$lang/labels.txt" ] || synth "$WORK/data/eval_$lang" --count "$EVAL_COUNT" --seed "$seed" --langs "$lang"
  [ -s "$WORK/data/val_$lang/labels.txt" ] || synth "$WORK/data/val_$lang" --count "$VAL_COUNT" --seed "$((seed + 50))" --langs "$lang"
  seed=$((seed + 1))
done
[ -s "$WORK/data/eval_clean/labels.txt" ] || synth "$WORK/data/eval_clean" --count "$((EVAL_COUNT * 6 / 10))" --seed 3 --langs "$LANGS" --no-augment
# validation during training: a mix of every language
mkdir -p "$WORK/data/val"
: > "$WORK/data/val/labels.txt"
for lang in "${LANG_LIST[@]}"; do sed "s#^#../val_$lang/#" "$WORK/data/val_$lang/labels.txt" >> "$WORK/data/val/labels.txt"; done
echo "::endgroup::"

# PaddleOCR epochs = chunks of CHUNK lines (fresh random subset each time)
read -r RATIO EPOCH_NUM < <(python -c "
import math; s, c, e = $SAMPLES, min($CHUNK, $SAMPLES), $EPOCHS
print(round(c / s, 6), max(1, math.ceil(e * s / c)))")
echo "$EPOCHS pass(es) over $SAMPLES lines = $EPOCH_NUM chunks of ~$((SAMPLES < CHUNK ? SAMPLES : CHUNK)) lines"

# CPU: Paddle runs its math on a single thread unless told otherwise
export FLAGS_paddle_num_threads="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

OVERRIDES=(
  "Global.model_name=$BASE_MODEL"
  "Global.epoch_num=$EPOCH_NUM"
  "Global.print_batch_step=$PRINT_STEP"
  "Global.eval_batch_epoch=1"
  "Global.save_epoch_step=1000000"
  "Global.character_dict_path=$DICT"
  "Global.save_model_dir=$WORK/output"
  "Global.save_res_path=$WORK/output/predicts.txt"
  "Optimizer.lr.learning_rate=$LR"
  "Train.dataset.data_dir=$WORK/data/train/"
  "Train.dataset.label_file_list=[$WORK/data/train/labels.txt]"
  "Train.dataset.ratio_list=[$RATIO]"
  "Train.loader.batch_size_per_card=$BATCH"
  "Train.loader.num_workers=2"
  "Eval.dataset.data_dir=$WORK/data/val/"
  "Eval.dataset.label_file_list=[$WORK/data/val/labels.txt]"
)
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  OVERRIDES+=("Global.checkpoints=$RESUME_CHECKPOINT" "Global.pretrained_model=")
else
  OVERRIDES+=("Global.pretrained_model=$WORK/pretrained/init.pdparams")
fi

echo "::group::training (budget ${TIME_BUDGET_MIN} min)"
cd "$PADDLEOCR_DIR"
set +e
timeout --signal=INT "${TIME_BUDGET_MIN}m" python "$HERE/train_cpu.py" -c "$CONFIG" -o "${OVERRIDES[@]}"
status=$?
set -e
echo "$status" > "$WORK/train_status"
echo "train.py exit status: $status (124/130 = stopped by the time budget)"
echo "::endgroup::"
[ "${PROBE:-0}" = 1 ] && exit 0

BEST="$WORK/output/best_accuracy"
[ -f "$BEST.pdparams" ] || BEST="$WORK/output/latest"
[ -f "$BEST.pdparams" ] || { echo "no checkpoint was written"; exit 1; }

echo "::group::export inference model from $BEST"
rm -rf "$WORK/export"
python "$HERE/train_cpu.py" export -c "$CONFIG" -o \
  "Global.model_name=$BASE_MODEL" \
  "Global.pretrained_model=$BEST" \
  "Global.character_dict_path=$DICT" \
  "Global.save_inference_dir=$WORK/export"
ls -la "$WORK/export"
echo "::endgroup::"
