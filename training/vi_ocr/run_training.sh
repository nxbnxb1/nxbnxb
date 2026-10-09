#!/usr/bin/env bash
# Fine-tuning of the PP-OCRv5 mobile recogniser on CPU runners; every step writes into $WORK.
#
#   run_training.sh prepare   dictionary, initial weights, validation and evaluation line sets
#   run_training.sh train     fresh synthetic lines (seed SEED) + training until TIME_BUDGET_MIN;
#                             the weights reached are saved to $WORK/output/final.pdparams
#   run_training.sh export    inference model from EXPORT_PARAMS (default: final weights)
#   run_training.sh all       the three in a row (one runner)
#
# Used by .github/workflows/_train.yml, which runs `train` on several runners at once and
# averages their weights between rounds, and by ocr-train-speed.yml.
#
# LANGS=vi,en     → from latin_PP-OCRv5_mobile_rec, compact Vietnamese + English dictionary
# LANGS=vi,en,ja  → from PP-OCRv5_mobile_rec, Japanese character set + Vietnamese letters
#
# Env: WORK, PADDLEOCR_DIR (PaddleOCR checkout), LANGS
#   prepare: VAL_COUNT, EVAL_COUNT (lines per language), RESUME_DICT (dictionary of a checkpoint)
#   train:   SAMPLES (lines generated for this run), SEED, EPOCHS (passes over them), BATCH,
#            CHUNK (lines per PaddleOCR "epoch": validation + checkpoint after each),
#            TIME_BUDGET_MIN, INIT_PARAMS (weights to start from; default: pretrained),
#            LR (default per product), LR_CONST=1 (constant LR instead of cosine), WARMUP
#            (fraction of a chunk), PRINT_STEP, NO_EVAL=1 (no validation during training),
#            PROBE=1 (speed measurement: keep nothing)
#   corpus:  CORPUS_MANIFEST + CORPUS_FILES (enterprise documents, scripts/collect_corpus.py):
#            training lines are then REAL_SHARE (default 0.5) real lines cut from documents of
#            the training companies (doc_lines.py, documents of shard SHARD=k/n) and the rest
#            synthetic; validation and evaluation also get real lines of the dev / test companies
#   loader:  LOADER_WORKERS (data loader processes, default 2)
#   CPU:     THREADS (default: all cores), FUSE / FREEZE / GTC (see train_cpu.py)
set -euo pipefail

CMD="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$PWD/vi_ocr_work}"
PADDLEOCR_DIR="${PADDLEOCR_DIR:?checkout of github.com/PaddlePaddle/PaddleOCR}"
LANGS="${LANGS:-vi,en}"
CONFIG="$HERE/vi_en_PP-OCRv5_mobile_rec.yml"
DICTS="$PADDLEOCR_DIR/ppocr/utils/dict"
DICT="$WORK/dict.txt"
WORKERS="$(nproc)"
IFS=',' read -ra LANG_LIST <<< "$LANGS"

if [[ ",$LANGS," == *",ja,"* ]]; then
  BASE_MODEL="PP-OCRv5_mobile_rec"
  OLD_DICT="$DICTS/ppocrv5_dict.txt"
  NAME="vi_en_ja_PP-OCRv5_mobile_rec"
  DEFAULT_LR="0.0002"  # gentler: keep what the model knows about kanji/kana that the data does not cover
  DICT_ARGS=(--langs "$LANGS" --base-dict "$OLD_DICT")
else
  BASE_MODEL="latin_PP-OCRv5_mobile_rec"
  OLD_DICT="$DICTS/ppocrv5_latin_dict.txt"
  NAME="vi_en_PP-OCRv5_mobile_rec"
  DEFAULT_LR="0.0003"
  DICT_ARGS=(--langs "$LANGS")
fi
mkdir -p "$WORK/pretrained" "$WORK/output" "$WORK/data"

synth() { (cd "$HERE" && python synth.py "$@" --dict "$DICT" --workers "$WORKERS"); }
corpus() { [ -n "${CORPUS_MANIFEST:-}" ] && [ -s "$CORPUS_MANIFEST" ] && [ -d "${CORPUS_FILES:-}" ]; }
real_lines() {  # out split count seed [shard]
  python "$HERE/doc_lines.py" "$1" --manifest "$CORPUS_MANIFEST" --files "$CORPUS_FILES" --dict "$DICT" \
    --langs "$LANGS" --split "$2" --count "$3" --seed "$4" --shard "${5:-1/1}" --workers "$WORKERS"
}

prepare() {
  local val="${VAL_COUNT:-200}" eval_count="${EVAL_COUNT:-1000}"
  {
    echo "NAME=$NAME"
    echo "BASE_MODEL=$BASE_MODEL"
    echo "FUSE=${FUSE:-0}"  # weights of a fused model continue and export fused
    echo "FREEZE=${FREEZE:-}"
    echo "GTC=${GTC:-1}"
    echo "LR=${LR:-$DEFAULT_LR}"
  } > "$WORK/model.env"

  echo "::group::dictionary + initial weights ($BASE_MODEL)"
  if [ -n "${RESUME_DICT:-}" ]; then
    cp "$RESUME_DICT" "$DICT"  # continued weights keep the dictionary they were trained with
  else
    python "$HERE/charset.py" "$DICT" "${DICT_ARGS[@]}"
  fi
  if [ ! -s "$WORK/pretrained/base.pdparams" ]; then
    curl -fsSL --retry 6 --retry-delay 10 --retry-all-errors -o "$WORK/pretrained/base.pdparams" \
      "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/${BASE_MODEL}_pretrained.pdparams"
  fi
  python "$HERE/init_weights.py" --pretrained "$WORK/pretrained/base.pdparams" --old-dict "$OLD_DICT" \
    --new-dict "$DICT" --out "$WORK/pretrained/init.pdparams"
  echo "::endgroup::"

  echo "::group::validation and evaluation lines ($LANGS)"
  local seed=10 lang
  for lang in "${LANG_LIST[@]}"; do
    # evaluation sets (evaluate.py, after training) and validation sets (during training) differ
    [ -s "$WORK/data/eval_$lang/labels.txt" ] || synth "$WORK/data/eval_$lang" --count "$eval_count" --seed "$seed" --langs "$lang"
    [ -s "$WORK/data/val_$lang/labels.txt" ] || synth "$WORK/data/val_$lang" --count "$val" --seed "$((seed + 50))" --langs "$lang"
    seed=$((seed + 1))
  done
  [ -s "$WORK/data/eval_clean/labels.txt" ] || synth "$WORK/data/eval_clean" --count "$((eval_count * 6 / 10))" --seed 3 --langs "$LANGS" --no-augment
  if corpus; then  # lines of real documents of companies never trained on
    [ -s "$WORK/data/eval_real/labels.txt" ] || real_lines "$WORK/data/eval_real" test "$eval_count" 7 || true
    [ -s "$WORK/data/val_real/labels.txt" ] || real_lines "$WORK/data/val_real" dev "$val" 77 || true
  fi
  mkdir -p "$WORK/data/val"
  : > "$WORK/data/val/labels.txt"
  for lang in "${LANG_LIST[@]}"; do sed "s#^#../val_$lang/#" "$WORK/data/val_$lang/labels.txt" >> "$WORK/data/val/labels.txt"; done
  if [ -s "$WORK/data/val_real/labels.txt" ]; then
    sed "s#^#../val_real/#" "$WORK/data/val_real/labels.txt" >> "$WORK/data/val/labels.txt"
  fi
  echo "::endgroup::"
}

train() {
  local samples="${SAMPLES:-12000}" seed="${SEED:-1}" epochs="${EPOCHS:-100}" chunk="${CHUNK:-6000}"
  local threads="${THREADS:-$(nproc)}" ratio epoch_num
  [ -n "${threads}" ] || threads="$(nproc)"

  echo "::group::training lines (seed $seed, $samples lines)"
  rm -rf "$WORK/data/train"
  local real=0
  if corpus; then
    real=$(python -c "print(int($samples * float('${REAL_SHARE:-0.5}')))")
    real_lines "$WORK/data/train/real" train "$real" "$seed" "${SHARD:-1/1}" || real=0
  fi
  synth "$WORK/data/train/synth" --count "$((samples - real))" --seed "$seed" --langs "$LANGS"
  sed "s#^#synth/#" "$WORK/data/train/synth/labels.txt" > "$WORK/data/train/labels.txt"
  if [ -s "$WORK/data/train/real/labels.txt" ]; then
    sed "s#^#real/#" "$WORK/data/train/real/labels.txt" >> "$WORK/data/train/labels.txt"
  fi
  echo "$(grep -c '^real/' "$WORK/data/train/labels.txt" || true) real + $(grep -c '^synth/' "$WORK/data/train/labels.txt" || true) synthetic lines"
  echo "::endgroup::"

  # PaddleOCR epochs = chunks of CHUNK lines (a fresh random subset each time)
  read -r ratio epoch_num < <(python -c "
import math; s, c, e = $samples, min($chunk, $samples), $epochs
print(round(c / s, 6), max(1, math.ceil(e * s / c)))")

  # CPU: Paddle runs its math on a single thread unless told otherwise
  export FLAGS_paddle_num_threads="$threads" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
  export FUSE="${FUSE:-0}"
  local overrides=(
    "Global.model_name=$BASE_MODEL"
    "Global.epoch_num=$epoch_num"
    "Global.print_batch_step=${PRINT_STEP:-10}"
    "Global.eval_batch_epoch=1"
    "Global.save_epoch_step=1000000"
    "Global.character_dict_path=$DICT"
    "Global.save_model_dir=$WORK/output"
    "Global.save_res_path=$WORK/output/predicts.txt"
    "Global.checkpoints="
    "Optimizer.lr.learning_rate=${LR:-$DEFAULT_LR}"
    "Optimizer.lr.warmup_epoch=${WARMUP:-1}"
    "Train.dataset.data_dir=$WORK/data/train/"
    "Train.dataset.label_file_list=[$WORK/data/train/labels.txt]"
    "Train.dataset.ratio_list=[$ratio]"
    "Train.loader.batch_size_per_card=${BATCH:-32}"
    "Train.loader.num_workers=${LOADER_WORKERS:-2}"
    "Eval.dataset.data_dir=$WORK/data/val/"
    "Eval.dataset.label_file_list=[$WORK/data/val/labels.txt]"
    # evaluation keeps each line's aspect ratio (variable width), so one line per batch
    "Eval.loader.batch_size_per_card=1"
  )
  # NO_EVAL=1: no validation during training (rounds on several runners validate the average)
  [ "${NO_EVAL:-0}" = 1 ] && overrides+=("Global.eval_batch_epoch=" "Global.eval_batch_step=[1000000000,1000000000]")
  [ "${LR_CONST:-0}" = 1 ] && overrides+=("Optimizer.lr.name=Const")
  if [ -n "${INIT_PARAMS:-}" ]; then
    export INIT_PARAMS  # loaded by train_cpu.py after the model structure is set up
    overrides+=("Global.pretrained_model=")
  else
    overrides+=("Global.pretrained_model=$WORK/pretrained/init.pdparams")
  fi

  echo "::group::training (budget ${TIME_BUDGET_MIN:-300} min, $epoch_num chunks)"
  rm -f "$WORK/output/final.pdparams"
  local status
  set +e
  (cd "$PADDLEOCR_DIR" && timeout --kill-after=5m --signal=INT "${TIME_BUDGET_MIN:-300}m" \
    python "$HERE/train_cpu.py" -c "$CONFIG" -o "${overrides[@]}")
  status=$?
  set -e
  echo "$status" > "$WORK/train_status"
  echo "train_cpu.py exit status: $status (124 = stopped by the time budget)"
  echo "::endgroup::"
  [ "${PROBE:-0}" = 1 ] && return 0
  [ -s "$WORK/output/final.pdparams" ] || { echo "no weights were saved"; exit 1; }
}

export_model() {
  local params="${EXPORT_PARAMS:-$WORK/output/final.pdparams}"
  echo "::group::export inference model from $params"
  rm -rf "$WORK/export"
  (cd "$PADDLEOCR_DIR" && python "$HERE/train_cpu.py" export -c "$CONFIG" -o \
    "Global.model_name=$BASE_MODEL" \
    "Global.pretrained_model=${params%.pdparams}" \
    "Global.character_dict_path=$DICT" \
    "Global.save_inference_dir=$WORK/export")
  ls -la "$WORK/export"
  echo "::endgroup::"
}

case "$CMD" in
  prepare) prepare ;;
  train) train ;;
  export) export_model ;;
  all)
    prepare
    train
    if [ "${PROBE:-0}" != 1 ]; then
      # one runner: the weights that scored best on the validation lines, if validation ran
      [ -s "$WORK/output/best_accuracy.pdparams" ] && export EXPORT_PARAMS="${EXPORT_PARAMS:-$WORK/output/best_accuracy.pdparams}"
      export_model
    fi ;;
  *) echo "usage: $0 prepare|train|export|all" >&2; exit 2 ;;
esac
