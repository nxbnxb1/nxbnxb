#!/usr/bin/env bash
# Free VLM inside the GitHub runner: Ollama serving an open-source vision-language model on CPU
# (default Qwen2.5-VL 3B). No API key, no cost; about a minute per picture on a 4-core runner.
# Exports the DOCEXTRACT_VLM_* settings for the following steps.
#
#   scripts/start_local_vlm.sh [ollama-model]
set -euo pipefail
model="${1:-qwen2.5vl:3b}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh >/dev/null
fi
sudo systemctl stop ollama >/dev/null 2>&1 || true  # run as this user so the model cache is ours
nohup ollama serve > /tmp/ollama.log 2>&1 &
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
  sleep 1
done
ollama pull "$model" >/dev/null
{
  echo "DOCEXTRACT_VLM_BASE_URL=http://127.0.0.1:11434/v1"
  echo "DOCEXTRACT_VLM_MODEL=$model"
  echo "DOCEXTRACT_VLM_TIMEOUT=900"
  echo "DOCEXTRACT_VLM_MAX_CONCURRENCY=1"
  echo "DOCEXTRACT_VLM_MAX_IMAGE_SIDE=896"
  echo "DOCEXTRACT_VLM_MAX_TOKENS=1024"
} >> "$GITHUB_ENV"
echo "Local VLM: $model via Ollama (free, CPU)" | tee -a "$GITHUB_STEP_SUMMARY"
