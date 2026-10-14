#!/usr/bin/env bash
# Step 4, run the selector against the measured device, with the quality model
# fitted to your own training runs rather than the placeholder coefficients.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"

BW="${BANDWIDTH_GBS:-68}"
TARGET="${TARGET_TOKENS_PER_S:-30}"
FIT_ARG=""
if [ -d results/runs ] && [ "$(ls -1 results/runs/*.json 2>/dev/null | wc -l)" -ge 3 ]; then
  FIT_ARG="--fit-from results/runs"
else
  echo "note: fewer than 3 training runs found, using the placeholder quality model"
fi

python -m bfas.select \
  --bandwidth-gbs "$BW" \
  --target-tokens-per-s "$TARGET" \
  --contexts 512 2048 4096 \
  --context-weights 0.5 0.35 0.15 \
  --memory-limit-gb 2.0 \
  --top-k 5 \
  $FIT_ARG \
  --out results/selection.json
