#!/usr/bin/env bash
# Step 1, the motivating table: price released small models with the cost
# model. Downloads configs only, not weights, so it takes seconds.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"

BW="${BANDWIDTH_GBS:-68}"

echo "== 16-bit weights =="
python -m bfas.evaluate --hf-models --contexts 512 2048 8192 --bandwidth-gbs "$BW"

echo
echo "== 4-bit weights, 8-bit cache: the realistic on-device setting =="
python -m bfas.evaluate --hf-models --contexts 512 2048 8192 \
  --bandwidth-gbs "$BW" --bits-w 4.5 --bits-kv 8
