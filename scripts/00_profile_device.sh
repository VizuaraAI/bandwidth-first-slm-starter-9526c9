#!/usr/bin/env bash
# Step 0, run this first and on every machine you report numbers from.
# Writes results/device_profile.json, which every later step reads.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"

python -m bfas.bandwidth \
  --working-set-mb 512 \
  --repeats 7 \
  --sweep \
  --torch-device "${TORCH_DEVICE:-cuda}" \
  --out results/device_profile.json
