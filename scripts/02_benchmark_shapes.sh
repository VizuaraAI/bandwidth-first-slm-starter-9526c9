#!/usr/bin/env bash
# Step 2, measure decode throughput for the reference shapes and compare the
# two candidate predictors. This is the evidence behind figure 2 panel (a).
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"

python -m bfas.benchmark \
  --shapes tiny-60m-mha tiny-60m-gqa2 tiny-60m-deep tiny-60m-wide tiny-60m-latent \
  --contexts 128 512 2048 \
  --new-tokens 32 \
  --device "${TORCH_DEVICE:-cpu}" \
  --out results/decode_benchmark.csv

python -m bfas.plots --rows results/decode_benchmark.csv --context 512
