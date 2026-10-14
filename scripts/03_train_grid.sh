#!/usr/bin/env bash
# Step 3, train the shape grid under one matched token budget. Build the token
# cache first; every shape then sees an identical stream.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"

TOKENS="${TOKEN_BUDGET:-50000000}"
CACHE="${TOKEN_CACHE:-data/tokens_fineweb_50M.npy}"
DEVICE="${TORCH_DEVICE:-cuda}"

if [ ! -f "$CACHE" ]; then
  python -m bfas.data --n-tokens "$TOKENS" --cache-dir data --tag fineweb
fi

for SHAPE in tiny-60m-mha tiny-60m-gqa2 tiny-60m-deep tiny-60m-wide tiny-60m-latent; do
  echo "=== $SHAPE ==="
  python -m bfas.train \
    --shape "$SHAPE" \
    --token-budget "$TOKENS" \
    --token-cache "$CACHE" \
    --seq-len 512 \
    --batch-size 8 \
    --grad-accum 4 \
    --device "$DEVICE" \
    --out-dir results/runs
done

python -m bfas.evaluate --runs results/runs --contexts 512 2048
