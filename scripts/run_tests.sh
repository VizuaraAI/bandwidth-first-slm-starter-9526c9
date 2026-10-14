#!/usr/bin/env bash
# The invariants the cost model must satisfy. Run before trusting a table.
set -euo pipefail
python tests/test_cost_model.py
