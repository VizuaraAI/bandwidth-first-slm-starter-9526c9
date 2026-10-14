# bandwidth-first-slm-starter

Starter repository for **Bandwidth-First Architecture Selection**: choosing
small language model shapes by bytes moved per decoded token instead of by
parameter count.

Naveen, this is your project scaffold. It is small on purpose: you should be
able to read all of `src/bfas/` in one sitting, and every file maps to a
section of the paper draft in `../paper/`.

## The idea in one paragraph

At batch size one, decoding a token means reading the weights the model uses
plus the whole key/value cache it attends to, with almost no arithmetic to hide
that traffic behind. So decode latency is `bytes per token / achievable
bandwidth`. A phone delivers roughly 50 to 90 GB/s; a data centre accelerator
delivers thousands. Parameter count only describes part of the first term, and
only at fixed precision. This project makes bytes per token the selection
variable, and shows that doing so changes which architecture you would ship.

## Does the premise hold? Check it in sixty seconds

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m bfas.evaluate --hf-models --contexts 4096 --bits-w 4.5 --bits-kv 8
```

That prices released models with the analytic cost model (configs only, no
weights downloaded). On the first run of this kit it showed, at context 4096
with 4-bit weights and 8-bit cache:

| Model | Parameters | Bytes per token | Cache share |
|---|---|---|---|
| Qwen2.5-1.5B | 1.54B | 885 MB | 6 percent |
| OLMo-2-1B | 1.48B | 943 MB | 27 percent |
| SmolLM2-1.7B | 1.71B | 1300 MB | 30 percent |
| Pythia-410m | 0.51B | 436 MB | 44 percent |
| Qwen2.5-0.5B | 0.49B | 290 MB | 8 percent |

Three models within 15 percent of each other in parameter count span 1.5x in
bytes moved, and a model with *more* parameters (Qwen2.5-0.5B) moves *fewer*
bytes than a smaller one (Pythia-410m). That gap is the paper.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src          # or: pip install -e .
python tests/test_cost_model.py
```

Python 3.10 or newer. A GPU helps for the training grid in weeks 3 and 4 but is
not needed for anything else; the cost model, the selector and the CPU decode
benchmark all run on a laptop.

## The first three things to run

```bash
# 1. Measure what your machine can actually deliver. Never quote a spec sheet.
bash scripts/00_profile_device.sh

# 2. Price published models and see the ranking come apart.
bash scripts/01_price_published_models.sh

# 3. Measure real decode throughput across shapes and compare the two
#    candidate predictors (this is figure 2a).
bash scripts/02_benchmark_shapes.sh
```

Then, once you have a GPU session:

```bash
bash scripts/03_train_grid.sh    # the matched-token-budget shape grid
bash scripts/04_select.sh        # the selector, with the quality model fitted
```

## Layout

```
src/bfas/
  bandwidth.py   measure achievable memory bandwidth (STREAM-style)
  arch.py        ArchSpec: the architecture shape, including bits per weight
  cost_model.py  bytes per decoded token, roofline predictions, quality per byte
  model.py       a small decoder-only LM built from an ArchSpec (GQA, latent KV)
  data.py        token cache, fixed-order batching, offline synthetic corpus
  train.py       train one shape under a fixed token budget
  benchmark.py   measured decode throughput, achieved GB/s, roofline efficiency
  evaluate.py    build the paper's tables from configs and from runs
  select.py      the bandwidth-first selector and its three baselines
  plots.py       figures from your measured results
configs/default.yaml     one config, every knob, nothing hidden
scripts/                 the five steps above, in order
experiments/log_template.md   one row per run, mapping to the paper's tables
tests/test_cost_model.py the invariants the cost model must satisfy
```

## What maps to what

| Repo | Paper |
|---|---|
| `cost_model.bytes_per_token` | Equation 1, Section 3.1 |
| `evaluate.py --hf-models` | Table 1, published models mispriced |
| `benchmark.py` | Table 2, cost model validation, and Figure 2a |
| `train.py` over the shape grid | the iso-bandwidth protocol, Section 3.2 |
| `select.py` | Table 3, selection rules |
| the ablation sweep you add in week 6 | Table 4 |

## Three things to be careful about

1. **Measure, do not quote.** `achievable_gbs` in the config starts as `null`
   on purpose. Fill it from `bfas.bandwidth` on the machine you report on.
2. **Report roofline efficiency with every throughput number.** If the engine
   reaches only a third of achievable bandwidth, your result is a statement
   about the engine, not about the architecture.
3. **The quality model in `select.py` is a placeholder until you fit it.** Run
   the training grid, then pass `--fit-from results/runs`. Say in the paper
   which figures used the fitted version.

## Data

No data files are committed. `bfas.data` streams FineWeb-Edu and caches a flat
token array locally. With no network, it falls back to a small synthetic Markov
corpus whose only purpose is proving the training loop works: loss should fall
from about 5.5 nats toward 2.5 within a few hundred steps. Never report a
number measured on it.
