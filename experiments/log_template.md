# Experiment log

One row per run. The columns are the columns of the paper's tables, so a filled
log is a filled paper. Copy a block, fill it in, never delete a row (a failed
run is a result about the method).

Conventions: bytes are reported at a stated context length; `BRE` is bandwidth
roofline efficiency, achieved GB/s divided by achievable GB/s; `quality` is
`1 / log(validation perplexity)`, higher is better.

---

## Device profiles

Fill one row per machine you report numbers from, by running
`bash scripts/00_profile_device.sh`.

| Date | Machine | Device | Achievable GB/s | Method | Fitted engine efficiency | Notes |
|---|---|---|---|---|---|---|
|  |  | cpu |  | numpy STREAM add |  |  |
|  |  | cuda |  | torch fp16 reduction |  |  |

---

## Table 1 in the paper: published models priced by the cost model

Produced by `bash scripts/01_price_published_models.sh`. Source file:
`results/offtheshelf_cost.csv`.

| Model | Params (B) | Weight MB/token | Cache MB/token | Total MB/token | Cache share | Predicted tok/s | Context | Weight bits | Cache bits |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  | 4096 | 4.5 | 8 |

Crossover context (where cache traffic overtakes weight traffic), one per model:

| Model | Crossover context | Comment |
|---|---|---|
|  |  |  |

---

## Table 2 in the paper: cost model validation

Produced by `bash scripts/02_benchmark_shapes.sh`. Source file:
`results/decode_benchmark.csv`.

| Run ID | Date | Device | Context | Shapes measured | MAPE (%) | Spearman (bytes) | Spearman (params) | Mean BRE | Notes |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  | 512 |  |  |  |  |  |  |
|  |  |  | 2048 |  |  |  |  |  |  |
|  |  |  | 8192 |  |  |  |  |  |  |

---

## The trained shape grid: quality under a matched token budget

Produced by `bash scripts/03_train_grid.sh`. Source files:
`results/runs/*.json`, `results/trained_shapes.csv`.
Every row must share the same token budget, the same stream and the same seed,
or the comparison is not controlled. State the budget here once:

- Token budget: __________ tokens, seq len ______, seed ______, data: FineWeb-Edu

| Run ID | Shape | Frontier | Params (M) | MB/token @2048 | Cache share | Val PPL | Quality | Quality per GB | Wall clock | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
|  | tiny-60m-mha | iso-parameter |  |  |  |  |  |  |  |  |
|  | tiny-60m-gqa2 | both |  |  |  |  |  |  |  |  |
|  | tiny-60m-deep | iso-parameter |  |  |  |  |  |  |  |  |
|  | tiny-60m-wide | iso-parameter |  |  |  |  |  |  |  |  |
|  | tiny-60m-latent | iso-bytes |  |  |  |  |  |  |  |  |
|  | small-140m-gqa4 | iso-bytes |  |  |  |  |  |  |  |  |

---

## Table 3 in the paper: selection rules

Produced by `bash scripts/04_select.sh`. Source file: `results/selection.json`.
Record whether the quality model was fitted; an unfitted row is a sanity check,
not a result.

| Run ID | Date | Device GB/s | Target tok/s | Quality model | Rule | Shape chosen | MB/token | Predicted tok/s | Meets target | Quality | Regret |
|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  | 30 | placeholder | bandwidth-first |  |  |  |  |  |  |
|  |  |  | 30 | placeholder | parameter-count |  |  |  |  |  |  |
|  |  |  | 30 | placeholder | flop-first |  |  |  |  |  |  |
|  |  |  | 30 | fitted | bandwidth-first |  |  |  |  |  |  |

---

## Table 4 in the paper: shape knob ablation

One knob varied at a time from a stated reference shape. Reference shape:
__________________

| Run ID | Knob | Value from | Value to | Delta MB/token @512 | @2048 | @8192 | Delta quality | Delta tok/s per quality point | Notes |
|---|---|---|---|---|---|---|---|---|---|
|  | GQA group count |  |  |  |  |  |  |  |  |
|  | head dimension |  |  |  |  |  |  |  |  |
|  | KV latent rank |  |  |  |  |  |  |  |  |
|  | weight bits |  |  |  |  |  |  |  |  |
|  | cache bits |  |  |  |  |  |  |  |  |
|  | depth over width |  |  |  |  |  |  |  |  |

---

## Run journal

Free text, one entry per working session. This is where the paper's discussion
section comes from, so write down the surprises while they are fresh.

### YYYY-MM-DD, run ID ____
- What I expected:
- What happened:
- Roofline efficiency, and what it says about engine versus shape:
- What I changed as a result:
- What to run next:
