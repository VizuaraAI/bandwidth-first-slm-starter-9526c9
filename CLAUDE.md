# CLAUDE.md

Instructions for the coding agent working in this repository with Naveen.

## What this project is

An empirical systems and machine learning study called **BFAS: Bandwidth-First
Architecture Selection**. The claim: autoregressive decode on bandwidth-limited
devices (mobile class, 50 to 90 GB/s) is bound by bytes moved per token, not by
parameter count, so architecture selection should be stated in bytes. The
deliverables are an analytic cost model, a calibration study against measured
decode throughput, an iso-bandwidth comparison protocol, and a selector that
maps a device plus a latency target to an architecture shape.

Target: a workshop paper by early 2027, arXiv preprint first. The paper draft
lives in `../paper/main.tex` and every repo artifact maps to a section of it.

## Project layout

```
src/bfas/{bandwidth,arch,cost_model,model,data,train,benchmark,evaluate,select,plots}.py
configs/default.yaml      the single source of configuration
scripts/00..04            the pipeline, in order
experiments/log_template.md   the experiment log, one row per run
tests/test_cost_model.py  invariants, run them before trusting a table
results/                  generated, not committed except the CSV tables
```

Read `src/bfas/cost_model.py` first. It is the centre of the project; every
other module either feeds it or consumes it.

## Conventions to follow

- **No em dashes in any prose, in code comments or in the paper.** Use commas,
  colons, parentheses or semicolons.
- Bytes are counted in bytes and reported in MB or GB with the unit in the
  column name. Bandwidth is GB/s with `1e9` bytes per GB; memory sizes use
  `1024**3`. Both conventions appear in this codebase on purpose, so keep each
  one where it already is and never silently convert between them.
- Every throughput number is reported with the context length it was measured
  at and the roofline efficiency it achieved. A tokens-per-second number with
  no context length attached is not a result.
- `ArchSpec` is frozen. Build variants with `spec.with_(...)`, never by mutating.
- New measurement code writes a CSV under `results/` and prints a table. The
  CSV is the audit trail between the paper and the run.
- Prefer small, readable functions over clever ones. A reviewer must be able to
  check the byte accounting by reading it.

## State of the code

Working and verified:

- the cost model agrees exactly with `sum(p.numel())` for models built by
  `model.py`, which is the check that keeps the accounting honest;
- `model.py` trains (single-batch overfit reaches near-zero loss) and supports
  grouped-query attention and a low-rank joint key/value cache;
- `benchmark.py` produces measured tokens per second and roofline efficiency;
- `select.py` produces three distinct selections from the three rules, with the
  parameter-count rule missing the latency target, which is the result the
  paper predicts.

Known caveats to keep in mind, and to state in the paper:

- The quality model in `select.py` is a placeholder with assumed coefficients
  until `--fit-from results/runs` is used. Anything produced before fitting is
  a sanity check, not a result.
- `model.py` is plain PyTorch with no fused kernels. The latent-cache variant
  moves fewer bytes but currently measures *slower* in wall clock because keys
  and values are reconstructed from the latent every step. That is an engine
  artifact, not a shape result. Report roofline efficiency so it is visible,
  and say so explicitly when discussing latent caches.
- Bandwidth measured inside a container or a shared VM can be far below the
  host's real figure. Profile on the machine you will report on.

## Next steps, in order

1. Build the 50M token FineWeb-Edu cache and run the shape grid
   (`scripts/03_train_grid.sh`). Fill the quality column of the paper's tables.
2. Fit the quality model from those runs and re-run the selector. Report both
   the fitted and the placeholder selection, and the regret of each rule.
3. Add the ablation sweep: vary one shape knob at a time from a reference
   shape, at contexts 512, 2048 and 8192. This fills Table 4 and Figure 2c.
4. Add a second device to the calibration study so Table 2 has a contrast row,
   and sweep bandwidth on one machine by varying `OMP_NUM_THREADS`.
5. Add a production engine (llama.cpp or MLC) as a second runtime, so the
   roofline efficiency column is not a statement about PyTorch alone.
6. Extend the cost model with the speculative decoding divisor on the weight
   term and test the prediction that speculation shifts the optimum toward
   larger, more cache-compressed models.

## How to help well here

- When adding an experiment, add its row to `experiments/log_template.md` in
  the same change. An unlogged run does not exist.
- When a number in the paper changes, change the CSV it came from and say which
  run produced it. Never edit a number in `main.tex` without a run behind it.
- If you are about to write "approximately" or "roughly" about a measurement,
  stop and measure it instead.
- If a result looks too good, check roofline efficiency and the context length
  first. Those two explain most surprises in this project.
