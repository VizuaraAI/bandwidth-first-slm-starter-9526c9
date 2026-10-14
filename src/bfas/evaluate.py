"""Evaluation and table building.

Two jobs:

    --hf-models   price real published models with the cost model, which is how
                  the paper's motivating table is produced. Only the config is
                  needed, so this runs in seconds and without weights.
    --runs        aggregate completed training runs into the paper's main
                  results table, including quality per byte.

Keep the CSV files this writes under version control. They are the audit trail
between a claim in the paper and the run that produced it.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Optional, Sequence

from . import cost_model
from .arch import ArchSpec, from_hf_config

MB = 1024.0 ** 2

# Published small models worth pricing. They span a wide range of cache
# geometries at similar parameter counts, which is what makes them a useful
# test of the argument: at roughly 1.5B parameters this list already contains a
# 1.5x spread in bytes moved per token.
# All of these load without an access request. Gated families (Llama, Gemma)
# work too once you have accepted their licence and run `huggingface-cli login`;
# add them with --hf-models. Models that cannot be fetched are skipped with a
# message rather than crashing the run.
DEFAULT_HF_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen2.5-0.5B",
    "EleutherAI/pythia-410m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "allenai/OLMo-2-0425-1B",
    "Qwen/Qwen2.5-1.5B",
    "HuggingFaceTB/SmolLM2-1.7B",
    "Qwen/Qwen2.5-3B",
    "microsoft/Phi-3-mini-4k-instruct",
]


def price_hf_models(
    model_ids: Sequence[str],
    contexts: Sequence[int] = (512, 2048, 8192),
    bandwidth_gbs: float = 68.0,
    bits_w: Optional[float] = None,
    bits_kv: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Run the cost model over published configs.

    The interesting column is `kv_fraction`: it shows the crossover context at
    which cache traffic overtakes weight traffic, which is different for every
    model and is invisible in a parameter count.
    """
    from transformers import AutoConfig

    rows: List[Dict[str, object]] = []
    for model_id in model_ids:
        try:
            config = AutoConfig.from_pretrained(model_id)
        except Exception as error:  # offline, gated repo, or unknown id
            print(f"  skipped {model_id}: {type(error).__name__}: {error}")
            continue
        spec = from_hf_config(config, name=model_id)
        overrides = {}
        if bits_w is not None:
            overrides["bits_w"] = bits_w
        if bits_kv is not None:
            overrides["bits_kv"] = bits_kv
        if overrides:
            spec = spec.with_(**overrides)
        for context in contexts:
            row = cost_model.summarize(spec, context, bandwidth_gbs)
            row["kv_per_context_token_kb"] = (
                cost_model.kv_bytes_per_token(spec, 1) / 1024.0
            )
            rows.append(row)
        print(f"  {spec.describe()}")
    return rows


def collect_runs(run_dir: str = "results/runs", context_len: int = 2048) -> List[Dict[str, object]]:
    """Turn training run records into table rows."""
    rows: List[Dict[str, object]] = []
    if not os.path.isdir(run_dir):
        print(f"no run directory at {run_dir}")
        return rows
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(run_dir, name)) as handle:
            record = json.load(handle)
        spec = ArchSpec(**record["spec"])
        budget = cost_model.bytes_per_token(spec, context_len)
        rows.append(
            {
                "shape": spec.name,
                "params_m": round(record["params_m"], 1),
                "val_ppl": round(record["final"]["val_ppl"], 3),
                "quality": round(record["quality"], 4),
                "bytes_per_token_mb": round(budget.total / MB, 2),
                "kv_fraction": round(budget.kv_fraction, 3),
                "quality_per_gb": round(
                    cost_model.quality_per_byte(record["quality"], spec, context_len), 4
                ),
                "tokens_seen_m": round(record["tokens_seen"] / 1e6, 1),
                "wall_clock_min": round(record["wall_clock_s"] / 60.0, 1),
            }
        )
    return rows


def write_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    if not rows:
        print(f"nothing to write to {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def print_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    if not rows:
        return
    widths = {c: max(len(c), *(len(f"{row.get(c, '')}") for row in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3g}"
            values.append(str(value).ljust(widths[column]))
        print("  ".join(values))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the paper's tables")
    parser.add_argument("--hf-models", nargs="*", default=None,
                        help="price published configs (default list if the flag is bare)")
    parser.add_argument("--runs", default=None, help="aggregate training runs from this directory")
    parser.add_argument("--contexts", nargs="*", type=int, default=[512, 2048, 8192])
    parser.add_argument("--bandwidth-gbs", type=float, default=68.0)
    parser.add_argument("--bits-w", type=float, default=None, help="e.g. 4.5 for 4-bit groups")
    parser.add_argument("--bits-kv", type=float, default=None)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    did_something = False

    if args.hf_models is not None:
        model_ids = args.hf_models or DEFAULT_HF_MODELS
        print(f"pricing {len(model_ids)} published configs at {args.bandwidth_gbs:.0f} GB/s")
        rows = price_hf_models(
            model_ids,
            contexts=tuple(args.contexts),
            bandwidth_gbs=args.bandwidth_gbs,
            bits_w=args.bits_w,
            bits_kv=args.bits_kv,
        )
        if rows:
            print()
            print_table(
                rows,
                [
                    "name",
                    "params_m",
                    "context_len",
                    "weight_mb",
                    "kv_mb",
                    "bytes_per_token_mb",
                    "kv_fraction",
                    "pred_tokens_per_s",
                ],
            )
            write_csv(rows, os.path.join(args.out_dir, "offtheshelf_cost.csv"))
        did_something = True

    if args.runs:
        rows = collect_runs(args.runs, context_len=max(args.contexts))
        if rows:
            print()
            print_table(
                rows,
                [
                    "shape",
                    "params_m",
                    "val_ppl",
                    "bytes_per_token_mb",
                    "kv_fraction",
                    "quality_per_gb",
                ],
            )
            write_csv(rows, os.path.join(args.out_dir, "trained_shapes.csv"))
            best = max(rows, key=lambda r: r["quality_per_gb"])
            print(f"\nbest quality per byte: {best['shape']} "
                  f"({best['quality_per_gb']} per GB per token)")
        did_something = True

    if not did_something:
        parser.print_help()


if __name__ == "__main__":
    main()
