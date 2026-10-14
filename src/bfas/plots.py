"""Figures from measured results.

These produce the two panels the paper leans on, from your CSV files rather
than from placeholders. Run them after `benchmark.py` has written
`results/decode_benchmark.csv`.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#1b6ca8"
ACCENT_D = "#0f4c78"
GREY = "#8c8c8c"
INK = "#1a1a1a"


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path) as handle:
        return list(csv.DictReader(handle))


def predictor_comparison(
    rows: Sequence[Dict[str, str]], context_len: int, out_path: str
) -> None:
    """The headline figure: throughput against parameter count, and against
    bytes per token. If the second panel is tight and the first is not, the
    paper's central claim holds on your hardware."""
    subset = [row for row in rows if int(row["context_len"]) == context_len]
    if len(subset) < 3:
        print(f"not enough rows at context {context_len}")
        return

    params = [float(row["params_m"]) for row in subset]
    bytes_mb = [float(row["pred_bytes_per_token_mb"]) for row in subset]
    tok_s = [float(row["tokens_per_s"]) for row in subset]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), dpi=200)
    axes[0].scatter(params, tok_s, s=30, color=GREY, edgecolors=INK, linewidths=0.5)
    axes[0].set_xlabel("parameters (M)")
    axes[0].set_ylabel("decode throughput (tokens/s)")
    axes[0].set_title("by parameter count")

    axes[1].scatter(bytes_mb, tok_s, s=30, color=ACCENT, edgecolors=ACCENT_D, linewidths=0.5)
    axes[1].set_xlabel("bytes per token (MB)")
    axes[1].set_title("by bytes per token")

    for ax in axes:
        ax.set_ylim(bottom=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"decode throughput at context {context_len}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    print(f"wrote {out_path}")


def context_scaling(rows: Sequence[Dict[str, str]], out_path: str) -> None:
    """Throughput against context length, one line per shape. The slope is the
    cache term, and shapes with compressed caches should visibly flatten."""
    by_shape: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_shape.setdefault(row["name"], []).append(row)

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    for index, (name, shape_rows) in enumerate(sorted(by_shape.items())):
        shape_rows = sorted(shape_rows, key=lambda r: int(r["context_len"]))
        ax.plot(
            [int(r["context_len"]) for r in shape_rows],
            [float(r["tokens_per_s"]) for r in shape_rows],
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            color=ACCENT if index % 2 == 0 else GREY,
            label=name,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length")
    ax.set_ylabel("decode throughput (tokens/s)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6.5, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot measured results")
    parser.add_argument("--rows", default="results/decode_benchmark.csv")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    rows = read_rows(args.rows)
    predictor_comparison(rows, args.context, os.path.join(args.out_dir, "predictors.png"))
    context_scaling(rows, os.path.join(args.out_dir, "context_scaling.png"))


if __name__ == "__main__":
    main()
