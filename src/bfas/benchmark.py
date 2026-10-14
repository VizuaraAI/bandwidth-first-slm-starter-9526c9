"""Measured decode throughput, and the comparison against the prediction.

Three things are measured per (shape, context) pair:

    tokens per second        what the user feels
    achieved GB/s            tokens/s x predicted bytes per token
    roofline efficiency      achieved GB/s / achievable GB/s

The third number is the honesty check. Values far below one mean the engine is
not saturating memory, so any architecture conclusion drawn from that run is a
statement about the engine and not about the shape.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import torch

from . import cost_model
from .arch import ArchSpec, REFERENCE_SHAPES
from .bandwidth import DeviceProfile, measure_host_bandwidth
from .model import BFASModel, build_model

GB = 1024.0 ** 3


@dataclass
class DecodeMeasurement:
    name: str
    context_len: int
    new_tokens: int
    tokens_per_s: float
    ms_per_token: float
    prefill_ms: float
    pred_bytes_per_token_mb: float
    achieved_gbs: float
    roofline_efficiency: float
    params_m: float
    device: str

    def row(self) -> Dict[str, object]:
        return asdict(self)


@torch.no_grad()
def measure_decode(
    model: BFASModel,
    context_len: int = 512,
    new_tokens: int = 32,
    device: str = "cpu",
    warmup: int = 1,
    repeats: int = 3,
    achievable_gbs: Optional[float] = None,
) -> DecodeMeasurement:
    """Time prefill and per-token decode separately.

    They are separated because they are bound by different things: prefill is
    compute-bound and parallel over the prompt, decode is bandwidth-bound and
    serial. Mixing them into one latency number is how the bandwidth signal
    gets hidden.
    """
    spec = model.spec
    model.eval().to(device)
    prompt = torch.randint(0, spec.vocab_size, (1, context_len), device=device)

    for _ in range(warmup):
        model.generate(prompt, max_new_tokens=4)

    best_decode = float("inf")
    best_prefill = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        out = model(prompt)
        caches = out["caches"]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        prefill = time.perf_counter() - start
        best_prefill = min(best_prefill, prefill)

        next_id = out["logits"][:, -1:, :].argmax(dim=-1)
        start = time.perf_counter()
        for _ in range(new_tokens):
            step = model(next_id, caches=caches)
            caches = step["caches"]
            next_id = step["logits"][:, -1:, :].argmax(dim=-1)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        best_decode = min(best_decode, time.perf_counter() - start)

    tokens_per_s = new_tokens / best_decode
    predicted = cost_model.bytes_per_token(spec, context_len).total
    achieved_gbs = tokens_per_s * predicted / 1e9
    efficiency = (
        achieved_gbs / achievable_gbs if achievable_gbs else float("nan")
    )
    return DecodeMeasurement(
        name=spec.name,
        context_len=context_len,
        new_tokens=new_tokens,
        tokens_per_s=tokens_per_s,
        ms_per_token=1000.0 * best_decode / new_tokens,
        prefill_ms=1000.0 * best_prefill,
        pred_bytes_per_token_mb=predicted / (1024.0 ** 2),
        achieved_gbs=achieved_gbs,
        roofline_efficiency=efficiency,
        params_m=cost_model.total_params(spec) / 1e6,
        device=device,
    )


def sweep(
    specs: Sequence[ArchSpec],
    contexts: Sequence[int] = (128, 512, 2048),
    new_tokens: int = 32,
    device: str = "cpu",
    dtype: str = "float32",
    achievable_gbs: Optional[float] = None,
) -> List[DecodeMeasurement]:
    """Benchmark every shape at every context. This produces the table the
    paper's main result is built from."""
    rows: List[DecodeMeasurement] = []
    max_context = max(contexts) + new_tokens + 8
    for spec in specs:
        model = build_model(spec, max_seq_len=max_context, dtype=dtype)
        for context_len in contexts:
            row = measure_decode(
                model,
                context_len=context_len,
                new_tokens=new_tokens,
                device=device,
                achievable_gbs=achievable_gbs,
            )
            rows.append(row)
            print(
                f"{spec.name:<20} ctx={context_len:<6} "
                f"{row.tokens_per_s:>7.1f} tok/s  "
                f"{row.pred_bytes_per_token_mb:>7.1f} MB/tok  "
                f"BRE={row.roofline_efficiency:.2f}"
            )
        del model
    return rows


def fit_efficiency(rows: Sequence[DecodeMeasurement], achievable_gbs: float) -> float:
    """One fitted constant: the fraction of achievable bandwidth this engine
    reaches, averaged over shapes. Fit it once, then use the cost model to
    predict shapes you have not built yet."""
    values = [row.achieved_gbs / achievable_gbs for row in rows]
    return sum(values) / len(values) if values else float("nan")


def rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman correlation, written out so there is no scipy dependency.

    The headline comparison of the paper is two of these: parameter count
    against throughput, and bytes per token against throughput.
    """
    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2.0 + 1.0
            for position in range(index, stop + 1):
                out[order[position]] = average
            index = stop + 1
        return out

    if len(xs) < 3:
        return float("nan")
    rank_x, rank_y = ranks(xs), ranks(ys)
    n = len(xs)
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    var_x = sum((a - mean_x) ** 2 for a in rank_x) ** 0.5
    var_y = sum((b - mean_y) ** 2 for b in rank_y) ** 0.5
    return cov / (var_x * var_y) if var_x and var_y else float("nan")


def write_rows(rows: Sequence[DecodeMeasurement], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].row().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.row())
    print(f"wrote {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark decode throughput by shape")
    parser.add_argument("--shapes", nargs="*", default=list(REFERENCE_SHAPES))
    parser.add_argument("--contexts", nargs="*", type=int, default=[128, 512, 2048])
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--vocab-size", type=int, default=None, help="shrink for quick runs")
    parser.add_argument("--profile", default="results/device_profile.json")
    parser.add_argument("--out", default="results/decode_benchmark.csv")
    args = parser.parse_args()

    if os.path.exists(args.profile):
        achievable = DeviceProfile.from_json(args.profile).achievable_gbs
        print(f"using measured bandwidth from {args.profile}: {achievable:.1f} GB/s")
    else:
        achievable = measure_host_bandwidth(256.0, 5).achievable_gbs
        print(f"measured bandwidth now: {achievable:.1f} GB/s")

    specs = [REFERENCE_SHAPES[name] for name in args.shapes]
    if args.vocab_size:
        specs = [spec.with_(vocab_size=args.vocab_size) for spec in specs]

    rows = sweep(
        specs,
        contexts=tuple(args.contexts),
        new_tokens=args.new_tokens,
        device=args.device,
        dtype=args.dtype,
        achievable_gbs=achievable,
    )
    write_rows(rows, args.out)

    print("\nrank correlation with measured tokens/s (higher magnitude is a better predictor):")
    for context_len in args.contexts:
        subset = [row for row in rows if row.context_len == context_len]
        if len(subset) < 3:
            continue
        by_params = rank_correlation(
            [row.params_m for row in subset], [row.tokens_per_s for row in subset]
        )
        by_bytes = rank_correlation(
            [row.pred_bytes_per_token_mb for row in subset],
            [row.tokens_per_s for row in subset],
        )
        print(
            f"  ctx={context_len:<6} parameters: {by_params:+.2f}   "
            f"bytes per token: {by_bytes:+.2f}"
        )

    fitted = fit_efficiency(rows, achievable)
    print(f"\nfitted engine efficiency: {fitted:.3f} of achievable bandwidth")
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(
            {"achievable_gbs": achievable, "fitted_efficiency": fitted},
            handle,
            indent=2,
        )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
