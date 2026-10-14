"""The bandwidth-first selector.

Given a device, a context length distribution and a latency target, enumerate
architecture shapes, keep the ones that fit the implied bytes-per-token budget,
and return the feasible shape with the highest predicted quality.

The quality predictor here is a stand-in: a log-parameter law with small
corrections for aspect ratio and cache compression. It exists so the search
machinery can be exercised on day one. Replace `QualityModel` with a curve
fitted to measured validation loss from the training grid as soon as those runs
exist, and report both the proxy and the fitted version so the difference is
visible.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import cost_model
from .arch import ArchSpec

GB = 1024.0 ** 3
MB = 1024.0 ** 2


# --------------------------------------------------------------------------- #
# Quality side of the trade-off
# --------------------------------------------------------------------------- #

@dataclass
class QualityModel:
    """A placeholder quality predictor, to be replaced by a fitted curve.

    Form: quality grows with the log of non-embedding parameters, with a mild
    penalty for extreme aspect ratios, a penalty for aggressive KV compression
    and a penalty for low weight precision. Coefficients are plausible rather
    than measured, and `fitted` records which is which so a figure made from a
    proxy can never be mistaken for a figure made from data.
    """

    a: float = 0.0
    b: float = 0.115
    aspect_penalty: float = 0.004
    aspect_target: float = 22.0
    latent_penalty: float = 0.03
    bits_penalty: float = 0.055
    bits_reference: float = 16.0
    fitted: bool = False

    def __call__(self, spec: ArchSpec) -> float:
        parts = cost_model.param_breakdown(spec)
        non_embedding = max(parts["total"] - parts["embedding"] - parts["lm_head"], 1)
        score = self.a + self.b * math.log10(non_embedding)
        score -= self.aspect_penalty * abs(spec.depth_width_ratio - self.aspect_target)
        if spec.kv_latent_rank:
            full = 2 * spec.n_kv_head * spec.head_dim
            compression = full / max(spec.kv_latent_rank, 1)
            score -= self.latent_penalty * math.log2(max(compression, 1.0))
        if spec.bits_w < self.bits_reference:
            score -= self.bits_penalty * math.log2(self.bits_reference / spec.bits_w)
        return score

    @staticmethod
    def fit_from_runs(run_dir: str = "results/runs") -> "QualityModel":
        """Fit `a` and `b` by least squares on measured runs.

        Needs at least three completed training runs. The returned model has
        `fitted=True`, which is what the paper should report.
        """
        points: List[Tuple[float, float]] = []
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"no run directory at {run_dir}")
        for name in sorted(os.listdir(run_dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(run_dir, name)) as handle:
                record = json.load(handle)
            spec = ArchSpec(**record["spec"])
            parts = cost_model.param_breakdown(spec)
            non_embedding = max(parts["total"] - parts["embedding"] - parts["lm_head"], 1)
            points.append((math.log10(non_embedding), float(record["quality"])))

        if len(points) < 3:
            raise ValueError(f"need at least 3 runs to fit, found {len(points)}")
        n = len(points)
        mean_x = sum(x for x, _ in points) / n
        mean_y = sum(y for _, y in points) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in points)
        var = sum((x - mean_x) ** 2 for x, _ in points)
        slope = cov / var if var else 0.0
        return QualityModel(a=mean_y - slope * mean_x, b=slope, fitted=True)


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #

@dataclass
class SearchSpace:
    """The shape knobs to enumerate. Keep the grid small and interpretable: the
    point is to show that the binding constraint changes the answer, not to win
    a search benchmark."""

    n_layers: Sequence[int] = (12, 16, 24, 32)
    d_models: Sequence[int] = (512, 768, 1024, 1536, 2048)
    n_kv_heads: Sequence[int] = (1, 2, 4, 8)
    head_dims: Sequence[int] = (48, 64, 96)
    kv_latent_ranks: Sequence[Optional[int]] = (None, 128, 256)
    weight_bits: Sequence[float] = (16.0, 8.0, 4.5)
    kv_bits: Sequence[float] = (16.0, 8.0)
    ffn_multiple: float = 2.67
    vocab_size: int = 49152
    min_params_m: float = 20.0
    max_params_m: float = 2000.0

    def enumerate(self) -> Iterable[ArchSpec]:
        combos = itertools.product(
            self.n_layers,
            self.d_models,
            self.n_kv_heads,
            self.head_dims,
            self.kv_latent_ranks,
            self.weight_bits,
            self.kv_bits,
        )
        for n_layer, d_model, n_kv_head, head_dim, latent, bits_w, bits_kv in combos:
            n_head = max(1, d_model // head_dim)
            if n_head % n_kv_head != 0 or n_head < n_kv_head:
                continue
            if latent is not None and latent >= 2 * n_kv_head * head_dim:
                continue  # a latent larger than the cache it replaces is pointless
            d_ff = int(round(d_model * self.ffn_multiple / 64)) * 64
            spec = ArchSpec(
                name=(
                    f"L{n_layer}_d{d_model}_kv{n_kv_head}_hd{head_dim}"
                    f"_r{latent or 0}_w{bits_w:g}_kvb{bits_kv:g}"
                ),
                n_layer=n_layer,
                d_model=d_model,
                n_head=n_head,
                n_kv_head=n_kv_head,
                d_ff=d_ff,
                vocab_size=self.vocab_size,
                d_head=head_dim,
                kv_latent_rank=latent,
                bits_w=bits_w,
                bits_kv=bits_kv,
            )
            params_m = cost_model.total_params(spec) / 1e6
            if self.min_params_m <= params_m <= self.max_params_m:
                yield spec


# --------------------------------------------------------------------------- #
# Constraints and selection
# --------------------------------------------------------------------------- #

@dataclass
class DeviceConstraint:
    """The deployment target, stated in the units that bind."""

    achievable_gbs: float
    engine_efficiency: float = 0.7
    target_tokens_per_s: float = 30.0
    context_lens: Sequence[int] = (512, 2048, 4096)
    context_weights: Optional[Sequence[float]] = None
    memory_limit_gb: Optional[float] = 2.0

    @property
    def byte_budget(self) -> float:
        return cost_model.byte_budget_for_slo(
            self.target_tokens_per_s, self.achievable_gbs, self.engine_efficiency
        )

    def weights(self) -> Sequence[float]:
        if self.context_weights:
            total = sum(self.context_weights)
            return [w / total for w in self.context_weights]
        return [1.0 / len(self.context_lens)] * len(self.context_lens)

    def expected_bytes(self, spec: ArchSpec) -> float:
        """Bytes per token averaged over the context length distribution. A
        selector tuned to a single context length overfits to it, and real
        traffic is a mixture."""
        return sum(
            weight * cost_model.bytes_per_token(spec, context).total
            for context, weight in zip(self.context_lens, self.weights())
        )

    def worst_case_bytes(self, spec: ArchSpec) -> float:
        return cost_model.bytes_per_token(spec, max(self.context_lens)).total

    def feasible(self, spec: ArchSpec) -> bool:
        if self.expected_bytes(spec) > self.byte_budget:
            return False
        if self.memory_limit_gb is not None:
            resident = cost_model.resident_bytes(spec, max(self.context_lens))
            if resident > self.memory_limit_gb * GB:
                return False
        return True


@dataclass
class Candidate:
    spec: ArchSpec
    quality: float
    expected_bytes_mb: float
    worst_case_bytes_mb: float
    params_m: float
    pred_tokens_per_s: float
    feasible: bool

    def as_row(self) -> Dict[str, object]:
        row = {key: value for key, value in asdict(self.spec).items()}
        row.update(
            {
                "quality": self.quality,
                "expected_bytes_mb": self.expected_bytes_mb,
                "worst_case_bytes_mb": self.worst_case_bytes_mb,
                "params_m": self.params_m,
                "pred_tokens_per_s": self.pred_tokens_per_s,
                "feasible": self.feasible,
            }
        )
        return row


def score_candidates(
    space: SearchSpace,
    constraint: DeviceConstraint,
    quality: Optional[QualityModel] = None,
) -> List[Candidate]:
    quality = quality or QualityModel()
    candidates = []
    for spec in space.enumerate():
        expected = constraint.expected_bytes(spec)
        worst = constraint.worst_case_bytes(spec)
        candidates.append(
            Candidate(
                spec=spec,
                quality=quality(spec),
                expected_bytes_mb=expected / MB,
                worst_case_bytes_mb=worst / MB,
                params_m=cost_model.total_params(spec) / 1e6,
                pred_tokens_per_s=constraint.achievable_gbs
                * 1e9
                * constraint.engine_efficiency
                / expected,
                feasible=constraint.feasible(spec),
            )
        )
    return candidates


SelectionRule = Callable[[List[Candidate], DeviceConstraint], Candidate]


def bandwidth_first(candidates: List[Candidate], constraint: DeviceConstraint) -> Candidate:
    """Ours: best predicted quality among shapes that fit the byte budget."""
    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        raise ValueError("no shape fits this budget, relax the SLO or the memory limit")
    return max(feasible, key=lambda c: c.quality)


def parameter_count_first(
    candidates: List[Candidate], constraint: DeviceConstraint
) -> Candidate:
    """The common practice baseline: pick the largest model whose weights fit
    in memory at the chosen precision, ignoring cache traffic entirely."""
    budget_params_m = (
        (constraint.memory_limit_gb or 2.0) * GB * 8.0 / 16.0 / 1e6
        if constraint.memory_limit_gb
        else 1e9
    )
    affordable = [c for c in candidates if c.params_m <= budget_params_m]
    pool = affordable or candidates
    return max(pool, key=lambda c: c.params_m)


def flop_first(candidates: List[Candidate], constraint: DeviceConstraint) -> Candidate:
    """A compute-centric baseline: maximize quality subject to a flops-per-token
    ceiling, which is the right rule for prefill and the wrong one for decode."""
    costs = sorted(
        cost_model.decode_flops_per_token(c.spec, max(constraint.context_lens))
        for c in candidates
    )
    ceiling = costs[len(costs) // 2]  # median arithmetic cost as the budget
    affordable = [
        c
        for c in candidates
        if cost_model.decode_flops_per_token(c.spec, max(constraint.context_lens)) <= ceiling
    ]
    pool = affordable or candidates
    return max(pool, key=lambda c: c.quality)


RULES: Dict[str, SelectionRule] = {
    "bandwidth-first": bandwidth_first,
    "parameter-count": parameter_count_first,
    "flop-first": flop_first,
}


def compare_rules(
    space: SearchSpace,
    constraint: DeviceConstraint,
    quality: Optional[QualityModel] = None,
) -> Dict[str, Dict[str, object]]:
    """Run every rule on the same candidate set.

    Two numbers matter per rule: whether its pick meets the latency target at
    all, and how much quality it gives up when it does. That pair is the
    paper's selector table. A negative regret is not a win: it means the rule
    selected a shape outside the byte budget, buying quality it cannot deliver
    at the target latency.
    """
    candidates = score_candidates(space, constraint, quality)
    best_feasible_quality = max(
        (c.quality for c in candidates if c.feasible), default=float("nan")
    )
    out: Dict[str, Dict[str, object]] = {}
    for name, rule in RULES.items():
        pick = rule(candidates, constraint)
        out[name] = {
            "shape": pick.spec.describe(),
            "params_m": round(pick.params_m, 1),
            "expected_bytes_mb": round(pick.expected_bytes_mb, 1),
            "pred_tokens_per_s": round(pick.pred_tokens_per_s, 1),
            "meets_slo": bool(pick.pred_tokens_per_s >= constraint.target_tokens_per_s),
            "quality": round(pick.quality, 4),
            "quality_regret": round(best_feasible_quality - pick.quality, 4),
        }
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Select an architecture under a bandwidth budget")
    parser.add_argument("--bandwidth-gbs", type=float, default=68.0, help="achievable, measured")
    parser.add_argument("--engine-efficiency", type=float, default=0.7)
    parser.add_argument("--target-tokens-per-s", type=float, default=30.0)
    parser.add_argument("--contexts", nargs="*", type=int, default=[512, 2048, 4096])
    parser.add_argument("--context-weights", nargs="*", type=float, default=None)
    parser.add_argument("--memory-limit-gb", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fit-from", default=None, help="run dir to fit the quality model")
    parser.add_argument("--out", default="results/selection.json")
    args = parser.parse_args()

    constraint = DeviceConstraint(
        achievable_gbs=args.bandwidth_gbs,
        engine_efficiency=args.engine_efficiency,
        target_tokens_per_s=args.target_tokens_per_s,
        context_lens=tuple(args.contexts),
        context_weights=args.context_weights,
        memory_limit_gb=args.memory_limit_gb,
    )

    quality = QualityModel()
    if args.fit_from:
        quality = QualityModel.fit_from_runs(args.fit_from)
        print(f"fitted quality model from {args.fit_from}: a={quality.a:.4f} b={quality.b:.4f}")
    else:
        print("using the placeholder quality model, fit it with --fit-from once runs exist")

    space = SearchSpace()
    candidates = score_candidates(space, constraint, quality)
    feasible = [c for c in candidates if c.feasible]

    print(f"\ndevice: {constraint.achievable_gbs:.0f} GB/s achievable, "
          f"engine efficiency {constraint.engine_efficiency:.2f}")
    print(f"latency target: {constraint.target_tokens_per_s:.0f} tok/s")
    print(f"implied byte budget: {constraint.byte_budget / MB:.1f} MB per token")
    print(f"candidates: {len(candidates)}, feasible: {len(feasible)}")

    print(f"\ntop {args.top_k} feasible shapes by predicted quality:")
    for candidate in sorted(feasible, key=lambda c: -c.quality)[: args.top_k]:
        print(
            f"  {candidate.spec.describe()}\n"
            f"     {candidate.params_m:6.1f}M params  "
            f"{candidate.expected_bytes_mb:6.1f} MB/tok  "
            f"{candidate.pred_tokens_per_s:6.1f} tok/s  q={candidate.quality:.4f}"
        )

    print("\nselection rules on the same candidate set:")
    comparison = compare_rules(space, constraint, quality)
    for name, row in comparison.items():
        flag = "meets SLO" if row["meets_slo"] else "MISSES SLO"
        print(
            f"  {name:<16} {row['pred_tokens_per_s']:>7.1f} tok/s  {flag:<10} "
            f"regret={row['quality_regret']:+.4f}  {row['shape']}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(
            {
                "constraint": asdict(constraint),
                "byte_budget_mb": constraint.byte_budget / MB,
                "quality_model": asdict(quality),
                "n_candidates": len(candidates),
                "n_feasible": len(feasible),
                "rules": comparison,
                "top_feasible": [
                    c.as_row() for c in sorted(feasible, key=lambda c: -c.quality)[: args.top_k]
                ],
            },
            handle,
            indent=2,
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
