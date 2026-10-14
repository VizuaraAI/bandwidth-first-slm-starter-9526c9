"""The analytic bytes-per-token cost model.

This is the centre of the project. One decoded token requires the engine to
read every weight it needs and every cached key/value it attends to. At batch
size one there is almost no arithmetic to hide that traffic behind, so

    seconds per token  ~=  bytes moved per token / achievable bandwidth

Parameter count only predicts the first of the two terms, and only when every
weight is stored at the same precision. Everything in this module exists to
make the full byte count explicit and checkable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from .arch import ArchSpec

GB = 1024.0 ** 3
MB = 1024.0 ** 2


# --------------------------------------------------------------------------- #
# Parameter accounting
# --------------------------------------------------------------------------- #

def param_breakdown(spec: ArchSpec) -> Dict[str, int]:
    """Parameters by component, counted the way the model is actually built."""
    d, h, hd = spec.d_model, spec.n_head, spec.head_dim
    q = d * h * hd
    o = h * hd * d
    if spec.kv_latent_rank:
        # Down-project the residual into one shared latent, then up-project it
        # into per-head keys and values.
        kv = d * spec.kv_latent_rank + spec.kv_latent_rank * 2 * spec.n_kv_head * hd
    else:
        kv = 2 * d * spec.n_kv_head * hd
    attn = q + kv + o
    # SwiGLU feed-forward: gate, up, down.
    ffn_per_expert = 3 * d * spec.d_ff
    ffn = ffn_per_expert * spec.n_expert
    router = d * spec.n_expert if spec.n_expert > 1 else 0
    norms = 2 * d  # two RMSNorms per block
    per_layer = attn + ffn + router + norms

    embed = spec.vocab_size * d
    head = 0 if spec.tie_embeddings else spec.vocab_size * d
    return {
        "embedding": embed,
        "lm_head": head,
        "attn": attn * spec.n_layer,
        "ffn": ffn * spec.n_layer,
        "router": router * spec.n_layer,
        "norms": norms * spec.n_layer + d,
        "per_layer": per_layer,
        "total": embed + head + per_layer * spec.n_layer + d,
    }


def total_params(spec: ArchSpec) -> int:
    return param_breakdown(spec)["total"]


def active_params(spec: ArchSpec) -> int:
    """Parameters whose bytes are actually read to produce one token.

    Three corrections to the headline parameter count, and they are the reason
    parameter count misranks models:

    1. The input embedding table is a lookup. One row is read, not the table.
    2. The output head is read in full, every token, for every vocabulary
       entry. Large-vocabulary small models pay a disproportionate price here.
    3. In a mixture of experts only the routed experts are read, even though
       all of them must be resident in memory.
    """
    d = spec.d_model
    parts = param_breakdown(spec)
    dense_ffn_per_layer = 3 * d * spec.d_ff * spec.n_expert_active
    attn_per_layer = (parts["attn"] // spec.n_layer) if spec.n_layer else 0
    router_per_layer = (parts["router"] // spec.n_layer) if spec.n_layer else 0
    per_layer = attn_per_layer + dense_ffn_per_layer + router_per_layer + 2 * d
    head = spec.vocab_size * d  # tied or not, the head matrix is read
    embed_row = d
    return head + embed_row + per_layer * spec.n_layer + d


# --------------------------------------------------------------------------- #
# Byte accounting
# --------------------------------------------------------------------------- #

@dataclass
class ByteBudget:
    """Bytes read to decode one token at a given context length."""

    weight_bytes: float
    kv_bytes: float
    activation_bytes: float
    context_len: int

    @property
    def total(self) -> float:
        return self.weight_bytes + self.kv_bytes + self.activation_bytes

    @property
    def kv_fraction(self) -> float:
        return self.kv_bytes / self.total if self.total else 0.0

    def as_dict(self) -> Dict[str, float]:
        out = asdict(self)
        out["total"] = self.total
        out["kv_fraction"] = self.kv_fraction
        out["total_gb"] = self.total / GB
        return out


def weight_bytes_per_token(spec: ArchSpec) -> float:
    return active_params(spec) * spec.bits_w / 8.0


def kv_bytes_per_token(spec: ArchSpec, context_len: int) -> float:
    """Cache traffic for one decode step at `context_len` tokens of history.

    The whole cache is streamed once per step, so this grows linearly in
    context. A low-rank joint cache stores one latent vector per layer per
    token instead of separate keys and values, which removes the factor of two
    and decouples the cache from head count.
    """
    if context_len <= 0:
        return 0.0
    if spec.kv_latent_rank:
        elements = context_len * spec.n_layer * spec.kv_latent_rank
    else:
        elements = 2 * context_len * spec.n_layer * spec.n_kv_head * spec.head_dim
    return elements * spec.bits_kv / 8.0


def activation_bytes_per_token(spec: ArchSpec) -> float:
    """Residual stream and logit traffic. Small, but not zero: the logit vector
    alone is `vocab_size` floats, which is a megabyte at a 150k vocabulary."""
    residual = spec.n_layer * 4 * spec.d_model * 4.0  # fp32 reads/writes per block
    logits = spec.vocab_size * 4.0
    return residual + logits


def bytes_per_token(spec: ArchSpec, context_len: int = 1024) -> ByteBudget:
    return ByteBudget(
        weight_bytes=weight_bytes_per_token(spec),
        kv_bytes=kv_bytes_per_token(spec, context_len),
        activation_bytes=activation_bytes_per_token(spec),
        context_len=context_len,
    )


def resident_bytes(spec: ArchSpec, context_len: int = 1024) -> float:
    """Peak memory footprint: all weights plus the full cache. A shape can fit
    in RAM and still be too slow, or be fast and not fit. Both constraints are
    reported so they are never confused."""
    weights = total_params(spec) * spec.bits_w / 8.0
    return weights + kv_bytes_per_token(spec, context_len)


# --------------------------------------------------------------------------- #
# Compute accounting, for the contrast with prefill
# --------------------------------------------------------------------------- #

def decode_flops_per_token(spec: ArchSpec, context_len: int = 1024) -> float:
    """Multiply-accumulate pairs counted as two flops."""
    mac = active_params(spec) - spec.d_model
    attn_scores = 2.0 * spec.n_layer * spec.n_head * spec.head_dim * context_len
    return 2.0 * mac + 2.0 * attn_scores


def arithmetic_intensity(spec: ArchSpec, context_len: int = 1024) -> float:
    """Flops per byte. Decode lands far to the left of the roofline ridge on
    every accelerator, which is the formal statement of the premise."""
    budget = bytes_per_token(spec, context_len)
    return decode_flops_per_token(spec, context_len) / budget.total


# --------------------------------------------------------------------------- #
# Roofline predictions
# --------------------------------------------------------------------------- #

def roofline_tokens_per_s(
    spec: ArchSpec,
    context_len: int,
    bandwidth_gbs: float,
    efficiency: float = 1.0,
) -> float:
    """Predicted decode throughput.

    `bandwidth_gbs` should be the *achievable* bandwidth measured by
    `bfas.bandwidth`, not the number on the spec sheet. `efficiency` is the
    one free parameter of the model: the fraction of achievable bandwidth the
    runtime reaches. Fit it once per (device, runtime) pair, then hold it fixed
    across architectures. If a single fitted constant explains throughput
    across shapes, the cost model is doing real work.
    """
    budget = bytes_per_token(spec, context_len)
    bytes_per_s = bandwidth_gbs * 1e9 * efficiency
    return bytes_per_s / budget.total


def time_per_output_token_ms(
    spec: ArchSpec, context_len: int, bandwidth_gbs: float, efficiency: float = 1.0
) -> float:
    return 1000.0 / roofline_tokens_per_s(spec, context_len, bandwidth_gbs, efficiency)


def byte_budget_for_slo(
    target_tokens_per_s: float, bandwidth_gbs: float, efficiency: float = 1.0
) -> float:
    """Invert the roofline: the bytes-per-token ceiling implied by a latency
    service level objective on a given device. This number, not a parameter
    count, is what the selector is handed."""
    return bandwidth_gbs * 1e9 * efficiency / target_tokens_per_s


def quality_per_byte(quality: float, spec: ArchSpec, context_len: int = 1024) -> float:
    """Quality per gigabyte of traffic per token. Use any higher-is-better
    quality score; for perplexity pass a converted score such as
    `1.0 / log(ppl)` so the metric stays higher-is-better."""
    return quality / (bytes_per_token(spec, context_len).total / GB)


def bandwidth_roofline_efficiency(
    measured_tokens_per_s: float,
    spec: ArchSpec,
    context_len: int,
    achievable_bandwidth_gbs: float,
) -> float:
    """Fraction of achievable bandwidth the running engine actually uses.

    This is the diagnostic that separates a shape problem from an engine
    problem. A low value means the runtime is leaving bandwidth on the table;
    a high value means the shape itself is the limit and only the byte count
    can be changed.
    """
    achieved = measured_tokens_per_s * bytes_per_token(spec, context_len).total
    return achieved / (achievable_bandwidth_gbs * 1e9)


def summarize(spec: ArchSpec, context_len: int, bandwidth_gbs: float) -> Dict[str, float]:
    """One row of the paper's main table."""
    budget = bytes_per_token(spec, context_len)
    return {
        "name": spec.name,
        "params_m": total_params(spec) / 1e6,
        "active_params_m": active_params(spec) / 1e6,
        "context_len": context_len,
        "weight_mb": budget.weight_bytes / MB,
        "kv_mb": budget.kv_bytes / MB,
        "bytes_per_token_mb": budget.total / MB,
        "kv_fraction": budget.kv_fraction,
        "resident_mb": resident_bytes(spec, context_len) / MB,
        "arithmetic_intensity": arithmetic_intensity(spec, context_len),
        "pred_tokens_per_s": roofline_tokens_per_s(spec, context_len, bandwidth_gbs),
    }
