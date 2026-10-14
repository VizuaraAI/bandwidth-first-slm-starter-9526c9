"""BFAS: bandwidth-first architecture selection for small language models.

Modules, in the order you will use them:

    bandwidth   measure achievable memory bandwidth on this machine
    arch        ArchSpec, the architecture shape under study
    cost_model  bytes moved per decoded token, and roofline predictions
    model       a small decoder-only language model built from an ArchSpec
    data        token cache and fixed-order batching
    train       train one shape under a fixed token budget
    benchmark   measured decode throughput and roofline efficiency
    evaluate    build the paper's tables
    select      the bandwidth-first selector and its baselines
    plots       figures from measured results
"""

from .arch import ArchSpec, REFERENCE_SHAPES, from_hf_config
from .cost_model import (
    ByteBudget,
    active_params,
    bandwidth_roofline_efficiency,
    byte_budget_for_slo,
    bytes_per_token,
    kv_bytes_per_token,
    quality_per_byte,
    resident_bytes,
    roofline_tokens_per_s,
    summarize,
    total_params,
    weight_bytes_per_token,
)

__version__ = "0.1.0"

__all__ = [
    "ArchSpec",
    "ByteBudget",
    "REFERENCE_SHAPES",
    "active_params",
    "bandwidth_roofline_efficiency",
    "byte_budget_for_slo",
    "bytes_per_token",
    "from_hf_config",
    "kv_bytes_per_token",
    "quality_per_byte",
    "resident_bytes",
    "roofline_tokens_per_s",
    "summarize",
    "total_params",
    "weight_bytes_per_token",
]
