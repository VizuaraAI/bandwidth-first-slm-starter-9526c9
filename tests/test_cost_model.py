"""Checks on the cost model.

Run with `python -m pytest tests` or plainly with
`python tests/test_cost_model.py`. These are the invariants that must hold for
the paper's argument to be sound, so run them before trusting any table.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bfas import cost_model
from bfas.arch import ArchSpec, REFERENCE_SHAPES

MHA = ArchSpec("mha", n_layer=12, d_model=512, n_head=8, n_kv_head=8, d_ff=1408)
GQA = MHA.with_(name="gqa", n_kv_head=2)
MQA = MHA.with_(name="mqa", n_kv_head=1)
LATENT = MHA.with_(name="latent", kv_latent_rank=128)


def test_cache_scales_with_kv_heads():
    """Grouped-query attention cuts cache traffic exactly in proportion to the
    key/value head count. If this fails, the cache term is wrong."""
    full = cost_model.kv_bytes_per_token(MHA, 1024)
    quarter = cost_model.kv_bytes_per_token(GQA, 1024)
    eighth = cost_model.kv_bytes_per_token(MQA, 1024)
    assert abs(full / quarter - 4.0) < 1e-9
    assert abs(full / eighth - 8.0) < 1e-9


def test_cache_is_linear_in_context():
    one = cost_model.kv_bytes_per_token(GQA, 1)
    many = cost_model.kv_bytes_per_token(GQA, 4096)
    assert abs(many - 4096 * one) < 1e-6
    assert cost_model.kv_bytes_per_token(GQA, 0) == 0.0


def test_latent_cache_is_smaller_and_head_independent():
    """A rank-128 latent cache is cheaper than 8 fp16 key/value heads of width
    64, and does not change when head count changes."""
    assert cost_model.kv_bytes_per_token(LATENT, 2048) < cost_model.kv_bytes_per_token(MHA, 2048)
    wider = LATENT.with_(name="latent-wide", n_kv_head=4)
    assert cost_model.kv_bytes_per_token(LATENT, 2048) == cost_model.kv_bytes_per_token(wider, 2048)


def test_weight_bytes_track_precision():
    full = cost_model.weight_bytes_per_token(MHA)
    half = cost_model.weight_bytes_per_token(MHA.with_(name="int8", bits_w=8.0))
    assert abs(full / half - 2.0) < 1e-9


def test_active_params_excludes_the_embedding_table():
    """The embedding table is a lookup (one row read), while the output head is
    read in full. With untied weights that makes active strictly smaller than
    total; with tied weights the table is the head, so almost everything is
    read and the ratio is one."""
    untied = MHA.with_(name="untied", tie_embeddings=False)
    assert cost_model.active_params(untied) < cost_model.total_params(untied)
    assert cost_model.active_params(untied) > untied.vocab_size * untied.d_model
    tied_ratio = cost_model.active_params(MHA) / cost_model.total_params(MHA)
    assert 0.99 < tied_ratio < 1.01


def test_parameter_count_can_misrank_bytes():
    """The claim the paper rests on: a larger model can move fewer bytes."""
    small_wide_cache = ArchSpec("small", 16, 512, 8, 8, 1408, d_head=64)
    large_thin_cache = ArchSpec("large", 16, 768, 12, 1, 2048, d_head=64, bits_w=4.5)
    assert cost_model.total_params(large_thin_cache) > cost_model.total_params(small_wide_cache)
    at_long_context = 4096
    assert (
        cost_model.bytes_per_token(large_thin_cache, at_long_context).total
        < cost_model.bytes_per_token(small_wide_cache, at_long_context).total
    )


def test_roofline_inverts_consistently():
    """Predicted throughput and the byte budget are two sides of one equation."""
    bandwidth, efficiency, target = 68.0, 0.7, 30.0
    budget = cost_model.byte_budget_for_slo(target, bandwidth, efficiency)
    spec = GQA
    predicted = cost_model.roofline_tokens_per_s(spec, 2048, bandwidth, efficiency)
    actual_bytes = cost_model.bytes_per_token(spec, 2048).total
    assert (predicted >= target) == (actual_bytes <= budget + 1e-6)


def test_cache_share_grows_with_context():
    short = cost_model.bytes_per_token(MHA, 128).kv_fraction
    long = cost_model.bytes_per_token(MHA, 8192).kv_fraction
    assert long > short


def test_reference_shapes_are_constructible():
    for name, spec in REFERENCE_SHAPES.items():
        assert spec.name == name
        assert cost_model.total_params(spec) > 0
        assert cost_model.summarize(spec, 512, 68.0)["pred_tokens_per_s"] > 0


if __name__ == "__main__":
    failures = 0
    for key, value in sorted(list(globals().items())):
        if key.startswith("test_") and callable(value):
            try:
                value()
                print(f"pass  {key}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL  {key}: {error}")
    raise SystemExit(1 if failures else 0)
