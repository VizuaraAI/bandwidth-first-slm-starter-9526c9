"""Architecture shapes.

An `ArchSpec` is the complete description of a decoder-only language model
shape, including the two knobs that most papers leave implicit: how many bits
a weight costs and how many bits a cached key or value costs. Those two knobs
are first-class here because they move bytes, and bytes are what decode pays
for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional


@dataclass(frozen=True)
class ArchSpec:
    """A decoder-only transformer shape.

    Attributes:
        n_layer: number of transformer blocks.
        d_model: residual stream width.
        n_head: number of query heads.
        n_kv_head: number of key/value heads. Equal to `n_head` is multi-head
            attention, 1 is multi-query attention, anything between is
            grouped-query attention.
        d_ff: inner width of the feed-forward block (per expert).
        vocab_size: tokenizer vocabulary size.
        d_head: per-head dimension. Defaults to `d_model // n_head`.
        kv_latent_rank: if set, keys and values are stored as a single joint
            latent vector of this rank per layer per token (low-rank cache,
            in the spirit of multi-head latent attention) instead of as
            `n_kv_head * d_head` keys plus the same number of values.
        tie_embeddings: whether the input embedding and the output head share
            weights. This matters for bytes: the output head is read in full
            on every decoded token.
        bits_w: bits per stored weight (16, 8, 4, 2 ...). Fractional values are
            allowed because real quantization formats carry scales and zeros,
            so 4-bit group quantization costs about 4.5 bits per weight.
        bits_kv: bits per stored key/value element.
        n_expert: number of feed-forward experts (1 means dense).
        n_expert_active: experts routed per token.
    """

    name: str
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    d_ff: int
    vocab_size: int = 49152
    d_head: Optional[int] = None
    kv_latent_rank: Optional[int] = None
    tie_embeddings: bool = True
    bits_w: float = 16.0
    bits_kv: float = 16.0
    n_expert: int = 1
    n_expert_active: int = 1

    def __post_init__(self) -> None:
        if self.d_head is None:
            if self.d_model % self.n_head != 0:
                raise ValueError(
                    f"{self.name}: d_model={self.d_model} is not divisible by "
                    f"n_head={self.n_head}, so d_head must be given explicitly"
                )
            object.__setattr__(self, "d_head", self.d_model // self.n_head)
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"{self.name}: n_head={self.n_head} must be a multiple of "
                f"n_kv_head={self.n_kv_head}"
            )
        if self.n_expert_active > self.n_expert:
            raise ValueError(f"{self.name}: cannot route more experts than exist")
        for field_name in ("n_layer", "d_model", "n_head", "n_kv_head", "d_ff"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{self.name}: {field_name} must be positive")

    @property
    def head_dim(self) -> int:
        assert self.d_head is not None
        return self.d_head

    @property
    def group_size(self) -> int:
        """Query heads per key/value head."""
        return self.n_head // self.n_kv_head

    @property
    def depth_width_ratio(self) -> float:
        """Aspect ratio of the network. MobileLLM-style deep-and-thin designs
        have a high value here; wide-and-shallow designs a low one."""
        return self.n_layer / self.d_model * 1000.0

    def with_(self, **changes: Any) -> "ArchSpec":
        """Return a copy with some fields replaced (handy inside sweeps)."""
        if "d_head" not in changes and "n_head" in changes:
            changes["d_head"] = self.head_dim
        return replace(self, **changes)

    def describe(self) -> str:
        kv = (
            f"latent r={self.kv_latent_rank}"
            if self.kv_latent_rank
            else f"{self.n_kv_head} kv heads"
        )
        moe = "" if self.n_expert == 1 else f", MoE {self.n_expert_active}/{self.n_expert}"
        return (
            f"{self.name}: L={self.n_layer} d={self.d_model} h={self.n_head} "
            f"({kv}) d_head={self.head_dim} d_ff={self.d_ff} "
            f"w{self.bits_w:g}b kv{self.bits_kv:g}b{moe}"
        )


def from_hf_config(config: Any, name: Optional[str] = None, **overrides: Any) -> ArchSpec:
    """Build an `ArchSpec` from a Hugging Face model config.

    This is how the project gets honest shapes for off-the-shelf models instead
    of guessed ones:

        from transformers import AutoConfig
        spec = from_hf_config(AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B"))
    """

    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            value = getattr(config, key, None)
            if value is not None:
                return value
        return default

    n_head = int(pick("num_attention_heads", "n_head", "num_heads"))
    d_model = int(pick("hidden_size", "n_embd", "d_model"))
    head_dim = pick("head_dim")
    spec = ArchSpec(
        name=name or getattr(config, "name_or_path", None) or "hf-model",
        n_layer=int(pick("num_hidden_layers", "n_layer", "num_layers")),
        d_model=d_model,
        n_head=n_head,
        n_kv_head=int(pick("num_key_value_heads", default=n_head)),
        d_ff=int(pick("intermediate_size", "ffn_dim", default=4 * d_model)),
        vocab_size=int(pick("vocab_size", default=49152)),
        d_head=int(head_dim) if head_dim else None,
        tie_embeddings=bool(pick("tie_word_embeddings", default=False)),
        n_expert=int(pick("num_local_experts", "num_experts", default=1)),
        n_expert_active=int(pick("num_experts_per_tok", default=1)),
    )
    return spec.with_(**overrides) if overrides else spec


# A few reference shapes. The small ones are trainable on one GPU inside the
# project's compute budget; the larger entries exist so the cost model can be
# checked against models the student can download and benchmark.
REFERENCE_SHAPES = {
    "tiny-60m-mha": ArchSpec("tiny-60m-mha", 12, 512, 8, 8, 1408),
    "tiny-60m-gqa2": ArchSpec("tiny-60m-gqa2", 12, 512, 8, 2, 1408),
    "tiny-60m-deep": ArchSpec("tiny-60m-deep", 24, 384, 6, 2, 1024),
    "tiny-60m-wide": ArchSpec("tiny-60m-wide", 8, 640, 10, 2, 1728, d_head=64),
    "tiny-60m-latent": ArchSpec("tiny-60m-latent", 12, 512, 8, 8, 1408, kv_latent_rank=128),
    "small-140m-gqa4": ArchSpec("small-140m-gqa4", 16, 768, 12, 4, 2048, d_head=64),
}
