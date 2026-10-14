"""A small decoder-only language model whose shape is fully parameterized.

Every knob the cost model prices is a real knob here: depth, width, head count,
grouped-query head count, head dimension, feed-forward width, tied embeddings
and an optional low-rank joint key/value cache. That correspondence is the
point. A shape the selector proposes can be trained and benchmarked without
touching this file.

Deliberately plain PyTorch: no fused kernels, no flash attention, no
compilation. Decode throughput measured here is a shape comparison under a
fixed engine, which is what the paper claims. Absolute tokens per second on a
production runtime will be higher.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch import ArchSpec


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x is (batch, heads, seq, head_dim); cos/sin are (seq, head_dim/2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Grouped-query attention, optionally with a low-rank joint KV cache.

    In latent mode the residual is projected to a single latent vector of rank
    `kv_latent_rank` which is what the cache stores. Keys and values are
    reconstructed from it on the fly. The cache then costs
    `rank` elements per layer per token instead of `2 * n_kv_head * d_head`,
    and stops depending on head geometry at all.
    """

    def __init__(self, spec: ArchSpec) -> None:
        super().__init__()
        self.spec = spec
        d, hd = spec.d_model, spec.head_dim
        self.n_head, self.n_kv_head, self.head_dim = spec.n_head, spec.n_kv_head, hd
        self.group_size = spec.group_size
        self.latent_rank = spec.kv_latent_rank

        self.q_proj = nn.Linear(d, spec.n_head * hd, bias=False)
        if self.latent_rank:
            self.kv_down = nn.Linear(d, self.latent_rank, bias=False)
            self.kv_norm = RMSNorm(self.latent_rank)
            self.kv_up = nn.Linear(self.latent_rank, 2 * spec.n_kv_head * hd, bias=False)
        else:
            self.k_proj = nn.Linear(d, spec.n_kv_head * hd, bias=False)
            self.v_proj = nn.Linear(d, spec.n_kv_head * hd, bias=False)
        self.o_proj = nn.Linear(spec.n_head * hd, d, bias=False)

    def _keys_values(self, cached: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Turn whatever is in the cache into keys and values."""
        batch, seq, _ = cached.shape
        if self.latent_rank:
            kv = self.kv_up(self.kv_norm(cached))
        else:
            kv = cached
        kv = kv.view(batch, seq, 2, self.n_kv_head, self.head_dim)
        return kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

    def cache_entry(self, x: torch.Tensor) -> torch.Tensor:
        """The tensor that goes into the cache for these positions."""
        if self.latent_rank:
            return self.kv_down(x)
        return torch.cat([self.k_proj(x), self.v_proj(x)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = x.shape
        entry = self.cache_entry(x)
        cache = entry if cache is None else torch.cat([cache, entry], dim=1)
        total = cache.shape[1]

        q = self.q_proj(x).view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self._keys_values(cache)

        q = apply_rope(q, cos[total - seq : total], sin[total - seq : total])
        k = apply_rope(k, cos[:total], sin[:total])

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        causal = seq > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(batch, seq, self.n_head * self.head_dim)
        return self.o_proj(out), cache


class SwiGLU(nn.Module):
    def __init__(self, spec: ArchSpec) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(spec.d_model, spec.d_ff, bias=False)
        self.up_proj = nn.Linear(spec.d_model, spec.d_ff, bias=False)
        self.down_proj = nn.Linear(spec.d_ff, spec.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, spec: ArchSpec) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(spec.d_model)
        self.attn = Attention(spec)
        self.ffn_norm = RMSNorm(spec.d_model)
        self.ffn = SwiGLU(spec)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, cache


class BFASModel(nn.Module):
    """The model under study. `spec` is the single source of truth."""

    def __init__(self, spec: ArchSpec, max_seq_len: int = 8192) -> None:
        super().__init__()
        self.spec = spec
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(spec.vocab_size, spec.d_model)
        self.blocks = nn.ModuleList(Block(spec) for _ in range(spec.n_layer))
        self.final_norm = RMSNorm(spec.d_model)
        self.lm_head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)
        if spec.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(max_seq_len, spec.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        caches: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        x = self.embed(input_ids)
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)

        new_caches: List[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            cache = caches[index] if caches is not None else None
            x, cache = block(x, cos, sin, cache)
            new_caches.append(cache)

        logits = self.lm_head(self.final_norm(x))
        out = {"logits": logits, "caches": new_caches}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Greedy or sampled decode with a growing cache. This is the loop the
        benchmark times, one token at a time, batch size one by default,
        because that is the on-device setting."""
        out = self(input_ids)
        caches = out["caches"]
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = out["logits"][:, -1, :]
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if generated.shape[1] >= self.max_seq_len:
                break
            out = self(next_id, caches=caches)
            caches = out["caches"]
        return generated

    def cache_bytes(self, context_len: int) -> int:
        """Measured cache size, to cross-check `cost_model.kv_bytes_per_token`.
        If these two disagree, trust neither until you know why."""
        per_token = (
            self.spec.kv_latent_rank
            if self.spec.kv_latent_rank
            else 2 * self.spec.n_kv_head * self.spec.head_dim
        )
        element_size = 2 if self.spec.bits_kv <= 16 else 4
        return per_token * self.spec.n_layer * context_len * element_size


def build_model(spec: ArchSpec, max_seq_len: int = 8192, dtype: str = "float32") -> BFASModel:
    model = BFASModel(spec, max_seq_len=max_seq_len)
    torch_dtype = getattr(torch, dtype)
    return model.to(torch_dtype) if torch_dtype != torch.float32 else model
