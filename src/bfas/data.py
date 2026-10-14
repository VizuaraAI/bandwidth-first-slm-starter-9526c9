"""Data loading for the controlled training runs.

The architecture comparison is only valid if every shape sees the same tokens
in the same order. So the pipeline is: tokenize once, cache a flat uint16
token array to disk, then hand every training run an identical stream. The
synthetic corpus exists so the whole pipeline can be smoke-tested on a laptop
with no network access.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_SUBSET = "sample-10BT"


def _token_cache_path(cache_dir: str, tag: str, n_tokens: int) -> str:
    return os.path.join(cache_dir, f"tokens_{tag}_{n_tokens // 1_000_000}M.npy")


def build_token_cache(
    n_tokens: int = 50_000_000,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    dataset_name: str = DEFAULT_DATASET,
    subset: Optional[str] = DEFAULT_SUBSET,
    cache_dir: str = "data",
    tag: str = "fineweb",
) -> str:
    """Stream a public corpus, tokenize it, store a flat token array.

    Streaming means no multi-gigabyte download sits in the repo. 50M tokens is
    a sensible first target: enough to separate architecture shapes on
    validation loss, small enough to retrain the whole grid overnight.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(cache_dir, exist_ok=True)
    path = _token_cache_path(cache_dir, tag, n_tokens)
    if os.path.exists(path):
        print(f"token cache already present: {path}")
        return path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    stream = load_dataset(dataset_name, name=subset, split="train", streaming=True)
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    buffer = np.empty(n_tokens, dtype=np.uint16)
    filled = 0
    for record in stream:
        ids = tokenizer(record["text"], add_special_tokens=False)["input_ids"]
        ids.append(eos)
        take = min(len(ids), n_tokens - filled)
        buffer[filled : filled + take] = np.asarray(ids[:take], dtype=np.uint16)
        filled += take
        if filled % 5_000_000 < take:
            print(f"  tokenized {filled / 1e6:.1f}M / {n_tokens / 1e6:.0f}M tokens")
        if filled >= n_tokens:
            break

    np.save(path, buffer[:filled])
    print(f"wrote {filled / 1e6:.1f}M tokens to {path}")
    return path


def synthetic_tokens(n_tokens: int = 400_000, vocab_size: int = 49152, seed: int = 0) -> np.ndarray:
    """A learnable stand-in corpus: an order-one Markov chain over a small
    alphabet, so a broken training loop is obvious immediately.

    What to expect when the pipeline works: loss starts near log(alphabet),
    about 5.5 nats, and falls toward the chain's conditional entropy of roughly
    2.5 nats within a few hundred optimizer steps on a small model. If loss
    sits flat at log(alphabet), something is wrong; if it drops, the loop is
    fine and any remaining flatness on real data is a learning-rate or budget
    question. Never report a number measured on this corpus.
    """
    rng = np.random.default_rng(seed)
    alphabet = min(256, vocab_size)
    transitions = rng.dirichlet(np.ones(alphabet) * 0.04, size=alphabet)
    tokens = np.empty(n_tokens, dtype=np.uint16)
    current = 0
    for index in range(n_tokens):
        tokens[index] = current
        current = int(rng.choice(alphabet, p=transitions[current]))
    return tokens


class TokenStream:
    """Fixed-length windows over a flat token array, with a held-out tail.

    The same seed gives the same batch order for every architecture, which is
    what makes the comparison controlled.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        seq_len: int = 512,
        batch_size: int = 8,
        val_fraction: float = 0.02,
        seed: int = 0,
    ) -> None:
        if tokens.size < seq_len * 4:
            raise ValueError("token array too small for this sequence length")
        split = int(tokens.size * (1.0 - val_fraction))
        self.train_tokens = tokens[:split]
        self.val_tokens = tokens[split:]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def _batch(self, source: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        highest = source.size - self.seq_len - 1
        starts = rng.integers(0, highest, size=self.batch_size)
        windows = np.stack([source[s : s + self.seq_len + 1] for s in starts])
        return torch.from_numpy(windows.astype(np.int64))

    def train_batches(self, steps: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(steps):
            window = self._batch(self.train_tokens, self.rng)
            yield window[:, :-1], window[:, 1:]

    def val_batches(self, steps: int = 20) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(1234)  # fixed, so validation loss is comparable
        for _ in range(steps):
            window = self._batch(self.val_tokens, rng)
            yield window[:, :-1], window[:, 1:]

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len


def load_tokens(path: Optional[str], synthetic_size: int = 400_000) -> np.ndarray:
    """Load a cached token array, or fall back to the synthetic corpus."""
    if path and os.path.exists(path):
        return np.load(path)
    print("no token cache found, using the synthetic corpus (smoke tests only)")
    return synthetic_tokens(synthetic_size)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the token cache")
    parser.add_argument("--n-tokens", type=int, default=50_000_000)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--tag", default="fineweb")
    args = parser.parse_args()

    build_token_cache(
        n_tokens=args.n_tokens,
        tokenizer_name=args.tokenizer,
        dataset_name=args.dataset,
        subset=args.subset,
        cache_dir=args.cache_dir,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
