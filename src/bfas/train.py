"""Train one architecture shape under a fixed token budget.

Controls that make the comparison fair, and that are easy to break by accident:

    same tokens, same order        one cached token array, one seed
    same token budget              step count set by tokens, not by epochs
    same optimizer settings         only the shape changes between runs
    same evaluation batches        fixed validation seed

Every run writes a JSON record into `results/runs/`. Those records are the rows
of the paper's quality column.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch

from . import cost_model
from .arch import ArchSpec, REFERENCE_SHAPES
from .data import TokenStream, load_tokens
from .model import build_model


@dataclass
class TrainConfig:
    shape: str = "tiny-60m-gqa2"
    token_budget: int = 20_000_000
    seq_len: int = 512
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
    token_cache: Optional[str] = None
    eval_every: int = 200
    eval_batches: int = 10
    out_dir: str = "results/runs"
    save_checkpoint: bool = False
    vocab_size: Optional[int] = None


def cosine_lr(step: int, total: int, config: TrainConfig) -> float:
    warmup = max(1, int(total * config.warmup_ratio))
    if step < warmup:
        return config.lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    floor = config.lr * config.min_lr_ratio
    return floor + 0.5 * (config.lr - floor) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: torch.nn.Module, stream: TokenStream, batches: int, device: str) -> Dict[str, float]:
    model.eval()
    losses = []
    for inputs, targets in stream.val_batches(batches):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)["logits"]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1)
        )
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {"val_loss": mean_loss, "val_ppl": math.exp(min(mean_loss, 20.0))}


def train(spec: ArchSpec, config: TrainConfig) -> Dict[str, object]:
    torch.manual_seed(config.seed)
    tokens = load_tokens(config.token_cache)
    stream = TokenStream(
        tokens,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    tokens_per_step = stream.tokens_per_step * config.grad_accum
    total_steps = max(1, config.token_budget // tokens_per_step)

    model = build_model(spec, max_seq_len=config.seq_len + 64, dtype=config.dtype)
    model.to(config.device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    params = cost_model.total_params(spec)
    print(spec.describe())
    print(
        f"parameters: {params / 1e6:.1f}M   "
        f"bytes/token at ctx {config.seq_len}: "
        f"{cost_model.bytes_per_token(spec, config.seq_len).total / 1024 ** 2:.1f} MB"
    )
    print(f"token budget: {config.token_budget / 1e6:.1f}M over {total_steps} steps")

    history = []
    started = time.perf_counter()
    step = 0
    batches = stream.train_batches(total_steps * config.grad_accum)
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(config.grad_accum):
            inputs, targets = next(batches)
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            logits = model(inputs)["logits"]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1)
            )
            (loss / config.grad_accum).backward()
            accumulated += loss.item() / config.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, total_steps, config)
        optimizer.step()
        step += 1

        if step % max(1, config.eval_every // 10) == 0:
            print(f"  step {step}/{total_steps}  train loss {accumulated:.4f}")
        if step % config.eval_every == 0 or step == total_steps:
            metrics = evaluate(model, stream, config.eval_batches, config.device)
            metrics.update({"step": step, "train_loss": accumulated})
            history.append(metrics)
            print(
                f"  step {step}/{total_steps}  val loss {metrics['val_loss']:.4f}  "
                f"ppl {metrics['val_ppl']:.2f}"
            )

    final = evaluate(model, stream, config.eval_batches * 2, config.device)
    record = {
        "shape": spec.name,
        "spec": {key: value for key, value in asdict(spec).items()},
        "config": asdict(config),
        "params_m": params / 1e6,
        "active_params_m": cost_model.active_params(spec) / 1e6,
        "bytes_per_token_mb": cost_model.bytes_per_token(spec, config.seq_len).total / 1024 ** 2,
        "steps": total_steps,
        "tokens_seen": total_steps * tokens_per_step,
        "wall_clock_s": time.perf_counter() - started,
        "history": history,
        "final": final,
        # Higher is better, so it can be divided by bytes directly.
        "quality": 1.0 / math.log(max(final["val_ppl"], 1.0001)),
    }
    record["quality_per_gb"] = cost_model.quality_per_byte(
        record["quality"], spec, config.seq_len
    )

    os.makedirs(config.out_dir, exist_ok=True)
    path = os.path.join(config.out_dir, f"{spec.name}.json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {path}")

    if config.save_checkpoint:
        ckpt = os.path.join(config.out_dir, f"{spec.name}.pt")
        torch.save({"spec": asdict(spec), "state_dict": model.state_dict()}, ckpt)
        print(f"wrote {ckpt}")

    return record


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train one shape under a fixed token budget")
    parser.add_argument("--shape", default="tiny-60m-gqa2", choices=sorted(REFERENCE_SHAPES))
    parser.add_argument("--token-budget", type=int, default=20_000_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--token-cache", default=None)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--out-dir", default="results/runs")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = TrainConfig(
        shape=args.shape,
        token_budget=args.token_budget,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        device=args.device,
        dtype=args.dtype,
        token_cache=args.token_cache,
        eval_every=args.eval_every,
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        save_checkpoint=args.save_checkpoint,
        seed=args.seed,
    )
    spec = REFERENCE_SHAPES[args.shape]
    if args.vocab_size:
        spec = spec.with_(vocab_size=args.vocab_size)
    train(spec, config)


if __name__ == "__main__":
    main()
