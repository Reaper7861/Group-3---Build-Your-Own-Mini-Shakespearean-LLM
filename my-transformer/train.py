from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path

import torch

from config import GPTConfig, get_config
from data import ShakespeareData
from model import GPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def estimate_loss(model: GPT, data: ShakespeareData, eval_iters: int) -> dict[str, float]:
    out = {}
    was_training = model.training
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(was_training)
    return out


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train(config: GPTConfig, data_path: Path, out_dir: Path) -> Path:
    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)
    set_seed(config.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    data = ShakespeareData(config, data_path)
    model = GPT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    log_path = out_dir / f"loss_log_{config.name}.csv"
    ckpt_path = out_dir / f"ckpt_{config.name}.pt"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        for step in range(config.max_steps):
            lr = config.learning_rate * (1 - step / config.max_steps)
            set_lr(optimizer, lr)

            if step % config.eval_interval == 0:
                losses = estimate_loss(model, data, config.eval_iters)
                writer.writerow(
                    {
                        "step": step,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                    }
                )
                f.flush()
                print(
                    f"step {step:4d} | train {losses['train']:.4f} | "
                    f"val {losses['val']:.4f} | lr {lr:.6f}"
                )

            x, y = data.get_batch("train")
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    final_losses = estimate_loss(model, data, config.eval_iters)
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        writer.writerow(
            {
                "step": config.max_steps,
                "train_loss": final_losses["train"],
                "val_loss": final_losses["val"],
            }
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "final_train_loss": final_losses["train"],
            "final_val_loss": final_losses["val"],
        },
        ckpt_path,
    )
    print(f"saved checkpoint: {ckpt_path}")
    print(f"saved loss log: {log_path}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train byte-level Shakespeare GPT.")
    parser.add_argument("--model", choices=["A", "B"], default="A")
    parser.add_argument("--data", type=Path, default=Path("input.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("my-transformer"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.model)
    updates = {
        "max_steps": args.max_steps,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "num_threads": args.num_threads,
    }
    config = replace(config, **{k: v for k, v in updates.items() if v is not None})
    train(config, args.data, args.out_dir)


if __name__ == "__main__":
    main()
