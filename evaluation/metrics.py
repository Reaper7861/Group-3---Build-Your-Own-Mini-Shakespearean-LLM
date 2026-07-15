from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my-transformer"))

from config import config_from_dict  # noqa: E402
from data import ShakespeareData  # noqa: E402
from model import GPT  # noqa: E402


@torch.no_grad()
def val_loss(checkpoint: Path, data_path: Path, eval_iters: int | None) -> tuple[float, float]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    config = config_from_dict(ckpt["config"])
    if eval_iters is None:
        eval_iters = config.eval_iters
    data = ShakespeareData(config, data_path)
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = data.get_batch("val")
        _, loss = model(x, y)
        losses[k] = loss.item()
    loss_value = losses.mean().item()
    return loss_value, math.exp(loss_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report validation loss and byte-level perplexity.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("input.txt"))
    parser.add_argument("--eval-iters", type=int, default=None)
    args = parser.parse_args()

    loss_value, perplexity = val_loss(args.checkpoint, args.data, args.eval_iters)
    print(f"val_loss: {loss_value:.4f}")
    print(f"byte_level_perplexity: {perplexity:.4f}")
    print("Note: byte-level perplexity is only meaningful for A vs B in this project.")


if __name__ == "__main__":
    main()
