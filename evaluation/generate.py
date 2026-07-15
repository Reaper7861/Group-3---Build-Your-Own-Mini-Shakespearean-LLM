from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my-transformer"))

from config import config_from_dict  # noqa: E402
from data import decode, encode  # noqa: E402
from model import GPT  # noqa: E402


@torch.no_grad()
def generate_ids(checkpoint: Path, prompt: str, temperature: float, new_tokens: int = 150) -> torch.Tensor:
    ckpt = torch.load(checkpoint, map_location="cpu")
    config = config_from_dict(ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    for _ in range(new_tokens):
        idx_cond = idx[:, -config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx[0]


def generate(checkpoint: Path, prompt: str, temperature: float, new_tokens: int = 150) -> str:
    return decode(generate_ids(checkpoint, prompt, temperature, new_tokens))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Sample from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    text = generate(args.checkpoint, args.prompt, args.temperature)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
