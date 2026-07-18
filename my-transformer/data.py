"""Loads Tiny Shakespeare as UTF-8 bytes and samples shifted training batches."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import torch


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int] | torch.Tensor) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    return bytes(ids).decode("utf-8", errors="replace")


def ensure_input(path: str | Path = "input.txt") -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)
    return path


def load_data(path: str | Path = "input.txt") -> tuple[torch.Tensor, torch.Tensor]:
    path = ensure_input(path)
    text = path.read_text(encoding="utf-8")
    ids = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(ids))
    return ids[:split], ids[split:]


class ShakespeareData:
    def __init__(self, config, data_path: str | Path = "input.txt"):
        self.config = config
        self.train_data, self.val_data = load_data(data_path)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.seed)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            raise ValueError("split must be 'train' or 'val'")

        max_start = len(data) - self.config.block_size
        if max_start <= 0:
            raise ValueError("dataset split is too short for this block_size")

        ix = torch.randint(
            0,
            max_start,
            (self.config.batch_size,),
            generator=self.generator,
        )
        x = torch.stack([data[i : i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.config.block_size + 1] for i in ix])
        return x.to(self.config.device), y.to(self.config.device)
