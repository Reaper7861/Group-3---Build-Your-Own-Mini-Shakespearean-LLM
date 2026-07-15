from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPTConfig:
    name: str
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    batch_size: int
    dropout: float = 0.1
    vocab_size: int = 256
    learning_rate: float = 1e-3
    max_steps: int = 3000
    eval_interval: int = 250
    eval_iters: int = 50
    seed: int = 4397
    device: str = "cpu"
    num_threads: int | None = None

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


CONFIGS: dict[str, GPTConfig] = {
    "A": GPTConfig(
        name="A",
        n_layer=2,
        n_head=4,
        n_embd=64,
        block_size=64,
        batch_size=32,
    ),
    "B": GPTConfig(
        name="B",
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=128,
        batch_size=16,
    ),
}


def get_config(name: str) -> GPTConfig:
    key = name.upper()
    if key not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown model config {name!r}. Choose one of: {valid}")
    return CONFIGS[key]


def config_from_dict(values: dict) -> GPTConfig:
    return GPTConfig(**values)
