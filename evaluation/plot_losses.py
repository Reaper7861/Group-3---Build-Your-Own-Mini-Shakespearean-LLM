"""Plots loss curves and final byte-level perplexity for Models A and B."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

a = pd.read_csv(ROOT / "my-transformer" / "loss_log_A.csv")
b = pd.read_csv(ROOT / "my-transformer" / "loss_log_B.csv")

plt.figure(figsize=(9, 5))

plt.plot(a["step"], a["train_loss"], label="Model A train", linestyle="--")
plt.plot(a["step"], a["val_loss"], label="Model A val")
plt.plot(b["step"], b["train_loss"], label="Model B train", linestyle="--")
plt.plot(b["step"], b["val_loss"], label="Model B val")

plt.title("Training and Validation Loss")
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

out = ROOT / "evaluation" / "loss_plot.png"
plt.savefig(out, dpi=200)
print(f"saved {out}")

final_losses = {
    "Model A": float(a.iloc[-1]["val_loss"]),
    "Model B": float(b.iloc[-1]["val_loss"]),
}
perplexities = {name: math.exp(loss) for name, loss in final_losses.items()}

plt.figure(figsize=(6, 4))
bars = plt.bar(perplexities.keys(), perplexities.values(), color=["#4c78a8", "#f58518"])
plt.title("Final Byte-Level Perplexity")
plt.ylabel("Perplexity = exp(validation loss)")
plt.grid(axis="y", alpha=0.3)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )
plt.tight_layout()

out = ROOT / "evaluation" / "perplexity_plot.png"
plt.savefig(out, dpi=200)
print(f"saved {out}")
