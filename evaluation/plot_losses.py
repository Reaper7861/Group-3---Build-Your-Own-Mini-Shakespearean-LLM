from pathlib import Path

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