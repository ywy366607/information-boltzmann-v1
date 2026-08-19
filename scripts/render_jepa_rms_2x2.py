"""Bar chart: JEPA+RMS vs Bayes+RMS (same protocol, both use RMSNorm Δ)."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("present/figs/jepa_bayes_rms_2x2.png")

cells = [
    ("JEPA + RMSNorm(Δ)", 90.39, 95.6, 75.9, "#38bdf8"),
    ("Bayes + RMSNorm(Δ)", 97.59, 95.9, 96.9, "#fbbf24"),
]

fig, ax = plt.subplots(figsize=(10.4, 5.5), dpi=160)
fig.patch.set_facecolor("#0b1120")
ax.set_facecolor("#0f172a")
labels = ["Overall", "1px OCR", "Kinks"]
x = np.arange(len(labels))
width = 0.32
for i, (name, acc, ocr, kinks, col) in enumerate(cells):
    vals = [acc, ocr, kinks]
    rects = ax.bar(
        x + (i - 0.5) * width, vals, width, color=col, alpha=0.92,
        label=name, edgecolor="#0b1120", linewidth=0.3,
    )
    for r in rects:
        ax.annotate(
            f"{r.get_height():.1f}",
            xy=(r.get_x() + r.get_width() / 2, r.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=8.5, color="#f8fafc", fontweight="bold",
        )
ax.set_xticks(x)
ax.set_xticklabels(labels, color="#e2e8f0")
ax.set_ylabel("Accuracy (%)", color="#cbd5e1")
ax.set_ylim(0, 114)
ax.set_title("Same protocol, both RMSNorm(Δ)  ·  2000 steps, N=3", color="white", fontsize=12, fontweight="bold")
ax.tick_params(colors="#94a3b8")
ax.grid(True, ls=":", alpha=0.3, color="#64748b", axis="y")
ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=10)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"saved {OUT}")
