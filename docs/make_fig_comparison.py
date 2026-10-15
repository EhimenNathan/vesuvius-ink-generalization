"""Grouped bar chart: AUC per held-out source, our models vs the organizers' models (identical pixels, reverse order).
Values from results/*/ensemble_results.json (four sources) and the PHerc1447 evaluation logs (EXPERIMENTS.md)."""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sources = ["0841\n(scroll)", "0009B\n(fragment, IR)", "0500P2\n(fragment, IR)", "MAN5", "PHerc1447\n(fresh scroll)"]
series = [  # (label, color in fixed categorical order, values; None = not evaluated)
    ("ours B0-256 seed 43 (5.5 M)", "#2a78d6", [0.753, 0.901, 0.875, 0.533, 0.745]),
    ("organizer ink_9um seed 42 (34 M)", "#eb6834", [0.703, 0.868, 0.829, 0.619, 0.674]),
    ("ours B0-256 seed 42 (5.5 M)", "#1baf7a", [0.730, 0.885, 0.885, 0.538, 0.733]),
    ("organizer ink_9um seed 43 (34 M)", "#eda100", [0.720, 0.868, 0.852, 0.553, 0.682]),
    ("organizer v8-in (83 M)", "#e87ba4", [None, None, None, None, 0.832]),
]
fig, ax = plt.subplots(figsize=(11, 4.6))
w = 0.16; x = np.arange(len(sources))
best = [max(v[j] for _, _, v in series if v[j] is not None) for j in range(len(sources))]
for i, (lab, col, vals) in enumerate(series):
    xs = x + (i - 2) * w
    for j, (xx, a) in enumerate(zip(xs, vals)):
        if a is None: continue
        ax.bar(xx, a - 0.5, w - 0.02, bottom=0.5, color=col, label=lab if j == 0 or (i == 4 and j == 4) else None, linewidth=0)
        if a == best[j]: ax.text(xx, a + 0.006, f"{a:.3f}", ha="center", va="bottom", fontsize=8, color="#0b0b0b")
ax.axhline(0.5, color="#52514e", lw=0.8); ax.text(len(sources) - 0.45, 0.505, "chance", fontsize=8, color="#52514e", ha="right")
ax.set_xticks(x); ax.set_xticklabels(sources, fontsize=9); ax.set_ylim(0.45, 1.02); ax.set_ylabel("ROC AUC (reverse depth order)")
ax.grid(axis="y", color="#e4e3dd", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=8, ncol=3, loc="upper left", frameon=False)
ax.set_title("Held-out sources never used for training by any model shown (v8-in: 0841/MAN5 evaluation pending)", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "figures" / "comparison_auc_by_source.png", dpi=160, facecolor="white"); print("ok")
