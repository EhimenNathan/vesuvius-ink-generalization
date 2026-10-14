"""Figures for the depth-pyramid defect: per-slice hypothesis test (all 25 volumes) and a visual 0841 w00 example."""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
EV = HERE.parent / "release" / "pyramid-repair-evidence"
sys.path.insert(0, str(EV / "tools"))
from verify_all_pyramids import Arr, BASE   # noqa: E402

res = json.loads((EV / "evidence" / "pyramid_pixel_all.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, flagged, title in [(axes[0], True, "18 volumes declaring z-scale 1 (flagged)"),
                           (axes[1], False, "7 volumes declaring z-scale 4 (controls)")]:
    for r in res:
        if r["survey_flagged"] != flagged: continue
        k = [row["slice"] for row in r["rows"]][1:]
        ax.plot(k, [row["truncated_r"] for row in r["rows"]][1:], color="#1f77b4", alpha=.55, lw=1.2)
        ax.plot(k, [row["pooled_r"] for row in r["rows"]][1:], color="#d62728", alpha=.55, lw=1.2)
    ax.set_title(title, fontsize=10); ax.set_xlabel("level-2 slice k"); ax.axhline(0, color="k", lw=.5)
    ax.grid(alpha=.3)
axes[0].set_ylabel("Pearson r with published level-2 slice")
axes[0].plot([], [], color="#1f77b4", label="truncated: XY-pool of level-0 slice k")
axes[0].plot([], [], color="#d62728", label="depth-pooled: mean of level-0 slices 4k..4k+3")
axes[0].legend(fontsize=8, loc="center right")
fig.suptitle("Ink-label pyramids: level 2 keeps the first 17 of 65 slices in all 18 flagged volumes (25/25 agree)", fontsize=11)
fig.tight_layout(); fig.savefig(HERE / "figures" / "pyramid_hypothesis_test.png", dpi=150); plt.close(fig)

# visual example: 0841 w00, one 128x128 level-2 region near the centre
base = BASE + "841/w00/w00.zarr/"
a0, a2 = Arr(base, 0), Arr(base, 2)
y2, x2, R = 1920, 2304, 128   # the region of the original single-region proof (L2 chunk 0.15.18)
L2 = a2.read(0, 17, y2, y2 + R, x2, x2 + R).astype(np.float32)
L0 = a0.read(0, 65, 4 * y2, 4 * (y2 + R), 4 * x2, 4 * (x2 + R)).astype(np.float32)
xy = L0.reshape(65, R, 4, R, 4).mean(axis=(2, 4))
k = 12
iso = xy[4 * k:4 * k + 4].mean(0)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.4]})
for ax, im, t in [(axes[0], L2[k], f"published level 2, slice {k}"),
                  (axes[1], xy[k], f"level 0 slice {k}, XY-pooled\nr = {np.corrcoef(L2[k].ravel(), xy[k].ravel())[0,1]:.5f}"),
                  (axes[2], iso, f"level 0 slices {4*k}-{4*k+3}, pooled\n(what z-scale 4 would mean) r = {np.corrcoef(L2[k].ravel(), iso.ravel())[0,1]:.3f}")]:
    ax.imshow(im, cmap="gray", vmin=np.percentile(L2, 1), vmax=np.percentile(L2, 99)); ax.set_title(t, fontsize=9); ax.axis("off")
prof0 = L0.mean(axis=(1, 2)); prof2 = L2.mean(axis=(1, 2))
axes[3].plot(np.arange(65), prof0, "k-", label="level 0 (65 slices, full depth)")
axes[3].plot(np.arange(17), prof2, "o", color="#1f77b4", ms=4, label="level 2 plotted at index k: matches slices 0-16")
axes[3].plot(4 * np.arange(17) + 1.5, prof2, "x", color="#d62728", ms=4, label="level 2 if it were depth-pooled")
axes[3].axvspan(0, 16, color="#1f77b4", alpha=.08)
axes[3].set_xlabel("level-0 depth index"); axes[3].set_ylabel("mean intensity"); axes[3].legend(fontsize=7)
axes[3].set_title("only the first quarter of the depth range survives", fontsize=9)
fig.suptitle("PHerc0841 w00 (public ink-label bucket): published level 2 = first 17 of 65 slices", fontsize=11)
fig.tight_layout(); fig.savefig(HERE / "figures" / "pyramid_0841_w00_example.png", dpi=150); plt.close(fig)
print("ok")
