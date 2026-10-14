"""Whole-segment prediction maps (no hand-picked regions): labels + each model, reverse depth, identical downsampling.

Usage: python make_fig_predictions.py <segment> <member:title,...> [downsample]
Maps from kout/bundles/*/maps/<alias>__<segment>__<dir>.npz; labels from kout/bundles/main/labels/data3/<segment>.zarr.
v8-in maps use the direction chosen by its label-free rule (stored in result_v8in.json) when present.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, zarr, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
seg = sys.argv[1]; members = [m.split(":", 1) for m in sys.argv[2].split(",")]
ds = int(sys.argv[3]) if len(sys.argv) > 3 else 8
L = ROOT / "kout/bundles/main/labels/data3" / f"{seg}.zarr"
ink = zarr.open_array(str(L / "ink"), "r")[:] > 0; sup = zarr.open_array(str(L / "sup"), "r")[:] > 0
ys, xs = np.nonzero(sup); Y0, Y1, X0, X1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
v8dir = {}
for f in glob.glob(str(ROOT / "kout/bundles/*/result_v8in.json")):
    for n, info in json.load(open(f))["per_segment"].items():
        if "label_free_choice" in info: v8dir[n] = info["label_free_choice"]

def load(alias):
    d = v8dir.get(seg, "rev") if alias == "v8in" else "rev"
    hits = glob.glob(str(ROOT / f"kout/bundles/*/maps/{alias}__{seg}__{d}.npz"))
    if len(hits) != 1: return None, d
    z = np.load(hits[0]); p = z["p"].astype(np.float32) / 255; y0, y1, x0, x1 = z["bbox"]
    full = np.full(sup.shape, np.nan, np.float32); full[y0:y1, x0:x1] = p
    if "cover" in z.files:
        c = np.unpackbits(z["cover"])[:int(np.prod(z["cover_shape"]))].reshape(z["cover_shape"]).astype(bool)
        full[y0:y1, x0:x1][~c] = np.nan
    return full, d

def shrink(a):
    a = a[Y0:Y1, X0:X1]; h, w = a.shape[0] // ds * ds, a.shape[1] // ds * ds
    return np.nanmean(a[:h, :w].reshape(h // ds, ds, w // ds, ds), axis=(1, 3))

panels = [("labels (ink within supervision)", shrink(np.where(sup, ink.astype(np.float32), np.nan)), None)]
for alias, title in members:
    m, d = load(alias)
    if m is None: print("missing", alias); continue
    ok = sup & np.isfinite(m)
    auc = roc_auc_score(ink[ok], m[ok])
    panels.append((f"{title} ({d})\nAUC {auc:.3f} on its covered pixels", shrink(np.where(sup, m, np.nan)), auc))
n = len(panels); h, w = panels[0][1].shape
fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2 * h / w + 0.9))
cm = plt.get_cmap("magma").copy(); cm.set_bad("0.65")
for ax, (t, im, _) in zip(np.atleast_1d(axes), panels):
    ax.imshow(im, cmap=cm, vmin=0, vmax=1, interpolation="nearest"); ax.set_title(t, fontsize=8); ax.axis("off")
fig.suptitle(f"{seg}: whole supervised area ({ds}x downsampled)\n"
             "grey = outside supervision or not covered by the model", fontsize=9)
fig.tight_layout()
out = ROOT / "docs/figures" / f"pred_{seg}.png"; fig.savefig(out, dpi=130, facecolor="white"); print("wrote", out)
