"""Kaggle GPU kernel: organizer v8-in (YoussefMoNader/ink-8um-v8in) on our held-out test inputs, identical label grid.

Inputs: our prepared stores (data3b = 0841 at 9.362 um, 33 slices; data3 = MAN5/0009B/0500P2 at ~9.6/8.8 um, 17 slices).
Per segment: crop the supervised bbox (+margin), resample to v8-in's native 8.64 um (24 layers centred on the render
centre, linear in depth, bilinear in XY), run the released inference recipe unchanged (ink8um.predict_stack: clip 0-200,
/255, 64 px tiles fully inside coverage, stride 21, Gaussian blend), resize the probability map back to our grid.
Direction: predeclared label-free rule = higher prediction spread on a fixed pilot tile subset; both directions are
reported when the time budget allows. Scoring uses supervised pixels covered by v8-in's tiling (coverage reported).
Exposure: 0841, MAN5 unseen by v8-in; 0009B used for v8-in validation (other segment); 0500P2 in v8-in training.
"""
import glob, json, os, subprocess, sys, time
subprocess.run("pip install -q 'zarr==2.18.7' 'numcodecs<0.16' safetensors", shell=True)
import numpy as np, torch, zarr, cv2
from huggingface_hub import snapshot_download
T0 = time.time(); OUT = "/kaggle/working"; BUDGET = 11.0 * 3600
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
REV = "d89166b41a3f5fad7749b3d7c0fdd1bd3695d844"
repo = snapshot_download("YoussefMoNader/ink-8um-v8in", revision=REV, allow_patterns=["*.json", "*.safetensors", "ink8um/*"],
                         ignore_patterns=["training/*"], local_dir="/kaggle/tmp/v8in")
sys.path.insert(0, repo)
from ink8um import InkDetector
from ink8um.inference import predict_stack, coverage_mask, tile_positions
base = InkDetector.from_pretrained(repo).eval()
log("v8-in loaded", sum(p.numel() for p in base.parameters()) / 1e6, "M params; GPUs", torch.cuda.device_count(),
    torch.cuda.get_device_name(0))
model = torch.nn.DataParallel(base.cuda()) if torch.cuda.device_count() > 1 else base.cuda()
for k in ("tile_size", "stride", "in_depth", "clip_max"): setattr(model, k, getattr(base, k))
model.reverse_layers = base.reverse_layers
UM = {"0841": 9.362, "MAN5": 9.596, "0009B": 9.6, "0500P2": 8.8}     # physical spacing of our prepared stores
EXPOSURE = {"0841": "unseen", "MAN5": "unseen", "0009B": "v8-in validation source (other segment)", "0500P2": "v8-in training source"}
TARGET_UM, DEPTH = 8.64, 24

class TileVol:
    def __init__(self, arr):
        self.shape = arr.shape; self.T = arr.chunks[1]; self.tiles = {}
        d = arr.store.dir_path() if hasattr(arr.store, "dir_path") else arr.store.path
        for f in os.listdir(os.path.join(d, arr.path)):
            if f.startswith("."): continue
            _, i, j = map(int, f.split("."))
            self.tiles[(i, j)] = arr[:, i * self.T:(i + 1) * self.T, j * self.T:(j + 1) * self.T]
    def crop(self, y0, y1, x0, x1):
        T = self.T; out = np.zeros((self.shape[0], y1 - y0, x1 - x0), np.uint8)
        for (i, j), t in self.tiles.items():
            ty0, tx0 = i * T, j * T
            sy0, sy1 = max(y0, ty0), min(y1, ty0 + t.shape[1]); sx0, sx1 = max(x0, tx0), min(x1, tx0 + t.shape[2])
            if sy0 < sy1 and sx0 < sx1:
                out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = t[:, sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
        return out

def build_stack(v, um):
    """(D,h,w) uint8 at `um` -> (H',W',24) uint8 at 8.64 um; returns stack and number of clamped layers."""
    D = v.shape[0]; c = (D - 1) / 2; r = TARGET_UM / um; s = um / TARGET_UM
    h2, w2 = int(round(v.shape[1] * s)), int(round(v.shape[2] * s))
    layers, clamped = [], 0
    for k in range(DEPTH):
        z = c + (k - (DEPTH - 1) / 2) * r
        if z < 0 or z > D - 1: clamped += 1
        z = min(max(z, 0.0), D - 1.0); z0 = int(np.floor(z)); z1 = min(z0 + 1, D - 1); f = z - z0
        sl = v[z0].astype(np.float32) * (1 - f) + v[z1].astype(np.float32) * f
        layers.append(np.clip(np.rint(cv2.resize(sl, (w2, h2), interpolation=cv2.INTER_LINEAR)), 0, 200).astype(np.uint8))
    return np.stack(layers, axis=2), clamped

def metrics(p, y):
    from sklearn.metrics import roc_auc_score, average_precision_score
    o = {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)), "n": int(len(y)), "pos": float(y.mean())}
    def f05(t):
        pr = p >= t; tp = (pr & y).sum(); fp = (pr & ~y).sum(); fn = (~pr & y).sum()
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        return 1.25 * prec * rec / max(1e-9, 0.25 * prec + rec)
    best = max((f05(t), t) for t in np.linspace(0.05, 0.95, 37))
    o["f05_best"], o["thr_best"] = float(best[0]), float(best[1]); return o

paths = sorted(glob.glob("/kaggle/input/**/data3b/*.zarr", recursive=True))
paths += [p for p in sorted(glob.glob("/kaggle/input/**/data3/*.zarr", recursive=True)) if not os.path.basename(p).startswith("pherc0841")]   # 0841 comes only from data3b
seen, jobs = set(), []
for p in paths:
    n = os.path.basename(p)[:-5]
    if n in seen: raise ValueError("duplicate store " + n)
    seen.add(n); jobs.append((n, p))
log("segments", [n for n, _ in jobs])
os.makedirs(f"{OUT}/maps", exist_ok=True)
results = {"model": "YoussefMoNader/ink-8um-v8in", "revision": REV, "target_um": TARGET_UM, "per_segment": {}}
rate = None
def run_segment(n, p, which):
    global rate
    g = zarr.open_group(p, "r"); sc = g.attrs["scroll"]; um = UM[sc]
    sup = g["sup"][:] > 0; ink = g["ink"][:] > 0
    if not sup.any(): return
    ys, xs = np.nonzero(sup); M = 64; H, W = sup.shape
    y0, y1, x0, x1 = max(0, ys.min() - M), min(H, ys.max() + 1 + M), max(0, xs.min() - M), min(W, xs.max() + 1 + M)
    v = TileVol(g["vol"]).crop(y0, y1, x0, x1)
    stack, clamped = build_stack(v, um); del v
    pts = tile_positions(coverage_mask(stack), 64, 21)
    info = {"scroll": sc, "exposure": EXPOSURE[sc], "um_in": um, "clamped_layers": clamped, "crop": [y0, y1, x0, x1],
            "stack_shape": list(stack.shape), "tiles": len(pts)}
    log(n, info)
    if which == "other":
        info = results["per_segment"][n]; chosen = info["label_free_choice"]
        todo = [d for d in ("fwd", "rev") if d != chosen]
    else:
        todo = None
    # label-free direction choice on a fixed, evenly spaced pilot subset of tiles
    sub = np.zeros(stack.shape[:2], np.uint8) if todo is None else None
    for _ in ([0] if todo is None else []):
        for (x, y) in pts[::max(1, len(pts) // 256)]: sub[y:y + 64, x:x + 64] = 255
        spread = {}
        for d in (False, True):
            pm = predict_stack(model, np.where(sub[..., None] > 0, stack, 0).astype(np.uint8), reverse=d, stride=21,
                               batch_size=32, num_workers=2)
            spread["rev" if d else "fwd"] = float(pm[pm > 0].std()) if (pm > 0).any() else 0.0
        chosen = max(spread, key=spread.get); info["pilot_spread"] = spread; info["label_free_choice"] = chosen
        results["per_segment"][n] = info; todo = [chosen]
    for dname in todo:
        est = len(pts) / rate if rate else 0
        if time.time() - T0 + 1.3 * est > BUDGET:
            log("skip", n, dname, "time budget"); info[dname] = "skipped (time budget)"; continue
        t = time.time()
        pm = predict_stack(model, stack, reverse=(dname == "rev"), stride=21, batch_size=32, num_workers=2)
        rate = len(pts) / max(1e-6, time.time() - t)
        cnt = np.zeros(stack.shape[:2], np.uint8)
        for (x, y) in pts: cnt[y:y + 64, x:x + 64] = 1
        h, w = y1 - y0, x1 - x0
        pb = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR)
        cov = cv2.resize(cnt, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        m = sup[y0:y1, x0:x1] & cov
        res = metrics(pb[m], ink[y0:y1, x0:x1][m]); res["coverage_of_supervised"] = float(m.sum() / sup.sum())
        res["tiles_per_s"] = rate; info[dname] = res
        np.savez_compressed(f"{OUT}/maps/v8in__{n}__{dname}.npz", p=(np.clip(pb, 0, 1) * 255).astype(np.uint8),
                            cover=np.packbits(cov), cover_shape=np.array(cov.shape), bbox=np.array([y0, y1, x0, x1]))
        log("EVAL", n, dname, json.dumps(res))
        json.dump(results, open(f"{OUT}/result_v8in.json", "w"), indent=1)
    del stack

for which, (n, p) in [("chosen", j) for j in jobs] + [("other", j) for j in jobs]:
    if which == "other" and "label_free_choice" not in results["per_segment"].get(n, {}): continue
    try: run_segment(n, p, which)
    except Exception as e:
        import traceback; traceback.print_exc(); log("SEGMENT FAILED", n, repr(e))
        results["per_segment"].setdefault(n, {})["error"] = repr(e)
json.dump(results, open(f"{OUT}/result_v8in.json", "w"), indent=1)
log("ALLDONE")
