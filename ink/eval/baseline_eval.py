"""Kaggle GPU kernel: score the released scrollprize/ink_9um checkpoints on the never-seen test sources.

Same data and metrics as train_v2.py: every supervised pixel of each test segment, both depth directions,
AUC / AP / best F0.5 / F0.5@thr. ink_9um outputs are label-smoothed (0.25..0.75), so F0.5@0.5 is reported on the
rescaled probability (p - 0.25) / 0.5 as the model card recommends.
"""
import glob, json, os, subprocess, sys, time
subprocess.run("pip install -q 'zarr==2.18.7' 'numcodecs<0.16' tifffile imagecodecs einops pynrrd cachetools", shell=True)
import numpy as np, zarr, tifffile
from numcodecs import Blosc

CONFIG = {"ckpts": ["hybrid_3d2d-seed42/step-075000.pth", "hybrid_3d2d-seed43/step-075000.pth"],
          "test_scrolls": ["0841", "MAN5", "0009B", "0500P2"], "batch": 64, "glob": "data*", "out_tag": "ink9um_baseline", "only": []}
OUT = "/kaggle/working"; TMP = "/tmp/b"; os.makedirs(TMP, exist_ok=True); T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
def sh(c):
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-4000:]); raise SystemExit("failed: " + c)
    return r
sh(f"git clone --depth 1 --filter=blob:none --sparse -b merge-ink-pipelines https://github.com/ScrollPrize/villa.git {TMP}/villa")
sh(f"cd {TMP}/villa && git sparse-checkout set ink-detection vesuvius")
sh(f"pip install -q --no-deps -e {TMP}/villa/vesuvius -e {TMP}/villa/ink-detection")
for ck in CONFIG["ckpts"]:
    sh(f"curl -sL -o {TMP}/{ck.replace('/', '_')} https://huggingface.co/scrollprize/ink_9um/resolve/main/{ck}")

def metrics(p, y):
    from sklearn.metrics import roc_auc_score, average_precision_score
    o = {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)), "n": int(len(y)), "pos": float(y.mean())}
    def f05(t):
        pr = p >= t; tp = (pr & y).sum(); fp = (pr & ~y).sum(); fn = (~pr & y).sum()
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        return 1.25 * prec * rec / max(1e-9, 0.25 * prec + rec)
    best = max((f05(t), t) for t in np.linspace(0.05, 0.95, 37))
    o["f05_best"], o["thr_best"], o["f05_at_0.5"] = float(best[0]), float(best[1]), float(f05(0.5))
    return o

res = {"config": CONFIG, "per_segment": {}, "pooled": {}}
pool = {}
os.makedirs(f"{OUT}/maps", exist_ok=True)
for p in sorted(glob.glob(f"/kaggle/input/**/{CONFIG['glob']}/*.zarr", recursive=True)):
    g = zarr.open_group(p, "r"); name = os.path.basename(p)[:-5]
    if g.attrs["scroll"] not in CONFIG["test_scrolls"] or name in res["per_segment"]: continue
    if CONFIG["only"] and name not in CONFIG["only"]: continue
    sup = g["sup"][:] > 0; ink = g["ink"][:] > 0
    ys, xs = np.nonzero(sup); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    y0, x0 = max(0, y0 - 64), max(0, x0 - 64); y1, x1 = min(sup.shape[0], y1 + 64), min(sup.shape[1], x1 + 64)
    vol = g["vol"][:, y0:y1, x0:x1]
    zp = f"{TMP}/{name}.zarr"
    zg = zarr.open_group(zp, mode="w")
    zg.create_dataset("0", data=vol, chunks=(vol.shape[0], 128, 128), compressor=Blosc(cname="zstd", clevel=1))
    zg.attrs["multiscales"] = [{"version": "0.4", "axes": [{"name": n, "type": "space"} for n in "zyx"],
                                "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}]}]
    m = sup[y0:y1, x0:x1]; yl = ink[y0:y1, x0:x1][m]
    res["per_segment"][name] = {}
    log(name, g.attrs["scroll"], vol.shape)
    for ck in CONFIG["ckpts"]:
        tag = ck.split("/")[0].split("-")[-1] + "_" + ck.split("-")[-1].split(".")[0]
        out = f"{TMP}/{name}_{tag}.tif"
        sh(f"cd {TMP}/villa/ink-detection && python -m koine_machines.inference.infer {zp} {TMP}/{ck.replace('/', '_')} {out} "
           f"--no-compile --num-workers 4 --batch-size {CONFIG['batch']} --direction both --layer-start 0 --layer-end {vol.shape[0]}")
        for dname, f in (("fwd", out), ("rev", out.replace(".tif", "_reverse.tif"))):
            pm = np.clip((tifffile.imread(f).astype(np.float32) / 255 - 0.25) / 0.5, 0, 1)
            pv = pm[m]
            r_ = metrics(pv, yl); res["per_segment"][name][f"{tag}_{dname}"] = r_; log("eval", name, tag, dname, r_)
            k = np.random.default_rng(0).choice(len(pv), size=min(len(pv), 300000), replace=False)
            pool.setdefault(f"{tag}_{dname}", ([], [], []))
            pool[f"{tag}_{dname}"][0].append(pv[k]); pool[f"{tag}_{dname}"][1].append(yl[k]); pool[f"{tag}_{dname}"][2].append(g.attrs["scroll"])
            np.savez_compressed(f"{OUT}/maps/ink9um_{tag}__{name}__{dname}.npz", p=(pm * 255).astype(np.uint8), bbox=np.array([y0, y1, x0, x1]))
for key, (ps, ys_, scs) in pool.items():
    res["pooled"][key] = metrics(np.concatenate(ps), np.concatenate(ys_)); log("POOLED", key, res["pooled"][key])
    for sc in sorted(set(scs)):
        idx = [i for i, c in enumerate(scs) if c == sc]
        res["pooled"][f"{key}_{sc}"] = metrics(np.concatenate([ps[i] for i in idx]), np.concatenate([ys_[i] for i in idx]))
        log("SCROLL", key, sc, res["pooled"][f"{key}_{sc}"])
json.dump(res, open(f"{OUT}/result_{CONFIG['out_tag']}.json", "w"), indent=1)
log("ALLDONE")
