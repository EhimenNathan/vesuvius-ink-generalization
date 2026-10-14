"""Kaggle CPU kernel: PHerc1447 w058 / w060 (YoussefMoNader/ink-8um-pherc1447-surfaces) as a fresh held-out test source.

Unseen by our models, by ink_9um and by the v8-in base model. Source render: 24 layers at 8.64 um, layer NN at normal
offset NN-11.5 voxels (ascending). Output /kaggle/working/data5/<name>.zarr in our test-store layout:
  vol (17, H, W) uint8 at 9.6 um: 17 depths spaced 9.6 um centred on offset 0 (linear between layers, then INTER_AREA
      resize by 8.64/9.6), stored in ascending offset order like the ink-bucket renders;
  ink = label > 127 (area-weighted >= 0.5 after resize); sup = certainty mask fully non-zero in the resized pixel;
  val = zeros; attrs scroll "1447", um 9.6.
Also writes maps/v8inRel__<name>__rev.npz: the publisher's released base-v8-in prediction (run with --reverse),
resized onto the same grid, in the map format of train_v2.py (uint8 in the supervised bbox).
"""
import json, os, subprocess, time, urllib.request
subprocess.run("pip install -q 'zarr==2.18.7' 'numcodecs<0.16' imagecodecs", shell=True)
import numpy as np, zarr, cv2, tifffile
from numcodecs import Blosc
T0 = time.time(); OUT = "/kaggle/working/data5"; MAPS = "/kaggle/working/maps"
os.makedirs(OUT, exist_ok=True); os.makedirs(MAPS, exist_ok=True)
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
REV = "7e4d918712a8d8642a6fdeee45b80217c1e98317"   # pinned dataset revision
B = f"https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/resolve/{REV}"
SRC_UM, DST_UM, D_OUT, N_LAYERS = 8.64, 9.6, 17, 24

def fetch(path, tries=8):
    for t in range(tries):
        try: return urllib.request.urlopen(f"{B}/{path}", timeout=600).read()
        except Exception as e:
            log("retry", path, repr(e)); time.sleep(min(60, 2 ** t))
    raise RuntimeError("failed " + path)
def png(path): return cv2.imdecode(np.frombuffer(fetch(path), np.uint8), cv2.IMREAD_GRAYSCALE)

summary = {}
for w in ("w058", "w060"):
    name = f"pherc1447-{w}"; t = time.time()
    layers = [tifffile.imread(__import__("io").BytesIO(fetch(f"{w}/layers/{i:02d}.tif"))) for i in range(N_LAYERS)]
    H0, W0 = layers[0].shape
    assert all(l.shape == (H0, W0) and l.dtype == np.uint8 for l in layers)
    s = SRC_UM / DST_UM; H, W = int(round(H0 * s)), int(round(W0 * s))
    c = (N_LAYERS - 1) / 2; r = DST_UM / SRC_UM
    vol = np.zeros((D_OUT, H, W), np.uint8); z_used = []
    for k in range(D_OUT):
        z = c + (k - (D_OUT - 1) / 2) * r; assert 0 <= z <= N_LAYERS - 1
        z0 = int(np.floor(z)); z1 = min(z0 + 1, N_LAYERS - 1); f = z - z0; z_used.append(round(z, 3))
        sl = layers[z0].astype(np.float32) * (1 - f) + layers[z1].astype(np.float32) * f
        vol[k] = np.clip(np.rint(cv2.resize(sl, (W, H), interpolation=cv2.INTER_AREA)), 0, 255).astype(np.uint8)
    del layers
    ink_f = cv2.resize((png(f"{w}/labels/inklabels.png") > 127).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    m_f = cv2.resize((png(f"{w}/labels/mask.png") > 0).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    ink = ((ink_f >= 0.5) * 255).astype(np.uint8); sup = ((m_f >= 0.999) * 255).astype(np.uint8)
    g = zarr.open_group(f"{OUT}/{name}.zarr", mode="w"); comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    g.create_dataset("vol", data=vol, chunks=(D_OUT, 128, 128), compressor=comp, fill_value=0)
    for k, v in (("ink", ink), ("sup", sup), ("val", np.zeros_like(sup))):
        g.create_dataset(k, data=v, chunks=(512, 512), compressor=comp)
    g.attrs.update({"scroll": "1447", "family": "pherc1447-v8in-surfaces", "source": f"{B}/{w}", "depth": D_OUT,
                    "um": DST_UM, "layer_positions": z_used, "depth_order": "ascending normal offset (as released)"})
    # released base v8-in prediction (zero-shot on PHerc1447) on the same grid
    ys, xs = np.nonzero(sup > 0); y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    pr = cv2.resize(png(f"{w}/predictions/v8in.png").astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    np.savez_compressed(f"{MAPS}/v8inRel__{name}__rev.npz", p=np.clip(np.rint(pr[y0:y1, x0:x1]), 0, 255).astype(np.uint8),
                        bbox=np.array([y0, y1, x0, x1]))
    sm = sup > 0
    summary[name] = {"src_shape": [H0, W0], "shape": [H, W], "sup_cm2": float(sm.sum() * (DST_UM * 1e-4) ** 2),
                     "ink_frac": float((ink[sm] > 0).mean()), "layer_positions": z_used, "s": round(time.time() - t)}
    log(name, json.dumps(summary[name]))
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
log("ALLDONE")
