"""Kaggle CPU kernel: prepare never-seen test sources (PHerc0841 x3, MAN5, fragments 0009B and 0500P2) from the
scrollprize `ink` label bucket.

Per segment: labels (full-res 2.4 um TIFF, *_v2 when present) are block-reduced 4x to the bucket zarr's level 2
(~9.6 um, 17 slices, already pooled in z), and only supervised 128-tiles (+1 margin) of level 2 are fetched.
Output /kaggle/working/data2/<name>.zarr with vol (17,H,W), ink, sup, val (H,W) uint8 - same layout as prep_kernel.
"""
import json, os, time, urllib.request, urllib.error, concurrent.futures as cf, subprocess, io
subprocess.run("pip install -q 'zarr==2.18.7' 'numcodecs<0.16' imagecodecs", shell=True)
import numpy as np, zarr, numcodecs, tifffile
from numcodecs import Blosc
from scipy.ndimage import binary_dilation

R = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink"
OUT = "/kaggle/working/data3"; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
SEGS = {   # name: (bucket dir, file stem, scroll)
    "pherc0841-w00": ("841/w00", "w00", "0841"),
    "pherc0841-ag4552896": ("841/auto_grown_20260220144552896", "auto_grown_20260220144552896", "0841"),
    "pherc0841-ag4252405": ("841/auto_grown_20260220174252405", "auto_grown_20260220174252405", "0841"),
    "phercman5-outer3": ("man5/MAN5_outer_3", "MAN5_outer_3", "MAN5"),
    "pherc0500p2-frag": ("0500p2/-1", "-1", "0500P2"),
    "pherc0009b-frag": ("0009b/auto_grown_20250919055754487_inp_hr_2um", "auto_grown_20250919055754487_inp_hr_2um", "0009B"),
}

def get(url, tries=8):
    for t in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=300).read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            ra = e.headers.get("Retry-After") if e.headers else None
            time.sleep(float(ra) if ra and ra.isdigit() else min(60, 2 ** t))
        except Exception:
            time.sleep(min(60, 2 ** t))
    raise RuntimeError("failed " + url)

def tif(url):
    b = get(url)
    if b is None: return None
    a = tifffile.imread(io.BytesIO(b))
    return a if a.ndim == 2 else a[..., 0] if a.shape[-1] in (3, 4) else a[a.shape[0] // 2]

def reduce4(m, H2, W2):
    """Block-mean 4x on the level-2 grid, in row strips (labels can be several gigapixels)."""
    out = np.zeros((H2, W2), np.float32)
    H, W = m.shape; W4 = min(W, W2 * 4) // 4 * 4
    for r0 in range(0, H2, 256):
        r1 = min(H2, r0 + 256); a = m[r0 * 4:r1 * 4, :W4] > 0
        hh = a.shape[0] // 4
        if hh == 0: break
        out[r0:r0 + hh, :W4 // 4] = a[:hh * 4].reshape(hh, 4, W4 // 4, 4).mean((1, 3))
    return out

summary = {}
import os as _os
OUT = _os.environ.get("PREP_OUT", OUT); _os.makedirs(OUT, exist_ok=True)
for name, spec in SEGS.items():
    d, stem, scroll = spec[:3]; lstem = spec[3] if len(spec) > 3 else stem     # label files may use a different stem
    t = time.time(); base = f"{R}/{d}/{stem}"; lbase = f"{R}/{d}/{lstem}"
    m0 = json.loads(get(f"{base}.zarr/0/.zarray"))
    Z0, H0, W0 = m0["shape"]; c0z, c0y, c0x = m0["chunks"]; sep0 = m0.get("dimension_separator", ".")
    codec0 = numcodecs.get_codec(m0["compressor"]) if m0.get("compressor") else None
    VZ = -(-Z0 // 4); H2, W2 = -(-H0 // 4), -(-W0 // 4)           # correct level-2 grid: x/y/z all pooled 4x
    assert c0z >= Z0, m0
    lab = {}; red = {}
    for kind in ("inklabels", "supervision_mask", "validation_mask"):
        a = tif(f"{lbase}_{kind}_v2.tif"); src = "v2"
        if a is None:
            a = tif(f"{lbase}_{kind}.tif"); src = "v1"
        lab[kind] = (None, src)
        log(name, kind, None if a is None else a.shape, src)
        red[kind] = reduce4(a, H2, W2) if a is not None else None
        del a; import gc; gc.collect()
    ink_f = red["inklabels"]; sup_f = red["supervision_mask"]
    val_f = red["validation_mask"] if red["validation_mask"] is not None else np.zeros((H2, W2), np.float32)
    ink = ((ink_f >= 0.5) * 255).astype(np.uint8)
    sup = ((sup_f >= 0.99) * 255).astype(np.uint8)          # fully supervised 4x4 blocks only
    val = ((val_f > 0) * 255).astype(np.uint8)
    T = 128
    ty, tx = -(-H2 // T), -(-W2 // T)
    sm = sup > 0
    tiles = np.zeros((ty, tx), bool)
    for i in range(ty):
        for j in range(tx):
            tiles[i, j] = sm[i * T:(i + 1) * T, j * T:(j + 1) * T].any()
    tiles = binary_dilation(tiles, iterations=1)
    g = zarr.open_group(f"{OUT}/{name}.zarr", mode="w")
    comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    vds = g.create_dataset("vol", shape=(VZ, H2, W2), chunks=(VZ, T, T), dtype=np.uint8, compressor=comp, fill_value=0)
    def l0_chunk(i, j):
        try:
            b = get(f"{base}.zarr/0/0{sep0}{i}{sep0}{j}")
        except RuntimeError as e:
            raise RuntimeError("Source chunk fetch failed; refusing to retain supervision over unknown input") from e
        if b is None: return None
        if codec0: b = codec0.decode(b)
        return np.frombuffer(b, np.uint8).reshape(c0z, c0y, c0x)[:Z0]
    def vt(k):
        """Level-2 tile (ti, tj): assemble the 512x512 level-0 block, mean-pool 4x in z, y, x."""
        ti, tj = k; y0, x0 = ti * T * 4, tj * T * 4; y1, x1 = min(H0, y0 + T * 4), min(W0, x0 + T * 4)
        blk = np.zeros((VZ * 4, T * 4, T * 4), np.float32); cnt = np.zeros((VZ * 4, 1, 1), np.float32)
        for ci in range(y0 // c0y, (y1 - 1) // c0y + 1):
            for cj in range(x0 // c0x, (x1 - 1) // c0x + 1):
                a = l0_chunk(ci, cj)
                if a is None: continue
                sy0, sy1 = max(y0, ci * c0y), min(y1, (ci + 1) * c0y); sx0, sx1 = max(x0, cj * c0x), min(x1, (cj + 1) * c0x)
                blk[:Z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[:, sy0 - ci * c0y:sy1 - ci * c0y, sx0 - cj * c0x:sx1 - cj * c0x]
        cnt[:Z0] = 1
        zs = blk.reshape(VZ, 4, T * 4, T * 4).sum(1) / np.maximum(cnt.reshape(VZ, 4, 1, 1).sum(1), 1)   # z mean (partial last group)
        pooled = np.rint(zs.reshape(VZ, T, 4, T, 4).mean((2, 4)))
        h, w = min(T, H2 - ti * T), min(T, W2 - tj * T)
        vds[:, ti * T:ti * T + h, tj * T:tj * T + w] = pooled[:, :h, :w].astype(np.uint8)
    with cf.ThreadPoolExecutor(12) as ex:
        list(ex.map(vt, [tuple(int(v) for v in k) for k in np.argwhere(tiles)]))
    for k, v in (("ink", ink), ("sup", sup), ("val", val)):
        g.create_dataset(k, data=v, chunks=(512, 512), compressor=comp)
    g.attrs.update({"scroll": scroll, "family": "ink-bucket-L0-pooled444", "source": base, "level": "2", "depth": int(VZ),
                    "label_versions": {k: v[1] for k, v in lab.items()}, "n_tiles": int(tiles.sum()), "um": 9.6})
    summary[name] = {"scroll": scroll, "depth": int(VZ), "tiles": int(tiles.sum()), "sup_cm2": float(sm.sum() * 9.6e-4 ** 2),
                     "ink_frac": float((ink[sm] > 0).mean()) if sm.any() else 0.0, "val_frac_of_sup": float((val[sm] > 0).mean()) if sm.any() else 0.0,
                     "s": round(time.time() - t)}
    log(name, json.dumps(summary[name]))
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
log("ALLDONE")
