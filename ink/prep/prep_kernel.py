"""Kaggle CPU kernel: build a compact ink_9um training set (supervised tiles only).

Per segment output (/kaggle/working/data/<seg>.npz-like zarr group):
  vol   uint8 (21, H, W)  sparse: only supervised tiles (+1-tile margin) are filled, rest 0
  ink   uint8 (H, W)       label plane (0/255)
  sup   uint8 (H, W)       supervision mask
  val   uint8 (H, W)       official validation mask (3 segments) else absent
  meta  json: scroll, source, tiles
Pooling follows villa scripts/prepare_9um_isotropic_input.py: level 2 of a 2.4 um surface render, centred 84 z planes,
rounded mean over 4 -> 21 slices. Native 9.362 um renders (28 slices): centred 21 slices (label plane 14 -> 10).
"""
import json, os, sys, time, urllib.request, urllib.error, concurrent.futures as cf, subprocess
subprocess.run("pip install -q 'zarr==2.18.7' 'numcodecs<0.16'", shell=True)
import numpy as np, zarr, numcodecs
from numcodecs import Blosc

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels"
OUT = "/kaggle/working/data"; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

A = "aligned-scrollprizeorg-21slices"; N = "native9-scrollprizeorg-21slices"
V139 = "2.399um-0.22m-78keV-volume-20260102150214.zarr"; V1667 = "2.399um-0.22m-78keV-volume-20251217075048.zarr"
VP4 = "2.4um-0.22m-78keV-volume-20260411134726.zarr"; V814 = "2.399um-0.22m-78keV-volume-20260309142202.zarr"
V9 = "9.362um-1.2m-113keV-volume-20250728140407.zarr"
SEGS = {
 "pherc0139-w016": (A, "0139", f"PHerc0139/segments/20250108000004-w029_2025010827/surface-volumes/{V139}"),
 "pherc0139-w017": (A, "0139", f"PHerc0139/segments/20250108000005-w030_2025010818/surface-volumes/{V139}"),
 "pherc0139-w028": (A, "0139", f"PHerc0139/segments/20260115000000-w044_2026011522/surface-volumes/{V139}"),
 "pherc0139-w029": (A, "0139", f"PHerc0139/segments/20260126000000-w045_2026012619/surface-volumes/{V139}"),
 "pherc0139-w035": (A, "0139", f"PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/{V139}"),
 "pherc0139-w039": (A, "0139", f"PHerc0139/segments/20260302000000-w039_2026030210/surface-volumes/{V139}"),
 "pherc0139-w040": (A, "0139", f"PHerc0139/segments/20250831000000-w040_2025083102/surface-volumes/{V139}"),
 "pherc0139-w041": (A, "0139", f"PHerc0139/segments/20260108000000-w041_2026010816/surface-volumes/{V139}"),
 "pherc0139-w043": (A, "0139", f"PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/{V139}"),
 "pherc0814-46527": (A, "0814", f"PHerc0814/segments/20260226000000-46527_2um_try2/surface-volumes/{V814}"),
 "pherc1667-w013": (A, "1667", f"PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/surface-volumes/{V1667}"),
 "pherc1667-w018": (A, "1667", f"PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/surface-volumes/{V1667}"),
 "pherc1667-w023": (A, "1667", f"PHerc1667/segments/20240304161941-w023_20240304161941_flatboi/surface-volumes/{V1667}"),
 "pherc1667-w028": (A, "1667", f"PHerc1667/segments/20251208130119-w028_20251208130119156_flatboi/surface-volumes/{V1667}"),
 "pherc1667-w029": (A, "1667", f"PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi/surface-volumes/{V1667}"),
 "pherc1667-w031": (A, "1667", f"PHerc1667/segments/20251223230000-w031_2025122323_flatboi/surface-volumes/{V1667}"),
 "phercparis4-w00": (A, "Paris4", f"PHercParis4/segments/20231016151002/surface-volumes/{VP4}"),
 "phercparis4-w01": (A, "Paris4", f"PHercParis4/segments/20230702185753/surface-volumes/{VP4}"),
 "phercparis4-w02": (A, "Paris4", f"PHercParis4/segments/20231031143852/surface-volumes/{VP4}"),
 "phercparis4-w03": (A, "Paris4", f"PHercParis4/segments/20231106155351/surface-volumes/{VP4}"),
 "phercparis4-w05": (A, "Paris4", f"PHercParis4/segments/20231012184424/surface-volumes/{VP4}"),
 "phercparis4-w06": (A, "Paris4", f"PHercParis4/segments/20231210121321/surface-volumes/{VP4}"),
 "phercparis4-w07": (A, "Paris4", f"PHercParis4/segments/20231007101619/surface-volumes/{VP4}"),
 "phercparis4-w09": (A, "Paris4", f"PHercParis4/segments/20230929220926/surface-volumes/{VP4}"),
 "w035": (N, "0139", f"PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/{V9}"),
 "w039": (N, "0139", f"PHerc0139/segments/20260302000000-w039_2026030210/surface-volumes/{V9}"),
 "w040": (N, "0139", f"PHerc0139/segments/20250831000000-w040_2025083102/surface-volumes/{V9}"),
 "w041": (N, "0139", f"PHerc0139/segments/20260108000000-w041_2026010816/surface-volumes/{V9}"),
 "w044": (N, "0139", f"PHerc0139/segments/20260115000000-w044_2026011522/surface-volumes/{V9}"),
}
ONLY = os.environ.get("ONLY")   # smoke test: comma list of segments
if ONLY:
    SEGS = {k: v for k, v in SEGS.items() if k in ONLY.split(",")}

FAILED = []
def get(url, tries=8, soft=False):
    for t in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            ra = e.headers.get("Retry-After") if e.headers else None
            time.sleep(float(ra) if ra and ra.isdigit() else min(60, 2 ** t))
        except Exception:
            time.sleep(min(60, 2 ** t))
    if soft:
        FAILED.append(url); return b"__FAILED__"
    raise RuntimeError("failed " + url)

def decode(meta, b):
    if b is None or b == b"__FAILED__":
        return None
    if meta.get("compressor"):
        b = numcodecs.get_codec(meta["compressor"]).decode(b)
    return np.frombuffer(b, np.dtype(meta["dtype"])).reshape(meta["chunks"])

def read_plane(base, plane, tiles=None):
    """Read one z plane (H, W) of a label zarr level 0; restrict to chunk grid cells where tiles is True."""
    meta = json.loads(get(f"{base}/0/.zarray"))
    Z, H, W = meta["shape"]; cz, cy, cx = meta["chunks"]; sep = meta.get("dimension_separator", ".")
    out = np.zeros((H, W), np.uint8); bad = []
    keys = [(i, j) for i in range(-(-H // cy)) for j in range(-(-W // cx)) if tiles is None or tiles[i, j]]
    zc = plane // cz
    def one(k):
        i, j = k
        b = get(f"{base}/0/{zc}{sep}{i}{sep}{j}", soft=True)
        if b == b"__FAILED__":
            bad.append((i, j)); return
        a = decode(meta, b)
        if a is not None:
            blk = a[plane - zc * cz]
            h, w = min(cy, H - i * cy), min(cx, W - j * cx)
            out[i * cy:i * cy + h, j * cx:j * cx + w] = blk[:h, :w]
    with cf.ThreadPoolExecutor(12) as ex:
        list(ex.map(one, keys))
    out_bad = np.zeros((-(-H // cy), -(-W // cx)), bool)
    for i, j in bad: out_bad[i, j] = True
    return out, meta, out_bad

summary = {}
for seg, (fam, scroll, svpath) in SEGS.items():
    t = time.time()
    lb = f"{HF}/{fam}/{seg}/{seg}"
    za = json.loads(get(f"{lb}_inklabels.zarr/.zattrs") or b"{}")
    plane = int(za.get("annotation_center_channel", 10 if fam == A else 14))
    sup, lmeta, bad_s = read_plane(f"{lb}_supervision_mask.zarr", plane)
    if sup.max() == 0:
        sup, _, bad_s = read_plane(f"{lb}_supervision_mask.zarr", 0)
    H, W = sup.shape
    T = 128
    ty, tx = -(-H // T), -(-W // T)
    sm = sup > 0
    tiles0 = np.zeros((ty, tx), bool)
    for i in range(ty):
        for j in range(tx):
            tiles0[i, j] = sm[i * T:(i + 1) * T, j * T:(j + 1) * T].any()
    from scipy.ndimage import binary_dilation
    tiles = binary_dilation(tiles0, iterations=1)
    ink, _, bad_i = read_plane(f"{lb}_inklabels.zarr", plane, tiles)
    for i, j in np.argwhere(bad_i):          # unknown ink -> drop supervision there (never train on a false negative)
        sup[i * T:(i + 1) * T, j * T:(j + 1) * T] = 0
    sm = sup > 0
    val = None
    if get(f"{lb}_validation_mask.zarr/0/.zarray") is not None:
        val, _, bad_v = read_plane(f"{lb}_validation_mask.zarr", plane, tiles)
        if val.max() == 0:
            val, _, bad_v = read_plane(f"{lb}_validation_mask.zarr", 0, tiles)
        for i, j in np.argwhere(bad_v):      # unknown validation membership -> exclude from training
            val[i * T:(i + 1) * T, j * T:(j + 1) * T] = 255
    log(seg, "labels", (H, W), "plane", plane, "sup frac %.3f ink frac(in sup) %.3f" % (sm.mean(), (ink[sm] > 0).mean() if sm.any() else 0),
        "tiles", int(tiles.sum()), "val" if val is not None else "")
    # surface volume
    level = "2" if fam == A else "0"
    vmeta = json.loads(get(f"{S3}/{svpath}/{level}/.zarray"))
    VZ, VH, VW = vmeta["shape"]; cz, cy, cx = vmeta["chunks"]; vsep = vmeta.get("dimension_separator", ".")
    assert (VH, VW) == (H, W) or abs(VH - H) <= 2, (seg, (VH, VW), (H, W))
    if fam == A:
        z0 = (VZ - 84) // 2; z1 = z0 + 84
    else:
        z0 = (VZ - 21 + 1) // 2; z1 = z0 + 21      # 28 -> 4..24 so plane 14 maps to 10
    assert cy == T and cx == T and cz >= VZ, vmeta
    g = zarr.open_group(f"{OUT}/{seg}.zarr", mode="w")
    comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    vds = g.create_dataset("vol", shape=(21, H, W), chunks=(21, T, T), dtype=np.uint8, compressor=comp, fill_value=0)
    def vt(k):
        i, j = k
        a = decode(vmeta, get(f"{S3}/{svpath}/{level}/0{vsep}{i}{vsep}{j}"))
        if a is None:
            return
        a = a[z0:z1].astype(np.float32)
        if fam == A:
            a = np.rint(a.reshape(21, 4, T, T).mean(1))
        h, w = min(T, H - i * T), min(T, W - j * T)
        vds[:, i * T:i * T + h, j * T:j * T + w] = a[:, :h, :w].astype(np.uint8)
    keys = [tuple(k) for k in np.argwhere(tiles)]
    with cf.ThreadPoolExecutor(24) as ex:
        list(ex.map(vt, keys))
    g.create_dataset("ink", data=ink, chunks=(512, 512), compressor=comp)
    g.create_dataset("sup", data=sup, chunks=(512, 512), compressor=comp)
    if val is not None:
        g.create_dataset("val", data=val, chunks=(512, 512), compressor=comp)
    g.attrs.update({"scroll": scroll, "family": fam, "source": svpath, "level": level, "z_range": [int(z0), int(z1)],
                    "label_plane": plane, "n_tiles": int(tiles.sum()), "um": 9.596 if fam == A else 9.362})
    summary[seg] = {"failed_label_chunks": int(bad_s.sum() + bad_i.sum()), "scroll": scroll, "tiles": int(tiles.sum()), "sup_cm2": float((sup > 0).sum() * (9.6e-4) ** 2),
                    "ink_frac": float((ink[sup > 0] > 0).mean()) if (sup > 0).any() else 0.0, "s": round(time.time() - t)}
    log(seg, json.dumps(summary[seg]))
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
os.system(f"du -sh {OUT}")
log("ALLDONE")
