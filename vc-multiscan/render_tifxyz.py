"""Render a surface volume from a tifxyz segment by sampling a remote OME-zarr scroll volume along surface normals.

Streams only the 128^3 chunks the surface touches (in-RAM LRU cache), tile by tile, so memory stays small.
Output: local OME-zarr group with level "0" of shape (n_slices, H, W) uint8, layer n_slices//2 on the surface,
same convention as public surface volumes (index increases along +normal).

Usage: render_tifxyz.py <tifxyz_dir> <volume s3 key prefix> <out.zarr> [--slices 31] [--tile 256] [--cache 300]
       [--flip-normal] [--crop y0 y1 x0 x1]   (crop in full-res output pixel coords)
"""
import argparse, json, os, time, urllib.request, urllib.error
from collections import OrderedDict
import concurrent.futures as cf
import numpy as np, tifffile, zarr
from scipy.ndimage import map_coordinates, zoom

B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
ap = argparse.ArgumentParser()
ap.add_argument("seg"); ap.add_argument("vol"); ap.add_argument("out")
ap.add_argument("--slices", type=int, default=31)
ap.add_argument("--tile", type=int, default=256)
ap.add_argument("--cache", type=int, default=300, help="chunks kept in RAM (2 MB each for uint8 128^3)")
ap.add_argument("--flip-normal", action="store_true")
ap.add_argument("--crop", type=int, nargs=4, default=None, help="y0 y1 x0 x1 in output pixel coords")
ap.add_argument("--align", action="store_true", help="rotate output so the scroll axis (+z) points up (fibers horizontal/vertical)")
ap.add_argument("--affine", default=None, help="reg.json with p_seg = M @ p_vol0 + t; samples are mapped into the volume by its inverse")
ap.add_argument("--offset", type=float, default=0.0, help="shift all sample depths along the normal (slices)")
ap.add_argument("--level", type=int, default=0, help="pyramid level of the volume to sample (coords divided by 2**level)")
a = ap.parse_args()

vol = a.vol.rstrip("/")
LV = a.level
meta = json.loads(urllib.request.urlopen(f"{B}/{vol}/{LV}/.zarray").read())
if a.affine:
    _r = json.load(open(a.affine)); _M = np.array(_r["M"], np.float64); _t = np.array(_r["t"], np.float64)
    AFF_INV = np.linalg.inv(_M); AFF_T = _t
else:
    AFF_INV = None
assert meta["compressor"] is None and meta["dtype"] == "|u1", meta
VZ, VY, VX = meta["shape"]; C = meta["chunks"][0]
assert meta["chunks"] == [C, C, C]
sep = meta.get("dimension_separator", ".")

# ---- surface geometry at full resolution (bilinear upsample of the 1/20 grid) ----
smeta = json.load(open(os.path.join(a.seg, "meta.json")))
step = int(round(1.0 / smeta["scale"][0]))
gx, gy, gz = (tifffile.imread(os.path.join(a.seg, f"{c}.tif")).astype(np.float32) for c in "xyz")
valid_g = (gx > 0) & (gy > 0) & (gz > 0)
gh, gw = gx.shape
H, W = (gh - 1) * step + 1, (gw - 1) * step + 1

# finite-difference normals on the coarse grid: n = d/du x d/dv (cross of row and column tangents)
P = np.stack([gx, gy, gz], -1)  # x,y,z order
du = np.zeros_like(P); dv = np.zeros_like(P)
du[:, 1:-1] = P[:, 2:] - P[:, :-2]; dv[1:-1] = P[2:] - P[:-2]
du[:, 0] = P[:, 1] - P[:, 0]; du[:, -1] = P[:, -1] - P[:, -2]
dv[0] = P[1] - P[0]; dv[-1] = P[-1] - P[-2]
N = np.cross(du, dv)
okn = valid_g.copy()
okn[:, 1:-1] &= valid_g[:, 2:] & valid_g[:, :-2]; okn[1:-1] &= valid_g[2:] & valid_g[:-2]
N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
# Sign matches the official vc_render_tifxyz surface volumes (verified on PHerc0800: r=0.975 at zero shift).
N = -N
if a.flip_normal:
    N = -N

# output -> full-res grid pixel mapping: g = cg + R (o - co)
if a.align:
    zr = np.nanmean(np.where(okn, dv[..., 2], np.nan)); zc = np.nanmean(np.where(okn, du[..., 2], np.nan))
    e = np.array([zr, zc]) / (np.hypot(zr, zc) + 1e-9)      # +z direction in (row, col) grid coords
    cos_, sin_ = -e[0], -e[1]                                  # R maps output-up (-1, 0) onto e
    Rg = np.array([[cos_, -sin_], [sin_, cos_]])
else:
    Rg = np.eye(2)
cg = np.array([(H - 1) / 2, (W - 1) / 2])
corners = np.array([[0, 0], [0, W - 1], [H - 1, 0], [H - 1, W - 1]], float) - cg
oc = corners @ np.linalg.inv(Rg).T                            # output coords of grid corners (centred)
Ho, Wo = (int(np.ceil(oc[:, 0].max() - oc[:, 0].min())) + 1, int(np.ceil(oc[:, 1].max() - oc[:, 1].min())) + 1)
co = np.array([(Ho - 1) / 2, (Wo - 1) / 2])
H, W = Ho, Wo
y0, y1, x0, x1 = a.crop if a.crop else (0, H, 0, W)
H2, W2 = y1 - y0, x1 - x0
D = a.slices
offs = np.arange(D, dtype=np.float32) - D // 2 + np.float32(a.offset)

g = zarr.open_group(a.out, mode="w")
from numcodecs import Blosc
outarr = g.create_dataset("0", shape=(D, H2, W2), chunks=(D, 128, 128), dtype=np.uint8,
                          compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE), fill_value=0)
g.attrs["multiscales"] = [{"version": "0.4", "axes": [{"name": n, "type": "space"} for n in "zyx"],
                           "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}]}]
g.attrs["render"] = {"segment": os.path.abspath(a.seg), "volume": vol, "slices": D, "flip_normal": a.flip_normal,
                     "crop": [y0, y1, x0, x1], "step": step, "affine": a.affine, "level": LV, "R_out_to_grid": Rg.tolist(),
                     "grid_centre": cg.tolist(), "out_centre": co.tolist()}

cache = OrderedDict()
stats = {"fetched": 0, "missing": 0}

def fetch(key):
    i, j, k = key
    try:
        b = urllib.request.urlopen(f"{B}/{vol}/{LV}/{i}{sep}{j}{sep}{k}", timeout=60).read()
        return key, np.frombuffer(b, np.uint8).reshape(C, C, C)
    except urllib.error.HTTPError:
        return key, None

pool = cf.ThreadPoolExecutor(16)

def ensure(keys):
    need = [k for k in keys if k not in cache]
    for k in keys:
        if k in cache:
            cache.move_to_end(k)
    for key, arr in pool.map(fetch, need):
        cache[key] = arr
        stats["fetched"] += 1
        stats["missing"] += arr is None
    while len(cache) > max(a.cache, len(keys)):
        cache.popitem(last=False)

def sample(zz, yy, xx):
    """Trilinear sample at float voxel coords (1-D arrays) from cached chunks."""
    out = np.zeros(zz.shape, np.float32)
    ci = np.stack([np.floor(zz).astype(np.int64) // C, np.floor(yy).astype(np.int64) // C, np.floor(xx).astype(np.int64) // C], 1)
    # each point needs the chunk of floor() and possibly +1 neighbours; build a 2x2x2-chunk local block per base chunk
    keys, inv = np.unique(ci, axis=0, return_inverse=True)
    inv = inv.ravel()
    # fetch every chunk this batch needs in one parallel round
    ensure(sorted({(i + a_, j + b_, k + c_) for i, j, k in keys for a_ in (0, 1) for b_ in (0, 1) for c_ in (0, 1)}))
    for n, (i, j, k) in enumerate(keys):
        sel = inv == n
        blk_keys = [(i + a_, j + b_, k + c_) for a_ in (0, 1) for b_ in (0, 1) for c_ in (0, 1)]
        ensure(blk_keys)
        blk = np.zeros((2 * C, 2 * C, 2 * C), np.uint8)
        for (ii, jj, kk) in blk_keys:
            arr = cache.get((ii, jj, kk))
            if arr is not None:
                blk[(ii - i) * C:(ii - i + 1) * C, (jj - j) * C:(jj - j + 1) * C, (kk - k) * C:(kk - k + 1) * C] = arr
        coords = np.stack([zz[sel] - i * C, yy[sel] - j * C, xx[sel] - k * C])
        out[sel] = map_coordinates(blk, coords, order=1, mode="constant", cval=0)
    return out

t0 = time.time()
T = a.tile
tiles = [(ty, tx) for ty in range(y0, y1, T) for tx in range(x0, x1, T)]
for n, (ty, tx) in enumerate(tiles):
    th, tw = min(T, y1 - ty), min(T, x1 - tx)
    oy, ox = np.meshgrid(np.arange(ty, ty + th, dtype=np.float64), np.arange(tx, tx + tw, dtype=np.float64), indexing="ij")
    gyf = cg[0] + Rg[0, 0] * (oy - co[0]) + Rg[0, 1] * (ox - co[1])
    gxf = cg[1] + Rg[1, 0] * (oy - co[0]) + Rg[1, 1] * (ox - co[1])
    FY, FX = gyf / step, gxf / step
    inside = (FY >= 0) & (FY <= gh - 1) & (FX >= 0) & (FX <= gw - 1)
    if not inside.any():
        continue
    gy0, gy1 = int(max(0, np.floor(FY[inside].min()))), int(min(gh - 1, np.ceil(FY[inside].max())))
    gx0, gx1 = int(max(0, np.floor(FX[inside].min()))), int(min(gw - 1, np.ceil(FX[inside].max())))
    if not valid_g[gy0:gy1 + 1, gx0:gx1 + 1].any():
        continue
    FY, FX = FY - gy0, FX - gx0
    def up(arr, order=1):
        return map_coordinates(arr[gy0:gy1 + 1, gx0:gx1 + 1], [FY, FX], order=order, mode="nearest")
    vmask = up(okn.astype(np.float32), order=0) > 0.5
    # require all 4 surrounding coarse nodes valid (bilinear support)
    vm4 = map_coordinates(okn[gy0:gy1 + 1, gx0:gx1 + 1].astype(np.float32), [FY, FX], order=1, mode="nearest") > 0.999
    vmask &= vm4 & inside
    if not vmask.any():
        continue
    px, py, pz = up(gx), up(gy), up(gz)
    nx, ny, nz = up(N[..., 0]), up(N[..., 1]), up(N[..., 2])
    nn = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-9
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    m = vmask.ravel()
    Z = (pz.ravel()[m][None] + offs[:, None] * nz.ravel()[m][None]).ravel()
    Y = (py.ravel()[m][None] + offs[:, None] * ny.ravel()[m][None]).ravel()
    X = (px.ravel()[m][None] + offs[:, None] * nx.ravel()[m][None]).ravel()
    if AFF_INV is not None:      # segment space -> volume level-0 space
        d = np.stack([Z - AFF_T[0], Y - AFF_T[1], X - AFF_T[2]])
        Z, Y, X = (AFF_INV @ d).astype(np.float32)
    if LV:
        Z, Y, X = Z / 2 ** LV, Y / 2 ** LV, X / 2 ** LV
    ok = (Z >= 0) & (Y >= 0) & (X >= 0) & (Z < VZ - 1) & (Y < VY - 1) & (X < VX - 1)
    vals = np.zeros(Z.shape, np.float32)
    vals[ok] = sample(Z[ok], Y[ok], X[ok])
    tile = np.zeros((D, th * tw), np.uint8)
    tile[:, m] = np.clip(vals.reshape(D, -1) + 0.5, 0, 255).astype(np.uint8)
    outarr[:, ty - y0:ty - y0 + th, tx - x0:tx - x0 + tw] = tile.reshape(D, th, tw)
    if n % 10 == 0:
        print(f"tile {n + 1}/{len(tiles)} fetched={stats['fetched']} ({stats['fetched'] * C ** 3 / 1e9:.2f} GB) "
              f"missing={stats['missing']} {time.time() - t0:.0f}s", flush=True)
print("done", a.out, (D, H2, W2), stats, f"{time.time() - t0:.0f}s", flush=True)
