#!/usr/bin/env python
"""Register two CT scans of the same Herculaneum scroll (similarity transform, coarse-to-fine).

Typical use: map a high-resolution partial scan (e.g. 2.4 um, ~36 mm of height) onto the full-scroll scan whose
coordinates the segments (tifxyz) live in (e.g. 9.36 um), so existing segments can be rendered from the better scan.

Model (zyx voxel coordinates):  p_fixed0 = M @ p_moving0 + t
  rotation about the scroll axis (z), optional in-plane mirror, optional z flip, xy scale, z scale, 3D translation.

Stages
  1. z offset/scale/flip from cross-section-area profiles along z (rotation-independent)
  2. in-plane rotation + mirror search by 2D NCC on matched axial slices
  3. Powell refinement of 3D NCC at --coarse-um (default 150 um)
  4. Powell refinement at half that resolution over the overlap
Outputs: <out>/reg.json (M, t, parameters, NCC scores) and overlay PNGs (red = fixed, green = warped moving).

Example
  python register_scans.py \\
    --fixed  PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr \\
    --moving PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr --out reg_1203
"""
import argparse, json, os, re, time, urllib.request, urllib.error, concurrent.futures as cf
import numpy as np
from scipy import ndimage, optimize
from PIL import Image

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--fixed", required=True, help="OME-zarr volume (S3 key under --base, or full http(s) URL)")
ap.add_argument("--moving", required=True)
ap.add_argument("--fixed-um", type=float, default=None, help="voxel size; parsed from the name ('9.362um') if omitted")
ap.add_argument("--moving-um", type=float, default=None)
ap.add_argument("--coarse-um", type=float, default=150.0)
ap.add_argument("--base", default="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com")
ap.add_argument("--threads", type=int, default=32)
ap.add_argument("--out", default="reg_out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
def log(*x): print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)

def url(v): return v.rstrip("/") if v.startswith("http") else f"{a.base}/{v.strip('/')}"
def vox_um(v, given):
    if given: return given
    m = re.search(r"(\d+\.\d+)um", v)
    if not m: raise SystemExit(f"cannot parse voxel size from {v}; pass --fixed-um/--moving-um")
    return float(m.group(1))
def zarray(v, level):
    try: return json.loads(urllib.request.urlopen(f"{url(v)}/{level}/.zarray").read())
    except urllib.error.HTTPError: return None

def choose(v, um, target):
    """Pick (level, extra block-mean factor) whose voxel size is closest to target."""
    best = None
    for L in range(0, 8):
        if zarray(v, L) is None: break
        for f in (1, 2, 3, 4):
            d = round(abs(np.log(um * 2 ** L * f / target)), 4)
            # ties go to the smaller extra factor = coarser stored level = far less data to stream
            if best is None or (d, f) < (best[0], best[2]): best = (d, L, f)
    return best[1], best[2]

def fetch(v, level, f=1, zr=None):
    import numcodecs
    m = zarray(v, level); Z, Y, X = m["shape"]; C = m["chunks"]; sep = m.get("dimension_separator", ".")
    codec = numcodecs.get_codec(m["compressor"]) if m.get("compressor") else None
    zr = (0, Z) if zr is None else (max(0, zr[0]), min(Z, zr[1]))
    out = np.zeros((zr[1] - zr[0], Y, X), np.uint8)
    keys = [(i, j, k) for i in range(zr[0] // C[0], -(-zr[1] // C[0])) for j in range(-(-Y // C[1])) for k in range(-(-X // C[2]))]
    def get(key):
        i, j, k = key
        try: b = urllib.request.urlopen(f"{url(v)}/{level}/{i}{sep}{j}{sep}{k}", timeout=120).read()
        except urllib.error.HTTPError: return
        if codec: b = codec.decode(b)
        c = np.frombuffer(b, np.dtype(m["dtype"])).reshape(C)
        z0, y0, x0 = i * C[0], j * C[1], k * C[2]
        zs, ze = max(z0, zr[0]), min(z0 + C[0], zr[1]); ye, xe = min(y0 + C[1], Y), min(x0 + C[2], X)
        out[zs - zr[0]:ze - zr[0], y0:ye, x0:xe] = c[zs - z0:ze - z0, :ye - y0, :xe - x0]
    with cf.ThreadPoolExecutor(a.threads) as ex: list(ex.map(get, keys))
    if f > 1:
        z, y, x = (s // f * f for s in out.shape)
        return out[:z, :y, :x].reshape(z // f, f, y // f, f, x // f, f).mean((1, 3, 5)).astype(np.float32)
    return out.astype(np.float32)

def ncc(x, y):
    m = (x > 0) & (y > 0)
    if m.sum() < 100: return -1.0
    x, y = x[m] - x[m].mean(), y[m] - y[m].mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))

um_f, um_m = vox_um(a.fixed, a.fixed_um), vox_um(a.moving, a.moving_um)
Lf, ff = choose(a.fixed, um_f, a.coarse_um); Lm, fm = choose(a.moving, um_m, a.coarse_um)
kf, km = 2 ** Lf * ff, 2 ** Lm * fm                      # level-0 voxels per coarse voxel
vf, vm = fetch(a.fixed, Lf, ff), fetch(a.moving, Lm, fm)
s_nom = (um_m * km) / (um_f * kf)
log(f"coarse fixed L{Lf}x{ff} {vf.shape} ({um_f * kf:.1f} um)  moving L{Lm}x{fm} {vm.shape} ({um_m * km:.1f} um)  s_nom {s_nom:.4f}")

# ---- stage 1: z alignment from area profiles ----
def area(v):
    thr = np.percentile(v[v > 0], 20) if (v > 0).any() else 1
    return (v > thr).mean((1, 2))
af, am = area(vf), area(vm); best = None
for s in np.linspace(s_nom * 0.97, s_nom * 1.03, 13):
    for sign in (1, -1):
        zm = np.arange(len(am))
        for z0 in range(-len(am), len(af)):
            zf = z0 + sign * s * (zm - (0 if sign > 0 else len(am) - 1))
            ok = (zf >= 0) & (zf <= len(af) - 1)
            if ok.sum() < 0.8 * min(len(am), len(af) / s): continue
            p, q = np.interp(zf[ok], np.arange(len(af)), af), am[ok]
            if p.std() < 1e-6 or q.std() < 1e-6: continue
            c = np.corrcoef(p, q)[0, 1]
            if best is None or c > best[0]: best = (c, s, sign, z0)
_, s_z, zsign, z0 = best
log("stage1 area-profile corr %.3f  z-scale %.4f  z-sign %+d  z0 %d" % best)
def toF(zmv): return z0 + zsign * s_z * (zmv - (0 if zsign > 0 else len(am) - 1))

# ---- stage 2: rotation / mirror ----
cm2, cf2 = np.array(vm.shape[1:]) / 2.0, np.array(vf.shape[1:]) / 2.0
def warp2d(img, th, mir):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); Mm = np.diag([1, -1]) if mir else np.eye(2)
    A = np.linalg.inv(Mm @ R) / s_nom
    return ndimage.affine_transform(img, A, offset=cm2 - A @ cf2, output_shape=vf.shape[1:], order=1)
res = []
for mir in (False, True):
    for th in np.deg2rad(np.arange(0, 360, 3)):
        sc = [ncc(warp2d(vm[int(round(z))], th, mir), vf[int(round(toF(z)))])
              for z in np.linspace(0.2, 0.8, 5) * (len(am) - 1) if 0 <= round(toF(z)) < len(af)]
        res.append((float(np.mean(sc)), th, mir))
res.sort(key=lambda r: -r[0]); _, th0, mir = res[0]
log("stage2 rotation top3", [(round(r[0], 3), round(float(np.rad2deg(r[1])), 1), r[2]) for r in res[:3]])

# ---- stage 3/4: 3D refinement ----
def Fmat(th, s, sz):
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    return np.diag([sz, s, s]) @ np.diag([zsign, 1, -1 if mir else 1]) @ R      # moving-centred -> fixed-centred (coarse)
def centres(p):
    th, s, tz, ty, tx, sz = p
    return np.array(vm.shape) / 2.0, np.array([toF(vm.shape[0] / 2.0), *cf2]) + np.array([tz, ty, tx])
def overlap_z(p):
    F = Fmat(p[0], p[1], p[5]); cmv, cfx = centres(p)
    e = [cfx[0] + F[0, 0] * (z - cmv[0]) for z in (0, vm.shape[0])]
    return max(0, int(min(e)) + 2), min(vf.shape[0], int(max(e)) - 2)
def warp(src, F, c_src, c_ref, origin, shape, stride):
    A = np.linalg.inv(F)
    return ndimage.affine_transform(src, A * stride, offset=c_src + A @ (np.asarray(origin, float) - c_ref), output_shape=shape, order=1)
def make_cost(src, ref, rf, rm, zoff, stride):
    """rf/rm: fine-per-coarse factors for fixed/moving; ref covers fixed z from zoff (fine units)."""
    refs = ref[::stride, ::stride, ::stride]
    def cost(p):
        F = Fmat(p[0], p[1], p[5]) * (rf / rm); cmv, cfx = centres(p)
        return -ncc(warp(src, F, cmv * rm, cfx * rf - np.array([zoff, 0, 0]), (0, 0, 0), refs.shape, stride), refs)
    return cost
p0 = np.array([th0, s_nom, 0, 0, 0, s_z]); zl, zh = overlap_z(p0)
r3 = optimize.minimize(make_cost(vm, vf[zl:zh], 1, 1, zl, 2), p0, method="Powell", options={"xtol": 1e-3, "ftol": 1e-5, "maxfev": 800})
log("stage3 ncc %.4f" % -r3.fun, np.round(r3.x, 5).tolist())

Lf2, ff2 = choose(a.fixed, um_f, a.coarse_um / 2); Lm2, fm2 = choose(a.moving, um_m, a.coarse_um / 2)
kf2, km2 = 2 ** Lf2 * ff2, 2 ** Lm2 * fm2; rf, rm = kf / kf2, km / km2
zl, zh = overlap_z(r3.x)
vf2 = fetch(a.fixed, Lf2, ff2, zr=(int(zl * rf * ff2), int(zh * rf * ff2))) if ff2 == 1 else fetch(a.fixed, Lf2, ff2)[int(zl * rf):int(zh * rf)]
vm2 = fetch(a.moving, Lm2, fm2)
log(f"fine fixed L{Lf2}x{ff2} {vf2.shape}  moving L{Lm2}x{fm2} {vm2.shape}")
c4 = make_cost(vm2, vf2, rf, rm, int(zl * rf), 2)
r4 = optimize.minimize(c4, r3.x, method="Powell", options={"xtol": 5e-4, "ftol": 1e-6, "maxfev": 500})
log("stage4 ncc %.4f" % -r4.fun, np.round(r4.x, 5).tolist())

# ---- export level-0 mapping ----
p = r4.x; F = Fmat(p[0], p[1], p[5]); cmv, cfx = centres(p)
M = (kf / km) * F; t = kf * cfx - M @ (km * cmv)
reg = {"fixed": a.fixed, "moving": a.moving, "note": "p_fixed_level0_zyx = M @ p_moving_level0_zyx + t",
       "M": M.tolist(), "t": t.tolist(),
       "params": {"theta_deg": float(np.rad2deg(p[0])), "s_xy": float(p[1]), "s_z": float(p[5]), "mirror": bool(mir),
                  "zsign": int(zsign), "t_coarse_zyx": [float(v) for v in p[2:5]]},
       "ncc": {"stage3": float(-r3.fun), "stage4": float(-r4.fun)}, "area_profile_corr": float(best[0]),
       "levels": {"coarse": [Lf, ff, Lm, fm], "fine": [Lf2, ff2, Lm2, fm2]}}
json.dump(reg, open(os.path.join(a.out, "reg.json"), "w"), indent=1)
log("wrote", os.path.join(a.out, "reg.json"))

def ov(x, y):
    x = x / max(1, np.percentile(x, 99.5)); y = y / max(1, np.percentile(y, 99.5))
    return (np.clip(np.stack([x, y, np.zeros_like(x)], -1), 0, 1) * 255).astype(np.uint8)
w = warp(vm2, F * (rf / rm), cmv * rm, cfx * rf - np.array([int(zl * rf), 0, 0]), (0, 0, 0), vf2.shape, 1)
zs = [int(q * (vf2.shape[0] - 1)) for q in (0.25, 0.5, 0.75)]
Image.fromarray(np.concatenate([ov(vf2[z], w[z]) for z in zs], 1)).save(os.path.join(a.out, "overlay_axial.png"))
Image.fromarray(ov(vf2[:, vf2.shape[1] // 2], w[:, vf2.shape[1] // 2])).save(os.path.join(a.out, "overlay_coronal.png"))
log("done")
