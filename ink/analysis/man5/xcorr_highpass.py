"""Label-registration test robust to low-frequency confounds: normalized cross-correlation of high-passed
prediction and high-passed label inside the supervision mask, over shifts up to +-R px. A real registration error
appears as a sharp peak at the same (dy,dx) for independent models; low-frequency agreement does not."""
import json, sys, numpy as np, zarr
from scipy import ndimage
L, R, sig, outp = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]; maps = sys.argv[5:]   # label zarr group dir
ink = zarr.open_array(f"{L}/ink", "r")[:] > 0; sup = zarr.open_array(f"{L}/sup", "r")[:] > 0
def hp(a, m):
    a = a.astype(np.float64) * m; w = ndimage.gaussian_filter(m.astype(np.float64), sig)
    mu = ndimage.gaussian_filter(a, sig) / np.maximum(w, 1e-6); return (a - mu) * m
out = {}
for f in maps:
    z = np.load(f); p = z["p"].astype(np.float64) / 255; y0, y1, x0, x1 = map(int, z["bbox"])
    # pad label window by R so shifted labels exist
    Y0, Y1, X0, X1 = max(0, y0 - R), min(ink.shape[0], y1 + R), max(0, x0 - R), min(ink.shape[1], x1 + R)
    lab = ink[Y0:Y1, X0:X1].astype(np.float64); ms = sup[Y0:Y1, X0:X1]
    mp = sup[y0:y1, x0:x1]
    a = hp(p, mp); b = hp(lab, ms)
    A = np.zeros_like(b); A[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = a
    Ma = np.zeros_like(b); Ma[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = mp
    F = np.fft.rfft2; I = lambda q: np.fft.irfft2(q, s=b.shape)
    num = I(np.conj(F(A)) * F(b))                        # sum_p A(p) b(p+s)
    na = I(np.conj(F(A * A)) * F(ms.astype(float)))      # sum A^2 over overlap
    nb = I(np.conj(F(Ma)) * F(b * b))
    ncc = num / np.sqrt(np.maximum(na * nb, 1e-12))
    shifts = {}
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            shifts[(dy, dx)] = ncc[dy % b.shape[0], dx % b.shape[1]]
    best = max(shifts, key=shifts.get); vals = np.array(list(shifts.values()))
    out[f] = dict(zero=float(shifts[(0, 0)]), best=list(best), best_ncc=float(shifts[best]),
                  median=float(np.median(vals)), sd=float(vals.std()),
                  peak_z=float((shifts[best] - np.median(vals)) / vals.std()))
    print(f.split("/")[-1], json.dumps(out[f]), flush=True)
json.dump(out, open(outp, "w"), indent=1)
