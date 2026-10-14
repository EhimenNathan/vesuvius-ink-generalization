"""Label-free pixel test of every surveyed volume whose level 2 has fewer depth slices than level 0.

For one interior region per volume, each published level-2 slice k is compared against two hypotheses built
from the level-0 data:
  truncated: XY 4x4 mean of level-0 slice k            (first 17 of 65 slices kept, depth not pooled)
  pooled:    4x4x4 mean of level-0 slices 4k..4k+3      (depth pooled, as a declared z-scale of 4 implies)
A volume is classified by the hypothesis with the higher correlation on the majority of slices. Volumes whose
metadata declares z-scale 1 (flagged by survey_pyramids.py) should come out "truncated"; volumes declaring
z-scale 4 serve as negative controls and should come out "pooled".

Usage: python verify_all_pyramids.py [survey.json] [out.json]
Requires numpy, requests, numcodecs. Downloads a few compressed chunks per volume (~1-10 MB each).
"""
import json, sys, time
from pathlib import Path
import numpy as np, requests, numcodecs

HERE = Path(__file__).resolve().parents[1]
SURVEY = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "evidence" / "pyramid_survey.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "evidence" / "pyramid_pixel_all.json"
BASE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/"
S = requests.Session()

def get(url, ok404=False):
    for attempt in range(6):
        r = S.get(url, timeout=120)
        if r.status_code == 404 and ok404: return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * 2 ** attempt); continue
        r.raise_for_status(); return r
    r.raise_for_status()

class Arr:
    def __init__(self, base, level):
        self.base = f"{base}{level}/"; md = get(self.base + ".zarray").json(); self.md = md
        if md.get("filters"): raise ValueError("unsupported filters")
        if md.get("order", "C") != "C": raise ValueError("unsupported order")
        self.shape, self.chunks, self.dtype = md["shape"], md["chunks"], np.dtype(md["dtype"])
        self.sep = md.get("dimension_separator", ".")
        self.codec = numcodecs.get_codec(md["compressor"]) if md.get("compressor") else None
        self.fill = md.get("fill_value") or 0; self.missing = 0
    def chunk(self, idx):
        r = get(self.base + self.sep.join(map(str, idx)), ok404=True)
        if r is None:
            self.missing += 1; return np.full(self.chunks, self.fill, self.dtype)
        raw = self.codec.decode(r.content) if self.codec else r.content
        return np.frombuffer(raw, self.dtype).reshape(self.chunks)
    def read(self, z0, z1, y0, y1, x0, x1):
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), self.dtype); cz, cy, cx = self.chunks
        for iz in range(z0 // cz, (z1 - 1) // cz + 1):
            for iy in range(y0 // cy, (y1 - 1) // cy + 1):
                for ix in range(x0 // cx, (x1 - 1) // cx + 1):
                    c = self.chunk((iz, iy, ix))
                    a0, a1 = max(z0, iz * cz), min(z1, (iz + 1) * cz)
                    b0, b1 = max(y0, iy * cy), min(y1, (iy + 1) * cy)
                    d0, d1 = max(x0, ix * cx), min(x1, (ix + 1) * cx)
                    out[a0 - z0:a1 - z0, b0 - y0:b1 - y0, d0 - x0:d1 - x0] = \
                        c[a0 - iz * cz:a1 - iz * cz, b0 - iy * cy:b1 - iy * cy, d0 - ix * cx:d1 - ix * cx]
        return out

def corr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0: return None
    return float(np.corrcoef(a, b)[0, 1])

def test_volume(entry, roi=32):
    base = BASE + entry["zarr"] + "/"
    a0, a2 = Arr(base, 0), Arr(base, 2)
    Z0, H0, W0 = a0.shape; Z2, H2, W2 = a2.shape
    if a2.shape[1] * 4 > H0 + 4 or a2.shape[2] * 4 > W0 + 4: raise ValueError("unexpected XY factor")
    # Label-free region choice: scan candidate L2 positions outward from the centre; keep the first region whose
    # level-0 data is >90% nonzero in every slice (papyrus present at all depths).
    cands = [(H2 // 2 + dy * roi * 3, W2 // 2 + dx * roi * 3) for r in range(6)
             for dy in range(-r, r + 1) for dx in range(-r, r + 1) if max(abs(dy), abs(dx)) == r]
    for tries, (cy, cx) in enumerate(cands[:40]):
        y2, x2 = cy - cy % 8, cx - cx % 8
        if y2 < 0 or x2 < 0 or y2 + roi > H2 or x2 + roi > W2: continue
        L0 = a0.read(0, Z0, y2 * 4, (y2 + roi) * 4, x2 * 4, (x2 + roi) * 4).astype(np.float64)
        if (L0 > 0).mean(axis=(1, 2)).min() < 0.9: continue
        L2 = a2.read(0, Z2, y2, y2 + roi, x2, x2 + roi).astype(np.float64)
        xy = L0.reshape(Z0, roi, 4, roi, 4).mean(axis=(2, 4))
        rows = []
        for k in range(Z2):
            iso = xy[4 * k:min(4 * k + 4, Z0)].mean(axis=0)
            rows.append(dict(slice=k, truncated_r=corr(L2[k], xy[k]), pooled_r=corr(L2[k], iso),
                             truncated_mae=float(np.abs(L2[k] - xy[k]).mean()), pooled_mae=float(np.abs(L2[k] - iso).mean())))
        valid = [r for r in rows if r["truncated_r"] is not None and r["pooled_r"] is not None]
        t_wins = sum(r["truncated_r"] > r["pooled_r"] for r in valid)
        # slice 0 is identical under both hypotheses; it carries no evidence
        informative = [r for r in valid if r["slice"] > 0]
        t_inf = sum(r["truncated_r"] > r["pooled_r"] for r in informative)
        verdict = ("truncated" if t_inf >= 0.9 * len(informative) else
                   "pooled" if t_inf <= 0.1 * len(informative) else "ambiguous") if informative else "no-data"
        return dict(zarr=entry["zarr"], survey_flagged=bool(entry["flags"]), declared_scales=entry["scales"][:3],
                    l0_shape=a0.shape, l2_shape=a2.shape, l0_chunks=a0.chunks, l2_region_yx=[y2, x2, roi],
                    candidates_tried=tries + 1, missing_chunks=a0.missing + a2.missing, verdict=verdict,
                    truncated_wins=t_inf, informative_slices=len(informative),
                    median_truncated_r=float(np.median([r["truncated_r"] for r in informative])),
                    median_pooled_r=float(np.median([r["pooled_r"] for r in informative])), rows=rows)
    return dict(zarr=entry["zarr"], survey_flagged=bool(entry["flags"]), verdict="no-region",
                note="no fully covered interior region among the candidates")

def main():
    survey = json.loads(SURVEY.read_text())
    todo = [e for e in survey if e["levels"][2][0] < e["levels"][0][0]]
    results = []
    for e in todo:
        try: res = test_volume(e)
        except Exception as ex: res = dict(zarr=e["zarr"], survey_flagged=bool(e["flags"]), verdict="error", error=repr(ex))
        results.append(res)
        print(f"{res['verdict']:>10}  flagged={res['survey_flagged']!s:5}  "
              f"trunc_r={res.get('median_truncated_r', float('nan')):.5f}  pool_r={res.get('median_pooled_r', float('nan')):.5f}  "
              f"{res.get('truncated_wins', '-')}/{res.get('informative_slices', '-')}  {e['zarr']}", flush=True)
        OUT.write_text(json.dumps(results, indent=1))
    agree = sum((r["verdict"] == "truncated") == r["survey_flagged"] for r in results if r["verdict"] in ("truncated", "pooled"))
    decided = sum(r["verdict"] in ("truncated", "pooled") for r in results)
    print(f"metadata flag agrees with pixel verdict on {agree}/{decided} decided volumes ({len(results)} tested)")

if __name__ == "__main__":
    main()
