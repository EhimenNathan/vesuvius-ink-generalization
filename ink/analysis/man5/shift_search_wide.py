"""Wide label-shift search on saved MAN5 probability maps (label-registration diagnostic, exploratory).
prediction(y,x) scored against label(y+dy,x+dx); sample fixed from pixels supervised under every shift tested."""
import json, sys, numpy as np, zarr
from sklearn.metrics import roc_auc_score
root = sys.argv[1]; R = int(sys.argv[2]); step = int(sys.argv[3]); maps = sys.argv[4:]
L = f"{root}/labels/data3/phercman5-outer3.zarr"; g = {k: zarr.open_array(f"{L}/{k}", "r") for k in ("ink", "sup")}
ink = g["ink"][:] > 0; sup = g["sup"][:] > 0; H, W = sup.shape
rng = np.random.default_rng(42)
ys, xs = np.nonzero(sup)
keep = (ys >= R) & (ys < H - R) & (xs >= R) & (xs < W - R)
ys, xs = ys[keep], xs[keep]
out = {}
for f in maps:
    z = np.load(f); p = z["p"]; y0, y1, x0, x1 = z["bbox"]
    inb = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
    yy, xx = ys[inb], xs[inb]
    k = rng.choice(len(yy), size=min(len(yy), 150000), replace=False); yy, xx = yy[k], xx[k]
    pv = p[yy - y0, xx - x0].astype(np.float32)
    grid = {}
    for dy in range(-R, R + 1, step):
        for dx in range(-R, R + 1, step):
            ly, lx = yy + dy, xx + dx
            m = sup[ly, lx]
            grid[(dy, dx)] = roc_auc_score(ink[ly, lx][m], pv[m])
    best = max(grid, key=grid.get)
    out[f] = {"zero": grid[(0, 0)], "best": list(best), "best_auc": grid[best],
              "on_boundary": max(abs(best[0]), abs(best[1])) == R - (R % step),
              "grid": {f"{a},{b}": v for (a, b), v in grid.items()}}
    print(f.split("/")[-1], "zero %.4f best %s %.4f" % (grid[(0, 0)], best, grid[best]), "boundary" if out[f]["on_boundary"] else "interior", flush=True)
json.dump(out, open(f"{root}/man5_alignment_wide_R{R}.json", "w"))
