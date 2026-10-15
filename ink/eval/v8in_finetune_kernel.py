"""Kaggle GPU kernel: fine-tune the organizers' v8-in on our corrected training inventory (V8IN_FINETUNE_PROTOCOL_20261014.md).

Data: vesuvius-ink-prep stores (data/<seg>.zarr: vol (21,H,W) uint8 at 9.6 um, ink/sup/val; scrolls 0139, 1667, Paris4,
0814). Patches are resampled on the fly to v8-in's 8.64 um x 24 layers. Recipe follows the released v8-in config.
Outputs: ft_best.safetensors (best validation AUC), ft_last.safetensors, ft_log.json. Nothing from a test source is read.
"""
import glob, json, math, os, random, subprocess, sys, time
subprocess.run("pip install -q 'zarr==2.18.7' 'numcodecs<0.16' safetensors", shell=True)
import numpy as np, torch, torch.nn.functional as F, zarr, cv2
from huggingface_hub import snapshot_download
CONFIG = dict(steps=1200, accum=8, batch_per_gpu=4, lr=1e-5, min_lr=1e-6, wd=1e-6, warmup=50, val_every=150,
              val_tiles=768, probe_tiles=512, infer_batch=8, budget_s=10.5 * 3600, seed=20261014, workers=4, order="auto",
              scroll_w={"0139": 29, "1667": 22, "Paris4": 11, "0814": 2})
CONFIG.update(globals().get("CONFIG_OVERRIDE", {}))
T0 = time.time(); OUT = "/kaggle/working"
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
REV = "d89166b41a3f5fad7749b3d7c0fdd1bd3695d844"
repo = snapshot_download("YoussefMoNader/ink-8um-v8in", revision=REV, allow_patterns=["*.json", "*.safetensors", "ink8um/*"],
                         ignore_patterns=["training/*"], local_dir="/kaggle/tmp/v8in")
sys.path.insert(0, repo)
from ink8um import InkDetector
from safetensors.torch import save_file
base = InkDetector.from_pretrained(repo)
NG = torch.cuda.device_count(); log("GPUs", NG, torch.cuda.get_device_name(0), "params", sum(p.numel() for p in base.parameters()) / 1e6)
model = base.cuda()
_net = torch.nn.DataParallel(model) if NG > 1 else model
class _Fwd:
    """Logits at tile resolution (as ink8um.predict_stack does when the decoder output differs)."""
    def __call__(self, x):
        o = _net(x)
        return o if o.shape[-1] == x.shape[-1] else F.interpolate(o, size=x.shape[-2:], mode="bilinear")
    def eval(self): _net.eval()
    def train(self): _net.train()
net = _Fwd()

# ---------------- data ----------------
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
        for i in range(max(0, y0 // T), (y1 - 1) // T + 1):
            for j in range(max(0, x0 // T), (x1 - 1) // T + 1):
                t = self.tiles.get((i, j))
                if t is None: continue
                ty0, tx0 = i * T, j * T
                sy0, sy1 = max(y0, ty0), min(y1, ty0 + t.shape[1]); sx0, sx1 = max(x0, tx0), min(x1, tx0 + t.shape[2])
                if sy0 < sy1 and sx0 < sx1:
                    out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = t[:, sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
        return out

TEST = {"0841", "MAN5", "0009B", "0500P2", "1447", "0343P"}
segs = {}
for p in sorted(glob.glob("/kaggle/input/**/data/*.zarr", recursive=True)):
    name = os.path.basename(p)[:-5]
    if name in segs: raise ValueError("duplicate store " + name)
    g = zarr.open_group(p, "r"); sc = g.attrs["scroll"]
    if sc in TEST: raise ValueError("test source in training inputs: " + name)
    if sc not in CONFIG["scroll_w"]: log("skip unknown scroll", name, sc); continue
    sup = g["sup"][:] > 0; ink = g["ink"][:] > 0; val = (g["val"][:] > 0) if "val" in g else np.zeros_like(sup)
    segs[name] = dict(scroll=sc, vol=TileVol(g["vol"]), ink=ink, sup=sup, val=val & sup)
    log("loaded", name, sc, segs[name]["vol"].shape, "train px", int((sup & ~val).sum()), "val px", int((val & sup).sum()))
n_val = sum(int(s["val"].sum()) for s in segs.values())
VAL_STORES = []
if n_val < 100000:
    VAL_STORES = [n for n in ("phercparis4-w09", "pherc1667-w031") if n in segs]
    for n in VAL_STORES: segs[n]["val"] = segs[n]["sup"].copy()
    log("validation fallback: whole stores", VAL_STORES)
log("validation pixels", sum(int(s["val"].sum()) for s in segs.values()))
cands = {}
for n, s in segs.items():
    if n in VAL_STORES: continue
    m = s["sup"] & ~s["val"]; ys, xs = np.nonzero(m[::8, ::8])
    if len(ys): cands.setdefault(s["scroll"], []).append((n, ys * 8, xs * 8))
scrolls = sorted(cands); sw = np.array([CONFIG["scroll_w"][k] for k in scrolls], float); sw /= sw.sum()
log("sampling", dict(zip(scrolls, sw.round(3))))

S_IN, S_OUT, D_IN, D_OUT = 9.6, 8.64, 21, 24
ZPOS = [min(max((D_IN - 1) / 2 + (k - (D_OUT - 1) / 2) * S_OUT / S_IN, 0.0), D_IN - 1.0) for k in range(D_OUT)]
def to_v8(v, out_px):
    """(21,h,w) uint8 at 9.6 um -> (24,out_px,out_px) float32 at 8.64 um (linear in depth, bilinear in XY)."""
    res = np.zeros((D_OUT, out_px, out_px), np.float32)
    for k, z in enumerate(ZPOS):
        z0 = int(math.floor(z)); z1 = min(z0 + 1, D_IN - 1); f = z - z0
        sl = v[z0].astype(np.float32) * (1 - f) + v[z1].astype(np.float32) * f
        res[k] = cv2.resize(sl, (out_px, out_px), interpolation=cv2.INTER_LINEAR)
    return res
def crop2d(a, y0, x0, n):
    out = np.zeros((n, n), a.dtype); H, W = a.shape
    sy0, sx0 = max(0, y0), max(0, x0); sy1, sx1 = min(H, y0 + n), min(W, x0 + n)
    if sy0 < sy1 and sx0 < sx1: out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out

ORDER = [False]   # reverse layers? fixed below from training labels
def make_tile(s, cy, cx, out_px):
    n_in = int(round(out_px * S_OUT / S_IN)); y0, x0 = cy - n_in // 2, cx - n_in // 2
    v = s["vol"].crop(y0, y0 + n_in, x0, x0 + n_in)
    x = to_v8(v, out_px)
    if ORDER[0]: x = x[::-1].copy()
    lab = cv2.resize(crop2d(s["ink"], y0, x0, n_in).astype(np.uint8), (out_px, out_px), interpolation=cv2.INTER_NEAREST)
    return x, lab

def augment(x, y, m):
    """x (24,80,80) float, y,m (80,80) -> center 64 crop after v8-in-style augmentation."""
    P = 64; n = x.shape[1]; c = n / 2
    if random.random() < 0.75:
        ang = random.uniform(0, 360); sc = 1 + random.uniform(-0.1, 0.1)
        M = cv2.getRotationMatrix2D((c, c), ang, sc); M[:, 2] += np.random.uniform(-0.15, 0.15, 2) * P
        x = np.stack([cv2.warpAffine(l, M, (n, n), flags=cv2.INTER_LINEAR, borderValue=0) for l in x])
        y = cv2.warpAffine(y, M, (n, n), flags=cv2.INTER_NEAREST, borderValue=0)
        m = cv2.warpAffine(m, M, (n, n), flags=cv2.INTER_NEAREST, borderValue=0)
    o = (n - P) // 2; x, y, m = x[:, o:o + P, o:o + P], y[o:o + P, o:o + P], m[o:o + P, o:o + P]
    if random.random() < 0.5: x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
    if random.random() < 0.5: x, y, m = x[:, ::-1, :], y[::-1, :], m[::-1, :]
    x = np.ascontiguousarray(x); y = np.ascontiguousarray(y); m = np.ascontiguousarray(m)
    if random.random() < 0.4:
        if random.random() < 0.5: x = np.stack([cv2.GaussianBlur(l, (0, 0), random.uniform(0.3, 1.2)) for l in x])
        else:
            k = random.choice([3, 5]); ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1 / k
            ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), random.uniform(0, 180), 1), (k, k))
            ker /= max(ker.sum(), 1e-6); x = np.stack([cv2.filter2D(l, -1, ker) for l in x])
    if random.random() < 0.5:
        for _ in range(random.randint(1, 2)):
            hh, ww = random.randint(1, int(0.2 * P)), random.randint(1, int(0.2 * P))
            yy, xx = random.randint(0, P - hh), random.randint(0, P - ww); x[:, yy:yy + hh, xx:xx + ww] = 0
    if random.random() < 0.6:    # v8-in "fourth augment": depth crop 90-100%, random paste offset, up to 2 layer cutouts
        D = x.shape[0]; cn = random.randint(max(1, math.ceil(D * 0.9)), D); st = random.randint(0, D - cn)
        ps = random.randint(0, D - cn); tmp = np.zeros_like(x); tmp[ps:ps + cn] = x[st:st + cn]
        if random.random() < 0.6:
            idx = np.arange(ps, ps + cn); np.random.shuffle(idx); tmp[idx[:random.randint(0, min(2, cn))]] = 0
        x = tmp
    return x, y, m

class Stream(torch.utils.data.IterableDataset):
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); r = (wi.id if wi else 0) + 1
        random.seed(CONFIG["seed"] + 1000 * r + int(time.time())); np.random.seed((CONFIG["seed"] + 7919 * r + int(time.time())) % 2**32)
        B = CONFIG["batch_per_gpu"] * max(1, NG)
        while True:
            X = np.zeros((B, 24, 64, 64), np.float32); Y = np.zeros((B, 64, 64), np.float32); Mk = np.zeros((B, 64, 64), np.float32)
            for b in range(B):
                sc = scrolls[np.random.choice(len(scrolls), p=sw)]; n, ys, xs = random.choice(cands[sc]); s = segs[n]
                i = np.random.randint(len(ys)); cy, cx = int(ys[i]) + random.randint(-24, 24), int(xs[i]) + random.randint(-24, 24)
                x, lab = make_tile(s, cy, cx, 80)
                n_in = int(round(80 * S_OUT / S_IN))
                msk = cv2.resize(crop2d(s["sup"] & ~s["val"], cy - n_in // 2, cx - n_in // 2, n_in).astype(np.uint8), (80, 80),
                                 interpolation=cv2.INTER_NEAREST)
                x, lab, msk = augment(x, lab.astype(np.float32), msk.astype(np.float32))
                X[b] = np.clip(x, 0, 200); Y[b] = lab; Mk[b] = msk
            yield torch.from_numpy(X / 255.0).float(), torch.from_numpy(Y), torch.from_numpy(Mk)

# ---------------- validation (fixed tiles on validation pixels, no augmentation) ----------------
rng = np.random.default_rng(CONFIG["seed"]); vt = []
for n, s in segs.items():
    ys, xs = np.nonzero(s["val"][::16, ::16])
    for k in rng.permutation(len(ys))[:max(1, CONFIG["val_tiles"] // max(1, len(segs)) * 3)]: vt.append((n, int(ys[k]) * 16, int(xs[k]) * 16))
vt = [vt[i] for i in rng.permutation(len(vt))[:CONFIG["val_tiles"]]]
def val_tiles():
    Xs, Ys, Ms = [], [], []
    for n, cy, cx in vt:
        s = segs[n]; x, lab = make_tile(s, cy, cx, 64); n_in = int(round(64 * S_OUT / S_IN))
        m = cv2.resize(crop2d(s["val"], cy - n_in // 2, cx - n_in // 2, n_in).astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
        Xs.append(np.clip(x, 0, 200) / 255.0); Ys.append(lab); Ms.append(m)
    return np.stack(Xs).astype(np.float32), np.stack(Ys), np.stack(Ms).astype(bool)
@torch.no_grad()
def val_auc():
    from sklearn.metrics import roc_auc_score
    Xv, Yv, Mv = val_tiles(); net.eval(); ps = []
    for i in range(0, len(Xv), CONFIG["infer_batch"]):
        with torch.autocast("cuda", dtype=torch.float16):
            ps.append(torch.sigmoid(net(torch.from_numpy(Xv[i:i + CONFIG["infer_batch"]]).cuda()).float())[:, 0].cpu().numpy())
    p = np.concatenate(ps); net.train()
    y = Yv[Mv] > 0
    return float(roc_auc_score(y, p[Mv])) if 0 < y.mean() < 1 else float("nan")

# depth order from training labels only (base model, 512 training tiles, both orders)
from sklearn.metrics import roc_auc_score
probe = []
for _ in range(CONFIG["probe_tiles"]):
    sc = scrolls[np.random.choice(len(scrolls), p=sw)]; n, ys, xs = random.choice(cands[sc]); i = np.random.randint(len(ys)); probe.append((n, int(ys[i]), int(xs[i])))
order_auc = {}
for rev in (False, True):
    ORDER[0] = rev; Xs, Ys = [], []
    for n, cy, cx in probe:
        x, lab = make_tile(segs[n], cy, cx, 64); Xs.append(np.clip(x, 0, 200) / 255.0); Ys.append(lab)
    Xs = np.stack(Xs).astype(np.float32); Ys = np.stack(Ys); ps = []
    with torch.no_grad():
        net.eval()
        for i in range(0, len(Xs), CONFIG["infer_batch"]):
            with torch.autocast("cuda", dtype=torch.float16):
                ps.append(torch.sigmoid(net(torch.from_numpy(Xs[i:i + CONFIG["infer_batch"]]).cuda()).float())[:, 0].cpu().numpy())
    order_auc["rev" if rev else "fwd"] = float(roc_auc_score(Ys.ravel() > 0, np.concatenate(ps).ravel()))
ORDER[0] = (order_auc["rev"] > order_auc["fwd"]) if CONFIG["order"] == "auto" else CONFIG["order"] == "rev"
torch.cuda.empty_cache()
log("ORDER (training labels)", order_auc, "-> reverse" if ORDER[0] else "-> as stored")

# ---------------- training ----------------
opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["wd"])
scaler = torch.cuda.amp.GradScaler()
def lr_at(step):
    if step < CONFIG["warmup"]: return CONFIG["lr"] * (step + 1) / CONFIG["warmup"]
    t = (step - CONFIG["warmup"]) / max(1, CONFIG["steps"] - CONFIG["warmup"])
    return CONFIG["min_lr"] + 0.5 * (CONFIG["lr"] - CONFIG["min_lr"]) * (1 + math.cos(math.pi * min(1.0, t)))
def loss_fn(logits, y, m):
    t = y * 0.75 + (1 - y) * 0.25                                 # label smoothing 0.25, as v8-in
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    bce = (bce * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)
    p = torch.sigmoid(logits) * m; inter = (p * y).sum((1, 2)); union = p.sum((1, 2)) + (y * m).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (union + 1)
    keep = (m.sum((1, 2)) > 0).float()
    return ((0.5 * dice + 0.5 * bce) * keep).sum() / keep.sum().clamp_min(1)
def save(tag):
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, f"{OUT}/ft_{tag}.safetensors")
history = {"config": CONFIG, "order_auc": order_auc, "reverse_layers": ORDER[0], "val_stores": VAL_STORES, "val": [], "loss": []}
v0 = val_auc(); history["val"].append((0, v0)); best = v0; log("VAL step 0 (base v8-in)", v0)
json.dump(history, open(f"{OUT}/ft_log.json", "w"), indent=1)
loader = iter(torch.utils.data.DataLoader(Stream(), batch_size=None, num_workers=CONFIG["workers"], prefetch_factor=4, persistent_workers=True))
net.train(); t_step = time.time(); step = 0; run = []
while step < CONFIG["steps"]:
    if time.time() - T0 > CONFIG["budget_s"]: log("time budget reached at step", step); break
    for g_ in opt.param_groups: g_["lr"] = lr_at(step)
    opt.zero_grad(set_to_none=True)
    for _ in range(CONFIG["accum"]):
        X, Y, M = (a.cuda(non_blocking=True) for a in next(loader))
        with torch.autocast("cuda", dtype=torch.float16):
            lg = net(X)[:, 0]
        loss = loss_fn(lg.float(), Y, M) / CONFIG["accum"]
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
        scaler.scale(loss).backward(); run.append(loss.item() * CONFIG["accum"])
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); step += 1
    if step == 1: log("peak GPU memory GB", [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(NG)])
    if step % 25 == 0:
        log(f"step {step} loss {np.mean(run[-25 * CONFIG['accum']:]):.4f} lr {lr_at(step):.2e} {(time.time() - t_step) / step:.2f}s/step")
        history["loss"].append((step, float(np.mean(run[-25 * CONFIG['accum']:]))))
    if step % CONFIG["val_every"] == 0 or step == CONFIG["steps"]:
        v = val_auc(); history["val"].append((step, v)); log("VAL step", step, v, "best", best)
        if v > best: best = v; save("best"); history["best_step"] = step
        json.dump(history, open(f"{OUT}/ft_log.json", "w"), indent=1)
save("last"); history["last_step"] = step; history["best_val"] = best
json.dump(history, open(f"{OUT}/ft_log.json", "w"), indent=1)
log("FT DONE best", best, "at", history.get("best_step", 0), "(0 = base model unbeaten)")
