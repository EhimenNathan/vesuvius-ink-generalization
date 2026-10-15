"""Kaggle GPU kernel: cross-scroll ink detection with held-out never-seen sources.

Data (outputs of vesuvius-ink-prep / vesuvius-ink-prep2 attached as kernel sources):
  /kaggle/input/**/data*/<seg>.zarr  vol (21|17, H, W) uint8 @ ~9.6 um, ink/sup/val (H, W) uint8, attrs.scroll
Protocol: train on every scroll not in CONFIG["test_scrolls"] (+ never on official validation-mask pixels); evaluate on all
supervised pixels of the test scrolls, in both depth directions. Metrics: AUC, AP, best F0.5 (+threshold), F0.5 at 0.5.
Models
  B0  ink_9um-like local 3D stem -> small 2D U-Net, full-res output, 128 px patches
  V1  3D stem (in-plane stride 2) -> pretrained timm encoder + FPN, 1/8-res logits upsampled, 256 px patches
"""
import glob, hashlib, json, math, os, random, subprocess, sys, time
subprocess.run("pip install -q 'zarr==2.18.7' 'numcodecs<0.16'", shell=True)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, zarr

CONFIG = {
    "model": "V1", "encoder": "resnet34", "stem_stride": 2, "stem_ch": 16,
    "test_scrolls": ["0841", "MAN5", "0009B", "0500P2"],
    "train_scrolls": None,       # explicit allowlist for folds; None preserves the historical complement split
    "input_manifest": None,     # explicit name/pattern/scroll/role entries; required for matched format pilots
    "mix_format_prob": 0.0,     # paired representation replacement; paired data must share the canonical grid and labels
    "loader_workers": 4,
    "prefetch_factor": 4,
    "iters": 6000, "batch": 24, "lr": 2e-4, "wd": 1e-2, "seed": 42, "patch": None,
    "loss": "bce_dice",          # bce_dice | focal_dice
    "label_smooth": 0.1, "focal_gamma": 2.0, "dice_w": 0.5,
    "cutmix": 0.0,               # probability of CutMix per batch
    "z_jitter": 2,
    "zflip": 0.0,                # probability of reversing depth order in training (direction-invariant model)
    "dec_ch": (256, 128, 64, 32, 16),
    "backbone_kw": {},           # extra timm kwargs for V3 (e.g. {"img_size": 256, "dynamic_img_size": true})
    "rgb_proj": True,            # V3: project 3D-stem features to 3 channels so RGB-pretrained weights are used as-is
    "drop_path": 0.1,
    "video": "r2plus1d_18",      # V4: torchvision video backbone r3d_18 | r2plus1d_18 | mc3_18 (Kinetics-400 weights)
    "groups": 5,                 # V5: number of slice groups (3 slices each) fed to the shared 2D encoder
    "tta": 8, "eval_stride_frac": 0.5, "save_maps": True, "tag": "",
    "eval_normalization_depth": "legacy_full", # input_window matches training; legacy_full reproduces old runs
    # self-training on the unlabeled test scrolls (transductive; test labels are never used to make pseudo-labels)
    "init_ckpt": "",
    "swa_start": 0.0,            # >0: also keep an equal-weight SWA average of snapshots after this fraction of training
    "swa_every": 500,            # snapshot interval (steps) for SWA
    "eval_swa": True,            # evaluate the SWA model as an extra stage (BatchNorm statistics recomputed first)             # glob under /kaggle/input: load these weights (EMA) instead of / before training
    "st_rounds": 0, "st_iters": 2000, "st_lr": 5e-5, "st_hi": 0.8, "st_lo": 0.2, "st_frac": 0.3, "st_min_area": 30,
}
CONFIG.update(globals().get("CONFIG_OVERRIDE", {}))   # set by the dataset launcher (push.py --via-dataset)
OUT = "/kaggle/working"; T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
log("CONFIG", json.dumps(CONFIG))
random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
dev = "cuda"
P = CONFIG["patch"] or (128 if CONFIG["model"] == "B0" else 256)
if isinstance(CONFIG.get("dec_ch"), list): CONFIG["dec_ch"] = tuple(CONFIG["dec_ch"])
D = 17                                                     # model input depth
TAG = CONFIG["tag"] or f"{CONFIG['model']}_{CONFIG['encoder']}_{CONFIG['loss']}_s{CONFIG['seed']}"

class TileVol:
    """In-RAM sparse volume of the (depth,128,128) tiles present in the zarr store; missing tiles read as zeros."""
    def __init__(self, arr):
        self.shape = arr.shape; self.T = arr.chunks[1]; self.tiles = {}
        d = arr.store.dir_path() if hasattr(arr.store, "dir_path") else arr.store.path
        for f in os.listdir(os.path.join(d, arr.path)):
            if f.startswith("."): continue
            _, i, j = map(int, f.split("."))
            self.tiles[(i, j)] = arr[:, i * self.T:(i + 1) * self.T, j * self.T:(j + 1) * self.T]
    def __getitem__(self, idx):
        _, ys, xs = idx; T = self.T
        y0, y1 = ys.start, min(ys.stop, self.shape[1]); x0, x1 = xs.start, min(xs.stop, self.shape[2])
        out = np.zeros((self.shape[0], y1 - y0, x1 - x0), np.uint8)
        for i in range(y0 // T, (y1 - 1) // T + 1):
            for j in range(x0 // T, (x1 - 1) // T + 1):
                t = self.tiles.get((i, j))
                if t is None: continue
                ty0, tx0 = i * T, j * T
                sy0, sy1 = max(y0, ty0), min(y1, ty0 + t.shape[1]); sx0, sx1 = max(x0, tx0), min(x1, tx0 + t.shape[2])
                out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = t[:, sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
        return out

def validate_input_manifest(manifest, train_sources, test_sources):
    if train_sources is None: raise ValueError("An explicit manifest requires declared training sources")
    if set(train_sources) & set(test_sources): raise ValueError("Manifest source lists overlap")
    by_name = {}
    for entry in manifest:
        if not all(k in entry for k in ["name", "pattern", "scroll", "role"]): raise ValueError("Incomplete manifest entry")
        if entry["name"] in by_name: raise ValueError("Duplicate manifest name")
        role = entry["role"]
        if role not in ["train", "test", "paired"]: raise ValueError("Invalid manifest role")
        allowed = test_sources if role == "test" else train_sources
        if entry["scroll"] not in allowed: raise ValueError("Manifest source is not authorized for its role")
        by_name[entry["name"]] = entry
    for entry in manifest:
        if entry.get("paired_with"):
            pair = by_name.get(entry["paired_with"])
            if entry["role"] != "train" or pair is None or pair["role"] != "paired" or pair["scroll"] != entry["scroll"]:
                raise ValueError("Invalid cross-source or cross-role representation pair")
    referenced = {e["paired_with"] for e in manifest if e.get("paired_with")}
    if {e["name"] for e in manifest if e["role"] == "paired"} != referenced:
        raise ValueError("Unreferenced paired representation")
    return by_name

def validate_pair_inputs(name, a, b):
    if a["vol"].shape != b["vol"].shape or a["vol"].shape[0] != 17:
        raise ValueError("Paired volume grids or depths differ")
    for key in ["ink", "sup", "val"]:
        if not np.array_equal(a[key], b[key]): raise ValueError("Paired supervision differs: " + key)
    aa, bb = a["attrs"], b["attrs"]
    for attrs in [aa, bb]:
        if attrs.get("registration_status") != "sample_validated_affine" or attrs.get("canonical_segment") != name:
            raise ValueError("Paired coordinate provenance is missing or inconsistent")
    if aa.get("representation") != "paired_original" or bb.get("representation") != "paired_corrected":
        raise ValueError("Paired representations are mislabeled")
    if not aa.get("pair_signature") or aa["pair_signature"] != bb.get("pair_signature"):
        raise ValueError("Paired registration signatures differ")
    if not np.isclose(aa["um"], bb["um"], rtol=1e-6): raise ValueError("Paired canonical spacing differs")

segs = {}; manifest_entries = {}; PAIRS = {}
import re as _re0
if CONFIG["input_manifest"] is not None:
    manifest_entries = validate_input_manifest(CONFIG["input_manifest"], CONFIG["train_scrolls"], CONFIG["test_scrolls"])
    input_paths = []
    for name, entry in manifest_entries.items():
        matches = sorted(set(glob.glob(entry["pattern"], recursive=True)))
        if len(matches) != 1: raise ValueError("Missing or ambiguous manifest input: " + name)
        input_paths.append((name, matches[0]))
    if len({os.path.realpath(p) for _, p in input_paths}) != len(input_paths):
        raise ValueError("A store appears more than once in the manifest")
else:
    input_paths = [(os.path.basename(p)[:-5], p) for p in sorted(glob.glob("/kaggle/input/**/data*/*.zarr", recursive=True))]
if not 0 <= CONFIG["mix_format_prob"] <= 1: raise ValueError("mix_format_prob must be in [0,1]")
if CONFIG["mix_format_prob"] and not manifest_entries: raise ValueError("Format mixing requires a manifest")
for name, p in input_paths:
    if CONFIG.get("skip_regex") and _re0.search(CONFIG["skip_regex"], p):
        log("skipping (skip_regex)", p); continue
    if name in segs:
        raise ValueError("Duplicate segment store: " + name + "; select one representation explicitly with skip_regex")
    g = zarr.open_group(p, "r")
    if manifest_entries and g.attrs["scroll"] != manifest_entries[name]["scroll"]:
        raise ValueError("Input source disagrees with manifest: " + name)
    sup = g["sup"][:] > 0; ink = g["ink"][:] > 0
    val = (g["val"][:] > 0) if "val" in g else np.zeros_like(sup)
    tv = TileVol(g["vol"]); region = np.zeros_like(sup)
    for (i, j) in tv.tiles: region[i * tv.T:(i + 1) * tv.T, j * tv.T:(j + 1) * tv.T] = True
    attrs = dict(g.attrs)
    # The first audited builder identifies representations through its output directories.
    folder = os.path.basename(os.path.dirname(p))
    if folder in ["paired_original", "paired_corrected"] and "representation" not in attrs:
        attrs["representation"] = folder
        attrs["representation_provenance"] = "audited builder output directory"
    segs[name] = {"scroll": g.attrs["scroll"], "vol": tv, "ink": ink, "sup": sup, "val": val, "region": region, "attrs": attrs}
    log("loaded", name, g.attrs["scroll"], segs[name]["vol"].shape, "tiles", len(segs[name]["vol"].tiles), "sup px", int(sup.sum()))
for name, entry in manifest_entries.items():
    if entry.get("paired_with"):
        other = entry["paired_with"]; a, b = segs[name], segs[other]
        validate_pair_inputs(name, a, b)
        PAIRS[name] = other
if PAIRS and (CONFIG["z_jitter"] != 0 or CONFIG["st_rounds"] != 0):
    raise ValueError("This matched pilot requires fixed depth windows and no transductive self-training")
if CONFIG["train_scrolls"] is not None:
    if set(CONFIG["train_scrolls"]) & set(CONFIG["test_scrolls"]):
        raise ValueError("Training and held-out source lists overlap")
    train_names = [n for n, s in segs.items() if s["scroll"] in CONFIG["train_scrolls"]
                   and (not manifest_entries or manifest_entries[n]["role"] == "train")]
    if CONFIG["iters"] > 0:
        missing_sources = set(CONFIG["train_scrolls"]) - {segs[n]["scroll"] for n in train_names}
        if missing_sources: raise ValueError("Declared training sources missing: " + str(sorted(missing_sources)))
    if CONFIG["iters"] > 0 and not train_names: raise ValueError("No declared training sources were loaded")
else:
    train_names = [n for n, s in segs.items() if s["scroll"] not in CONFIG["test_scrolls"]]
test_names = [n for n, s in segs.items() if s["scroll"] in CONFIG["test_scrolls"]
              and (not manifest_entries or manifest_entries[n]["role"] == "test")]
log("train", len(train_names), sorted({segs[n]["scroll"] for n in train_names}), "test", len(test_names), sorted({segs[n]["scroll"] for n in test_names}))
if manifest_entries and {segs[n]["scroll"] for n in test_names} != set(CONFIG["test_scrolls"]):
    raise ValueError("Declared held-out sources missing from manifest inputs")

cands = {}
for n in train_names:
    s = segs[n]; m = s["sup"] & ~s["val"]
    ys, xs = np.nonzero(m[::8, ::8]); ys, xs = ys * 8, xs * 8
    if len(ys): cands.setdefault(s["scroll"], []).append((n, ys, xs))
scroll_w = {"0139": 29, "1667": 22, "Paris4": 11, "0814": 2}
scrolls = list(cands); w = np.array([scroll_w.get(k, 8) for k in scrolls], float); w /= w.sum()
log("sampling weights", dict(zip(scrolls, np.round(w, 3))))

PL = {}   # name -> {"ink","sup","rev","ys","xs"} pseudo-label sources (filled by self-training)
def sample_batch(B):
    """Returns X (B,D,P,P) with per-sample z jitter already applied, Y, M."""
    X = np.zeros((B, D, P, P), np.uint8); Y = np.zeros((B, P, P), np.uint8); M = np.zeros((B, P, P), np.uint8)
    for b in range(B):
        if PL and random.random() < CONFIG["st_frac"]:
            n = random.choice(list(PL)); q = PL[n]; s = segs[n]; H, W = s["sup"].shape
            i = np.random.randint(len(q["ys"])); y0 = int(np.clip(q["ys"][i] - P // 2, 0, max(0, H - P))); x0 = int(np.clip(q["xs"][i] - P // 2, 0, max(0, W - P)))
            v = s["vol"][:, y0:y0 + P, x0:x0 + P]; h, wd = v.shape[1:]; c0 = (v.shape[0] - D) // 2; v = v[c0:c0 + D]
            X[b, :, :h, :wd] = v[::-1] if q["rev"] else v
            Y[b, :h, :wd] = q["ink"][y0:y0 + h, x0:x0 + wd]; M[b, :h, :wd] = q["sup"][y0:y0 + h, x0:x0 + wd]
            continue
        sc = scrolls[np.random.choice(len(scrolls), p=w)]
        n, ys, xs = random.choice(cands[sc]); s = segs[n]; H, W = s["sup"].shape
        if n in PAIRS:
            # Both arms consume the same RNG draw and sample the same canonical coordinates.
            use_paired = random.random() < CONFIG["mix_format_prob"]
            if use_paired: s = segs[PAIRS[n]]
        i = np.random.randint(len(ys)); cy, cx = ys[i] + np.random.randint(-P // 4, P // 4), xs[i] + np.random.randint(-P // 4, P // 4)
        y0 = int(np.clip(cy - P // 2, 0, max(0, H - P))); x0 = int(np.clip(cx - P // 2, 0, max(0, W - P)))
        h, wd = min(P, H - y0), min(P, W - x0)
        v = s["vol"][:, y0:y0 + h, x0:x0 + wd]; Dz = v.shape[0]
        c0 = (Dz - D) // 2; j = CONFIG["z_jitter"]
        off = c0 + (np.random.randint(-min(j, c0), min(j, Dz - D - c0) + 1) if Dz > D else 0)
        X[b, :, :h, :wd] = v[off:off + D]
        Y[b, :h, :wd] = s["ink"][y0:y0 + h, x0:x0 + wd]
        M[b, :h, :wd] = s["sup"][y0:y0 + h, x0:x0 + wd] & ~s["val"][y0:y0 + h, x0:x0 + wd]
    return X, Y, M

def normalize(x):
    """Per-sample robust normalisation over non-zero voxels: (x - median) / (1.4826 * MAD), from a 64k-voxel subsample."""
    B = x.shape[0]; flat = x.reshape(B, -1)
    idx = torch.randint(0, flat.shape[1], (65536,), device=x.device)
    sub = flat[:, idx]; valid = sub > 0
    subn = torch.where(valid, sub, torch.nan)
    med = torch.nanmedian(subn, 1).values
    mad = torch.nanmedian((subn - med[:, None]).abs(), 1).values * 1.4826 + 1e-3
    ok = valid.sum(1) > 100
    med = torch.where(ok, med, torch.zeros_like(med)); mad = torch.where(ok, mad, torch.ones_like(mad))
    out = ((x - med[:, None, None, None]) / mad[:, None, None, None]).clamp(-5, 5) * (x > 0)
    return torch.where(ok[:, None, None, None], out, torch.zeros_like(out))

def augment(x, y, m):
    B = x.shape[0]
    if CONFIG["zflip"] > 0:
        fl = torch.rand(B, device=x.device) < CONFIG["zflip"]
        x = torch.where(fl[:, None, None, None], x.flip(1), x)
    k = random.randint(0, 3)
    x, y, m = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2)), torch.rot90(m, k, (1, 2))
    if random.random() < 0.5: x, y, m = x.flip(3), y.flip(2), m.flip(2)
    if random.random() < 0.5: x, y, m = x.flip(2), y.flip(1), m.flip(1)
    r = lambda *s: torch.rand(*s, device=x.device)
    x = x * (1 + 0.4 * (r(B, 1, 1, 1) - 0.5)) + 0.4 * (r(B, 1, 1, 1) - 0.5)          # contrast / brightness
    g = torch.exp(0.3 * (r(B, 1, 1, 1) - 0.5))                                         # gamma on the normalised signal
    x = torch.sign(x) * x.abs().pow(g)
    if random.random() < 0.25:
        x = x.masked_fill(r(B, D, 1, 1) < 0.1, 0)                                      # depth-channel dropout
    if random.random() < 0.2:
        x = F.avg_pool2d(x, 3, 1, 1)                                                   # mild blur
    x = x + 0.05 * torch.randn_like(x)
    if CONFIG["cutmix"] > 0 and random.random() < CONFIG["cutmix"]:
        perm = torch.randperm(B, device=x.device); lam = random.uniform(0.3, 0.7)
        hh, ww = int(P * math.sqrt(lam)), int(P * math.sqrt(lam)); yy, xx = random.randint(0, P - hh), random.randint(0, P - ww)
        x[:, :, yy:yy + hh, xx:xx + ww] = x[perm, :, yy:yy + hh, xx:xx + ww]
        y[:, yy:yy + hh, xx:xx + ww] = y[perm, yy:yy + hh, xx:xx + ww]; m[:, yy:yy + hh, xx:xx + ww] = m[perm, yy:yy + hh, xx:xx + ww]
    return x, y, m

class Stem(nn.Module):
    def __init__(self, c=16, stride=1):
        super().__init__()
        self.f = nn.Sequential(nn.Conv3d(1, c, 3, stride=(1, stride, stride), padding=1, bias=False), nn.InstanceNorm3d(c, affine=True), nn.LeakyReLU(0.01, True),
                               nn.Conv3d(c, c, 3, padding=1, bias=False), nn.InstanceNorm3d(c, affine=True), nn.LeakyReLU(0.01, True))
        self.att = nn.Conv3d(c, 1, 1); nn.init.zeros_(self.att.weight); nn.init.zeros_(self.att.bias)
        self.out_ch = 2 * c
    def forward(self, x):
        f = self.f(x); wts = torch.softmax(self.att(f).float(), 2).to(f.dtype)
        return torch.cat([(f * wts).sum(2), f.amax(2)], 1)

def cbr(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
class SmallUNet(nn.Module):
    def __init__(self, cin, ch=(32, 64, 128, 256, 320)):
        super().__init__()
        self.enc = nn.ModuleList(); c = cin
        for o in ch:
            self.enc.append(nn.Sequential(cbr(c, o), cbr(o, o))); c = o
        self.up = nn.ModuleList([nn.Sequential(cbr(ch[i + 1] + ch[i], ch[i]), cbr(ch[i], ch[i])) for i in range(len(ch) - 1)])
        self.head = nn.Conv2d(ch[0], 1, 1)
    def forward(self, x):
        feats = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2)); feats.append(x)
        x = feats[-1]
        for i in range(len(self.up) - 1, -1, -1):
            x = self.up[i](torch.cat([F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False), feats[i]], 1))
        return self.head(x)

class B0(nn.Module):
    def __init__(self):
        super().__init__(); self.stem = Stem(16); self.net = SmallUNet(self.stem.out_ch)
    def forward(self, x): return self.net(self.stem(x[:, None]))

class _REBN(nn.Module):
    def __init__(self, i, o, d=1):
        super().__init__(); self.c = nn.Conv2d(i, o, 3, padding=d, dilation=d, bias=False); self.b = nn.BatchNorm2d(o)
    def forward(self, x): return F.relu(self.b(self.c(x)), True)
def _up(x, ref): return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
class RSU(nn.Module):
    """U2-Net residual U-block RSU-L (Qin et al. 2020); dilated=True gives RSU-4F (no pooling, dilations 1,2,4,8)."""
    def __init__(self, L, i, m, o, dilated=False):
        super().__init__(); self.L, self.dil = L, dilated
        self.inp = _REBN(i, o)
        ds = [1, 2, 4, 8] if dilated else [1] * (L - 1) + [2]
        self.enc = nn.ModuleList([_REBN(o if k == 0 else m, m, ds[k]) for k in range(L)])
        self.dec = nn.ModuleList([_REBN(2 * m, o if k == 0 else m, ds[k]) for k in range(L - 1)])
    def forward(self, x):
        hin = self.inp(x); h = hin; e = []
        for k, c in enumerate(self.enc):
            if k > 0 and not self.dil and k < self.L - 1: h = F.max_pool2d(h, 2, ceil_mode=True)
            h = c(h); e.append(h)
        d = e[-1]
        for k in range(self.L - 2, -1, -1):
            d = self.dec[k](torch.cat([d if self.dil else _up(d, e[k]), e[k]], 1))
        return d + hin
class U2Body(nn.Module):
    """U2-Net 2D body (6 encoder stages, 5 decoder stages, fused side outputs); widths scaled by mid/out."""
    def __init__(self, cin, m=24, o=96):
        super().__init__()
        self.e = nn.ModuleList([RSU(7, cin, m, o), RSU(6, o, m, o), RSU(5, o, m, o), RSU(4, o, m, o), RSU(4, o, m, o, True), RSU(4, o, m, o, True)])
        self.d = nn.ModuleList([RSU(7, 2 * o, m, o), RSU(6, 2 * o, m, o), RSU(5, 2 * o, m, o), RSU(4, 2 * o, m, o), RSU(4, 2 * o, m, o, True)])
        self.side = nn.ModuleList([nn.Conv2d(o, 1, 3, padding=1) for _ in range(6)]); self.fuse = nn.Conv2d(6, 1, 1)
    def forward(self, x):
        e = []
        for k, blk in enumerate(self.e):
            if k > 0: x = F.max_pool2d(x, 2, ceil_mode=True)
            x = blk(x); e.append(x)
        d = [e[-1]]; x = e[-1]
        for k in range(4, -1, -1):
            x = self.d[k](torch.cat([_up(x, e[k]), e[k]], 1)); d.insert(0, x)
        sides = [_up(self.side[k](d[k]), d[0]) for k in range(6)]
        return self.fuse(torch.cat(sides, 1))
class V6(nn.Module):
    """B0's 3D stem + U2-Net 2D body (U2-Net ablation, 2026-10-15)."""
    def __init__(self):
        super().__init__(); self.stem = Stem(16); self.net = U2Body(self.stem.out_ch, CONFIG.get("u2_mid", 24), CONFIG.get("u2_out", 96))
    def forward(self, x): return self.net(self.stem(x[:, None]))

class V1(nn.Module):
    def __init__(self, enc):
        super().__init__()
        import timm
        self.stem = Stem(CONFIG["stem_ch"], stride=CONFIG["stem_stride"])
        self.enc = timm.create_model(enc, pretrained=True, features_only=True, in_chans=self.stem.out_ch, out_indices=(1, 2, 3, 4))
        chs = self.enc.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in chs])
        self.fuse = nn.Sequential(cbr(128, 128), nn.Dropout2d(0.1), nn.Conv2d(128, 1, 1))
    def forward(self, x):
        f = self.enc(self.stem(x[:, None]))
        y = self.lat[-1](f[-1])
        for i in range(len(f) - 2, -1, -1):
            y = F.interpolate(y, size=f[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](f[i])
        return self.fuse(y)

class V2(nn.Module):
    """3D stem at full resolution -> pretrained timm encoder (strides 2..32) -> U-Net decoder back to full resolution."""
    def __init__(self, enc):
        super().__init__()
        import timm
        self.stem = Stem(CONFIG["stem_ch"], stride=1)
        self.enc = timm.create_model(enc, pretrained=True, features_only=True, in_chans=self.stem.out_ch)
        chs = self.enc.feature_info.channels()                     # e.g. resnet34: 64,64,128,256,512 at strides 2..32
        dc = list(CONFIG["dec_ch"])[:len(chs)]
        self.dec = nn.ModuleList()
        prev = chs[-1]
        for i in range(len(chs) - 2, -1, -1):
            o = dc[len(chs) - 2 - i]
            self.dec.append(nn.Sequential(cbr(prev + chs[i], o), cbr(o, o))); prev = o
        self.final = nn.Sequential(cbr(prev + self.stem.out_ch, 32), cbr(32, 32))
        self.head = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        s0 = self.stem(x[:, None]); f = self.enc(s0)
        y = f[-1]
        for k, i in enumerate(range(len(f) - 2, -1, -1)):
            y = F.interpolate(y, size=f[i].shape[-2:], mode="bilinear", align_corners=False)
            y = self.dec[k](torch.cat([y, f[i]], 1))
        y = F.interpolate(y, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.final(torch.cat([y, s0], 1)))

class V3(nn.Module):
    """Hybrid for any timm backbone (CNN, Swin, ViT/DINOv2/DINOv3):
    full-res 3D stem -> {full-res skip, stride-2 and stride-4 conv skips} + backbone features (any strides),
    fused coarse-to-fine by a U-Net-style decoder that ends at full resolution."""
    def __init__(self, enc):
        super().__init__()
        import timm
        self.stem = Stem(CONFIG["stem_ch"], stride=1); c0 = self.stem.out_ch
        self.down1 = nn.Sequential(nn.Conv2d(c0, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), cbr(64, 64))
        self.down2 = nn.Sequential(nn.Conv2d(64, 96, 3, 2, 1, bias=False), nn.BatchNorm2d(96), nn.ReLU(True), cbr(96, 96))
        kw = dict(CONFIG.get("backbone_kw") or {})
        try:
            self.enc = timm.create_model(enc, pretrained=True, features_only=True, in_chans=3 if CONFIG["rgb_proj"] else c0,
                                         drop_path_rate=CONFIG["drop_path"], **kw)
        except TypeError:
            self.enc = timm.create_model(enc, pretrained=True, features_only=True, in_chans=3 if CONFIG["rgb_proj"] else c0, **kw)
        self.proj = nn.Conv2d(c0, 3, 1) if CONFIG["rgb_proj"] else nn.Identity()
        import re as _re
        mm = _re.search(r"patch(\d+)", enc)
        self.mult = int(mm.group(1)) if mm and "swin" not in enc else 32       # ViT: patch size; CNN/Swin: stride 32
        chs = self.enc.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, 192, 1) for c in chs])
        self.fuse_bb = nn.Sequential(cbr(192 * len(chs), 256), cbr(256, 256))          # all backbone maps resized to stride 8
        self.up4 = nn.Sequential(cbr(256 + 96, 128), cbr(128, 128))
        self.up2 = nn.Sequential(cbr(128 + 64, 64), cbr(64, 64))
        self.up1 = nn.Sequential(cbr(64 + c0, 32), cbr(32, 32))
        self.head = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        s0 = self.stem(x[:, None]); H, W = s0.shape[-2:]
        d1 = self.down1(s0); d2 = self.down2(d1)
        z = self.proj(s0)
        # backbones with patch tokens or window attention need sizes that are multiples of their stride
        ph, pw = (-H) % self.mult, (-W) % self.mult
        z = F.pad(z, (0, pw, 0, ph))
        fs = self.enc(z)
        fs = [f if f.ndim == 4 and f.shape[1] in (c,) else f.permute(0, 3, 1, 2) for f, c in zip(fs, self.enc.feature_info.channels())]
        tgt = (H // 8, W // 8)
        g = torch.cat([F.interpolate(l(f), size=tgt, mode="bilinear", align_corners=False) for l, f in zip(self.lat, fs)], 1)
        y = self.fuse_bb(g)
        y = self.up4(torch.cat([F.interpolate(y, size=d2.shape[-2:], mode="bilinear", align_corners=False), d2], 1))
        y = self.up2(torch.cat([F.interpolate(y, size=d1.shape[-2:], mode="bilinear", align_corners=False), d1], 1))
        y = self.up1(torch.cat([F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False), s0], 1))
        return self.head(y)

class UDec(nn.Module):
    """U-Net decoder over a list of 2D maps (any channel counts, decreasing resolution) + a full-res skip."""
    def __init__(self, chs, c_full, widths=(256, 128, 96, 64)):
        super().__init__()
        self.blocks = nn.ModuleList(); prev = chs[-1]
        for k, c in enumerate(reversed(chs[:-1])):
            o = widths[min(k, len(widths) - 1)]
            self.blocks.append(nn.Sequential(cbr(prev + c, o), cbr(o, o))); prev = o
        self.final = nn.Sequential(cbr(prev + c_full, 32), cbr(32, 32)); self.head = nn.Conv2d(32, 1, 1)
    def forward(self, feats, full):
        y = feats[-1]
        for blk, f in zip(self.blocks, reversed(feats[:-1])):
            y = blk(torch.cat([F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False), f], 1))
        y = F.interpolate(y, size=full.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.final(torch.cat([y, full], 1)))

class V4(nn.Module):
    """Pretrained 3D video ResNet (Kinetics) on the 17-slice stack, mean-pooled over depth per stage (3rd-place style),
    U-Net decoder with a full-resolution 3D-stem skip."""
    def __init__(self):
        super().__init__()
        import torchvision.models.video as tvv
        ctor = {"r3d_18": (tvv.r3d_18, tvv.R3D_18_Weights), "r2plus1d_18": (tvv.r2plus1d_18, tvv.R2Plus1D_18_Weights),
                "mc3_18": (tvv.mc3_18, tvv.MC3_18_Weights)}[CONFIG["video"]]
        net = ctor[0](weights=ctor[1].KINETICS400_V1)
        self.vstem, self.layers = net.stem, nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.full = Stem(CONFIG["stem_ch"], stride=1)
        self.dec = UDec([64, 64, 128, 256, 512], self.full.out_ch)
    def forward(self, x):
        full = self.full(x[:, None])
        v = self.vstem(x[:, None].expand(-1, 3, -1, -1, -1).contiguous())
        feats = [v.mean(2)]
        for l in self.layers:
            v = l(v); feats.append(v.mean(2))
        return self.dec(feats, full)

class V5(nn.Module):
    """2.5D (2nd-place style): shared pretrained 2D encoder applied to groups of 3 slices, 3D conv across groups,
    mean+max pooled over groups, U-Net decoder with a full-resolution 3D-stem skip."""
    def __init__(self, enc):
        super().__init__()
        import timm
        self.G = CONFIG["groups"]
        self.enc = timm.create_model(enc, pretrained=True, features_only=True, in_chans=3)
        chs = self.enc.feature_info.channels()
        self.mix = nn.ModuleList([nn.Sequential(nn.Conv3d(c, c, (3, 1, 1), padding=(1, 0, 0), groups=max(1, c // 32), bias=False),
                                                nn.BatchNorm3d(c), nn.ReLU(True)) for c in chs])
        self.red = nn.ModuleList([nn.Conv2d(2 * c, c, 1) for c in chs])
        self.full = Stem(CONFIG["stem_ch"], stride=1)
        self.dec = UDec(chs, self.full.out_ch)
    def forward(self, x):
        B, Dz, H, W = x.shape; G = self.G
        c0 = (Dz - 3 * G) // 2; xs = x[:, c0:c0 + 3 * G].reshape(B * G, 3, H, W)
        fs = self.enc(xs); feats = []
        for f, mix, red in zip(fs, self.mix, self.red):
            f = f.reshape(B, G, *f.shape[1:]).permute(0, 2, 1, 3, 4)          # B, C, G, h, w
            f = mix(f); feats.append(red(torch.cat([f.mean(2), f.amax(2)], 1)))
        return self.dec(feats, self.full(x[:, None]))

model = {"B0": B0, "V1": lambda: V1(CONFIG["encoder"]), "V2": lambda: V2(CONFIG["encoder"]), "V3": lambda: V3(CONFIG["encoder"]), "V4": V4, "V5": lambda: V5(CONFIG["encoder"]), "V6": V6}[CONFIG["model"]]().to(dev)
if CONFIG["model"] in ("V3", "V4", "V5"):
    pre = model.enc if hasattr(model, "enc") else nn.ModuleList([model.vstem, model.layers])
    enc_ids = {id(p) for p in pre.parameters()}
    groups = [{"params": [p for p in model.parameters() if id(p) in enc_ids], "lr": CONFIG["lr"] * CONFIG.get("enc_lr_mult", 0.25)},
              {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": CONFIG["lr"]}]
else:
    groups = [{"params": list(model.parameters()), "lr": CONFIG["lr"]}]
if CONFIG.get("init_ckpt"):
    _ck = sorted(glob.glob(f"/kaggle/input/**/{CONFIG['init_ckpt']}", recursive=True))
    assert _ck, "init_ckpt not found: " + CONFIG["init_ckpt"]
    if len(_ck) != 1: raise ValueError("Ambiguous init_ckpt: " + str(_ck))
    model.load_state_dict(torch.load(_ck[0], map_location="cpu", weights_only=False)["state"]); log("loaded weights", _ck[0])
opt = torch.optim.AdamW(groups, weight_decay=CONFIG["wd"])
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g_["lr"] for g_ in groups], total_steps=max(1, CONFIG["iters"]), pct_start=0.1)
scaler = torch.cuda.amp.GradScaler()
ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995))

def loss_fn(logits, y, m):
    if logits.shape[-1] != y.shape[-1]:
        f = y.shape[-1] // logits.shape[-1]
        ym = F.avg_pool2d((y * m)[:, None].float(), f)[:, 0]; mm = F.avg_pool2d(m[:, None].float(), f)[:, 0]
        y = ym / mm.clamp_min(1e-6); m = (mm > 0.5).float()
    lg = logits[:, 0].float(); ls = CONFIG["label_smooth"]; ys = y.float() * (1 - ls) + ls / 2
    ce = F.binary_cross_entropy_with_logits(lg, ys, reduction="none")
    if CONFIG["loss"] == "focal_dice":
        pt = torch.exp(-ce); ce = (1 - pt) ** CONFIG["focal_gamma"] * ce * 2.0
    l1 = (ce * m).sum() / m.sum().clamp_min(1)
    p = torch.sigmoid(lg) * m; inter = (p * y).sum(); dice = 1 - (2 * inter + 1) / (p.sum() + (y * m).sum() + 1)
    return l1 + CONFIG["dice_w"] * dice

log("model", CONFIG["model"], CONFIG["encoder"], "params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), "patch", P)
class BatchStream(torch.utils.data.IterableDataset):
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); sd = CONFIG["seed"] * 1000 + (wi.id if wi else 0)
        random.seed(sd); np.random.seed(sd)
        while True:
            yield sample_batch(CONFIG["batch"])
loader = iter(torch.utils.data.DataLoader(BatchStream(), batch_size=None, num_workers=CONFIG["loader_workers"],
             prefetch_factor=CONFIG["prefetch_factor"], persistent_workers=True)) if CONFIG["iters"] > 0 else None
model.train(); hist = []; t_it = time.time()
target_stream_digest = hashlib.sha256()
swa = torch.optim.swa_utils.AveragedModel(model) if CONFIG.get("swa_start", 0) > 0 and CONFIG["iters"] > 0 else None; swa_n = 0
for it in range(CONFIG["iters"]):
    X, Y, M = (a.numpy() if torch.is_tensor(a) else a for a in next(loader))
    target_stream_digest.update(np.ascontiguousarray(Y).tobytes())
    target_stream_digest.update(np.ascontiguousarray(M).tobytes())
    x = normalize(torch.from_numpy(X).to(dev).float()); y = torch.from_numpy(Y).to(dev).float(); m = torch.from_numpy(M).to(dev).float()
    x, y, m = augment(x, y, m)
    with torch.autocast("cuda", dtype=torch.float16):
        lg = model(x)
    loss = loss_fn(lg, y, m)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite training loss at step {it + 1}")
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); ema.update_parameters(model)
    hist.append(loss.item())
    if swa is not None and it + 1 >= int(CONFIG["swa_start"] * CONFIG["iters"]) and (it + 1) % CONFIG["swa_every"] == 0:
        swa.update_parameters(model); swa_n += 1
    if (it + 1) % 500 == 0:
        log(f"it {it + 1} loss {np.mean(hist[-500:]):.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t_it) / 500:.2f}s/it"); t_it = time.time()
if swa is not None and swa_n > 0:
  try:
      # recompute BatchNorm running statistics for the averaged weights on fresh training batches
      for mod in swa.modules():
          if isinstance(mod, nn.modules.batchnorm._BatchNorm): mod.reset_running_stats(); mod.momentum = None
      swa.train()
      with torch.no_grad():
          for _ in range(100):
              X, Y, M = (a.numpy() if torch.is_tensor(a) else a for a in next(loader))
              x = normalize(torch.from_numpy(X).to(dev).float()); y_ = torch.from_numpy(Y).to(dev).float(); m_ = torch.from_numpy(M).to(dev).float()
              x, _, _ = augment(x, y_, m_)
              with torch.autocast("cuda", dtype=torch.float16):
                  swa(x)
      swa.eval(); log(f"SWA: averaged {swa_n} snapshots, BatchNorm statistics recomputed")
      torch.save({"state": swa.module.state_dict(), "config": CONFIG, "patch": P}, f"{OUT}/model_{TAG}_swa.pt")
  except Exception as e:
    log("SWA FAILED (continuing with EMA):", repr(e)); swa = None
if CONFIG["iters"] == 0: ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995))
torch.save({"state": ema.module.state_dict(), "config": CONFIG, "patch": P}, f"{OUT}/model_{TAG}.pt")

# ---------------- evaluation ----------------
def tta_logits(net, x, n):
    outs = []
    for k in range(4 if n >= 4 else max(1, n)):
        for fl in ((False, True) if n >= 8 else (False,)):
            xi = torch.rot90(x, k, (2, 3)); xi = xi.flip(3) if fl else xi
            o = net(xi); o = o.flip(3) if fl else o
            outs.append(torch.rot90(o, -k, (2, 3)))
    return torch.stack(outs).mean(0)

def predict(net, vol, sup, reverse=False):
    net.eval(); H, W = sup.shape
    ys, xs = np.nonzero(sup); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    st = int(P * CONFIG["eval_stride_frac"])
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    win = np.outer(np.hanning(P + 2)[1:-1], np.hanning(P + 2)[1:-1]).astype(np.float32) + 1e-3
    def starts(lo, hi, size):
        # Preserve the historical lattice for covered regions; repair only gaps.
        result = list(range(max(0, lo - P // 4), max(1, hi - P // 2), st))
        if not result: result = [min(lo, max(0, size - P))]
        if result[-1] + P < hi: result.append(min(hi - P, max(0, size - P)))
        return result
    if st < 1 or st > P: raise ValueError("Evaluation stride must be in [1, patch]")
    sy, sx = starts(y0, y1, H), starts(x0, x1, W)
    batch = []
    def flush():
        X = np.stack([b[2] for b in batch]); x = torch.from_numpy(X).to(dev).float()
        Dz = x.shape[1]; c0 = (Dz - D) // 2
        if Dz < D: raise ValueError("Volume has fewer depth slices than the model requires")
        if CONFIG["eval_normalization_depth"] == "input_window":
            x = normalize(x[:, c0:c0 + D])
        elif CONFIG["eval_normalization_depth"] == "legacy_full":
            x = normalize(x)[:, c0:c0 + D]
        else: raise ValueError("Unknown eval_normalization_depth")
        if reverse: x = x.flip(1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            pr = torch.sigmoid(F.interpolate(tta_logits(net, x, CONFIG["tta"]).float(), size=(P, P), mode="bilinear", align_corners=False))[:, 0].cpu().numpy()
        for (yy, xx, _), p in zip(batch, pr):
            h, wd = min(P, H - yy), min(P, W - xx)
            acc[yy:yy + h, xx:xx + wd] += p[:h, :wd] * win[:h, :wd]; cnt[yy:yy + h, xx:xx + wd] += win[:h, :wd]
        batch.clear()
    for yy in sy:
        for xx in sx:
            if not sup[yy:yy + P, xx:xx + P].any(): continue
            v = vol[:, yy:yy + P, xx:xx + P]; X = np.zeros((v.shape[0], P, P), np.uint8); X[:, :v.shape[1], :v.shape[2]] = v
            batch.append((yy, xx, X))
            if len(batch) == 16: flush()
    if batch: flush()
    if np.any(sup & (cnt <= 0)): raise RuntimeError("Evaluation left requested pixels uncovered")
    return (acc / np.maximum(cnt, 1e-6)).astype(np.float32), (y0, y1, x0, x1)

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

def evaluate(net, stage):
    res = {"per_segment": {}, "pooled": {}}; pool = {"fwd": ([], [], []), "rev": ([], [], [])}; full = {}
    os.makedirs(f"{OUT}/maps", exist_ok=True)
    for n in test_names:
        s = segs[n]
        if not s["sup"].any(): continue
        res["per_segment"][n] = {}; full[n] = {}
        for dname, rev in (("fwd", False), ("rev", True)):
            cover = s["region"] if CONFIG["st_rounds"] else s["sup"]
            pm, bb = predict(net, s["vol"], cover, rev)
            if CONFIG["st_rounds"]: full[n][dname] = pm.astype(np.float16)     # keep only when self-training needs it
            m = s["sup"]; p, y = pm[m], s["ink"][m]
            res["per_segment"][n][dname] = metrics(p, y); log(stage, "eval", n, dname, res["per_segment"][n][dname])
            k = np.random.default_rng(0).choice(len(p), size=min(len(p), 300000), replace=False)
            pool[dname][0].append(p[k]); pool[dname][1].append(y[k]); pool[dname][2].append(s["scroll"])
            if CONFIG["save_maps"]:
                y0, y1, x0, x1 = bb
                np.savez_compressed(f"{OUT}/maps/{TAG}_{stage}__{n}__{dname}.npz", p=(pm[y0:y1, x0:x1] * 255).astype(np.uint8), bbox=np.array(bb))
                if dname == "fwd":
                    from PIL import Image
                    vis = np.concatenate([pm[y0:y1, x0:x1], s["ink"][y0:y1, x0:x1].astype(np.float32) * m[y0:y1, x0:x1]], 1)
                    im = Image.fromarray((np.clip(vis, 0, 1) * 255).astype(np.uint8)); im.thumbnail((2400, 2400)); im.save(f"{OUT}/pred_{TAG}_{stage}_{n}.png")
            del pm, p, y; import gc; gc.collect()
    for dname, (ps, ys_, scs) in pool.items():
        if ps:
            res["pooled"][dname] = metrics(np.concatenate(ps), np.concatenate(ys_)); log(stage, "POOLED", dname, res["pooled"][dname])
            for sc in sorted(set(scs)):
                idx = [i for i, c in enumerate(scs) if c == sc]
                res["pooled"][f"{dname}_{sc}"] = metrics(np.concatenate([ps[i] for i in idx]), np.concatenate([ys_[i] for i in idx]))
                log(stage, "SCROLL", dname, sc, res["pooled"][f"{dname}_{sc}"])
    return res, full

def make_pseudo_labels(full):
    """Confident teacher pixels -> pseudo labels. Direction is chosen WITHOUT labels (higher prediction spread)."""
    from scipy import ndimage
    info = {}
    PL.clear()
    for n, d in full.items():
        s = segs[n]; reg = s["region"]
        sd = {k: float(v[reg].astype(np.float32).std()) for k, v in d.items()}
        dsel = max(sd, key=sd.get); pm = d[dsel].astype(np.float32)
        hi, lo = pm > CONFIG["st_hi"], pm < CONFIG["st_lo"]
        lab, nl = ndimage.label(hi); sizes = ndimage.sum(hi, lab, range(1, nl + 1))
        hi = np.isin(lab, np.nonzero(np.asarray(sizes) >= CONFIG["st_min_area"])[0] + 1)
        conf = (hi | lo) & reg
        ys, xs = np.nonzero(conf[::8, ::8]); ys, xs = ys * 8, xs * 8
        if len(ys) == 0: continue
        PL[n] = {"ink": hi.astype(np.uint8), "sup": conf.astype(np.uint8), "rev": dsel == "rev", "ys": ys, "xs": xs}
        info[n] = {"dir_chosen": dsel, "spread": sd, "conf_frac": float(conf[reg].mean()), "ink_frac_conf": float(hi[conf].mean()) if conf.any() else 0.0}
        log("pseudo-labels", n, info[n])
    return info

net = ema.module
results = {"config": CONFIG, "tag": TAG, "target_stream_sha256": target_stream_digest.hexdigest()}
log("TARGET_STREAM_SHA256 " + results["target_stream_sha256"])
def _save_results():
    json.dump(results, open(f"{OUT}/result_{TAG}.json", "w"), indent=1)
# An exception would mark the kernel ERROR and discard the trained checkpoints, so evaluation failures are logged instead.
t_eval0 = time.time()
try:
    results["teacher"], full = evaluate(net, "teacher")
except Exception as e:
    import traceback; log("TEACHER EVAL FAILED:", repr(e)); traceback.print_exc(); results["teacher_error"] = repr(e); full = {}
_save_results()
if swa is not None and swa_n > 0 and CONFIG.get("eval_swa", True) and time.time() - T0 + 1.2 * (time.time() - t_eval0) > CONFIG.get("session_budget_s", 41400):
    log("SWA eval skipped: elapsed", int(time.time() - T0), "s exceeds the session budget"); results["swa_skipped"] = "time budget"
elif swa is not None and swa_n > 0 and CONFIG.get("eval_swa", True):
    try:
        results["swa"], _ = evaluate(swa.module, "swa")
    except Exception as e:
        log("SWA eval failed:", repr(e))
    _save_results()
for r in range(CONFIG["st_rounds"]):
    results[f"pl_round{r + 1}"] = make_pseudo_labels(full)
    model.load_state_dict(ema.module.state_dict()); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["st_lr"], weight_decay=CONFIG["wd"])
    loader = iter(torch.utils.data.DataLoader(BatchStream(), batch_size=None, num_workers=4, prefetch_factor=4, persistent_workers=True))
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995)); hist = []
    for it in range(CONFIG["st_iters"]):
        X, Y, M = (a.numpy() if torch.is_tensor(a) else a for a in next(loader))
        x = normalize(torch.from_numpy(X).to(dev).float()); y = torch.from_numpy(Y).to(dev).float(); m = torch.from_numpy(M).to(dev).float()
        x, y, m = augment(x, y, m)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x)
        loss = loss_fn(lg, y, m)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); ema.update_parameters(model)
        hist.append(loss.item())
        if (it + 1) % 500 == 0: log(f"st{r + 1} it {it + 1} loss {np.mean(hist[-500:]):.4f}")
    results[f"student{r + 1}"], full = evaluate(ema.module, f"student{r + 1}")
    torch.save({"state": ema.module.state_dict(), "config": CONFIG, "patch": P}, f"{OUT}/model_{TAG}_st{r + 1}.pt")
json.dump(results, open(f"{OUT}/result_{TAG}.json", "w"), indent=1)
log("ALLDONE")
