"""Kaggle CPU kernel: collect selected probability maps + label stores (ink/sup only) into one zip for local scoring.

RULES: (kernel slug, filename regex, member alias). Maps are renamed <alias>__<segment>__<dir>.npz; a duplicate
(alias, segment, dir) is an error. Labels: data3b (0841 at 9.362 um) and data3 without 0841 (mis-scaled there).
"""
import glob, json, os, re, shutil, zipfile
RULES = json.loads(os.environ.get("BUNDLE_RULES", "null") or "null") or globals().get("RULES")
OUT = "/kaggle/working"; STAGE = "/kaggle/tmp/bundle"
os.makedirs(f"{STAGE}/maps", exist_ok=True); os.makedirs(f"{STAGE}/labels", exist_ok=True)
manifest = {"maps": {}, "labels": {}}
for slug, pattern, alias in RULES:
    hits = [p for p in glob.glob(f"/kaggle/input/**/{slug}/**/*.npz", recursive=True) if re.fullmatch(pattern, os.path.basename(p))]
    if not hits: raise RuntimeError(f"no maps for rule {slug} {pattern}")
    for p in hits:
        m = re.fullmatch(r".+?__(.+?)__(fwd|rev)\.npz", os.path.basename(p))
        key = f"{alias}__{m.group(1)}__{m.group(2)}.npz"
        if key in manifest["maps"]: raise RuntimeError(f"duplicate {key}: {manifest['maps'][key]} {p}")
        shutil.copyfile(p, f"{STAGE}/maps/{key}"); manifest["maps"][key] = p
stores = sorted(glob.glob("/kaggle/input/**/data3b/*.zarr", recursive=True))
stores += [p for p in sorted(glob.glob("/kaggle/input/**/data3/*.zarr", recursive=True)) if not os.path.basename(p).startswith("pherc0841")]
for p in stores:
    n = os.path.basename(p)
    if n in manifest["labels"]: raise RuntimeError("duplicate label store " + n)
    d = f"{STAGE}/labels/data3/{n}"; os.makedirs(d)
    for f in (".zgroup", ".zattrs"):
        if os.path.exists(f"{p}/{f}"): shutil.copyfile(f"{p}/{f}", f"{d}/{f}")
    for a in ("ink", "sup"): shutil.copytree(f"{p}/{a}", f"{d}/{a}")
    manifest["labels"][n] = p
json.dump(manifest, open(f"{STAGE}/manifest.json", "w"), indent=1)
with zipfile.ZipFile(f"{OUT}/bundle.zip", "w", zipfile.ZIP_STORED) as z:
    for root, _, files in os.walk(STAGE):
        for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), STAGE))
print("BUNDLE", len(manifest["maps"]), "maps", len(manifest["labels"]), "label stores", os.path.getsize(f"{OUT}/bundle.zip") // 2**20, "MB")
