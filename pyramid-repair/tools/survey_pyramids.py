"""Survey OME-Zarr pyramids of every segment in the scrollprize `ink` label bucket: per level shape, chunks and the
declared scale, and flag levels whose depth is shorter than level 0 while the declared z-scale says "not downsampled"
(i.e. the depth axis was truncated instead of pooled)."""
import json, sys, urllib.request, urllib.error
H = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/ink"
R = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink"
def ls(path):
    try: return json.loads(urllib.request.urlopen(f"{H}/{path}?recursive=false&limit=500", timeout=60).read())
    except Exception: return []
def js(url):
    try: return json.loads(urllib.request.urlopen(url, timeout=60).read())
    except Exception: return None
rows = []
for top in [e["path"].split("/")[-1] for e in ls("") if e["type"] == "directory"]:
    for seg in [e["path"] for e in ls(top) if e["type"] == "directory"]:
        for z in [e["path"] for e in ls(seg.replace("ink/", "", 1)) if e["type"] == "directory" and e["path"].endswith(".zarr")
                  and not any(t in e["path"] for t in ("label", "mask", "max_"))]:
            rel = z.replace("ink/", "", 1)
            at = js(f"{R}/{rel}/.zattrs") or {}
            try: scales = [d["coordinateTransformations"][0]["scale"] for d in at["multiscales"][0]["datasets"]]
            except Exception: scales = []
            lv = []
            for l in range(6):
                m = js(f"{R}/{rel}/{l}/.zarray")
                if not m: break
                lv.append((m["shape"], m["chunks"]))
            if not lv: continue
            z0 = lv[0][0][0]; flags = []
            for l, (sh, ch) in enumerate(lv[1:], 1):
                zs = scales[l][0] if l < len(scales) else None
                if sh[0] < z0 and zs == 1.0: flags.append(f"L{l}: depth {sh[0]}/{z0} but z-scale 1 (TRUNCATED)")
            row = {"zarr": rel, "levels": [s for s, _ in lv], "chunks0": lv[0][1], "scales": scales[:len(lv)], "flags": flags}
            rows.append(row)
            print(("BAD " if flags else "ok  ") + rel, "L0", lv[0][0], "L2", lv[2][0] if len(lv) > 2 else None, "|", "; ".join(flags[:2]), flush=True)
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "pyramid_survey.json", "w"), indent=1)
print("segments", len(rows), "with truncated depth levels:", sum(bool(r["flags"]) for r in rows))
