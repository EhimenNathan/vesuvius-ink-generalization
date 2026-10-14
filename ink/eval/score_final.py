"""Local scoring of the frozen protocol (FINAL_EVAL_PROTOCOL_20261014.md) with ensemble_audited.py.

Maps: kout/bundles/*/maps/<alias>__<segment>__<dir>.npz.  Labels: kout/bundles/main/labels/data3/*.zarr.
Usage: python score_final.py <run-name> <members comma> <direction> [combine] [segments comma]
Writes kout/final_scores/<run-name>/ensemble_results.json.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
name, members, direction = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
combine = sys.argv[4] if len(sys.argv) > 4 else "mean"
segments = sys.argv[5].split(",") if len(sys.argv) > 5 else []
CONFIG_OVERRIDE = dict(members=members, segments=segments, direction=direction, combine=combine, max_combo=1,
                       map_glob=str(ROOT / "kout/bundles/*/maps/*.npz"),
                       label_glob=str(ROOT / "kout/bundles/main/labels/data3/*.zarr"), max_pixels=300000, seed=0)
OUT = str(ROOT / "kout/final_scores" / name)
src = (ROOT / "train/ensemble_audited.py").read_text().replace("if __name__ == '__main__': main()", "")
exec(compile(src, "ensemble_audited.py", "exec"))
res = main()
for key, r in res.items():
    ps = " ".join(f"{s}={v['auc']:.4f}" for s, v in r["per_source"].items())
    print(f"{key:60s} pooled={r['pooled']['auc']:.4f} macro={r['macro_source_auc']:.4f} AP={r['pooled']['ap']:.4f} | {ps}")
