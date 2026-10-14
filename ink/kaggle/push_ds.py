"""Push a kernel whose code lives in a private dataset (for accounts where large script pushes time out).

Usage: push_ds.py <slug> <code.py> '<json CONFIG overrides>' <source kernel slugs comma list> [--cpu] [--upload]
--upload refreshes the dataset <user>/vesuvius-code with the current code file before pushing.
The kernel itself is a short, readable launcher: it sets CONFIG_OVERRIDE and exec()s the script from the dataset.
"""
import json, os, shutil, sys, tempfile, time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
USER = api.get_config_value("username")
slug, codef, cfg, srcs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), [s for s in sys.argv[4].split(",") if s]
DS = f"{USER}/vesuvius-code"
if "--upload" in sys.argv:
    d = tempfile.mkdtemp(); shutil.copy(codef, d)
    json.dump({"title": "vesuvius-code", "id": DS, "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(d, "dataset-metadata.json"), "w"))
    mine = [x.ref for x in api.dataset_list(mine=True)]
    if DS in mine:
        api.dataset_create_version(d, "update code", dir_mode="zip", quiet=True)
    else:
        api.dataset_create_new(d, public=False, dir_mode="zip", quiet=True)
    for _ in range(60):
        if str(api.dataset_status(DS)).lower().endswith("ready"): break
        time.sleep(10)
    print("dataset ready", DS)
fn = os.path.basename(codef)
launcher = f'''# Launcher: runs {fn} from the private dataset {DS} with the settings below.
import glob, json
CONFIG_OVERRIDE = json.loads({json.dumps(json.dumps(cfg))})
_src = sorted(glob.glob("/kaggle/input/**/{fn}", recursive=True))[0]
print("running", _src, "with", CONFIG_OVERRIDE, flush=True)
exec(compile(open(_src).read(), _src, "exec"))
'''
k = tempfile.mkdtemp(); open(os.path.join(k, "launcher.py"), "w").write(launcher)
json.dump({"id": f"{USER}/{slug}", "title": slug, "code_file": "launcher.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": "--cpu" not in sys.argv, "enable_internet": True,
           "dataset_sources": [DS], "competition_sources": [], "kernel_sources": [s if "/" in s else f"{USER}/{s}" for s in srcs]},
          open(os.path.join(k, "kernel-metadata.json"), "w"))
r = api.kernels_push(k); print("pushed", slug, getattr(r, "url", r), getattr(r, "error", ""))
