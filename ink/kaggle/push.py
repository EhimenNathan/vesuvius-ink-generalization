"""Push a kernel. Usage: push.py <slug> <code.py> '<json CONFIG overrides>' <source kernel slugs comma list> [--cpu] [--no-internet]"""
import json, os, sys, tempfile
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
USER = api.get_config_value("username")
slug, codef, cfg, srcs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), [s for s in sys.argv[4].split(",") if s]
code = open(codef).read()
anchor = 'log("CONFIG"' if 'log("CONFIG"' in code else 'OUT = "/kaggle/working"'
if cfg:   # only inject when there are overrides (scripts without CONFIG stay untouched)
    if "CONFIG_OVERRIDE" in code:
        code = "import json\nCONFIG_OVERRIDE = json.loads(" + repr(json.dumps(cfg)) + ")\n" + code
    else:
        if anchor not in code:
            raise ValueError("CONFIG injection anchor not found")
        code = code.replace(anchor, "CONFIG.update(json.loads(" + repr(json.dumps(cfg)) + "))\n" + anchor, 1)
k = tempfile.mkdtemp(); fn = os.path.basename(codef); open(os.path.join(k, fn), "w").write(code)
json.dump({"id": f"{USER}/{slug}", "title": slug, "code_file": fn, "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": "--cpu" not in sys.argv, "enable_internet": "--no-internet" not in sys.argv,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": [s if "/" in s else f"{USER}/{s}" for s in srcs]},
          open(os.path.join(k, "kernel-metadata.json"), "w"))
r = api.kernels_push(k); print("pushed", slug, getattr(r, "url", r), getattr(r, "error", ""))
