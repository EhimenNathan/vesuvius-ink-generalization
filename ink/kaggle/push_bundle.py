"""Push bundle_maps_kernel.py as a CPU kernel with the given rules. Usage: push_bundle.py <slug> <rules.json> <sources,comma>"""
import json, os, sys, tempfile
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate(); USER = api.get_config_value("username")
slug, rules, srcs = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3].split(",")
code = "RULES = " + json.dumps(rules) + "\n" + open(os.path.join(os.path.dirname(__file__), "bundle_maps_kernel.py")).read()
d = tempfile.mkdtemp(); open(os.path.join(d, "bundle.py"), "w").write(code)
json.dump({"id": f"{USER}/{slug}", "title": slug, "code_file": "bundle.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": False, "enable_internet": False, "dataset_sources": [], "competition_sources": [],
           "kernel_sources": [f"{USER}/{s}" for s in srcs]}, open(os.path.join(d, "kernel-metadata.json"), "w"))
r = api.kernels_push(d); print("pushed", slug, getattr(r, "url", r), getattr(r, "error", ""))
