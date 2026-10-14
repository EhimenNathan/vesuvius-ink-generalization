"""Print filtered log lines of a kernel. Usage: getlog.py <slug> [regex]"""
import json, re, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
USER = api.get_config_value("username")
r = api.kernels_logs(f"{USER}/" + sys.argv[1]); t = r if isinstance(r, str) else getattr(r, "log", str(r))
try: t = "".join(e.get("data", "") for e in json.loads(t))
except Exception: pass
pat = sys.argv[2] if len(sys.argv) > 2 else r"^\[|Error|Traceback"
for l in t.splitlines():
    if re.search(pat, l): print(l[:400])
