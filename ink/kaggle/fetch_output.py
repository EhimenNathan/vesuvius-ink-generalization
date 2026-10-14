"""Download one kernel output file via its signed URL with curl (resumable). Usage: fetch_output.py <owner/slug> <filename> <dest>"""
import subprocess, sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api = KaggleApi(); api.authenticate()
owner, slug = sys.argv[1].split("/"); name, dest = sys.argv[2], sys.argv[3]
with api.build_kaggle_client() as k:
    req = ApiListKernelSessionOutputRequest(); req.user_name = owner; req.kernel_slug = slug; req.page_size = 200
    files = k.kernels.kernels_api_client.list_kernel_session_output(req).files
url = next(f.url for f in files if f.file_name == name)
import os, zipfile
def valid():
    if not name.endswith(".zip"): return True
    try: return zipfile.ZipFile(dest).testzip() is None
    except Exception: return False
for attempt in range(30):
    # resume after a dropped connection (single writer); start over only if a finished file is not a valid zip
    r = subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "5", "-o", dest, url])
    if r.returncode == 0:
        if valid(): print("ok", dest, os.path.getsize(dest)); break
        print("complete but invalid; restarting from zero"); os.remove(dest)
    else:
        print("curl exit", r.returncode, "resuming at", os.path.getsize(dest) if os.path.exists(dest) else 0)
