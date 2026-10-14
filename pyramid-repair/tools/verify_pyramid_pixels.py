"""Independent bounded pixel comparison: one 0841 L0 chunk vs published L2."""
import json
from pathlib import Path
import numpy as np
import requests,numcodecs
ROOT=Path(__file__).resolve().parents[1]/'evidence'
ROOT.mkdir(parents=True,exist_ok=True)
BASE='https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/841/w00/w00.zarr/'
def get(path):
    r=requests.get(BASE+path,timeout=60);r.raise_for_status();return r
m0=get('0/.zarray').json();m2=get('2/.zarray').json()
# This reproducer is intentionally scoped to the pinned region and encoding.
assert m0['shape']==[65,15872,18944] and m2['shape']==[17,3968,4736], 'Upstream shape changed; review before comparison'
for md in (m0,m2):
    assert not md.get('filters') and md.get('order','C')=='C', 'Unsupported chunk encoding'
    assert md['chunks'][0]>=md['shape'][0], 'Expected a single depth chunk'
def chunk(level,md,iy,ix):
    key=f'0.{iy}.{ix}';raw=get(f'{level}/{key}').content
    if md.get('compressor'):raw=numcodecs.get_codec(md['compressor']).decode(raw)
    return np.frombuffer(raw,dtype=np.dtype(md['dtype'])).reshape(md['chunks'])[:md['shape'][0]],key
# One aligned interior region chosen from dimensions, without looking at labels.
iy=max(0,m2['shape'][1]//2//m2['chunks'][1]);ix=max(0,m2['shape'][2]//2//m2['chunks'][2])
b,key2=chunk(2,m2,iy,ix)
cy,cx=m0['chunks'][1:]
source_y=iy*m2['chunks'][1]*4;source_x=ix*m2['chunks'][2]*4
a,key0=chunk(0,m0,source_y//cy,source_x//cx)
xy=a.reshape(a.shape[0],cy//4,4,cx//4,4).mean(axis=(2,4))
actual=b[:m2['shape'][0],:xy.shape[1],:xy.shape[2]].astype(float)
pooled=np.array([xy[i:min(i+4,len(xy))].mean(axis=0) for i in range(0,len(xy),4)])
def corr(a,b):
    a,b=a.ravel(),b.ravel()
    if a.std()==0 or b.std()==0:return None
    return float(np.corrcoef(a,b)[0,1])
rows=[]
for k in range(min(len(actual),len(pooled))):
    rows.append(dict(slice=k,xy_only_same_index_correlation=corr(actual[k],xy[k]),
                     isotropic_pool_correlation=corr(actual[k],pooled[k]),
                     xy_only_mae=float(np.mean(np.abs(actual[k]-xy[k]))),
                     isotropic_pool_mae=float(np.mean(np.abs(actual[k]-pooled[k])))))
report=dict(source=BASE,l0_chunk=key0,l2_chunk=key2,l0_shape=m0['shape'],l2_shape=m2['shape'],
            roi_l2_shape=list(actual.shape),rows=rows,
            scope='One aligned interior region, label-free. This does not prove the same pixel behavior in all flagged volumes.')
(ROOT/'pyramid_pixel_proof.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2),flush=True)
