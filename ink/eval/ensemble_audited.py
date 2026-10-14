"""CPU ensemble scorer: fixed common pixels, duplicate rejection, explicit provenance.

Uses existing uint8 maps, so results need not exactly match float32 online scores.
Rank averaging uses within-segment midranks; 0.5 is a percentile cutoff.
The exhaustive combinations are diagnostic, not a license to pick on test labels.
"""
import glob, itertools, json, os, re
import numpy as np
import zarr
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

CONFIG = dict(members=[], segments=[], direction='rev', max_combo=4, combine='mean',
              map_glob='/kaggle/input/**/maps/*.npz', label_glob='/kaggle/input/**/data*/*.zarr',
              skip_regex='', member_aliases={}, max_pixels=300000, seed=0)
CONFIG.update(globals().get('CONFIG_OVERRIDE', {}))
OUT = globals().get('OUT', '/kaggle/working')

def metrics(p, y):
    if not len(y) or np.unique(y).size != 2:
        raise ValueError('Metric population needs both ink and background')
    def f05(t):
        pr = p >= t
        tp = np.count_nonzero(pr & y); fp = np.count_nonzero(pr & ~y); fn = np.count_nonzero(~pr & y)
        return float(1.25 * tp / max(1e-12, 1.25 * tp + fp + .25 * fn))
    best, threshold = max((f05(t), float(t)) for t in np.linspace(.05, .95, 37))
    return dict(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                n=len(y), positive_fraction=float(y.mean()), f05_at_0_5=f05(.5),
                diagnostic_oracle_f05=best, diagnostic_oracle_threshold=threshold)

def main():
    if CONFIG['combine'] not in ('mean','rank','logit'): raise ValueError('Unknown combination method')
    if CONFIG['direction'] not in ('fwd','rev','both'): raise ValueError('Unknown direction')
    if CONFIG['max_pixels'] < 1: raise ValueError('max_pixels must be positive')
    maps = {}
    directions = ['fwd', 'rev'] if CONFIG['direction'] == 'both' else [CONFIG['direction']]
    for path in sorted(glob.glob(CONFIG['map_glob'], recursive=True)):
        if CONFIG['skip_regex'] and re.search(CONFIG['skip_regex'], path): continue
        match = re.fullmatch(r'(.+?)__(.+?)__(fwd|rev)\.npz', os.path.basename(path))
        if not match: continue
        member, seg, direction = match.groups()
        member = CONFIG['member_aliases'].get(member, member)
        if CONFIG['members'] and member not in CONFIG['members']: continue
        if CONFIG['segments'] and seg not in CONFIG['segments']: continue
        if direction not in directions: continue
        key = member, seg, direction
        if key in maps: raise ValueError(f'Duplicate map {key}: {maps[key]} and {path}')
        maps[key] = path
    members = CONFIG['members'] or sorted({k[0] for k in maps})
    segments = CONFIG['segments'] or sorted({k[1] for k in maps})
    if not members or not segments: raise ValueError('No maps selected')
    for key in itertools.product(members, segments, directions):
        if key not in maps: raise ValueError(f'Missing map {key}; specify a complete cohort')
    labels = {}
    for path in sorted(glob.glob(CONFIG['label_glob'], recursive=True)):
        if CONFIG['skip_regex'] and re.search(CONFIG['skip_regex'], path): continue
        seg = os.path.basename(path)[:-5]
        if seg not in segments: continue
        if seg in labels: raise ValueError(f'Duplicate label store {seg}: {labels[seg]} and {path}')
        labels[seg] = path
    values, targets, sources, population = {}, {}, {}, {}
    for seg in segments:
        if seg not in labels: raise ValueError(f'Missing labels for {seg}')
        g = zarr.open_group(labels[seg], mode='r')
        ink, sup = g['ink'][:] > 0, g['sup'][:] > 0
        yy, xx = np.nonzero(sup)
        common = np.ones(len(yy), bool)
        loaded = {}
        for member, direction in itertools.product(members, directions):
            with np.load(maps[member, seg, direction], allow_pickle=False) as z:
                p, bb = z['p'], z['bbox'].astype(int)
                # optional coverage mask (e.g. v8-in tiles only fully covered windows): uncovered pixels leave the population
                cover = np.unpackbits(z['cover'])[:int(np.prod(z['cover_shape']))].reshape(z['cover_shape']).astype(bool)                     if 'cover' in z.files else None
            y0, y1, x0, x1 = bb
            if p.shape != (y1-y0, x1-x0): raise ValueError('Map/bbox shape mismatch')
            inside = (yy>=y0)&(yy<y1)&(xx>=x0)&(xx<x1)
            v = np.full(len(yy), np.nan, np.float32)
            v[inside] = p[yy[inside]-y0, xx[inside]-x0]
            if cover is not None:
                if cover.shape != p.shape: raise ValueError('Coverage/map shape mismatch')
                v[inside] = np.where(cover[yy[inside]-y0, xx[inside]-x0], v[inside], np.nan)
            if np.issubdtype(p.dtype, np.integer):
                if p.dtype != np.uint8: raise ValueError('Only uint8 integer maps are supported')
                v /= 255
            if np.any(np.isfinite(v) & ((v < 0) | (v > 1))): raise ValueError('Scores outside [0,1]')
            common &= np.isfinite(v)
            loaded[member, direction] = v
        indices = np.flatnonzero(common)
        population[seg] = dict(supervised_pixels=len(yy), common_pixels=len(indices), labels=labels[seg])
        if len(indices) > CONFIG['max_pixels']:
            indices = np.sort(np.random.default_rng(CONFIG['seed']).choice(indices, CONFIG['max_pixels'], replace=False))
        population[seg]['sampled_pixels'] = len(indices)
        targets[seg] = ink[yy[indices], xx[indices]]
        sources[seg] = g.attrs['scroll']
        for key, v in loaded.items(): values[key[0], seg, key[1]] = v[indices]
    def score(combo, direction):
        predictions = {}
        for seg in segments:
            vs = [values[m,seg,direction] for m in combo]
            if CONFIG['combine'] == 'rank' and len(combo)>1:
                vs = [(rankdata(v, method='average')-.5)/len(v) for v in vs]
            if CONFIG['combine'] == 'logit' and len(combo)>1:
                v = np.mean([np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1)) for p in vs],axis=0)
                p = 1/(1+np.exp(-v))
            else: p = np.mean(vs,axis=0)
            predictions[seg] = p
        per_segment = {s:metrics(predictions[s],targets[s]) for s in segments}
        per_source = {}
        for sc in sorted(set(sources.values())):
            ss = [s for s in segments if sources[s]==sc]
            per_source[sc] = metrics(np.concatenate([predictions[s] for s in ss]),np.concatenate([targets[s] for s in ss]))
        return dict(pooled=metrics(np.concatenate(list(predictions.values())),np.concatenate([targets[s] for s in segments])),
                    per_segment=per_segment,per_source=per_source,
                    macro_source_auc=float(np.mean([r['auc'] for r in per_source.values()])))
    results = {}
    for direction in directions:
        combos = [c for k in range(1,min(CONFIG['max_combo'],len(members))+1) for c in itertools.combinations(members,k)]
        if len(members)>CONFIG['max_combo']: combos.append(tuple(members))
        for combo in combos:
            key = direction+'|'+'+'.join(combo)
            results[key] = score(combo,direction)
            print(key,results[key]['pooled']['auc'],flush=True)
    os.makedirs(OUT,exist_ok=True)
    with open(os.path.join(OUT,'ensemble_results.json'),'w') as f:
        json.dump(dict(config=CONFIG,population=population,map_paths={'|'.join(k):v for k,v in maps.items()},
                       rank_threshold_is_not_calibrated_probability=True,results=results),f,indent=2)
    return results

if __name__ == '__main__': main()
