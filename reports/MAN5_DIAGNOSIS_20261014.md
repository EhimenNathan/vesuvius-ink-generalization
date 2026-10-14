# MAN5 failure diagnosis (2026-10-14)

Every model we have scored, ours and the organizer's ink_9um, sits near chance on MAN5_outer_3 (AUC 0.50–0.62).
These checks rule out the three cheap explanations. All are label-free, except the shift checks, which use labels
only diagnostically and never to select a model or shift.

## 1. Label registration: no translation within ±128 px (±1.2 mm) explains it

`man5/xcorr_highpass.py` computes the normalized cross-correlation of the high-passed prediction map against the
high-passed label inside the supervision mask, for every shift up to ±128 px. A real registration error appears as a
sharp peak at the same shift for independent models. Low-frequency agreement, which is what drove Codex's AUC-vs-shift
grid toward its ±8 px boundary, is removed by the high-pass.

| map (sigma 32) | NCC at (0,0) | best shift | best NCC | peak z |
|---|---:|---|---:|---:|
| **positive control: 0009B, B0 s42 rev (AUC 0.884)** | **0.531** | (6, 6) | 0.562 | 5.3 |
| MAN5 B0 s42 fwd | −0.011 | (11, −68) | 0.084 | 2.6 |
| MAN5 B0 s42 rev | −0.010 | (−106, −87) | 0.111 | 2.4 |
| MAN5 ink_9um s42 fwd | 0.031 | (−58, −128) | 0.122 | 3.2 |
| MAN5 ink_9um s42 rev | 0.029 | (58, −18) | 0.130 | 2.6 |
| MAN5 ink_9um s43 fwd | −0.008 | (−124, −29) | 0.091 | 2.7 |
| MAN5 ink_9um s43 rev | 0.099 | (−31, 35) | 0.134 | 2.2 |

At sigma 8 the control gives a peak at (6, 4) with z 8.4; MAN5 gives z 3–4, the size expected of the maximum of noise
over ~66k shifts. The MAN5 maxima scatter across the search window and disagree between models. The control also shows
a consistent ~6 px offset on 0009B, a small registration detail worth noting for that source.

Codex's ±8 px AUC grid (`kout/cpu_audit_20261010/man5_alignment.json`) and a ±96 px AUC grid
(`man5/shift_search_wide.py`) both drift toward far offsets with model-dependent optima ((−8,−72) for B0;
(−96,−88) for ink_9um s42), the signature of low-frequency confounding rather than misregistration.

## 2. Physical scale: correct (9.596 µm), not the cause

The only MAN5 scan on the open-data bucket is `PHercMAN5/volumes/20260311104824-2.399um-0.2m-78keV-masked.zarr`
(shape 31640×17495×17495). The segment's tifxyz bbox (x 2411–15204, y 6051–14701, z 5027–28674) lies inside it. The
render is 1 voxel per pixel: the tifxyz grid (scale 0.05, 1206×1495 points) × 20 = 24120×29900, exactly the render L0
shape. So the L0 voxel is 2.399 µm and our 4×4×4 pooling gives 9.596 µm, the intended scale. The level pyramid is
genuinely depth-pooled (pixel test, `release/pyramid-repair-evidence`), so MAN5 is not affected by the truncation bug.

## 3. Depth: papyrus present across the window

Codex's three label-free depth profiles peak at pooled indices 7, 10 and 6 of 17, so the sheet sits inside the window.

## What remains, stated as hypotheses only

- **Acquisition:** MAN5 is the only test source scanned at 0.2 m propagation distance and 78 keV (0841: 1.2 m, 113 keV).
  Shorter propagation gives weaker phase contrast, which is the usual carrier of carbon-ink signal.
- **Labels:** MAN5 labels are registered photographs (v1). Ink visible in photographs need not be visible in CT, and a
  non-rigid photo-to-surface warp would not be caught by a translation search.

Neither is tested here. Until one is, MAN5 is reported as an open failure for every model, not tuned against.
