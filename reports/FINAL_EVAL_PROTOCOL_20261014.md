# Final evaluation protocol: frozen 2026-10-14, before any 30k or v8-in result was seen

Written while `vesuvius-ink-final-b0p256-30k` (main) and `vesuvius-v8in-eval` (second account) were RUNNING, with no
output from either available. Nothing below may be changed after their results are seen; any later deviation is
reported as a deviation.

## Test population
Never-seen (by our models and by ink_9um) sources at verified physical scale: 0841 ×3 segments (data3b, 9.362 µm),
MAN5_outer_3 (9.596 µm), 0009B (≈9.6 µm), 0500P2 (≈8.8 µm). All supervised pixels; one common population per
comparison (`ensemble_audited.py`, intersection of finite maps, 300k-pixel seeded sample per segment for the
pooled numbers; per-segment metrics on the same sample).

## Fixed choices
- Depth order: **reverse** for every model of ours and for ink_9um (the `ink/` renders are depth-reversed relative to
  the training representation; prior art: Andreas Hadjigeorgiou, villa #1867). Forward reported as a diagnostic only.
- v8-in direction: its label-free rule in `v8in_eval_kernel.py` (higher prediction spread on a fixed pilot tile
  subset). Both directions are reported when available.
- Checkpoint: EMA primary for every model; SWA (30k run only) is secondary.
- No smoothing, no test-tuned threshold. F0.5 at 0.5 is the deployable number; best-threshold F0.5 is diagnostic.
- Combination: mean of probabilities primary; rank mean secondary.

## Headline (ours only)
1. Single model: **B0-256 30k s44 EMA**.
2. Ensemble: mean of **B0-256 s42, B0-256 s43, V3+DINOv3 ViT-S, B0-256 30k s44** (all EMA).
Compared against ink_9um s42, s43 and their mean ensemble on identical pixels. Primary statistic: macro-source AUC
over the four sources (0841, MAN5, 0009B, 0500P2), with pooled AUC, AP and per-source values alongside. MAN5 is
reported, never dropped.

## Secondary
- Ours + ink_9um (practical recommendation, not a superiority claim).
- **v8-in head-to-head** on the pixels v8-in's tiling covers (coverage fraction reported). Exposure: 0841 and MAN5
  unseen by v8-in (clean comparison; MAN5 depth only 163 of v8-in's 207 µm, 6 layers edge-clamped); 0009B is a v8-in
  validation source (another segment); 0500P2 is in v8-in training. The clean v8-in claim uses **0841 and MAN5 only**.

## Disclosure
Steps, batch, patch, seed, parameter count and wall time for every model. Our 4 test sources have informed
development since early October (choice of patch size, loss, backbone), so they are "held-out sources", not an
untouched final test; Paris4 is a development fold.
