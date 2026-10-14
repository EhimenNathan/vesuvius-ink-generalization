# Pilot diagnosis and source-restoration control — 2026-10-11

The CPU audit completed. The matched-format pilot had substantially less training coverage than the historical recipe. This is a plausible explanation for its weak baseline, not an established cause of the AUC difference. A private 3,000-step control is now RUNNING on Kaggle; no new AUC is available and no 30,000-step run has started.

## Evidence

CPU job: `ehimennathan/vesuvius-pilot-cpu-diagnosis-20261011` (COMPLETE). Four CPU workers audited 31 stores. Training counts use `sup & ~val`; held-out Paris4 ink labels were not read in this audit. Local compact evidence is in `pilot_cpu_diagnosis_summary.json` and the cached job log. The full `pilot_input_audit.json` remains in Kaggle outputs; its direct download timed out.

| Source | Historical stores | Restricted pilot stores | Historical training-pixel sum | Pilot training-pixel sum | Retained | Sampling probability |
|---|---:|---:|---:|---:|---:|---:|
| 0139 | 14 | 9 | 12,577,127 | 7,556,784 | 60.08% | 54.72% |
| 1667 | 6 | 1 | 27,315,006 | 1,198,292 | 4.39% | 41.51% |
| 0814 | 1 | 1 | 428,993 | 428,993 | 100% | 3.77% |

These are store-level pixel sums. Overlapping acquisitions and surfaces mean they are not counts of unique physical pixels. Five omitted native 0139 views correspond to surfaces also represented in aligned 0139 inputs; restoring them adds representations, not five independent sources.

The pilot concentrated 1667 sampling on one canonical surface. At 3,000 steps and batch 16, its expected patch presentations per 1667 store increased from approximately 3,321 to 19,925. These are expectations under the sampler, not measured unique patches.

Historical logs also reveal confounds: stronger older runs used 6,000 steps and seed 42 or 43, versus 3,000 and seed 314159 in the pilot; depth jitter and some losses differed. Their AUCs cannot establish an isolated format or coverage effect.

The matched-format gain was small and fragile: mean AUC 0.589991 to 0.591795, with AP slightly declining. Leaving out one segment gives mean deltas from -0.001631 to +0.004432; leaving out w00 reverses the sign. This is descriptive sensitivity, not an uncertainty interval or evidence of independent segment samples.

Label-free Paris4 normalization checks found median shifts of approximately 0.022–0.078 core MAD units and full/core MAD ratios of 0.965–1.000. These population statistics do not reproduce the stochastic Torch normalization estimator and do not show a normalization-induced AUC change.

## Controlled next experiment

Private job: `ehimennathan/vesuvius-source-restoration-3k-20261011`, version 1, verified RUNNING with no reported failure. Protocol: `source_restoration_protocol.json`; full input manifest: `source_restoration_manifest.json`.

Restore the five omitted 1667 stores and five native 0139 views. Keep the same audited canonical original 1667-w029 input. Keep B0, patch 256, seed 314159, focal/Dice loss, label smoothing 0.1, batch 16, 3,000 steps, learning rate 0.0002, weight decay 0.01, source weights, zero depth jitter/flip, EMA, and zero mixed-format replacement. No SWA or self-training. The corrected pair remains declared for provenance but replacement probability is zero.

Training loads 21 original training stores plus the declared paired store. Held-out inputs are excluded from the training phase to limit host memory. One training process saves its EMA checkpoint before any evaluation. Target-stream hashes will differ from the restricted pilot by design because the training inventory changes.

Evaluate separately, one segment/model at a time, on the same eight Paris4 segments. Primary remains forward-depth EMA, unsmoothed, TTA 4, 17-slice input-window normalization, identical normalization seeds, and equal-segment mean AUC. The restricted baseline reference is 0.5899907357566443. Use the same supervised scoring population. Preserve full checkpoint defaults, including evaluation stride 0.5.

This single-seed control tests the effect of restoring the inventory jointly. It cannot separate the contribution of 1667 coverage from native 0139 views, establish a mixed-format gain, or prove generalization beyond the previously used Paris4 development holdout. A longer run remains gated on the result and a defensible follow-up.
