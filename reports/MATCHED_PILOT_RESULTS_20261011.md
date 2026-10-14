# Matched-format pilot result — 2026-10-11

Evaluation version 2 completed successfully. Both B0 arms trained for 3,000 steps with seed 314159 and identical training-target streams. The EMA checkpoints were evaluated on the same eight original-format Paris4 segments, with identical normalization subsamples, forward depth order and no smoothing. All 88,625,409 supervised pixels were used in the per-segment metrics. This count does not represent independent observations.

| Measure | Baseline | Mixed format | Difference |
|---|---:|---:|---:|
| Primary: equal-segment mean AUC | 0.589991 | 0.591795 | +0.001804 |
| Secondary: pooled sampled AUC | 0.585251 | 0.588524 | +0.003273 |
| Equal-segment mean AP | 0.362034 | 0.361041 | −0.000994 |

The pooled statistic uses the same deterministic 300,000-pixel sample from each segment (2.4 million total), not all pixels. Primary arithmetic and per-segment scoring counts/class proportions were independently checked against the persisted log. Both target-stream digests match.

| Paris4 segment | Baseline AUC | Mixed AUC | Difference |
|---|---:|---:|---:|
| w00 | 0.571077 | 0.596928 | +0.025851 |
| w01 | 0.582577 | 0.582651 | +0.000074 |
| w02 | 0.526726 | 0.510136 | −0.016591 |
| w03 | 0.554314 | 0.558344 | +0.004030 |
| w05 | 0.586648 | 0.578949 | −0.007698 |
| w06 | 0.569348 | 0.569708 | +0.000361 |
| w07 | 0.676292 | 0.673735 | −0.002557 |
| w09 | 0.652943 | 0.663910 | +0.010967 |

Mixed training improved five of eight segments, with three regressions. This is a small, uneven single-seed result; it does not establish a reproducible generalization gain or justify immediate promotion to 30,000 steps. AP slightly declined. Reverse-depth scores remain secondary and were not used to select the headline result.

Only one training surface, 1667-w029, has both representations. Paris4 was isolated from this run's training, but has informed previous development and is not an untouched final test. These 3,000-step Paris4 scores are not directly comparable to the older 12,000-step four-source macro scores. This pilot did not run the organizer's ink_9um baseline.

Follow-up: the CPU input audit completed and a predeclared 3,000-step source-restoration control is RUNNING. The restricted pilot retained only 4.39% of the historical 1667 store-level supervised training-pixel sum while preserving its 41.51% sampling probability. This is a candidate explanation, not a demonstrated cause. See `PILOT_DIAGNOSIS_20261011.md` and `source_restoration_protocol.json`. The 30,000-step run remains gated; no new AUC is available yet.

Evidence: `matched_pilot_results.json`, `vesuvius-matched-evaluation-20261010_output.json`, and https://www.kaggle.com/code/ehimennathan/vesuvius-matched-evaluation-20261010 (version 2).
