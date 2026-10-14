# Depth-pyramid defect evidence and physical-scale-aware repair

Local review draft, 2026-10-11. Nothing in this bundle has been published by this task.

Some ink-label surface pyramids have shortened depth while declaring unchanged depth scale. A 33-volume metadata survey flags 18 volumes. **All 18 are now pixel-verified** (2026-10-14, `tools/verify_all_pyramids.py`): in one label-free interior region per volume, every informative level-2 slice k matches the XY-pooled level-0 slice k (median r 0.857–0.99999) and not the 4×4×4 depth-pooled slab (median r −0.16 to 0.20). So level 2 holds the first 17 of 65 slices, i.e. only the top quarter of the depth range. The 7 surveyed volumes that do declare a z-scale of 4 (0139 w035–w044, MAN5) act as negative controls: all 7 match depth pooling (median r 0.80–0.96) and not truncation. Metadata flag and pixel verdict agree on 25/25 volumes; Paris4 (65 slices at every level) is not affected. The earlier single-region 0841 w00 proof (r 0.999885–0.999918 over 17 slices) is kept as `evidence/pyramid_pixel_proof.json`.

## Inspect existing evidence

`evidence/pyramid_survey.json` lists paths, shapes, declared scales and flags. `evidence/pyramid_pixel_all.json` (+ `.log`) holds the per-volume, per-slice correlations and errors for both hypotheses. `evidence/pyramid_pixel_proof.json` records chunk keys, all per-slice correlations and mean absolute errors for the verified region. `evidence/summary.json` gives independently recomputed counts and bounds. `SHA256.json` pins file contents.

## Repeat the all-volume pixel test

```bash
python tools/verify_all_pyramids.py evidence/pyramid_survey.json fresh_pixel_all.json
```

Needs NumPy, requests and numcodecs; downloads one level-0 region (65×128×128) and one level-2 region per volume, about 2 minutes per volume. Regions are chosen without labels: the first candidate near the centre whose level-0 data are >90% non-zero at every depth.

## Repeat the metadata survey

```bash
python tools/survey_pyramids.py fresh_survey.json
```

The survey uses public Hugging Face metadata and Python's standard library. The preserved historical script skips failed requests; check the inspected-volume count and network logs before comparing surveys. A smaller or missing survey is not evidence that defects were repaired. The snapshot is the evidence behind the historical 18/33 claim.

## Build a corrected input from fine-resolution data

To repeat the fixed-region pixel comparison, install NumPy, requests and numcodecs, then run:

```bash
python tools/verify_pyramid_pixels.py
```

This fetches metadata and two compressed chunks, compares same-index XY pooling with isotropic pooling, and writes `evidence/pyramid_pixel_proof.json`. Preserve the supplied snapshot before rerunning. Shape or encoding changes fail explicitly and require review. This reproducer is scoped to the stated 0841 region; it is not a generic verifier for every OME-Zarr layout.

Requires NumPy, Zarr 2 (`zarr==2.18.7`) and `numcodecs<0.16`. Download/stage the relevant source array first. The example uses verified isotropic source spacing of 4.681 micrometers, yielding 9.362 micrometers after pooling:

```bash
python tools/pool_surface.py source.zarr corrected.zarr --dataset 0 --voxel-um 4.681 --spacing-evidence path/to/scan_metadata.json
```

For independently verified 2.399-micrometer spacing, use `--voxel-um 2.399`, yielding 9.596 micrometers. Three Z,Y,X spacing values are accepted for anisotropic input. Do not infer physical spacing solely from relative pyramid scale factors. The script selects integer factors within its declared tolerance, refuses an existing output, streams spatial blocks, records actual output spacing and provenance, and marks completion in output attributes. It does not prove input spacing, align labels, fix sheet geometry or infer depth direction.

## Scientific scope

Recovered organizer-model controls on 1667 improved from roughly 0.54 to 0.91–0.93 AUC after corrected preparation. That source was in its training corpus. The model control is corroborating pipeline evidence, not an unseen-scroll generalization result or an isolated causal ablation of every preprocessing change. Full control artifacts remain in the project experiment records and are not bundled here.

Depth-order dependence and window checks have prior art by Andreas Hadjigeorgiou and Paulo Camillo. This package focuses on the depth-pyramid defect and physical-scale-safe preparation. Visual depth profiles, upstream review and community adoption remain release work. `tools/verify_all_pyramids.py` was run fresh against the public bucket on 2026-10-14; the single-region `tools/verify_pyramid_pixels.py` reproducer has not been rerun since packaging. Model performance on MAN5 remains weak.

Code/data redistribution rights require review before public release; this draft does not assign a new license to upstream metadata or scroll data.
