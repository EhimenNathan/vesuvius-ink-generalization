# ink/ label bucket: in 18 of 33 surface volumes, pyramid level 2 holds only the first 17 of 65 depth slices while declaring z-scale 1

**In one sentence:** in 18 of the 33 surface volumes of the Hugging Face `scrollprize/datasets` `ink/` bucket, level 2 is the first 17 of the 65 rendered slices pooled only in XY, yet `.zattrs` declares an unchanged z-scale of 1, so anything that reads it as a downsampled render sees only the outermost quarter of the depth. Levels 1 and 3–5 have the same shortened depths (33, 9, 5, 3) with z-scale 1; we pixel-tested level 2 only.

**Why this matters, and how fixing it helps**
- *Who is affected:* anyone who trains or evaluates an ink model from these volumes' coarser levels, for example by taking the 17-slice level 2 as a ready-made ~9.6 µm input (which is what we did at first).
- *What goes wrong, silently:* the data look valid (correct XY size, plausible intensities, no error), but the model receives the outer quarter of the papyrus depth instead of the full window it was trained on. Results then point to the model instead of the input. On these inputs the released `ink_9um` looks unable to read even a scroll it was trained on (AUC 0.54 on 1667, see below). We first wrongly concluded "ink_9um does not generalise" before tracing it to this.
- *What a fix enables:* correct training and evaluation on 18 labelled volumes, including held-out material used in cross-scroll comparisons (0841, 0009B, 0500P2), and comparable numbers across groups. The same model on the same scroll goes from AUC 0.54 to 0.91–0.93 once the input is rebuilt correctly.

**Affected (metadata flag and pixel test agree):** 0009B `auto_grown_20250919055754487_inp_hr_2um`, 0500P2 `-1`, 0814 `814_46527_2um_try2`, all three 0841 (`w00`, `auto_grown_20260220144552896`, `auto_grown_20260220174252405`), 1667 `w013 w018 w023 w028 w029 w031`, 0139 `w016 w017 w028 w029 w030`, and `unused/500p2a`. Paris4 (65 slices at every level) and the seven volumes that declare z-scale 2/4 (0139 w035–w044, MAN5) are correct.

**Evidence (label-free).** For one interior region per volume, each level-2 slice k is compared with (a) the level-0 slice k pooled 4×4 in XY and (b) level-0 slices 4k..4k+3 pooled 4×4×4:

| | median r with (a) truncation | median r with (b) depth pooling |
|---|---|---|
| 18 flagged volumes | 0.857 – 0.99999 | −0.16 – 0.20 |
| 7 volumes declaring z-scale 4 | −0.17 – 0.23 | 0.80 – 0.96 |

Example: `ink/841/w00/w00.zarr`, level 0 `[65,15872,18944]`, level 2 `[17,3968,4736]`; level-0 chunk `0.60.72` vs level-2 chunk `0.15.18`: all 17 level-2 slices match XY-pooled level-0 slices at the same index, r 0.999885–0.999918.

**Reproduce:** `python pyramid-repair/tools/verify_all_pyramids.py` (NumPy, requests, numcodecs; reads one region per volume from the public bucket, about 2 min per volume) in https://github.com/EhimenNathan/vesuvius-ink-generalization. Per-slice results, survey and checksums are in `pyramid-repair/evidence/`.

**Control measurement:** the released `ink_9um` scores AUC 0.54 on its own training scroll 1667 when fed the published level 2, and 0.91–0.93 when the input is rebuilt from level 0 (training-source control, not a generalization claim).

**Workaround:** rebuild from level 0 with a pooling factor chosen from the render's physical voxel size (`pyramid-repair/tools/pool_surface.py`). Note that the 0841 renders are 4.681 µm (`meta.json` volume `4.681um_113keV_1.2m_binmean_2`), so they need 2× pooling to reach ~9.4 µm, not 4×.

**Suggested fix:** regenerate levels 1–5 of these 18 volumes with depth pooling, or declare the actual crop in the coordinate transforms, and add a shape/scale consistency check to pyramid generation.

Prior work on depth direction and depth windows: @AndreasHad04 (#1867) and Paulo Camillo.
