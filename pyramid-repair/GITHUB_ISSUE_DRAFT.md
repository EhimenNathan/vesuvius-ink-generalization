Title: Ink-label surface pyramid loses depth coverage while declaring unchanged Z scale

We found a depth-coverage inconsistency in the public ink-label surface pyramids. A metadata survey of 33 volumes flags 18 whose coarser levels have shorter depth but a declared Z scale of 1. A label-free pixel test confirms all 18: in an interior region of each, level-2 slice k equals the XY-pooled level-0 slice k (median r 0.857–0.99999), not the depth-pooled slab (median r −0.16 to 0.20). Level 2 therefore holds the first 17 of 65 slices. The 7 surveyed volumes that declare Z scale 4 (0139 w035–w044, MAN5) behave as true depth pooling (median r 0.80–0.96), so metadata and pixels agree on 25/25 volumes.

Detailed example: `ink/841/w00/w00.zarr`, fine-resolution shape `[65,15872,18944]`, level-2 shape `[17,3968,4736]`. In the aligned region covered by fine chunk `0.60.72` and level-2 chunk `0.15.18`, each of the 17 level-2 slices matches XY-only pooling of the fine slice at the same index (Pearson r 0.999885–0.999918). The comparison against depth-plus-XY pooling differs increasingly with slice index. The attached JSON includes every slice correlation and mean absolute error.

This means a consumer selecting a central window from the shortened array can receive a different physical depth interval from the full render. Either the depth content or its declared coordinate transform needs correction, depending on the intended pyramid contract.

Our workaround reconstructs inputs from the fine-resolution source using independently verified voxel spacing, writes actual physical output spacing and preserves provenance. PHerc0841 requires factor 2 from 4.681 micrometers; a factor 4 assumption would produce an incorrect physical scale. Relative scale factors alone do not establish micrometer spacing.

Corroborating pipeline controls: released ink_9um evaluated on corrected 1667 inputs recovers from roughly 0.54 to 0.91–0.93 AUC. 1667 is a training source for that model; this is not a generalization claim or a complete isolation of the pyramid change from every preprocessing choice.

Proposed maintainer checks: confirm intended depth sampling and coordinate transforms, compare fine/coarse slices on additional affected surfaces, rebuild invalid levels or document the crop origin, and add shape/transform plus bounded pixel checks to pyramid generation.

Attachments prepared locally: survey JSON, all-volume pixel test JSON/log and tool, direct 0841 pixel comparison JSON, summary and checksums; physical-scale-aware streaming builder. No report has been posted yet. Credit for prior depth-direction/window investigations belongs to Andreas Hadjigeorgiou and Paulo Camillo.
