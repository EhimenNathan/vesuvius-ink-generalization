We found a depth-coverage inconsistency in the public ink-label surface pyramids and have prepared a reproducible report and workaround.

A metadata survey flags 18/33 volumes, and a label-free pixel test confirms all 18: level 2 keeps only the first 17 of 65 depth slices (XY-pooled, median r up to 0.99999) instead of depth-pooling them. The 7 volumes that declare z-scale 4 are correctly depth-pooled, so metadata and pixels agree on 25/25. Affected: 0009B, 0500P2, 0814, all three 0841, six 1667, 0139 w016, w017, w028, w029, w030, and unused/500p2a.

The workaround rebuilds model inputs from fine-resolution data using verified physical spacing, then records the actual output spacing. For example, 0841 is 4.681 micrometers and requires factor 2, not factor 4, to reach 9.362 micrometers. Organizer ink_9um controls on training source 1667 recover from approximately 0.54 to 0.91–0.93 AUC after corrected preparation; this is pipeline evidence, not unseen-scroll generalization.

We would welcome independent checks of additional regions and clarification of the intended pyramid depth contract. Prior work on depth direction and window selection by Andreas Hadjigeorgiou and Paulo Camillo should be credited.

[Insert public repository and issue links after publishing approval. This draft has not been posted.]
