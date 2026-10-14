# Organizer v8-in release: baseline and research update

2026-10-11. Restored evaluation verified RUNNING, failureMessage null. Its persisted log was empty at this check, so no restored AUC or per-segment scores are available. Latest audited Paris4 pilot mean AUC remains 0.591795 mixed versus 0.589991 restricted baseline. No 30,000-step run has started.

## Verified released sources

Model card and file inventory retrieved directly from public Hugging Face endpoints. Base-model repository revision: `d89166b41a3f5fad7749b3d7c0fdd1bd3695d844`; surface dataset revision: `7e4d918712a8d8642a6fdeee45b80217c1e98317`. No weights, large render arrays or 56 GB corpus downloaded during this check.

- [Base model](https://huggingface.co/YoussefMoNader/ink-8um-v8in): ResNet3D-50 plus 2D U-Net decoder, video-pretrained, 24x64x64 inputs clipped to 0–200 and divided by 255; InstanceNorm3d input; internal interpolation to 96x256x256 before encoding. Depth max collapse at four encoder levels. 83.4 million parameters; released epoch-6 checkpoint.
- [PHerc1447 surfaces](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces): 24 layers at native 8.64 micrometer spacing; both released models consume these layers reversed. Labels and certainty masks are available for w058/w060. w062 labels are excluded because re-flattening changed image coordinates.
- [Fine-tuned LOO-w062 model](https://huggingface.co/YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062): distinguish winding holdout from scroll holdout; it trains on the same scroll's other windings.

| PHerc1447 winding | Base v8-in AUC / AP | Fine-tuned LOO AUC / AP |
|---|---|---|
| w058 | 0.862 / 0.693 | 0.845 / 0.658 |
| w060 | 0.810 / 0.677 | 0.803 / 0.662 |
| w062 | No aligned released labels | No aligned released labels |

These are publisher-reported metrics against refined labels, inside certainty masks with ink threshold >127. We have not independently recomputed them. Base-model PHerc1447 source exclusion is documented by the publisher; LOO models hold out a winding, not the entire scroll. PNG predictions are quantized to uint8. Thin unpredicted coverage rims are documented; missing inference must not be silently dropped or mistaken for verified no-ink.

The base card documents 508,006 training patches plus 43,249 validation patches, from a broader training corpus than our restricted pilot. 0500P2 and a fragment are included; 0009B_ag132115 is a validation segment. Our old comparison sources are not all unseen by this baseline. Source and segment exposure must be audited before a shared generalization claim. Our 0.591795 Paris4 score cannot be compared numerically with its PHerc1447 scores.

## Research implications

The supplied organizer announcement reports less glue response in newer Paris4 models, signals on PHerc1447, and inadequate cross-scroll transfer of the linked PHerc0172 method. The Zenodo page returned HTTP 429; that paper was not independently reviewed here. The announcement also says PHerc1447 was withdrawn from First Letters eligibility while remaining a Grand Prize target. Treat the withdrawal as organizer-provided information and verify the official eligible list before prize submission.

Input resampling before the decoder and InstanceNorm are already part of the released v8-in method. They are useful controlled baseline choices, not our new invention. The training repository also includes `group_dro.py`; inspect the active recipe before describing worst-group optimization as novel or as already active. Broad data mixing is insufficient if included representations lack usable ink signal; inclusion needs evidence determined from training/development sources.

Our strongest potential differentiator remains trustworthy cross-render/scale generalization with diagnosed geometry and missing-depth failures. It must outperform or complement this new baseline on a matched population; proving a gain over ink_9um alone is now insufficient for a strong current-method claim.

## Proposed matched benchmark, before further tuning

1. Complete the ongoing Paris4 evaluation and retain its baseline reproduction check.
2. Pin the PHerc1447 dataset revision and check our own training inventories/checkpoints for PHerc1447 exposure. Freeze our chosen checkpoint, physical-scale adaptation, center-depth selection, normalization, depth convention and scoring mask before looking at our w058/w060 results. No adaptation or pseudo-label training on PHerc1447 in the source-held-out benchmark.
3. Independently reproduce base v8-in metrics from released predictions/labels first, with mask, threshold and quantization recorded. Resolve any discrepancy rather than optimizing until the reported numbers appear.
4. Run our frozen model on those same pixels. Respect the 8.64 micrometer input spacing; our prepared inputs use a different physical scale/depth window, so an explicitly defined adaptation is necessary. Keep predictions and labels aligned on the native surface grid. Check actual prediction coverage and record the intersection only as a secondary comparison if coverage differs; primary requested pixels remain fixed with uncovered pixels flagged.
5. Report w058 and w060 separately with AP, exact sample counts and fixed-region visual examples. These are one new scroll, not two independent scrolls. Human-refined annotation provenance limits claims of fully independent truth. Do not score w062 against unaligned older labels or model predictions.
6. Once inspected for development, this source no longer serves as an untouched test for later tuned models. Broader cross-scroll improvement needs additional reserved sources/independent annotations and replication.

Fine-tuning, teacher-assisted pseudo-labeling and zero-shot inference are separate experimental categories. Avoid combining them in a headline table without explicit exposure labels. A new model's visually cleaner output is valuable, but it requires representative examples and false-positive checks on glue/fibers, not selected attractive letters alone.
