# Ink detection experiments (cross-scroll generalisation)

## 2026-10-16 01:05 UTC: v8-in fine-tune VERDICT, negative (predeclared rule not met)
Identical pipeline (v8in_eval_kernel.py, reverse, batch 8), per-segment AUC, base -> fine-tuned:
0841 ag4252405 0.723 -> 0.709, ag4552896 0.798 -> 0.794, w00 0.784 -> 0.761 (source mean 0.768 -> 0.755);
MAN5 0.488 -> 0.442; PHerc1447 w058 0.793 -> 0.820, w060 0.760 -> 0.774 (mean 0.777 -> 0.797).
Macro over the three sources (segment means): base 0.678, fine-tune 0.665. The fine-tune wins 1 of 3 sources and loses
the macro, so the claim fails. The validation gain on held-out training stores (0.804 -> 0.835) did not transfer to
unseen scrolls. Also: base v8-in is the best single model on 0841 on identical pixels (0.768 vs Reader v2 0.756, our
B0 0.753), and our 9.6 um store round trip costs v8-in about 0.06 AUC on PHerc1447 (0.793 / 0.760 vs 0.862 / 0.810
from its native 8.64 um layers).

## 2026-10-15 22:55 UTC
- Base v8-in round-3 evaluation ran 8.7 h (the identical fine-tune job took 1 h 53 min) with no output: treated as hung
  and superseded by an identical relaunch (same code and config, so the comparison stays fair).
- U2-NetP Paris4 ablation (lop4-u2p) ended ERROR with an empty log: killed when the main account's GPU quota ran out.
  No result; the U2-Net question remains open (the Colab U2-NetP run on the patch pack was at chance on PHerc1447).
- Discord post drafted (submission/DISCORD_POST.md) for the user to post; issue villa#2107 still has no replies.

## 2026-10-15 16:05 UTC: fine-tuned v8-in, round 3 COMPLETE (reverse, batch 8)
0841 ag4252405 0.709 (AP 0.490), ag4552896 0.794 (0.694), w00 0.761 (0.584); MAN5 0.442 (0.246);
PHerc1447 w058 0.820 (0.654), w060 0.774 (0.610). Base v8-in through the identical pipeline still running; verdict
pending. (Base v8-in's released predictions on the native grid: 1447 0.862 / 0.810.)

## 2026-10-15 14:10 UTC: v8-in evaluations, round 2 failed; round 3 launched
- Batch-8 evaluations crashed on a JSON int64 bug after logging metrics; outputs (maps) lost. Logged metrics:
  fine-tuned v8-in, reverse (label-free rule chose rev): 0841 ag4252405 0.709, ag4552896 0.794, w00 0.761; MAN5 0.442.
  PHerc1447: the label-free rule chose fwd for BOTH models, giving chance (fine-tune 0.507 / 0.480; base 0.494 / 0.473).
- The original base v8-in run on 0841/MAN5/0009B/0500P2 (batch 32) failed with OOM on every segment: no base numbers.
- Protocol amended (direction fixed to reverse for all sources, both models; see protocol file). Round 3: fine-tune and
  base, 1447 + 0841 + MAN5, reverse, batch 8, JSON fixed. Main-account GPU quota exhausted (resets 2026-10-17 00:00 UTC).

## 2026-10-15: bug report filed; v8-in fine-tune run 2 done; test evaluation relaunched
- Pyramid bug filed as ScrollPrize/villa#2107 (08:27 UTC, with the user's own commentary and attestation), text also
  in the repo as pyramid-repair/BUG_REPORT.md.
- v8-in fine-tune run 2 (memory guard worked; stopped cleanly at step 538): held-out-store validation AUC 0.804 (base)
  -> 0.747 (75) -> 0.815 (150) -> **0.835 (225, selected)** -> 0.821 / 0.820 / 0.821 / 0.834. ft_best.safetensors saved.
- First test evaluation of the fine-tune and of base v8-in on PHerc1447: every segment hit CUDA OOM (inference batch
  32 over 2 T4 = 16 tiles of 96x256x256 per GPU). No result. Relaunched with batch 8. The 4.5 M U2-Net ablation
  (lop4-v6u2) was stopped to save main-account quota; the U2-NetP ablation (lop4-u2p) continues.

## 2026-10-15: identical-pixel ensembles with Reader v2, and letter shape (Kaggle CPU `vesuvius-score-ensembles`,
`vesuvius-letter-shape`; score pack dataset `ehimennathan/vesuvius-score-pack`)
Four sources (rev, 300k px/segment): Reader v2 alone macro 0.785, pooled 0.789, AP 0.645, 0841 0.756.
**Predeclared all-model mean (6 models): macro 0.790, pooled 0.778, AP 0.643; 0841 0.771 (+0.015 over the best
single model)**, 0009B 0.912, 0500P2 0.904, MAN5 0.571. Descriptive (post hoc): B0 s43 + Reader v2 macro 0.791,
pooled 0.792, AP 0.650, **0841 0.780**: our B0 is complementary to Reader v2 on the never-seen scroll.
PHerc1447: v8-in 0.832; predeclared all-model mean (7) 0.802 (weak members hurt); descriptive Reader v2 + v8-in
**0.839**, AP 0.694.
Letter shape (villa #1907 elongation, eval_forme.py as written; labels' own value in brackets), 0841 ag4252405 [25.0]:
B0 s42 27.8, B0 s43 22.8, Reader v2 25.8, ink_9um 35.6/36.7, mean 20.4; ag4552896 [21.8]: 26.3, 23.1, 14.8, 16.1/21.2,
17.4; w00 [20.0]: 27.7, 20.1, 11.2, 21.8/19.0, 14.8. PHerc1447 w058 [29.2]: 18.8, 19.1, 16.1, 16.4/18.2, v8-in 12.8;
w060 [29.4]: 8.6, 6.6, 8.6, 8.7/13.9, v8-in 8.9. With 5-9 letters per segment and speckle inflating the metric
(fragmented noise also scores high), no shape claim is made; ensembles that win on AUC lose elongation (blurring).

## 2026-10-15: Reader v2 on our corrected stores + ensemble rule (declared before any ensemble score was computed)
Reader v2 (domenicor046/reader-v2 step 40000, sha256 verified, run through villa's koine_machines infer exactly as
ink_9um; reverse order predeclared like every other model; forward is near chance everywhere):
0841 0.756 | PHerc1447 0.797 | MAN5 0.563 | 0009B 0.923 (exposed: other 0009B segments in its training) | 0500P2 0.896
(exposed: in its training). Its published 0841 AUC (0.824, own pipeline from the native 9.36 um scan) is higher than
on our 0841 inputs (pooled from the 4.681 um scan): it is sensitive to the input representation.
Ensemble rule, fixed now: primary = equal-weight mean of probabilities of EVERY model evaluated on a source (ours:
B0 s42, B0 s43, DINOv3; organizers': ink_9um s42, s43, v8-in where available; community: Reader v2), on identical
pixels, reverse order. No member selection on test labels. Subsets (ours-only, public-only, pairs) are descriptive.

## 2026-10-15: Colab lightweight results (negative) and a patch-pack validation caveat
- E1b U2-NetP (1.15 M, no upsampling, 60 min, 5,660 steps, about 1.4 epochs): pack-validation AUC 0.527 -> 0.620 (1k)
  -> 0.648 (2k) -> 0.701 (3k, selected) -> 0.561 (4k) -> 0.693 (5k). **PHerc1447: w058 0.495, w060 0.559, mean 0.527
  (chance)**; v8-in 0.862 / 0.810. The rule (beat v8-in on both) is clearly not met.
- v8-in as loaded on Colab reproduces its released PHerc1447 w058 map (r = 0.945 on a 768 px crop, mean 0.319 vs
  0.315), so loading and inference are correct.
- But v8-in scores **below chance on the pack's own validation patches** under every layout tried (512 patches:
  as stored 0.407, depth-flipped 0.125, H/W transposed 0.379, both 0.109, labels transposed 0.390; 4,096 patches:
  0.4635). Labels are ink = 255, 2.8 % positive (train 18.8 %). Either the validation patches do not align with their
  labels as I read them, or a convention is unstated; they are not usable as a checkpoint-selection proxy here, and
  E1b's selection was therefore unreliable as well.
- Throughput limits (about 200 patches/s, about 1.4 epochs/h on a T4) make training lightweight models from scratch on
  the pack non-competitive within free Colab. Line stopped; effort moved to Reader v2 / v8-in / fine-tune ensembles.
- Community models found (2026-10-15): Reader v2 (domenicor046/reader-v2; 0841 0.824 vs human labels, 0.858 vs team
  map; trained by distilling the team's ~2.4 um maps), Hecate (team; 0841 0.855). Our B0 (0.753 on 0841) is not
  competitive with them. villa #1907 shows every ~9 um map renders 0841 letters as blobs (elongation metric).

## 2026-10-15: lightweight models on the v8 patch pack (Colab T4), predeclared before any test result
- Data: `YoussefMoNader/ink-8um-v8-patchpack` rev d67b1c1 (508,006 train / 43,249 val patches, 24x64x64 at 8.64 um),
  the same data v8-in was trained on. Script `colab/lite_ink.py` (sha256 c4708c13..., identical on Colab).
- Models: our 3D stem (with v8-in's per-tile InstanceNorm3d input) + U2-NetP body (1.15 M) or B0 SmallUNet (5.5 M).
  Recipe: v8-in loss (0.5 Dice + 0.5 BCE, smoothing 0.25), fourth augment, rot90/flip, AdamW 2e-3 cosine, AMP, fixed
  time budget. U2 variants add deep supervision on all six side outputs (as in the U2-Net paper).
- Checkpoint selection: AUC on 4,096 fixed patches of the pack's own validation split (0009B, Frag3, Frag4) only.
- Test: PHerc1447 w058/w060 (dataset rev 7e4d918), run through v8-in's released `predict_stack` (64 px tiles,
  stride 21, layers reversed), AUC inside the certainty mask with label > 127, exactly the protocol that reproduces
  v8-in's published 0.862 / 0.810.
- Rule: a lightweight model "beats v8-in" only if its mean AUC exceeds 0.836 AND it beats v8-in on both windings.
- Throughput (T4, batch 128): U2-NetP 240 patches/s without in-model upsampling, 125 with 2x; B0 268 / 168.
  E1 (U2-NetP, 2x upsampling) was stopped after 100 steps for speed; E1b = U2-NetP, no upsampling, 60 min.

## 2026-10-15: U2-NetP (lightweight) ablation, predeclared before launch
Same as the V6 entry below but U2-NetP widths (mid 16, out 64): 1.15 M parameters total with the 3D stem (B0 5.54 M).
Same comparator and rule: counts as better than B0-256 only if the 8-segment Paris4 forward mean exceeds 0.721 and
at least 6/8 segments improve. Only if it passes is it trained on all four sources and scored on the test sources.
The 30k-step B0 run was stopped by the user after >12 h without completing (no outputs).

## 2026-10-15: PHerc1447 complete for our models and ink_9um (rev; pooled 300k/segment)
| model | PHerc1447 rev AUC | (fwd) |
|---|---:|---:|
| ours B0 s43 | 0.745 | 0.477 |
| ours B0 s42 | 0.733 | |
| ours DINOv3 | 0.690 | 0.497 |
| ink_9um s43 | 0.682 | 0.493 |
| ink_9um s42 | 0.674 | 0.486 |
| organizer v8-in (released zero-shot run) | **0.832** | |
Our B0 beats ink_9um on this fresh scroll too (+0.05 to +0.07); v8-in leads by ~0.09. Reverse order confirmed for all.
The `vesuvius-v8in-eval` session (0841/MAN5) shows a restart at 00:30 UTC; its first run's outputs are not available.
Open-problem mapping (scrollprize.org/2026_open_problems): our work addresses #5 cross-scroll generalization and #2
label/data quality; villa help-wanted #192/#193 ask for 3D labels, which this work does not provide.

## 2026-10-15: U2-Net ablation (user request), predeclared before launch
V6 = B0's 3D stem + U2-Net 2D body (RSU-7/6/5/4/4F/4F encoder, 5 RSU decoders, fused side outputs; mid 32, out 128;
4.53 M params vs B0 5.54 M). Single fused output, no deep supervision (deviation from the paper). Comparator: the
historical Paris4-held-out B0-256 Focal+Dice run, identical config (seed 42, 6000 iters, batch 16, TTA 4), 8-segment
forward mean AUC 0.7048 (w00 .622, w01 .700, w02 .676, w03 .680, w05 .715, w06 .699, w07 .742, w09 .805).
Rule: V6 counts as better only if its 8-segment mean exceeds 0.721 (0.7048 + seed noise 0.016) AND it improves at
least 6/8 segments. Caveat: trainer revisions since the B0 run (coverage repair, evaluation guards) leave defaults equal.

## 2026-10-14: v8-in fine-tune on our corrected inventory (protocol `V8IN_FINETUNE_PROTOCOL_20261014.md`)
More results on PHerc1447 (rev): DINOv3 w058 0.731, w060 0.660, pooled 0.690 (B0 s43 0.745; v8-in 0.832).
Smoke `vesuvius-v8in-ft-smoke` (12 steps) passed after one fix (inference batch 32 OOM on T4; now 8):
- No training store has official validation-mask pixels, so the predeclared fallback holds out phercparis4-w09 and
  pherc1667-w031 entirely (7.8 M validation pixels).
- Depth order from training labels: base v8-in reads our aligned training representation as stored (AUC 0.817 vs
  0.465 reversed), consistent with reversing the ink-bucket test renders.
- Base v8-in validation AUC 0.771 (768-tile protocol will differ slightly). Peak memory 8.5 GB per T4 at 2 tiles/GPU;
  about 4 tiles/s, i.e. 16 s per 64-tile step.
Full run `vesuvius-v8in-ft-run1` launched 23:10 UTC: 1800 steps x 64 tiles, LR 1e-5 cosine, validation every 150.

## 2026-10-14: identical-pixel scoring under the frozen protocol (30k and v8-in-on-our-sources still pending)

`train/score_final.py` (ensemble_audited, 300k seeded pixels per segment, reverse order, uint8 maps).
| | 0841 | 0009B | 0500P2 | MAN5 | macro (primary) | pooled | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 s42 | 0.730 | 0.885 | 0.885 | 0.538 | 0.759 | 0.737 | 0.587 |
| B0 s43 | 0.753 | 0.901 | 0.875 | 0.533 | 0.766 | 0.759 | 0.596 |
| DINOv3 | 0.729 | 0.875 | 0.870 | 0.504 | 0.744 | 0.738 | 0.576 |
| ours mean (s42+s43+DINOv3) | 0.761 | 0.899 | 0.894 | 0.526 | 0.770 | 0.760 | 0.616 |
| ink_9um s42 | 0.703 | 0.868 | 0.829 | 0.619 | 0.755 | 0.734 | 0.560 |
| ink_9um s43 | 0.720 | 0.868 | 0.852 | 0.553 | 0.748 | 0.732 | 0.566 |
| ink_9um mean | 0.728 | 0.883 | 0.864 | 0.621 | **0.774** | 0.751 | 0.598 |
| ours + ink_9um (secondary) | 0.764 | 0.902 | 0.897 | 0.568 | 0.783 | 0.769 | 0.631 |
**On the predeclared primary (macro over 4 sources) our ensemble does not beat the ink_9um ensemble (0.770 vs 0.774)**:
ours wins 0841/0009B/0500P2 and pooled AUC/AP; MAN5 decides the macro. Our single models beat each ink_9um seed.

PHerc1447 (fresh source; prep5, 9.6 um, 17 depths; gate: v8-in released predictions reproduce the publisher's
0.862/0.810 exactly on the native grid and 0.8625/0.8121 on our grid):
| | w058 | w060 | pooled | AP |
|---|---:|---:|---:|---:|
| B0 s43 (rev; fwd 0.47) | 0.788 | 0.720 | 0.745 | 0.540 |
| organizer v8-in base, released zero-shot run | **0.863** | **0.812** | **0.832** | **0.676** |
| mean of both | 0.857 | 0.791 | 0.813 | 0.658 |
v8-in beats our B0 on this unseen scroll by ~0.08 AUC, and averaging hurts. Our model claim is therefore limited to
"beats the released ink_9um", not "beats the organizers' current model". 0343P was dropped: its supervision mask
equals its ink labels (no negatives).

## 2026-10-14: restored-inventory result (forward evaluation COMPLETE)

`ehimennathan/vesuvius-restored-forward-20261011` completed. Baseline reproduction gate passed exactly
(delta 0.0 vs the matched-pilot evaluation). Same eight Paris4 segments, forward EMA, unsmoothed, TTA 4.

| Paris4 seg | restricted pilot base | restored inventory | diff |
|---|---:|---:|---:|
| w00 | 0.5711 | 0.6370 | +0.066 |
| w01 | 0.5826 | 0.6606 | +0.078 |
| w02 | 0.5267 | 0.6281 | +0.101 |
| w03 | 0.5543 | 0.6366 | +0.082 |
| w05 | 0.5866 | 0.6566 | +0.070 |
| w06 | 0.5693 | 0.6483 | +0.079 |
| w07 | 0.6763 | 0.7175 | +0.041 |
| w09 | 0.6529 | 0.7905 | +0.138 |
| **mean AUC** | **0.5900** | **0.6719** | **+0.082** |
| mean AP | 0.3620 | 0.4715 | +0.110 |

All 8 segments improve. The matched-format pilot's weak baseline was caused by its restricted training
inventory (4.39% of 1667 pixels, 10 stores omitted), not by the model. The restored 3k model is close to the
historical 6k-12k B0-256 runs (7-segment means 0.677-0.703), consistent with that. The mixed-format +0.0018 was measured on a
crippled baseline and is not carried forward. The older `vesuvius-restored-evaluation-20261011` (v1) ended ERROR
and is superseded by the forward job. Next: the final 30k run uses the full historical inventory (no manifest).

Trainer hardening (2026-10-14): `train_v2.py` evaluation failures are now logged and results saved instead of
raising (an ERROR status discards checkpoints), and SWA evaluation is skipped if it would overrun the 12 h session.
Codex's hash-pinned version is kept as `train_v2_codex_20261010.py` (sha256 ac6afaf7...).


## 2026-10-11: user-requested forward evaluation restart

User reports no visible active session and authorized a restart. Both checkpoint
jobs were rechecked COMPLETE. New private job
`ehimennathan/vesuvius-restored-forward-20261011` version 1 was accepted with
no source errors (kernelId 138112199). Its code dataset is
`ehimennathan/vesuvius-restored-forward-code-20261011`.

It preserves the same eight segments, two checkpoints, scoring masks, full
checkpoint settings, TTA 4, normalization seeds and baseline reproduction
gate. It scores forward depth only, halving the 32 full segment passes to 16.
The kernel logs `GPU_EXECUTION_CONFIRMED` after CUDA setup and progress every
25 inference batches. No new model training is required. Acceptance and the
generic RUNNING API response do not independently prove active GPU execution.
The old session remains unresolved; its session ID was unavailable and it
was not cancelled using an unverified identifier. No new AUC yet.

## 2026-10-11: restored training complete; evaluation submitted

The bounded dispatcher observed source-restoration training COMPLETE with no
failure message. Separate private evaluation version 1 was accepted as
`ehimennathan/vesuvius-restored-evaluation-20261011` (kernelId 138075050), with
no invalid sources. Subsequent live status requests timed out; evaluation
progress/completion and restored AUC are not yet verified. Dispatcher evidence
is in `restored_evaluation_dispatch.log`. The latest audited pilot AUC remains
0.591795 mixed versus 0.589991 restricted baseline.

Existing visual example: `kout/cpu_audit_20261010/b0/pred_H2Hfix_B0p256fd_teacher_phercman5-outer3.png`.
The generator concatenates forward-depth model probabilities on the left and
annotated ink within the supervision mask on the right. This is an earlier
B0 seed-42 checkpoint, not the newly restored model. MAN5 remains a weak
example; clear reference strokes must not be presented as model discoveries.

## 2026-10-11: CPU diagnosis complete; source-restoration control running

CPU job `ehimennathan/vesuvius-pilot-cpu-diagnosis-20261011` completed. The
restricted pilot retained 4.39% of historical 1667 store-level training-pixel
coverage while retaining 41.51% source sampling probability. Ten training
stores were omitted, including five native 0139 views that overlap aligned
surfaces. Historical recipes also differed in budget, seed and depth jitter;
coverage is a candidate cause, not an established causal result.

Private job `ehimennathan/vesuvius-source-restoration-3k-20261011` version 1
was accepted and verified RUNNING without a reported failure. It restores
the omitted inventory, keeping the pilot training hyperparameters and the
same canonical original w029. Training and evaluation are separate to limit
memory. Primary evaluation remains forward EMA, unsmoothed, on the same
eight Paris4 segments. No new AUC yet; no 30,000-step run launched.
See `PILOT_DIAGNOSIS_20261011.md` and `source_restoration_protocol.json`.

## PRIOR WORK and CORRECTIONS (2026-10-09), which must be reflected in any submission
- **Prior work:** AndreasHad04 (ScrollPrize/villa issue #1867, 2026-09-22; repo villa-apple-silicon) already published
  a held-out ink_9um evaluation on PHerc0841 ×3 + 500p2a, the depth-direction dependence (reverse on 0841, renders
  prefer reverse), a label-free direction rule (57/70 and 84/84 cases), window-shift sweeps and depth-shuffle analyses.
  pscamillo (ink9-depth-window, Aug 2026) studied the depth window size and z-jitter. Our "depth-order trap" and
  "held-out benchmark" are therefore NOT novel; credit them. Not found reported anywhere (villa issues searched): the
  **truncated level-1/2 depth pyramids** in the ink-label bucket (18/33 volumes).
- **Scale error in our test prep:** the PHerc0841 bucket renders are 4.681 µm (`volume: 4.681um_113keV_1.2m_binmean_2`),
  not 2.4 µm. Our 4×4×4 pooling therefore produced ~18.7 µm inputs for 0841 (2× too coarse). 0009B (~2.4 µm),
  0500P2 (~2.2 µm) and the 1667 controls (2.399 µm) are correct. Every model was compared on the identical inputs, so
  the relative ranking stands, but the absolute 0841 numbers are invalid until 0841 is rebuilt with 2×2×2 pooling. The
  pooling factor must be chosen from each render's source voxel size, not assumed.
- **Corrected 0841 (2×2×2 pooled → 9.362 µm, 33 slices, `vesuvius-ink-prep3b`):** V3 + DINOv3 ViT-S (rev)
  AUC **0.729**, AP 0.532, F0.5 0.546 (per segment ag4252405 / ag4552896 / w00 in the fwd table; rev pooled shown).
  On the too-coarse (18.7 µm) inputs it had scored 0.845, so absolute 0841 scores drop at the correct scale.
  **ink_9um on the same pixels (rev): seed42 0.703 / seed43 0.720 (mean 0.712), AP 0.519 / 0.524, F0.5 0.546 / 0.535**,
  which matches Andreas's independent 0.69–0.75, so our pipeline is validated. **At the correct scale, our 0841
  advantage is marginal (+0.017 AUC vs ink_9um's mean, F0.5 tied).** The earlier large 0841 gaps were inflated by
  the mis-scaled inputs. Correct-scale sources where we lead more clearly: the IR-labelled fragments 0009B and 0500P2
  (scaled correctly from the start). MAN5: ink_9um better. **B0-256 seed 43 on corrected 0841 (rev): AUC 0.753, AP 0.551, F0.5 0.550**, the best on
  0841 (+0.033 to +0.050 over the ink_9um seeds).

### CORRECT-SCALE never-seen summary (rev AUC; 0841 from prep3b, the others from prep3)
| model | 0841 | 0009B | 0500P2 | MAN5 |
|---|---|---|---|---|
| ours B0-256 s43 | **0.753** | **0.901** | 0.875 | 0.533 |
| ours B0-256 s42 | 0.730 | 0.885 | **0.887** | 0.538 |
| ours V3 + DINOv3 ViT-S | 0.729 | 0.875 | 0.870 | 0.504 |
| ink_9um s42 | 0.703 | 0.868 | 0.831 | **0.617** |
| ink_9um s43 | 0.720 | 0.868 | 0.854 | 0.552 |
At the correct scale, B0-256 is our strongest model on unseen data. DINOv3's Paris4 edge did not transfer to the
never-seen sources. Every one of our models beats both ink_9um checkpoints on both IR-labelled fragments; MAN5 remains
an open failure.

All inputs are at ~9.6 µm. Training data is the ink_9um labels from the same four scrolls ink_9um used (0139, 1667,
Paris4, 0814). Labels are pooled to the 21-slice "aligned" representation; the model sees 17 slices with ±2 z-jitter.
Official validation-mask pixels are never trained on.

## Protocol
- **Tuning folds** (leave one scroll out): Paris4 held out (8 segments, 82 cm² supervised), then 1667 held out. Metrics
  are computed on every supervised pixel of the held-out scroll: AUC, AP, best F0.5, F0.5 at 0.5. The forward depth
  direction is primary.
- **Never-seen confirmation set:** scrolls 0841 (3 segments) and MAN5, plus fragments 0009B and 0500P2, about 8 cm². These
  come from the scrollprize `ink` bucket and were never in ink_9um's training data, which gives a fair head-to-head
  against the released checkpoints. Used for confirmation only, not for tuning.

## Fold Paris4 (forward AUC; mean over the 7 segments shared by all runs, w00–w07)
| run | model | patch | loss | 7-seg mean AUC | pooled AUC | pooled F0.5 | notes |
|---|---|---|---|---|---|---|---|
| B0_lo-Paris4 | B0 (ink_9um-like 3D stem + U-Net) | 128 | BCE+Dice | 0.651 | 0.663 | 0.447 | baseline recipe |
| V1_lo-Paris4 | 3D stem s2 + ResNet34 FPN, 1/8 output | 256 | BCE+Dice | — | 0.628 | 0.401 | low-res output hurts at 9.6 µm |
| V2r34_lo-Paris4 | 3D stem + ResNet34 full-res U-Net | 256 | BCE+Dice | 0.677 | (OOM at eval) | — | |
| **B0p256_lo-Paris4** | B0 | **256** | BCE+Dice | **0.693** | (OOM at eval) | — | wins 7/7 segments vs B0-128 |
| V3dinov3cnxT_lo-Paris4 | V3 hybrid + DINOv3 ConvNeXt-T | 256 | BCE+Dice | 0.657 | 0.669 | 0.451 | |
| B0p512_lo-Paris4 | B0 | 512 | BCE+Dice | 0.683 | **0.698** | **0.505** | w09 0.816; 1.5× slower than 256 |
| B0p256fd_lo-Paris4 | B0 | 256 | Focal+Dice | 0.690 | **0.703** | 0.502 | neutral vs BCE+Dice on ranking; well calibrated (F0.5@0.5 = 0.501) |

| B0p256s43_lo-Paris4 | B0 (seed 43) | 256 | BCE+Dice | 0.677 | 0.689 | 0.487 | seed noise ≈ ±0.016 on the 7-segment mean |
| V3swinT_lo-Paris4 | V3 hybrid + Swin-T (in22k) | 256 | BCE+Dice | 0.664 | 0.683 | 0.446 | transformer backbone: no gain |
| V4r21d_lo-Paris4 | V4 R(2+1)D-18 video encoder | 256 | — | — | — | — | OOM on a T4 even at batch 8; 2.05 s/it; dropped |
| V5cnxT_lo-Paris4 | V5 2.5D slice groups + ConvNeXt-T (2nd-place style) | 256 | BCE+Dice | 0.596 | 0.603 | 0.369 | clearly worse |

| **V3dinov3vitS_lo-Paris4** | **V3 hybrid + DINOv3 ViT-S/16 (self-supervised)** | 256 | BCE+Dice | **0.733** | **0.742** | **0.514** | +0.04 to +0.056 over B0-256; wins 14/15 segment comparisons |

| V3dinov3vitS_s43_lo-Paris4 | same, seed 43 (replication) | 256 | BCE+Dice | 0.691 | 0.702 | 0.464 | **seed spread ±0.02; 2-seed mean 0.712 (7-seg) vs B0-256 0.685 → +0.03, not +0.05** |

**Correction (2026-10-09):** the first DINOv3 ViT-S seed (0.742) was a favourable draw. Over two seeds DINOv3 is about
+0.03 better than B0-256 on Paris4, with mixed per-segment wins against B0 seed 42.

**Head-to-head, V3 + DINOv3 ViT-S (12k it), never-seen corrected data, rev:** AUC **0.798**, AP **0.666**, F0.5
**0.659** (F0.5@0.5 0.657); 0009B 0.875, 0500P2 0.870, 0841 **0.845**, MAN5 0.504. That is the best single model, at
+0.033 AUC / +0.06 AP / +0.04 F0.5 over ink_9um's seed mean (0.765 / 0.606 / 0.618).

**Backbone conclusion:** CNN, Swin, Kinetics-video and 2.5D backbones do NOT beat the task-specific B0. The
**self-supervised DINOv3 ViT-S**, inside the V3 hybrid (full-resolution 3D stem + stride-2/4 skips + U-Net decoder),
DOES beat B0, by about 3× the seed noise. Next: head-to-head against ink_9um, plus seed replication.

**Replicated finding:** 256 px beats 128 px on 14/14 segment comparisons (2 seeds × 7 segments), +0.026 to +0.042 AUC.
256 vs 512 px and BCE vs Focal+Dice are within seed noise.

The reverse depth direction scores AUC ≈ 0.53 on every Paris4 run: the depth convention is consistent within the
training corpus.

## Released ink_9um on the never-seen set (pooled AUC)
| checkpoint | fwd | rev | best per source |
|---|---|---|---|
| seed42 step75k | 0.534 | 0.562 | 0500P2 rev 0.680 |
| seed43 step75k | 0.505 | 0.569 | 0841 rev 0.611 |

### Control (2026-10-08): the same pipeline on ink-bucket segments ink_9um WAS trained on (identical 17-slice L2 format)
| segment | fwd AUC s42 / s43 | rev AUC s42 / s43 |
|---|---|---|
| 0139 w043 | 0.50 / 0.57 | **0.92 / 0.94** |
| 1667 w028 | 0.54 / 0.47 | 0.54 / 0.54 |
| 1667 w029 | 0.46 / 0.46 | 0.56 / 0.54 |

- **Confirmed:** the ink-bucket renders store depth in the opposite order to ink_9um's training representation. The
  inference default (forward) drops AUC from 0.93 to 0.53 on a TRAINING segment.
- **Withdrawn:** "ink_9um is near chance on never-seen sources". The same format gives near-chance results on 1667, a
  training scroll, so the low scores are at least partly a representation mismatch (suspected depth offset or spacing;
  under investigation with `vesuvius-ink9um-ctrl-aligned`, which runs ink_9um on the aligned representation of the same
  segments).
- **Withdrawn:** "9× activity on sheet switches" (PHerc1203). That figure came from an orientation-dependent mask; with
  the rotation-invariant mask the effect is not consistent.

### ROOT CAUSE (2026-10-08): truncated depth in the published ink-label OME-Zarr pyramids
- Depth-intensity profiles (`vesuvius-depth-profiles`): aligned training renders show a papyrus sheet crossing
  (e.g. 72→93→63) with ink contrast up to +10 at the centre. Ink-bucket L2 for 1667 w029, 0841 w00, 0009B and 0500P2 is
  flat (e.g. 65→62) with ink contrast of only about 1–2. 0139 w043 (different chunking, 64×256×256) is fine; MAN5 shows a
  sheet.
- Direct check on 841/w00: **level-2 slice k equals level-0 slice k downsampled in xy (correlation 1.00 for k = 0..16)**,
  against ≤0.06 for depth-pooled slices. The `.zattrs` declare z-scale 1 at level 2. So level 2 holds only the first 17
  of 65 slices (41 µm at one end of a 156 µm stack); depth is truncated, not downsampled. The ink layers (cf. the
  bucket's own `max_22_42` files) are not in it.
- Consequence: any model fed level 2 (the ~9.6 µm level the 9 µm models expect) sees no papyrus. That explains the
  near-chance scores of BOTH ink_9um and our models on these sources, and on the 1667 control.
- Fix: rebuild the 9.6 µm representation from level 0 with 4×4×4 mean pooling (`train/prep3_kernel.py`), giving 17
  slices that span the full stack. On the same 841 tile the corrected depth profile shows structure (92→72) where level 2
  was flat (92→88).
- Survey of all bucket segments (`train/survey_pyramids.py` → `kout/pyramid_survey.json`): **18 of 33 segment volumes
  are truncated.** Level 1 keeps 33/65 slices and level 2 keeps 17/65 slices, while declaring z-scale 1. Affected:
  0009B; 0139 w016, w017, w028, w029, w030; 0500P2; 1667 w013, w018, w023, w028, w029, w031; 814; 841 (all three);
  unused/500p2a. Correct: 0139 w035–w044 and MAN5 (depth pooled, verified by data for w043 and MAN5), and Paris4 (all
  eight; level 2 keeps 65 slices, consistent with its metadata).

### Pipeline validation (aligned representation, ink_9um on segments it was trained on)
0139 w043 fwd 0.994 / 0.991; 1667 w028 fwd 0.913 / 0.937; 1667 w029 fwd 0.841 / 0.877 (seed42 / seed43). Reverse is
about 0.5. The evaluation pipeline is sound; the 1667 failure in bucket format came from the data.

### FIX VERIFIED end to end (released ink_9um on 1667 bucket segments it was trained on)
| segment | published truncated L2 (best dir) | **corrected L0→4×4×4 pooled, rev** (s42 / s43) | fwd (s42 / s43) |
|---|---|---|---|
| 1667 w028 | 0.54 | **0.926 / 0.956** | 0.500 / 0.543 |
| 1667 w029 | 0.56 | **0.897 / 0.912** | 0.485 / 0.561 |
| pooled | ~0.54 | **0.911 / 0.934** | 0.494 / 0.552 |
Correct depth pooling restores ink_9um fully (matching or exceeding its own aligned format, 0.84–0.94), and the
bucket's depth order is confirmed reversed.

### Head-to-head on CORRECTED never-seen data (L0 → 4×4×4 pooled, `vesuvius-ink-prep3`)
| source | ours B0-256 (trained on 0139/1667/Paris4/814 only), on truncated L2, rev | **ours, corrected data, rev** | ours, corrected, fwd |
|---|---|---|---|
| 0009B (fragment, IR labels) | 0.493 | **0.885** | 0.512 |
| 0500P2 (fragment, IR labels) | 0.646 | **0.885** | 0.552 |
| 0841 (3 segments) | 0.589 | **0.822** | 0.509 |
| MAN5 | 0.503 | 0.538 | 0.545 |
| pooled | 0.542 | **0.771** (F0.5 0.640) | 0.527 |
Depth-flip (orientation-invariant, zflip 0.5) model on corrected data: pooled fwd 0.719 / rev 0.725; 0009B
0.827/0.818, 0500P2 0.802/0.804, 0841 0.788/0.793, MAN5 0.470/0.493. Invariance costs about 0.05 AUC against the
correctly oriented model, but gains about 0.19 against it read the wrong way.

### HEAD-TO-HEAD, corrected never-seen data, identical pixels (`vesuvius-ink-h2h-score`), reverse = correct order
| model | AUC | AP | F0.5 | 0009B | 0500P2 | 0841 | MAN5 |
|---|---|---|---|---|---|---|---|
| ours B0-256 (1 seed, 12k it, 5.5M params) | **0.771** | **0.638** | **0.640** | **0.885** | **0.887** | **0.822** | 0.538 |
| ink_9um s42 (75k it) | 0.769 | 0.605 | 0.610 | 0.868 | 0.831 | 0.800 | **0.617** |
| ink_9um s43 | 0.761 | 0.606 | 0.626 | 0.868 | 0.854 | 0.799 | 0.552 |
| ink_9um s42+s43 | 0.779 | 0.638 | 0.650 | 0.883 | 0.866 | 0.811 | 0.620 |
| ours + ink_9um s42 + s43 | **0.785** | **0.657** | **0.661** | 0.895 | 0.887 | 0.827 | 0.594 |
| ours depth-flip (order-agnostic) | 0.725 | 0.566 | 0.555 | 0.818 | 0.806 | 0.792 | 0.492 |
| ink_9um, default forward order | 0.515–0.533 | ~0.28 | ~0.31 | | | | |
**Second seed (2026-10-09):** ours B0-256 seed 43 (rev): AUC **0.792**, AP 0.652, F0.5 0.633; 0009B 0.901,
0500P2 0.875, 0841 0.831, MAN5 0.533. **Seed means: ours 0.782 AUC / 0.645 AP / 0.637 F0.5 vs ink_9um 0.765 / 0.606 /
0.618.** Both of our seeds beat both ink_9um seeds on AUC and AP. The AUC margin (+0.017) is about the seed noise;
the AP margin (+0.04) is clearer. ink_9um still wins MAN5.

Reading (seed 42 only): single model vs single model, ours is at least on par on AUC (+0.002 / +0.010), ahead on AP (+0.03) and F0.5
(+0.01 to +0.03), and higher on 3 of 4 sources (both IR-labelled fragments and 0841), but lower on MAN5. The AUC gap
is within our seed noise (±0.016), so "surpassed" is NOT established. Ours with ink_9um gives the best result (0.785),
i.e. the models are complementary. When the orientation is unknown, the order-agnostic model beats ink_9um's default by
+0.19 AUC.

The never-seen sources use the opposite depth orientation, so a deployable model must handle orientation (see the
depth-flip augmentation experiment).

### Codex CPU audit (2026-10-10)

See `CPU_AUDIT_20261010.md` for evidence and limitations. Private CPU job
`ehimennathan/vesuvius-cpu-audit-20261010` completed; no GPU was used.

### Matched-format pilot launch, 2026-10-10

Verified result, 2026-10-11: evaluation v2 COMPLETE. Forward EMA equal-segment
Paris4 AUC: baseline 0.5899907358; mixed 0.5917952283; delta +0.0018044926.
Five/eight segments improved, three regressed. Mean AP 0.3620344870 vs
0.3610409427. Per-segment scoring populations match across arms; total
supervised pixels 88,625,409. This is a single-seed, one-paired-surface pilot;
no robust generalization claim or 30,000-step promotion. Details in
`MATCHED_PILOT_RESULTS_20261011.md` and `matched_pilot_results.json`.

Evaluation version 1 failed before scoring with missing default
`eval_stride_frac`. Version 2 uses the full saved checkpoint configurations,
audited against actual training logs (stride 0.5, all declared overrides match).
Private code dataset updated; evaluation version 2 accepted without source
errors. Both trained checkpoints remain intact. AUC pending; no retraining.

Training-only completion update: job COMPLETE, both arms finished 3,000 steps,
identical target-stream digest, both EMA checkpoints and result JSONs present.
Separate private evaluation job `ehimennathan/vesuvius-matched-evaluation-20261010`
version 1 accepted without source errors. The full primary Paris4 comparison
remains pending; no 30,000-step run has started.

Latest status: private-code-dataset upload succeeded using the explicitly
supplied main-account credentials. Job
`ehimennathan/vesuvius-matched-training-only-20261010` version 1 was accepted
with no invalid sources and reported RUNNING. The 482-byte launcher verifies
the mounted code SHA256. Training completion,
new target-stream validation and full primary AUC remain pending.

Recovery update: the original GPU preflight passed and both arms finished 3,000
steps with identical target-stream digest
`da10481a1ad0f877fe7713d970726591e974fefd1932350b5f4a60af8523ee98`.
Both exited with signal 9 during concurrent evaluation; host-memory exhaustion
is the working explanation, not a confirmed kernel OOM report. The full primary
comparison is unavailable. Two evaluation-only attempts failed to find the
original checkpoints in the failed job's mounted inputs. Direct later output
pages timed out. Default concurrent evaluation is now disabled. A training-only
launch commits checkpoints before a separate evaluator loads one segment and
one model at a time. Upload attempts experienced connection timeouts/resets;
successful launch must be verified rather than inferred. See
`MATCHED_PILOT_20261010.md` for the repaired protocol. No 30,000-step run started.

CPU registration and matched builder passed for 1667-w029; w028 was rejected.
The canonical pair has 1,198,292 shared training pixels and verified reversed
corrected depth against original offset 2. Private job
`ehimennathan/vesuvius-matched-format-pilot-20261010` version 1 was accepted
and reported RUNNING. Two 3,000-step B0 arms share seed, targets, source weights
and crops; actual target-stream hashes must match. Forward-depth EMA Paris4
equal-segment AUC is predeclared primary. Results and GPU preflight completion
are pending. See `MATCHED_PILOT_20261010.md` and
`matched_pilot_protocol.json`; no 30,000-step run has been launched.

- MAN5 shifts, fixed 150k diagnostic pixels: B0 forward 0.545794 to exploratory
  best 0.560061; reverse 0.5408 to 0.549100. Boundary optima; no established
  registration correction. No shifts applied to benchmark results.
- Three MAN5 intensity profiles show structure in the 17-slice pooled stack.
  Physical input spacing remains unresolved; assigned builder spacing is not
  independent evidence.
- Paris4 seed43 held-out fold, eight recorded segments: sigma=1 gives mean AUC
  0.691939 vs 0.691813 unsmoothed. Freeze sigma=1 as secondary; unsmoothed primary.
- Fresh label-free 0841 pixel comparison: all 17 L2 indices match XY-pooled L0
  slices at the same index, r=0.999885–0.999918. One region directly verified;
  18/33 is the metadata-survey count, not 18 independent pixel proofs.
- Recovered existing 1667 controls: seed42 reverse AUC 0.546749 to 0.911220;
  seed43 0.536072 to 0.934086. Training-source controls, not generalization.
- Launcher JSON handling, narrow inference coverage, duplicate-source ambiguity,
  and fail-soft source-download behavior repaired. New audited ensemble scorer
  freezes the scoring population and records provenance. CPU regression checks
  passed; modified neural inference has not been run on GPU.
- Prior pooled scores above include superseded 0841 spacing; do not use them for
  current superiority claims. Review corrected per-source results and recompute
  the complete benchmark with explicit input provenance.
