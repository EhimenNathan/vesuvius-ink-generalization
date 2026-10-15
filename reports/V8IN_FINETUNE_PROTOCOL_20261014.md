# v8-in fine-tune protocol: frozen 2026-10-14, before any fine-tuned model exists

Goal: improve the organizers' current best model (`YoussefMoNader/ink-8um-v8in`, epoch 6, rev d89166b) on scrolls that
neither it nor our data has seen, by fine-tuning it on our corrected training inventory, which it mostly lacks
(v8-in: 1 segment of 1667, 4 of 0139, 2 of Scroll 1; ours: 6, 14 and 8, about 119 cm² at 9.6 µm, rebuilt from L0 with
correct physical pooling). This is "v8-in + our data", credited as such; it is not a new architecture.

## Training (kernel `v8in_finetune_kernel.py`)
- Data: `vesuvius-ink-prep` stores (0139, 1667, Paris4, 0814 only). Supervised pixels `sup & ~val`. No 0841, MAN5,
  PHerc1447, 0009B or 0500P2 pixel is loaded.
- Representation: 9.6 µm / 21 slices resampled on the fly to v8-in's 8.64 µm / 24 layers. The depth order fed to
  the model is fixed at start by scoring base v8-in on 512 training tiles in both orders (training labels only).
- Recipe = the released v8-in recipe: 0.5 Dice + 0.5 BCE (smoothing 0.25), AdamW, cosine LR from 1e-5 to 1e-6 (half the
  original peak, because this is a fine-tune), weight decay 1e-6, AMP, gradient clip 1, flips, 360° rotation with
  ±10% scale and ±15% shift, blur (p 0.4), coarse dropout (p 0.5), fourth augment (p 0.6). Source sampling uses the proven `train_v2.py` scroll weights (0139 29 : 1667 22 : Paris4 11 : 0814 2); amended before any training, because sqrt(area) would give Scroll 1, which v8-in already knows best, half the batches.
- Checkpoint selection: AUC on the official validation-mask pixels of the training stores (never trained on),
  evaluated at step 0 (base model) and periodically. If the stores hold fewer than 100,000 validation-mask pixels, `phercparis4-w09` and `pherc1667-w031` are held out entirely as validation stores instead (declared here, before any training). The selected checkpoint is the best one on those pixels; if no
  checkpoint beats step 0 there, the fine-tune is reported as a failure and not evaluated further.

## Test (one inference pipeline for both models: `v8in_eval_kernel.py`, same pixels, same direction rule)
- Sources: 0841 (3 segments), MAN5, PHerc1447 w058 + w060. Primary: macro AUC over the three sources. Also per source,
  pooled AUC, AP.
- Claim requires the fine-tune to beat base v8-in on the macro and on at least 2 of 3 sources; otherwise reported as a
  null/negative result. 0009B and 0500P2 are reported separately as v8-in-exposed sources.
