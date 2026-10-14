# vc-multiscan

Render existing Vesuvius Challenge segments from a **different, higher-resolution scan** of the same scroll, and flag
where a segment actually lies on papyrus.

Several scrolls now have a full-scroll scan at ~9 µm, which is where segments (`tifxyz`) are traced, plus a partial
scan at ~2.4 µm. Ink models such as `scrollprize/ink_9um` are mostly trained on 2.4 µm data pooled to ~9.6 µm. This
tool lets you render a 9 µm-traced segment straight from the 2.4 µm scan without re-segmenting.

| Tool | What it does |
|---|---|
| `register_scans.py` | Coarse-to-fine similarity registration (rotation about the scroll axis, optional mirror or z-flip, xy/z scale, translation) between two OME-Zarr scans. Streams only the coarse pyramid levels. Writes `reg.json` (`p_fixed = M @ p_moving + t`) and overlay PNGs. |
| `render_tifxyz.py` | Streaming tifxyz → surface-volume renderer that reads only the 128³ chunks the surface touches. Options: `--affine reg.json` (sample another scan), `--level` (pyramid level, e.g. 2 ≈ 9.6 µm from a 2.4 µm scan), `--offset` (normal shift in slices), `--align` (rotate the output so the scroll axis points up), `--crop`. Writes OME-Zarr, so it works with `vesuvius.ink_detection` / `koine_machines` inference. |
| `papyrus_mask.py` | Surface-quality mask from a rendered layer. Real papyrus shows crossing fibres (strong 4-fold orientation coherence at any angle); sheet switches show wood-grain swirls. Use it to gate ink predictions. |

## Validation

| Check | Result |
|---|---|
| `render_tifxyz.py` vs the official `vc_render_tifxyz` surface volume (PHerc0800 segment `20251029010146`, same mesh and scan) | Pearson r = **0.975** at zero layer shift (mean over 21 layers) |
| `register_scans.py` PHerc1203 2.403 µm → 9.362 µm | rotation 0.02°, no mirror, scale 1.0266 (nominal ratio 1.0267); NCC **0.927** at ~150 µm, **0.905** at ~75 µm; sheets overlap visually (`docs/PHerc1203_overlay_*.png`) |
| Same segment patch rendered from both scans (PHerc1203 `auto_grown_20250925223153537`) | r = **0.91** after a 1.5-slice normal offset (`--offset 1.5`); the 2.4 µm render is visibly sharper (`docs/PHerc1203_render_9um_vs_2.4um.png`) |
| `papyrus_mask.py` on public PHerc1203 auto-grown segments | only 4–49% of each segment is clean papyrus. `ink_9um` high-confidence pixels concentrate outside the mask, i.e. on sheet switches (e.g. 0.26 outside vs 0.03 inside on `221856743`), so unmasked "activity" is a false-positive trap. |

## Usage

```bash
# 1. register the partial high-res scan to the full-scroll scan (CPU, ~1 h, streams ~1 GB)
python register_scans.py \
  --fixed  PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr \
  --moving PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr --out reg_1203

# 2. render a 9.362 um-traced segment from the 2.4 um scan at level 2 (~9.6 um, ink_9um's training representation)
python render_tifxyz.py <segment.tifxyz dir> PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr out.zarr \
  --affine reg_1203/reg.json --level 2 --offset 1.5 --align

# 3. run ink detection on out.zarr, then gate it with the papyrus mask of a rendered layer
python papyrus_mask.py layer.png papyrus.png
```

Registration for PHerc1203 is included in `examples/`. The 2.4 µm scan covers 9.362 µm slices z ≈ 7939–11824.

## Limitations
- Global similarity transform. Residual normal-direction offsets of ~1–2 slices remain, so sweep `--offset` or the
  model's z window.
- `--align` uses the mean z-gradient of the parametrization. Strongly distorted segments may stay rotated; check the
  rendered layer.
- `papyrus_mask.py` thresholds are relative to each segment's maximum coherence. Wavy-fibre papyrus can be under-detected.

## License
MIT
