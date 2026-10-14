"""Surface-quality mask from an aligned render layer: papyrus = local gradient energy concentrated near 0/90 deg
(crossing fibres); sheet-switch regions (wood-grain swirls) have spread orientations.
Usage: papyrus_mask.py layer.png out_mask.png [--sigma 6] [--thr 0.55]   (works at any scale)"""
import argparse, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
ap = argparse.ArgumentParser(); ap.add_argument("layer"); ap.add_argument("out"); ap.add_argument("--sigma", type=float, default=6)
ap.add_argument("--thr", type=float, default=0.55); a = ap.parse_args()
im = np.asarray(Image.open(a.layer).convert("L"), np.float32) / 255
valid = gaussian_filter((im > 0).astype(np.float32), 3) > 0.99
gy, gx = sobel(im, 0), sobel(im, 1)
th = np.arctan2(gy, gx); e = gx ** 2 + gy ** 2
# 4-fold symmetric orientation: cos(4*theta) = +1 at 0/90/180/270 deg
# rotation-invariant: coherence of the 4-fold orientation field (crossing fibres at any angle)
c4 = gaussian_filter(e * np.cos(4 * th), a.sigma * 3); s4 = gaussian_filter(e * np.sin(4 * th), a.sigma * 3)
den = gaussian_filter(e, a.sigma * 3) + 1e-9
score = np.hypot(c4, s4) / den        # in [0, 1]
angle = np.rad2deg(np.arctan2(s4, c4) / 4)   # dominant fibre angle (deg, mod 90)
mask = (score > a.thr * score[valid].max()) & valid
Image.fromarray((mask * 255).astype(np.uint8)).save(a.out)
Image.fromarray((np.clip(score, 0, 1) * 255 * valid).astype(np.uint8)).save(a.out.replace(".png", "_score.png"))
print("fibre angle median %.1f deg" % np.median(angle[mask]) if mask.any() else "no papyrus")
print("papyrus fraction %.3f  score max %.3f  median %.3f" % (mask[valid].mean(), score[valid].max(), np.median(score[valid])))
