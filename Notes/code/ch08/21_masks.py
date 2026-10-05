"""21_masks.py -- the controlled masks of the cut-sky study, their sky fractions and spectra.

Question: what do our masks look like, how much sky does each keep, and how is the mask's own
power spread over multipoles (the quantity W_l that will set the mode coupling)?
Computes: eight masks at nside 256 (full sky; caps keeping 10, 30 and 70 per cent; a cosine-
tapered version of the 10 per cent cap; a band cut |b| < 20 deg, binary and tapered; 400
point-source holes of radius 1 deg), their moments fsky and w_i, and W_l up to l = 1535 for
W and for W^2 (both are needed later).  Checks two exact identities:
    W_0 = 4 pi fsky^2 w1^2      and      sum_l (2l+1) W_l = 4 pi fsky w2   (Parseval),
the second summed to l = 767 = 3 nside - 1 (beyond that the pixel quadrature aliases).
Writes: data/ch08/masks.npz, figures/ch08/masks.pdf, figures/ch08/mask_spectra.pdf,
        results/ch08/21_masks.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES
import lib_masks as lm

setup()
NSIDE, LW = 256, 1535                    # W_l is needed up to l + l' = 2 x 767
OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08" / "masks.npz"

rng = rng_for("ch08", "21_masks")
centres = lm.hole_centres(400, rng)
MASKS = {
    "full":      np.ones(hp.nside2npix(NSIDE)),
    "cap10":     lm.cap(NSIDE, 0.10),
    "cap10apo":  lm.cap(NSIDE, 0.10, apo_deg=10.0),
    "cap30":     lm.cap(NSIDE, 0.30),
    "cap70":     lm.cap(NSIDE, 0.70),
    "band20":    lm.band(NSIDE, 20.0),
    "band20apo": lm.band(NSIDE, 20.0, apo_deg=5.0),
    "holes":     lm.holes(NSIDE, centres, 1.0),
}

if OUT.exists():
    z = np.load(OUT)
    WL = {k: z["wl_" + k] for k in MASKS}
    WL2 = {k: z["wl2_" + k] for k in MASKS}
else:
    WL = {k: lm.window_spectrum(w, LW) for k, w in MASKS.items()}
    WL2 = {k: lm.window_spectrum(w ** 2, LW) for k, w in MASKS.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, nside=NSIDE, centres=centres,
             **{"mask_" + k: w.astype(np.float32) for k, w in MASKS.items()},
             **{"wl_" + k: v for k, v in WL.items()}, **{"wl2_" + k: v for k, v in WL2.items()})
    print(f"[cache] wrote {OUT}")

# ---------------------------------------------------------------- moments and identities
nums = {}
NAMES = {"full": "Full", "cap10": "CapTen", "cap10apo": "CapTenApo", "cap30": "CapThirty",
         "cap70": "CapSeventy", "band20": "Band", "band20apo": "BandApo", "holes": "Holes"}
ell = np.arange(LW + 1)
for k, w in MASKS.items():
    fsky, wi = lm.mask_moments(w)
    w0_err = WL[k][0] / (4 * np.pi * fsky ** 2 * wi[1] ** 2) - 1
    # Parseval up to 3 nside - 1, the band the pixel quadrature resolves
    pars = np.sum(((2 * ell + 1) * WL[k])[:768]) / (4 * np.pi * fsky * wi[2])
    print(f"{k:10s} fsky={fsky:.4f} w1={wi[1]:.3f} w2={wi[2]:.3f} w4={wi[4]:.3f} "
          f"w2^2/w4={wi[2]**2/wi[4]:.3f}  W0 check {w0_err:+.1e}  Parseval(l<=767) {pars:.5f}")
    n = NAMES[k]
    nums[f"EightBfsky{n}"] = f"{fsky:.3f}"
    nums[f"EightBwtwo{n}"] = f"{wi[2]:.3f}"
    nums[f"EightBwfour{n}"] = f"{wi[4]:.3f}"
    nums[f"EightBweff{n}"] = f"{wi[2]**2/wi[4]:.3f}"
    nums[f"EightBfeff{n}"] = f"{fsky * wi[2]**2 / wi[4]:.3f}"
    nums[f"EightBParseval{n}"] = f"{pars:.4f}"
    nums[f"EightBWzero{n}"] = float(abs(w0_err))

# analytic sky fractions of the geometric masks
nums["EightBfskyBandExact"] = f"{1 - np.sin(20 * lm.DEG):.4f}"
nums["EightBfskyBandPix"] = f"{np.mean(MASKS['band20'] > 0):.4f}"
nums["EightBcapTenDeg"] = f"{np.degrees(np.arccos(1 - 2 * 0.10)):.2f}"
nums["EightBcapThirtyDeg"] = f"{np.degrees(np.arccos(1 - 2 * 0.30)):.2f}"
nums["EightBcapSeventyDeg"] = f"{np.degrees(np.arccos(1 - 2 * 0.70)):.2f}"
nums["EightBholesNaive"] = f"{400 * 2 * np.pi * (1 - np.cos(lm.DEG)) / (4 * np.pi):.4f}"
nums["EightBholesLost"] = f"{1 - np.mean(MASKS['holes'] > 0):.4f}"
nums["EightBnside"] = NSIDE
save_numbers("ch08", "21_masks", nums)

# ---------------------------------------------------------------- figure 1: the masks
show = [("cap10", "(a) cap, $f_{\\rm sky}=0.1$"), ("cap10apo", "(b) same cap, tapered over $10^\\circ$"),
        ("cap30", "(c) cap, $f_{\\rm sky}=0.3$"), ("band20", "(d) band cut $|b|<20^\\circ$"),
        ("band20apo", "(e) band cut, tapered over $5^\\circ$"), ("holes", "(f) 400 holes of radius $1^\\circ$")]
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("mask", ["#3a3936", "#9fc3ea"])   # 0 = cut (dark), 1 = kept
cmap.set_bad("white")                                                    # outside the ellipse


def moll(m):
    """Mollweide image of a map (NaN outside the ellipse), for imshow."""
    proj = hp.projector.MollweideProj(xsize=600)
    img = np.asarray(proj.projmap(m, lambda x, y, z: hp.vec2pix(NSIDE, x, y, z)), dtype=float)
    img[~np.isfinite(img) | (img < -1e20)] = np.nan
    return img


fig, axs = plt.subplots(2, 3, figsize=(8.0, 3.4))
for ax, (k, t) in zip(axs.ravel(), show):
    ax.imshow(moll(MASKS[k]), cmap=cmap, vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ax.set_title(t, fontsize=9)
    ax.axis("off")
savefig(fig, "ch08", "masks")

# ---------------------------------------------------------------- figure 2: mask spectra
# The band masks are north-south symmetric, so their odd multipoles vanish, and every binary
# edge rings; we therefore plot the average of W_l over logarithmic bins of multipoles.
fig, ax = plt.subplots(figsize=(6.4, 3.6))
edges = np.unique(np.r_[np.arange(2, 16, 2), np.round(np.geomspace(16, 768, 26)).astype(int)])
lc = np.sqrt(edges[:-1] * edges[1:])
for (k, lab), c in zip([("cap10", "cap 0.1"), ("cap10apo", "cap 0.1 tapered"), ("band20", "band"),
                        ("band20apo", "band tapered"), ("holes", "holes")],
                       [SERIES[0], SERIES[1], SERIES[2], SERIES[3], SERIES[4]]):
    y = np.array([WL[k][a:b].mean() for a, b in zip(edges[:-1], edges[1:])]) / WL[k][0]
    ax.loglog(lc, y, color=c, lw=1.4, marker="o", ms=2.5, label=lab)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$\mathcal{W}_\ell/\mathcal{W}_0$")
ax.set_ylim(1e-12, 2)
ax.legend(ncol=2, loc="lower left")
savefig(fig, "ch08", "mask_spectra")
