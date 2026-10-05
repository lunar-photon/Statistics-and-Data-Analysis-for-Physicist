"""02_mask.py -- from the Planck common mask to the windows of the analysis, and their coupling.

Question: the common mask is a 0/1 map.  Its holes are of two kinds -- the Galactic plane and
thousands of small discs around point sources.  How do we taper (apodise) each kind, what sky
fraction survives, and what coupling matrix does each window produce?
Steps:
  1. split the holes into connected pieces; pieces larger than 20 deg^2 are 'Galactic',
     the rest 'point sources' (this mirrors Planck 2018 V, Sect. 3.2.2, who apodise the two
     kinds differently: 4.71 deg FWHM for the Galactic part, 30' for the sources);
  2. distance from every kept pixel to the nearest hole of each kind; cosine taper of width
     2 deg (Galactic) and 0.5 deg (sources); the window is the product;
  3. variants used by the checks: binary (no taper), |b| > 30 deg cut, north and south halves;
  4. fsky, w2 = <W^2>/fsky and the coupling matrix M_ll' (l <= 1199) of each window.
Writes: data/ch13/masks_n512.npz, data/ch13/coupling_*.npz, figures/ch13/masks.pdf,
        results/ch13/02_mask.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers
import lib_planck as lp

setup()
NSIDE, LC = 512, 1199
t0 = time.time()
nums = {}
binary = hp.read_map(lp.PLANCK / f"common_mask_n{NSIDE}.fits", dtype=np.float64)
gal, ps, ncomp, nbig = lp.mask_components(binary, NSIDE)
nums.update({"TwelveAHoles": ncomp, "TwelveAHolesBig": nbig,
             "TwelveAFskyGal": f"{gal.mean():.3f}", "TwelveAFskyPS": f"{1 - ps.mean():.3f}"})
print(f"{ncomp} holes, {nbig} extended; Galactic part removes {1 - gal.mean():.3f}, sources {1 - ps.mean():.3f}")

W_main = lp.apodise(gal, NSIDE, 2.0) * lp.apodise(ps, NSIDE, 0.5)
masks = {
    "binary": binary,
    "main": W_main,
    "cut30": W_main * lp.latitude_taper(NSIDE, "both", 30.0, 2.0),
    "north": W_main * lp.latitude_taper(NSIDE, "north", 0.0, 2.0),
    "south": W_main * lp.latitude_taper(NSIDE, "south", 0.0, 2.0),
}
lp.DATA.mkdir(parents=True, exist_ok=True)
np.savez(lp.DATA / f"masks_n{NSIDE}.npz", **{k: v.astype(np.float32) for k, v in masks.items()},
         gal=gal.astype(np.float32), ps=ps.astype(np.float32))

names = {"binary": "Bin", "main": "Main", "cut": "Cut", "north": "North", "south": "South"}
for k, W in masks.items():
    fsky = np.mean(W > 0)
    w2 = np.mean(W ** 2) / fsky
    tag = names.get(k, names.get(k[:3], k))
    nums[f"TwelveAFsky{tag}"] = f"{fsky:.3f}"
    nums[f"TwelveAWtwo{tag}"] = f"{w2:.3f}"
    nums[f"TwelveAFskyEff{tag}"] = f"{np.mean(W ** 2) ** 2 / np.mean(W ** 4):.3f}"
    ts = time.time()
    lp.coupling(W, LC, name=k)
    print(f"{k}: fsky={fsky:.3f} w2={w2:.3f}  coupling {time.time() - ts:.0f} s", flush=True)

# ---------------------------------------------------------------- figure: the sky and the windows
maps = lp.load_maps(NSIDE)
sky = maps["full"] - np.mean(maps["full"][binary > 0])
fig = plt.figure(figsize=(8.0, 5.4))
hp.mollview(np.where(binary > 0, sky, hp.UNSEEN), fig=fig.number, sub=(2, 2, 1), title="(a) SMICA, common mask",
            min=-300, max=300, cmap="RdBu_r", unit=r"$\mu$K", cbar=True, notext=True)
hp.mollview(masks["main"], fig=fig.number, sub=(2, 2, 2), title=r"(b) main window: 2$^\circ$ / 0.5$^\circ$ tapers",
            min=0, max=1, cmap="Greys_r", cbar=True, notext=True)
hp.mollview(masks["cut30"], fig=fig.number, sub=(2, 2, 3), title=r"(c) extra cut $|b|>30^\circ$",
            min=0, max=1, cmap="Greys_r", cbar=True, notext=True)
hp.mollview(masks["north"] + 0.5 * masks["south"], fig=fig.number, sub=(2, 2, 4),
            title="(d) north (white) and south (grey)", min=0, max=1, cmap="Greys_r", cbar=True, notext=True)
savefig(fig, "ch13", "masks")

# zoom on the taper: a gnomonic cut-out around a source-rich region at high latitude
fig = plt.figure(figsize=(7.6, 3.1))
hp.gnomview(binary, rot=(-60, 45), xsize=300, reso=4.0, fig=fig.number, sub=(1, 2, 1), title="(a) binary common mask",
            min=0, max=1, cmap="Greys_r", notext=True, cbar=False)
hp.gnomview(masks["main"], rot=(-60, 45), xsize=300, reso=4.0, fig=fig.number, sub=(1, 2, 2),
            title="(b) apodised window", min=0, max=1, cmap="Greys_r", notext=True, cbar=False)
savefig(fig, "ch13", "mask_zoom")

nums["TwelveAMaskSeconds"] = f"{time.time() - t0:.0f}"
save_numbers("ch13", "02_mask", nums)
