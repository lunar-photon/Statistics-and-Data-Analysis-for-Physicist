"""42_pol_spectra.py -- windows, a noise model and the polarization cross-spectra of the real sky.

Question: through the SMICA polarization confidence mask, what are the EE, BB, EB, TE and TB
cross-spectra of the two half-mission maps (so that no noise bias enters), and what does the
noise of a half-mission polarization map look like?
Steps:
  1. window: split the holes of PMASK (nside 512) into Galactic and point-source parts, cosine
     tapers of 2 deg and 0.5 deg (as for temperature in 02_mask.py); also the binary mask;
     temperature uses the 13a window (common mask, same tapers); a stricter 'cut' window is the
     product of the two (it also removes the sky the temperature common mask removes);
  2. pseudo-a_lm of W x (Q, U) by the spin-2 transform -> E, B; of W_T x T -> T;
     binned cross-spectra of hm1 x hm2 (symmetrised for EB, TE, TB);
  3. noise model from d = (hm1 - hm2)/2: local variance v(p) of (Q_d^2 + U_d^2)/2 smoothed over
     2 deg, and the E and B spectra of d / sqrt(v), divided by <W^2>, averaged over 31 multipoles.
Writes: data/ch13/pol_windows_n512.npz, data/ch13/pol_noise_model.npz, data/ch13/pol_data_spectra.npz,
        figures/ch13/pol_maps.pdf, results/ch13/42_pol_spectra.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers
import lib_planck as lp
import lib_biref as lb

setup()
t0 = time.time()
N, L = lb.NSIDE, lb.LS
nums = {}
pol = lb.load_pol()
binary = pol["pmask"]

# ---------------------------------------------------------------- 1. windows
gal, ps, ncomp, nbig = lp.mask_components(binary, N)
WP = lp.apodise(gal, N, 2.0) * lp.apodise(ps, N, 0.5)
WT = lp.load_masks(N)["main"].astype(np.float64)
WC = WP * WT                     # a stricter window: also the temperature common mask (Galactic check)
np.savez(lb.DATA / f"pol_windows_n{N}.npz", main=WP.astype(np.float32), binary=binary.astype(np.float32),
         T=WT.astype(np.float32), cut=WC.astype(np.float32))
for k, W in (("Bin", binary), ("Main", WP), ("Cut", WC)):
    nums[f"ThirteenEFsky{k}"] = f"{np.mean(W > 0):.3f}"
    nums[f"ThirteenEFwtwo{k}"] = f"{np.mean(W ** 2):.3f}"
    nums[f"ThirteenEFskyEff{k}"] = f"{np.mean(W ** 2) ** 2 / np.mean(W ** 4):.3f}"
nums["ThirteenEHoles"] = ncomp
nums["ThirteenEHolesBig"] = nbig
nums["ThirteenEFskyGal"] = f"{gal.mean():.3f}"
nums["ThirteenEFskyTwin"] = f"{np.mean(WT > 0):.3f}"
print(nums, flush=True)


def pseudo_eb(W, q, u):
    _, e, b = hp.map2alm([np.zeros_like(q), W * q, W * u], lmax=L, iter=0, pol=True)
    return e, b


# ---------------------------------------------------------------- 2. data spectra
tmaps = lp.load_maps(N)
keepT = lp.load_masks(N)["binary"] > 0
vec = np.array(hp.pix2vec(N, np.arange(12 * N * N)))
halves = []
for k in ("hm1", "hm2"):
    t = tmaps[k].copy()
    m = t.copy(); m[~keepT] = hp.UNSEEN
    mono, dip = hp.fit_dipole(m)
    t = t - mono - dip @ vec
    aT = hp.map2alm(WT * t, lmax=L, iter=0)
    aE, aB = pseudo_eb(WP, pol[f"{k}_Q"], pol[f"{k}_U"])
    halves.append((aT, aE, aB))
spec = lb.cross_spectra(*halves)
dq, du = 0.5 * (pol["hm1_Q"] - pol["hm2_Q"]), 0.5 * (pol["hm1_U"] - pol["hm2_U"])
nE, nB = pseudo_eb(WP, dq, du)
null = {"EE": lb.binned(hp.alm2cl(nE)), "BB": lb.binned(hp.alm2cl(nB))}
hc = [pseudo_eb(WC, pol[f"{k}_Q"], pol[f"{k}_U"]) for k in ("hm1", "hm2")]
spec_cut = lb.cross_spectra((halves[0][0],) + hc[0], (halves[1][0],) + hc[1])
# the same through the binary mask (E-to-B leakage comparison on the data)
hb = [pseudo_eb(binary, pol[f"{k}_Q"], pol[f"{k}_U"]) for k in ("hm1", "hm2")]
spec_bin = {"EE": lb.binned(hp.alm2cl(hb[0][0], hb[1][0])), "BB": lb.binned(hp.alm2cl(hb[0][1], hb[1][1]))}
# unbinned EB and EE-BB for the figure of the estimator
ucl = {"EB": 0.5 * (hp.alm2cl(halves[0][1], halves[1][2]) + hp.alm2cl(halves[1][1], halves[0][2])),
       "EE": hp.alm2cl(halves[0][1], halves[1][1]), "BB": hp.alm2cl(halves[0][2], halves[1][2])}
np.savez(lb.DATA / "pol_data_spectra.npz", **{f"x_{k}": v for k, v in spec.items()},
         **{f"null_{k}": v for k, v in null.items()},
         **{f"cut_{k}": v for k, v in spec_cut.items()}, **{f"bin_{k}": v for k, v in spec_bin.items()},
         **{f"u_{k}": v for k, v in ucl.items()})
print("data spectra done", time.time() - t0, flush=True)

# ---------------------------------------------------------------- 3. noise model
keep = binary > 0
v = hp.smoothing(0.5 * (dq ** 2 + du ** 2), fwhm=2.0 * lp.DEG, iter=0)
v = np.where(keep, v / v[keep].mean(), 1.0)
v = np.clip(v, 0.05, None)
gE, gB = pseudo_eb(WP, dq / np.sqrt(v), du / np.sqrt(v))
w2 = np.mean(WP ** 2)
ng = {}
for k, a in (("EE", gE), ("BB", gB)):
    pcl = hp.alm2cl(a) / w2
    s = np.convolve(pcl, np.ones(31) / 31, mode="same")
    s[:31] = pcl[:31]
    s[-15:] = s[-16]
    s[:2] = 0.0
    ng[k] = s
nd = {"EE": hp.alm2cl(nE) / w2, "BB": hp.alm2cl(nB) / w2}
np.savez(lb.DATA / "pol_noise_model.npz", v=v.astype(np.float32), ngEE=ng["EE"], ngBB=ng["BB"],
         ndEE=nd["EE"], ndBB=nd["BB"])
# white-noise level of a half-mission map: N_l = (sigma theta)^2 -> sigma in muK arcmin
ell = np.arange(L + 1)
band = (ell > 600) & (ell < 900)
TP = pol["beam_P"] * pol["pixwin_P"]
nhalf = 2 * np.mean(nd["EE"][band] / TP[band] ** 2)
nums["ThirteenENoiseArcmin"] = f"{np.sqrt(nhalf) / lp.ARCMIN:.0f}"
nums["ThirteenEVmax"] = f"{v[keep].max():.1f}"
nums["ThirteenENoiseRatioEB"] = f"{np.mean(nd['BB'][band]) / np.mean(nd['EE'][band]):.2f}"
print(nums, flush=True)

# ---------------------------------------------------------------- figure: the maps
fig = plt.figure(figsize=(8.0, 5.4))
P = np.hypot(pol["full_Q"], pol["full_U"])
Psm = np.hypot(hp.smoothing(pol["full_Q"], fwhm=lp.DEG, iter=0), hp.smoothing(pol["full_U"], fwhm=lp.DEG, iter=0))
hp.mollview(np.where(keep, pol["full_Q"], hp.UNSEEN), fig=fig.number, sub=(2, 2, 1), min=-15, max=15,
            cmap="RdBu_r", title=r"(a) SMICA $Q$, PMASK", unit=r"$\mu$K", notext=True)
hp.mollview(np.where(keep, Psm, hp.UNSEEN), fig=fig.number, sub=(2, 2, 2), min=0, max=2,
            title=r"(b) $P=\sqrt{Q^2+U^2}$, smoothed $1^\circ$", unit=r"$\mu$K", notext=True)
hp.mollview(WP, fig=fig.number, sub=(2, 2, 3), min=0, max=1, cmap="Greys_r",
            title=r"(c) polarization window", notext=True)
hp.mollview(np.where(keep, v, hp.UNSEEN), fig=fig.number, sub=(2, 2, 4), min=0, max=3, cmap="viridis",
            title=r"(d) relative noise variance $v(p)$", notext=True)
savefig(fig, "ch13", "pol_maps")
nums["ThirteenESpecSeconds"] = f"{time.time() - t0:.0f}"
save_numbers("ch13", "42_pol_spectra", nums)
