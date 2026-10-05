"""b01_prepare.py -- from the Planck archive files to working temperature maps at four resolutions.

Question: the SMICA CMB map comes at N_side = 2048 with a 5' beam, the common mask at the same
resolution.  What are the maps, masks and noise model at the resolutions where we measure topology
(N_side 64, 128, 256, 512 with FWHM 160', 80', 40', 20', as in Planck 2018 VII, Table 1), and does
the map's power agree with the best-fit spectrum used for the simulations?

Computes: 1. the harmonic coefficients of the inpainted SMICA temperature map (muK), and for each
             working resolution the coefficients re-beamed to the new FWHM and pixel window;
          2. each mask degraded the way Planck does it (smoothed with the same beam, thresholded at 0.9);
          3. the noise spectrum from the half-mission half-difference map (hm1 - hm2)/2;
          4. a check of the map's power: pseudo-C_l / f_sky of the masked map against the best fit.
Writes:   data/ch13/b_prepared.npz (alms, masks, noise spectrum), figures/ch13/b_maps.pdf,
          figures/ch13/b_spectrum.pdf, results/ch13/b01_prepare.tex
Needs:    data/planck/COM_CMB_IQU-smica_2048_R3.00_{full,hm1,hm2}.fits and
          data/planck/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits (IRSA Planck archive).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, DATA
from camb_fiducial import load_fiducial
from lib_skytopo import windows, pixwin, remove_monodipole

setup()
PL = DATA / "planck"
OUT = DATA / "ch13" / "b_prepared.npz"
SCALES = [(64, 160.0), (128, 80.0), (256, 40.0), (512, 20.0)]     # (N_side, FWHM arcmin): Planck Table 1
NSIDE_IN, FWHM_IN = 2048, 5.0                                     # SMICA: N_side 2048, 5' Gaussian beam
LMAX_IN = 3 * 512 - 1                                             # enough for the finest working map
K2UK = 1e6


def read(name, field):
    m = hp.read_map(PL / name, field=field, dtype=np.float32)     # converts NESTED -> RING
    return m


if OUT.exists():
    z = dict(np.load(OUT, allow_pickle=True))
else:
    z = {}
    # ---- 1. the map: inpainted temperature (field 5, I_STOKES_INP), in muK
    t = read("COM_CMB_IQU-smica_2048_R3.00_full.fits", 5).astype(np.float64) * K2UK
    alm_in = hp.map2alm(t, lmax=LMAX_IN, iter=1)
    del t
    win_in = hp.gauss_beam(np.radians(FWHM_IN / 60), lmax=LMAX_IN) * pixwin(NSIDE_IN, LMAX_IN)
    mask2048 = read("COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits", 0).astype(np.float64)
    z["fsky2048"] = mask2048.mean()
    # ---- 3. noise: half-mission half-difference, pseudo-spectrum over the mask / f_sky
    h1 = read("COM_CMB_IQU-smica_2048_R3.00_hm1.fits", 0).astype(np.float64)
    h2 = read("COM_CMB_IQU-smica_2048_R3.00_hm2.fits", 0).astype(np.float64)
    hd = 0.5 * (h1 - h2) * K2UK * mask2048
    del h1, h2
    z["nl_hd"] = hp.anafast(hd, lmax=LMAX_IN, iter=0) / mask2048.mean()
    del hd
    # ---- 4. power check: pseudo-C_l of the masked (not inpainted) map / f_sky, beam and pixel removed
    t = read("COM_CMB_IQU-smica_2048_R3.00_full.fits", 0).astype(np.float64) * K2UK
    z["cl_pseudo"] = hp.anafast(t * mask2048, lmax=LMAX_IN, iter=0) / mask2048.mean() / win_in ** 2
    del t
    # ---- 2. masks and re-beamed coefficients at each working resolution
    alm_mask = hp.map2alm(mask2048, lmax=LMAX_IN, iter=0)
    del mask2048
    for nside, fwhm in SCALES:
        lmax = 3 * nside - 1
        win = windows(nside, fwhm, lmax)
        m = hp.alm2map(hp.almxfl(hp.resize_alm(alm_mask, LMAX_IN, LMAX_IN, lmax, lmax), win), nside, lmax=lmax)
        obs = m >= 0.9                                            # Planck: degrade, threshold at 0.9
        a = hp.resize_alm(alm_in, LMAX_IN, LMAX_IN, lmax, lmax)
        a = hp.almxfl(a, win / win_in[:lmax + 1])                 # 5' beam and 2048 pixel -> working ones
        a = remove_monodipole(a, obs, nside, lmax)                # fitted outside the mask, as for the data
        z[f"alm_{nside}"], z[f"obs_{nside}"], z[f"mask_{nside}"] = a, obs, m.astype(np.float32)
        print(nside, "fsky", obs.mean(), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **z)

# ---------------------------------------------------------------- numbers
ell = np.arange(LMAX_IN + 1)
_, clfid = load_fiducial("TT")
clfid = clfid[:LMAX_IN + 1]
nl = z["nl_hd"] / (hp.gauss_beam(np.radians(FWHM_IN / 60), lmax=LMAX_IN) * pixwin(NSIDE_IN, LMAX_IN)) ** 2
nums = {"BfskyFull": round(100 * float(z["fsky2048"]), 1)}
for nside, fwhm in SCALES:
    nums[f"Bfsky{['SixFour', 'OneTwoEight', 'TwoFiveSix', 'FiveOneTwo'][SCALES.index((nside, fwhm))]}"] = \
        round(100 * float(z[f"obs_{nside}"].mean()), 1)


def ratio(lo, hi):
    s = slice(lo, hi)
    return float(np.sum((2 * ell[s] + 1) * z["cl_pseudo"][s]) / np.sum((2 * ell[s] + 1) * clfid[s]))


nums.update({"BpowLow": round(ratio(2, 30), 2), "BpowMid": round(ratio(30, 800), 3),
             "BpowHigh": round(ratio(800, 1500), 3)})
s = slice(800, 1500)                                    # the same band with the noise power subtracted
nums["BpowHighNoNoise"] = round(float(np.sum((2 * ell[s] + 1) * (z["cl_pseudo"][s] - nl[s]))
                                      / np.sum((2 * ell[s] + 1) * clfid[s])), 3)
# noise-to-signal of the deconvolved spectrum at three multipoles, and noise level in muK arcmin
for l, k in [(500, "Five"), (1000, "Thousand"), (1500, "Fifteen")]:
    nums[f"BnoiseRatio{k}"] = round(float(np.mean(nl[l - 25:l + 25]) / np.mean(clfid[l - 25:l + 25])), 3)
nums["BnoiseUkArcmin"] = round(float(np.sqrt(np.mean(z["nl_hd"][400:800])) * 60 * 180 / np.pi), 0)
# how much of the working-map variance is noise at each resolution
for (nside, fwhm), k in zip(SCALES, ["SixFour", "OneTwoEight", "TwoFiveSix", "FiveOneTwo"]):
    lmax = 3 * nside - 1
    w = windows(nside, fwhm, lmax) ** 2
    nw = z["nl_hd"][:lmax + 1] / (hp.gauss_beam(np.radians(FWHM_IN / 60), lmax=lmax) * pixwin(NSIDE_IN, lmax)) ** 2 * w
    sw = clfid[:lmax + 1] * w
    l = np.arange(lmax + 1)
    nums[f"BnoiseVar{k}"] = round(100 * float(np.sum((2 * l + 1) * nw[:]) / np.sum((2 * l + 1) * sw)), 2)
    nums[f"BnoiseGrad{k}"] = round(100 * float(np.sum((2 * l + 1) * l * (l + 1) * nw) / np.sum((2 * l + 1) * l * (l + 1) * sw)), 2)
save_numbers("ch13", "b01_prepare", nums)
for k, v in nums.items():
    print(k, v)

# ---------------------------------------------------------------- figure 1: the map and the masks
fig = plt.figure(figsize=(10, 3.4))
t = hp.alm2map(z["alm_128"], 128, lmax=383)
obs = z["obs_128"]
tm = np.where(obs, t, hp.UNSEEN)
hp.mollview(tm, fig=fig.number, sub=(1, 2, 1), title=r"SMICA, $N_{\rm side}=128$, FWHM $80'$ ($\mu$K)",
            cmap="RdBu_r", min=-250, max=250, badcolor="0.75", cbar=True)
u = (t - t[obs].mean()) / t[obs].std()
ex = np.where(obs, (u > 1.0).astype(float), hp.UNSEEN)
hp.mollview(ex, fig=fig.number, sub=(1, 2, 2), title=r"excursion set $u>1$ (black)", cmap="Greys",
            min=0, max=1.3, badcolor="0.75", cbar=False)
savefig(fig, "ch13", "b_maps")

# ---------------------------------------------------------------- figure 2: power and noise
fig, ax = plt.subplots(figsize=(6.4, 3.6))
D = ell * (ell + 1) / (2 * np.pi)
edges = np.unique(np.concatenate([np.arange(2, 30, 2), np.geomspace(30, LMAX_IN, 40).astype(int)]))
cen = 0.5 * (edges[1:] + edges[:-1])
binned = lambda c: np.array([np.mean((D * c)[a:b]) for a, b in zip(edges[:-1], edges[1:])])
ax.plot(ell[2:], (D * clfid)[2:], color="k", ls="--", lw=1.2, label="best-fit spectrum (simulations)")
ax.plot(cen, binned(z["cl_pseudo"]), "o", ms=3, color=SERIES[0], label=r"SMICA map, pseudo-$C_\ell/f_{\rm sky}$")
ax.plot(cen, binned(nl), "-", color=SERIES[1], label="noise (half-mission half-difference)")
for (nside, fwhm), c in zip(SCALES, SERIES[2:6]):
    w = hp.gauss_beam(np.radians(fwhm / 60), lmax=LMAX_IN) ** 2
    ax.plot(ell[2:], (D * clfid * w)[2:], color=c, lw=0.9, label=fr"best fit $\times b_\ell^2$, {fwhm:.0f}$'$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1, 1e4)
ax.set_xlim(2, LMAX_IN)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
ax.legend(fontsize=7, loc="lower left")
savefig(fig, "ch13", "b_spectrum")
