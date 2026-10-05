"""24_camb_healpix.py -- from the CAMB spectrum to a HEALPix sky and back.

Question: what does the fiducial CAMB spectrum look like with its cosmic-variance band,
how does HEALPix cut the sphere into pixels, and does the round trip
C_l -> a_lm -> map -> a_lm -> C_hat_l give back what we put in?
Computes: D_l with the band D_l (1 +- sqrt(2/(2l+1))) and one sky's D_hat_l; HEALPix tilings at
N_side = 1, 2, 4; one synfast-style sky at N_side = 256; anafast on it versus alm2cl on the
input a_lm (iter = 0 and 3); our from-scratch alm2cl versus healpy's.
Writes: figures/ch07/dl_band.pdf, figures/ch07/healpix_tiling.pdf, figures/ch07/roundtrip.pdf,
        results/ch07/24_camb_healpix.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES
from camb_fiducial import load_fiducial
import lib_harmonic as lh

setup()
ell, cl = load_fiducial()                      # raw C_l in muK^2, lensed, l = 0..3000, C_0 = C_1 = 0
rng = rng_for("ch07", "24_camb_healpix")
fac = ell * (ell + 1) / (2 * np.pi)
Dl = fac * cl

# ---------------------------------------------------------------- (1) D_l and its cosmic-variance band
LMAX = 2500
alm = lh.synalm(cl, LMAX, rng)
chat = lh.alm2cl(alm, LMAX)
L = np.arange(2, LMAX + 1)
band = np.sqrt(2.0 / (2 * L + 1))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.fill_between(L, Dl[L] * (1 - band), Dl[L] * (1 + band), color=SERIES[0], alpha=0.25, lw=0,
                label=r"$D_\ell\,[1\pm\sqrt{2/(2\ell+1)}]$")
ax.plot(L, fac[L] * chat[L], ".", ms=1.6, color=SERIES[1], label=r"one sky: $\ell(\ell+1)\widehat C_\ell/2\pi$")
theory_line(ax, L, Dl[L], label=r"CAMB $D_\ell$")
ax.set_xscale("log")
ax.set_xlim(2, LMAX)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
ax.legend(loc="upper left", fontsize=8)
savefig(fig, "ch07", "dl_band")

# ---------------------------------------------------------------- (2) HEALPix tilings
fig = plt.figure(figsize=(9.0, 2.4))
for i, ns in enumerate([1, 2, 4]):
    npix = hp.nside2npix(ns)
    colour = rng.permutation(npix).astype(float)      # random colours make the tiles visible
    hp.mollview(colour, fig=fig.number, sub=(1, 3, i + 1), cmap="tab20", cbar=False, notext=True,
                title=rf"$N_{{\rm side}}={ns}$: {npix} pixels")
    hp.graticule(dpar=30, dmer=60, color="0.3", lw=0.3)
savefig(fig, "ch07", "healpix_tiling")

# ---------------------------------------------------------------- (3) round trip on a map
NS = 256
from scipy.ndimage import median_filter


def roundtrip(lm, it):
    """Draw a_lm up to lm, make the N_side = 256 map, analyse it again; return |C_map/C_input - 1|."""
    a = lh.synalm(cl, lm, rng)
    sky = hp.alm2map(a, NS, lmax=lm)                 # = synfast with our own random numbers
    c_in = lh.alm2cl(a, lm)
    c_map = hp.anafast(sky, lmax=lm, iter=it)
    return sky, a, c_in, np.abs(c_map[2:] / c_in[2:] - 1)

sky, a, c_in, err_2n_3 = roundtrip(2 * NS, 3)
our_vs_hp = np.max(np.abs(c_in[2:] - hp.alm2cl(a)[2:]) / hp.alm2cl(a)[2:])
_, _, _, err_2n_0 = roundtrip(2 * NS, 0)
_, _, _, err_3n_3 = roundtrip(3 * NS - 1, 3)
fig = plt.figure(figsize=(9.0, 3.2))
hp.mollview(sky, fig=fig.number, sub=(1, 2, 1), cmap="RdBu_r", title=rf"$N_{{\rm side}}={NS}$, $\ell\leq{2*NS}$",
            unit=r"$\mu$K", min=-350, max=350, notext=True)
ax = fig.add_axes([0.62, 0.17, 0.37, 0.72])
for e, lab, col in [(err_2n_0, r"band limit $2N_{\rm side}$, iter = 0", SERIES[1]),
                    (err_2n_3, r"band limit $2N_{\rm side}$, iter = 3", SERIES[0]),
                    (err_3n_3, r"band limit $3N_{\rm side}-1$, iter = 3", SERIES[2])]:
    Lx = np.arange(2, 2 + e.size)
    ax.plot(Lx, median_filter(e, 9), color=col, lw=1.1, label=lab)
ax.axvline(2 * NS, color="0.4", lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$|\widehat C_\ell^{\rm map}/\widehat C_\ell^{\rm input}-1|$")
ax.legend(loc="upper left", fontsize=7)
ax.set_title(r"round trip $a_{\ell m}\to$ map $\to a_{\ell m}$ (running median)")
savefig(fig, "ch07", "roundtrip")

save_numbers("ch07", "24_camb_healpix", {
    "SBpeakEll": int(np.argmax(Dl)),
    "SBpeakD": f"{Dl.max():.0f}",
    "SBlmaxBand": LMAX,
    "SBnpixTwoFiveSix": hp.nside2npix(256),
    "SBnpixFiveTwelve": hp.nside2npix(512),
    "SBnpixTwoK": f"{hp.nside2npix(2048):,}".replace(",", "{,}"),
    "SBpixTwoFiveSix": hp.nside2resol(256, arcmin=True),
    "SBpixFiveTwelve": hp.nside2resol(512, arcmin=True),
    "SBpixTwoK": hp.nside2resol(2048, arcmin=True),
    "SBourVsHp": our_vs_hp,
    "SBrtZero": np.median(err_2n_0),
    "SBrtThree": np.median(err_2n_3),
    "SBrtThreeMax": np.max(err_2n_3),
    "SBrtAlias": np.median(err_3n_3[: 2 * NS - 1]),
})
