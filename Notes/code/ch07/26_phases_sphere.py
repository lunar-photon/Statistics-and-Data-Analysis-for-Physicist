"""26_phases_sphere.py -- two skies with identical C_hat_l that look nothing alike.

Question: what does the full-sky C_hat_l throw away?
Computes: a sky made of 40 hot spots (strongly non-Gaussian); its a_lm; a second sky with the
same |a_lm| but random phases (m > 0) and random signs (m = 0).  Both skies have exactly the
same C_hat_l at every l, but different pixel histograms and different shapes.
Writes: figures/ch07/phases_maps.pdf, figures/ch07/phases_stats.pdf, results/ch07/26_phases_sphere.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES
import lib_harmonic as lh

setup()
rng = rng_for("ch07", "26_phases_sphere")
NS, LM, NSPOT, RAD = 128, 255, 40, np.radians(4.0)

# hot spots: Gaussian bumps of width 4 deg at isotropically random centres
vec = np.array(hp.pix2vec(NS, np.arange(hp.nside2npix(NS))))
cz = rng.uniform(-1, 1, NSPOT); ph = rng.uniform(0, 2 * np.pi, NSPOT)
cen = np.array([np.sqrt(1 - cz ** 2) * np.cos(ph), np.sqrt(1 - cz ** 2) * np.sin(ph), cz])
ang = np.arccos(np.clip(cen.T @ vec, -1, 1))            # (NSPOT, npix) angular distance to each centre
spots = np.exp(-0.5 * (ang / RAD) ** 2).sum(axis=0)
spots -= spots.mean()

alm = hp.map2alm(spots, lmax=LM, iter=3)
l, m = lh.getlm(LM)
phase = np.exp(2j * np.pi * rng.uniform(size=alm.size))
sign = rng.choice([-1.0, 1.0], size=alm.size)
alm_r = np.where(m == 0, sign * np.abs(alm), np.abs(alm) * phase)   # same |a_lm|, new phases
map_a = hp.alm2map(alm, NS, lmax=LM)                    # the spot sky, band-limited to l <= 255
map_b = hp.alm2map(alm_r, NS, lmax=LM)
ca, cb = lh.alm2cl(alm, LM), lh.alm2cl(alm_r, LM)

v = np.max(np.abs(map_a))
fig = plt.figure(figsize=(9.0, 2.9))
hp.mollview(map_a, fig=fig.number, sub=(1, 2, 1), cmap="RdBu_r", min=-v, max=v, cbar=False,
            notext=True, title="(a) 40 hot spots")
hp.mollview(map_b, fig=fig.number, sub=(1, 2, 2), cmap="RdBu_r", min=-v, max=v, cbar=False,
            notext=True, title=r"(b) same $|a_{\ell m}|$, random phases")
savefig(fig, "ch07", "phases_maps")

L = np.arange(1, 121)                                   # beyond l~120 the 4-degree spots have no power
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
ax[0].loglog(L, ca[L], color=SERIES[0], lw=2.4, label="hot-spot sky")
ax[0].loglog(L, cb[L], color=SERIES[1], lw=1.0, ls="--", label="phase-randomised sky")
ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel(r"$\widehat C_\ell$ (arbitrary units)")
ax[0].set_title(r"(a) identical $\widehat C_\ell$")
ax[0].legend(loc="lower left", fontsize=8)
bins = np.linspace(-v, v, 80)
ax[1].hist(map_a, bins=bins, density=True, color=SERIES[0], alpha=0.55, label="hot-spot sky")
ax[1].hist(map_b, bins=bins, density=True, color=SERIES[1], alpha=0.55, label="phase-randomised sky")
ax[1].set_yscale("log")
ax[1].set_xlabel("pixel value"); ax[1].set_ylabel("density")
ax[1].set_title("(b) different pixel histograms")
ax[1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
savefig(fig, "ch07", "phases_stats")

save_numbers("ch07", "26_phases_sphere", {
    "SBphNspot": NSPOT,
    "SBphMaxDiff": np.max(np.abs(ca[1:] - cb[1:]) / ca[1:]),
    "SBphSkewA": stats.skew(map_a),
    "SBphSkewB": stats.skew(map_b),
})
