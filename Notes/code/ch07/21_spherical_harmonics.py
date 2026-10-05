"""21_spherical_harmonics.py -- the Fourier modes of the sphere.

Question: what do the spherical harmonics Y_lm look like on the sky, and how does a
CMB-like sky get built up as we add higher multipoles?
Computes: (1) a gallery of Re Y_lm maps; (2) the dipole check a_10 = A sqrt(4 pi/3)
for the map A cos(theta); (3) one Gaussian sky with the fiducial C_l, truncated at
l_max = 4, 16, 64, 600.
Writes: figures/ch07/ylm_gallery.pdf, figures/ch07/truncation.pdf, results/ch07/21_spherical_harmonics.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from common import setup, savefig, save_numbers, rng_for
from camb_fiducial import load_fiducial
import lib_harmonic as lh

setup()
NSIDE = 64
theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))   # colatitude, longitude of pixel centres

# ---------------------------------------------------------------- (1) gallery of Re Y_lm
pairs = [(1, 0), (1, 1), (2, 1), (4, 0), (4, 2), (4, 4)]
fig = plt.figure(figsize=(9.0, 4.6))
for i, (l, m) in enumerate(pairs):
    y = np.real(sph_harm_y(l, m, theta, phi))                     # scipy: (degree, order, polar, azimuth)
    hp.mollview(y, fig=fig.number, sub=(2, 3, i + 1), title=rf"$\ell={l},\ m={m}$",
                cmap="RdBu_r", cbar=False, min=-np.abs(y).max(), max=np.abs(y).max(), notext=True)
savefig(fig, "ch07", "ylm_gallery")

# ---------------------------------------------------------------- (2) dipole check
A = 3.0                                                           # amplitude in muK (CMB dipole is ~3 mK; any A works)
dip = A * np.cos(theta)
alm_dip = hp.map2alm(dip, lmax=4, iter=3)
a10_num = alm_dip[hp.Alm.getidx(4, 1, 0)].real
a10_th = A * np.sqrt(4 * np.pi / 3)
others = np.max(np.abs(np.delete(alm_dip, hp.Alm.getidx(4, 1, 0))))

# ---------------------------------------------------------------- (3) one sky, truncated
ell, cl = load_fiducial()
LMAX, NS2 = 600, 256
rng = rng_for("ch07", "21_spherical_harmonics")
alm = lh.synalm(cl, LMAX, rng)
lvec, mvec = lh.getlm(LMAX)
fig = plt.figure(figsize=(9.0, 4.8))
vmax = {}
for i, lcut in enumerate([4, 16, 64, 600]):
    a = np.where(lvec <= lcut, alm, 0)
    sky = hp.alm2map(a, NS2, lmax=LMAX)
    v = np.percentile(np.abs(sky), 99.5)
    vmax[lcut] = v
    hp.mollview(sky, fig=fig.number, sub=(2, 2, i + 1), title=rf"$\ell\leq{lcut}$ (scale $\pm{v:.0f}\,\mu$K)",
                cmap="RdBu_r", min=-v, max=v, cbar=False, notext=True)
savefig(fig, "ch07", "truncation")

# mode counting
save_numbers("ch07", "21_spherical_harmonics", {
    "SBdipA": A,
    "SBdipTh": f"{a10_th:.6f}",
    "SBdipNum": f"{a10_num:.6f}",
    "SBdipOther": others,
    "SBnRealSix": (LMAX + 1) ** 2 - 4,          # real numbers in l = 2..600
    "SBnStoredSix": lh.nalm(LMAX),               # complex numbers stored by healpy (m >= 0, all l)
})
