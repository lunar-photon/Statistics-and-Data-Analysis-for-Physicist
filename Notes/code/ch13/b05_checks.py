"""b05_checks.py -- two checks of the topology code of lib_skytopo on the sphere.

Question 1: on an unmasked sphere, does beta1 = beta0 - chi of the hot set (8-connected pieces,
            chi = V - E + F of the closed hot pixels) agree with Alexander duality, which counts the
            holes of the hot set as the number of cold regions (4-connected) minus one?
Question 2: how long does one union-find pass (the H0 pairs of the hot set) take on an N_side = 512 map?

Computes: for full-sky Gaussian maps at N_side 64 and 128 with the best-fit spectrum (80' beam at 128,
          160' at 64), beta1 both ways at 61 thresholds in [-3, 3] where both the hot and the cold set are
          non-empty; the number of (map, threshold) cases and of disagreements; the wall time of
          h0_pairs on one N_side = 512 map (after numba compilation).
Writes:   results/ch13/b05_checks.tex
"""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
from common import save_numbers, rng_for
from camb_fiducial import load_fiducial
from lib_skytopo import windows, mesh, neighbours, entry_levels, chi_from_levels, h0_pairs, betti0_from_pairs

rng = rng_for("ch13", "b05_checks")


def gauss_map(c, nside, lmax):
    """A Gaussian map with spectrum c: a_l0 ~ N(0, C_l), Re and Im of a_lm (m > 0) ~ N(0, C_l / 2)."""
    ell, m = hp.Alm.getlm(lmax)
    s = np.sqrt(c[ell])
    re, im = rng.standard_normal(ell.size), rng.standard_normal(ell.size)
    return hp.alm2map(np.where(m == 0, s * re, s * (re + 1j * im) / np.sqrt(2)), nside, lmax=lmax)


_, cl = load_fiducial("TT")
nus = np.linspace(-3.0, 3.0, 61)
cases, bad, nmaps = 0, 0, 0
for nside, fwhm in ((64, 160.0), (128, 80.0)):
    lmax = 3 * nside - 1
    c = cl[:lmax + 1] * windows(nside, fwhm, lmax) ** 2
    c[:2] = 0.0
    obs = np.ones(hp.nside2npix(nside), dtype=bool)
    msh, nb8, nb4 = mesh(nside), neighbours(nside), neighbours(nside, edge_only=True)
    for k in range(10):
        t = gauss_map(c, nside, lmax)
        u = (t - t.mean()) / t.std()
        chi = chi_from_levels(entry_levels(u, obs, msh), nus)
        b, d = h0_pairs(u, obs, nb8)
        b0 = betti0_from_pairs(b, d, nus)
        bc, dc = h0_pairs(-u, obs, nb4)
        b0c = betti0_from_pairs(bc, dc, -nus)                      # cold regions {u <= nu}, edge-connected
        ok = (u > nus[:, None]).any(axis=1) & (u <= nus[:, None]).any(axis=1)
        cases += int(ok.sum())
        bad += int(np.sum((b0 - chi)[ok] != (b0c - 1)[ok]))
        nmaps += 1
    print(nside, "cases", cases, "disagreements", bad, flush=True)

# timing of one union-find pass at N_side = 512
nside = 512
lmax = 3 * nside - 1
c = cl[:lmax + 1] * windows(nside, 20.0, lmax) ** 2
c[:2] = 0.0
u = gauss_map(c, nside, lmax)
u = (u - u.mean()) / u.std()
obs = np.ones(u.size, dtype=bool)
nb8 = neighbours(nside)
h0_pairs(u[:1000].copy(), obs[:1000], np.full((1000, 8), -1, dtype=np.int64))   # compile on a tiny input
t0 = time.time()
h0_pairs(u, obs, nb8)
dt = time.time() - t0
print("h0 pass at 512:", dt, "s")
save_numbers("ch13", "b05_checks", {"DualMaps": nmaps, "DualCases": cases, "DualBad": bad,
                                     "HzeroTimeFiveOneTwo": round(dt, 2)})
