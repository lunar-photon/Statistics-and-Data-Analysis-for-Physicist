"""09_phases.py -- what the power spectrum throws away: the Fourier phases.

Question: two fields can have exactly the same P(k) and look nothing alike.  Where
does the difference live, and which simple statistics still see it?
Computes: a non-Gaussian field made of 70 thin rings (shells, as left by expanding
bubbles) on a 256^2 periodic grid.  Phase randomisation: keep every |delta_k|, replace
the phases by uniform random ones.  The two maps have identical P_hat(k) by
construction.  We compare their one-point PDFs (skewness) and the number of
connected regions of the excursion set {delta > nu} as a function of nu (scipy.ndimage.label).
Phase swap: amplitudes of a smooth Gaussian field + phases of the rings.
Writes: figures/ch07/phases_flat_maps.pdf, figures/ch07/phases_flat_stats.pdf, results/ch07/09_phases.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import skew
from lib_fields import gaussian_field, measure_pk, randomise_phases, swap_phases

rng = rng_for("ch07", "09_phases")
N, L = 256, 256.0
x = np.arange(N)
X, Y = np.meshgrid(x, x, indexing="ij")
rings = np.zeros((N, N))
for _ in range(70):
    cx, cy = rng.uniform(0, N, 2)
    R = rng.uniform(6, 22)
    dx = (X - cx + N / 2) % N - N / 2                  # periodic distances
    dy = (Y - cy + N / 2) % N - N / 2
    rings += np.exp(-((np.hypot(dx, dy) - R) ** 2) / (2 * 1.2**2))
rings = (rings - rings.mean()) / rings.std()

shuffled = randomise_phases(rings, rng)
shuffled = (shuffled - shuffled.mean()) / shuffled.std()
smoothg = gaussian_field(lambda k: np.exp(-k**2 * 3.0**2 / 2), N, L, 2, rng)
swapped = swap_phases(smoothg, rings)                  # |FFT| of a blob field, phases of the rings
swapped = (swapped - swapped.mean()) / swapped.std()
swapped2 = swap_phases(rings, smoothg)                 # |FFT| of the rings, phases of the blobs
swapped2 = (swapped2 - swapped2.mean()) / swapped2.std()

setup(9.6, 2.7)
fig, ax = plt.subplots(1, 4)
for a, m, t in zip(ax, [rings, shuffled, swapped2, swapped],
                   ["(a) rings", "(b) same $|\\delta_k|$, random phases",
                    "(c) $|\\delta_k|$ of rings, phases of blobs", "(d) $|\\delta_k|$ of blobs, phases of rings"]):
    a.imshow(m, cmap="RdBu_r", origin="lower", vmin=-3, vmax=3)
    a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.tight_layout()
savefig(fig, "ch07", "phases_flat_maps")

setup(9.6, 3.0)
fig, ax = plt.subplots(1, 3)
k, pa, nm = measure_pk(rings, L, nbins=30, log=True)
_, pb, _ = measure_pk(shuffled, L, nbins=30, log=True)
ax[0].loglog(k, pa, "o", ms=4, color=SERIES[0], label="(a) rings")
ax[0].loglog(k, pb, "x", ms=4, color=SERIES[1], label="(b) random phases")
ax[0].set_xlabel("$k$ (rad/cell)"); ax[0].set_ylabel(r"$\hat P(k)$"); ax[0].legend(fontsize=7)
bins = np.linspace(-3, 6, 61)
ax[1].hist(rings.ravel(), bins, density=True, histtype="step", color=SERIES[0], lw=1.5, label="(a)")
ax[1].hist(shuffled.ravel(), bins, density=True, histtype="step", color=SERIES[1], lw=1.5, label="(b)")
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\delta/\sigma$"); ax[1].set_ylabel("one-point PDF"); ax[1].legend(fontsize=7)
nus = np.linspace(-2, 3, 41)
def ncomp(m):
    return np.array([ndimage.label(m > nu)[1] for nu in nus])
ca, cb = ncomp(rings), ncomp(shuffled)
ax[2].plot(nus, ca, "-", color=SERIES[0], label="(a) rings")
ax[2].plot(nus, cb, "-", color=SERIES[1], label="(b) random phases")
ax[2].set_xlabel(r"threshold $\nu$ (units of $\sigma$)"); ax[2].set_ylabel(r"regions with $\delta>\nu\sigma$")
ax[2].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch07", "phases_flat_stats")

i1 = np.argmin(np.abs(nus - 1.0))
out = {"PkMaxDiff": np.max(np.abs(pa / pb - 1)),
       "SkewRings": skew(rings.ravel()), "SkewShuf": skew(shuffled.ravel()),
       "CompRingsOne": ca[i1], "CompShufOne": cb[i1],
       "NRings": 70}
save_numbers("ch07", "09_phases", {f"SevA{k}": v for k, v in out.items()})
