"""b02_grf_persistence.py -- persistence of a Gaussian random field, from scratch and with libraries.

Question: what does the persistence diagram of a smooth random field look like, do the
from-scratch union-find codes (islands, lakes) agree with gudhi, ripser and scikit-image, and
how long do they take?
Computes: one 256x256 Gaussian field; its H0 and H1 diagrams with lib_persist; the same with
gudhi.CubicalComplex and ripser.lower_star_img; bottleneck distances between the codes
(persim); Betti curves from the diagrams against direct counts with scipy.ndimage.label and
the islands with skimage.measure.label and the Euler characteristic from
skimage.measure.euler_number; run time against grid size.
Writes: figures/ch12/b_grf_diagram.pdf, figures/ch12/b_timing.pdf, results/ch12/b02_grf_persistence.tex
"""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, theory_line  # noqa: F401
from lib_fields import gaussian_field
from lib_persist import islands, lakes, betti_curve, to_sublevel, finite

import gudhi
import ripser
import persim
from skimage import measure

rng = rng_for("ch12", "b02_grf_persistence")
N, R = 256, 4.0                                           # grid and smoothing length (pixels)


def field(n, rng):
    P = lambda k: np.exp(-(k * R) ** 2)                   # Gaussian-smoothed white noise
    f = gaussian_field(P, n, float(n), d=2, rng=rng)
    return f / f.std()


f = field(N, rng)
t0 = time.perf_counter()
h0 = islands(f)
h1 = lakes(f)
t_ours = time.perf_counter() - t0

# ---------------------------------------------------------------- libraries
t0 = time.perf_counter()
cc = gudhi.CubicalComplex(top_dimensional_cells=-f)        # sublevel of -f = superlevel of f
cc.persistence()
t_gudhi = time.perf_counter() - t0
g0 = cc.persistence_intervals_in_dimension(0)
g1 = cc.persistence_intervals_in_dimension(1)
r0 = ripser.lower_star_img(-f)
dB0 = persim.bottleneck(to_sublevel(finite(h0)), finite(g0))
dB1 = persim.bottleneck(to_sublevel(h1), g1)
dBr = persim.bottleneck(to_sublevel(finite(h0)), finite(r0))
print(f"pairs: ours H0 {len(h0)} H1 {len(h1)}; gudhi H0 {len(g0)} H1 {len(g1)}; ripser H0 {len(r0)}")
print(f"bottleneck ours-gudhi H0 {dB0:.2e} H1 {dB1:.2e}; ours-ripser H0 {dBr:.2e}")

# ---------------------------------------------------------------- Betti curves two ways
nus = np.linspace(-3, 3, 61)
b0 = betti_curve(h0, nus)
b1 = betti_curve(h1, nus)
b0_direct, b1_direct, chi_sk, b0_sk = [], [], [], []
for nu in nus:
    land = f >= nu
    b0_direct.append(ndimage.label(land, structure=np.ones((3, 3)))[1])
    b0_sk.append(measure.label(land, connectivity=2).max())   # scikit-image, 8-neighbours
    lab, n = ndimage.label(~land)                          # water, 4-neighbours
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    b1_direct.append(n - np.count_nonzero(edge))           # water bodies not touching the sea
    chi_sk.append(measure.euler_number(land, connectivity=2))
b0_direct, b1_direct, chi_sk, b0_sk = map(np.array, (b0_direct, b1_direct, chi_sk, b0_sk))
mism = int(np.abs(b0 - b0_direct).max() + np.abs(b1 - b1_direct).max()
           + np.abs(b0 - b1 - chi_sk).max() + np.abs(b0 - b0_sk).max())
print("max mismatch of Betti curves / Euler characteristic:", mism)

# ---------------------------------------------------------------- monotone relabelling check
g = np.exp(0.8 * f)                                        # a lognormal-like field, same order
h0g = islands(g)
same = np.allclose(np.log(h0g[np.isfinite(h0g[:, 1])]) / 0.8, h0[np.isfinite(h0[:, 1])])

# ---------------------------------------------------------------- figure: field, diagram, curves
setup(7.4, 2.7)
fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
ax[0].imshow(f, cmap="RdBu_r", vmin=-3, vmax=3, origin="lower")
ax[0].contour(f, levels=[1.0], colors="k", linewidths=0.6, origin="lower")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title(r"field, contour $\nu=1$", fontsize=9)
fin = finite(h0)
ax[1].plot([-4, 4], [-4, 4], color=INK2, lw=0.8, ls=":")
ax[1].scatter(fin[:, 0], fin[:, 1], s=7, color=SERIES[0], label=r"islands $H_0$")
ax[1].scatter(h1[:, 0], h1[:, 1], s=7, color=SERIES[1], marker="s", label=r"lakes $H_1$")
ax[1].set_xlim(-3.5, 4)
ax[1].set_ylim(-4, 3.5)
ax[1].set_xlabel(r"birth $b$")
ax[1].set_ylabel(r"death $d$")
ax[1].legend(loc="upper left", fontsize=7, handletextpad=0.2)
ax[1].set_title("persistence diagram", fontsize=9)
ax[2].plot(nus, b0, color=SERIES[0], label=r"$\beta_0$ from diagram")
ax[2].plot(nus, b1, color=SERIES[1], label=r"$\beta_1$ from diagram")
ax[2].plot(nus[::4], b0_direct[::4], "o", ms=3, mfc="none", color=SERIES[0])
ax[2].plot(nus[::4], b1_direct[::4], "s", ms=3, mfc="none", color=SERIES[1])
ax[2].set_xlabel(r"level $\nu$")
ax[2].set_title("Betti curves", fontsize=9)
ax[2].plot([], [], "o", ms=3, mfc="none", color=INK2, label="direct count")
ax[2].set_ylim(0, 112)
ax[2].legend(fontsize=6.5, loc="upper center", ncol=1, handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
fig.tight_layout()
savefig(fig, "ch12", "b_grf_diagram")

# ---------------------------------------------------------------- timing against grid size
sizes = [64, 128, 256, 512, 1024]
tt_ours, tt_gudhi = [], []
for n in sizes:
    fn = field(n, rng)
    t0 = time.perf_counter()
    islands(fn)
    lakes(fn)
    tt_ours.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    c = gudhi.CubicalComplex(top_dimensional_cells=-fn)
    c.persistence()
    tt_gudhi.append(time.perf_counter() - t0)
    print(f"N={n}: ours {tt_ours[-1]:.3f} s, gudhi {tt_gudhi[-1]:.3f} s")
npix = np.array(sizes) ** 2
slope = np.polyfit(np.log(npix[1:]), np.log(tt_ours[1:]), 1)[0]
setup(4.2, 3.0)
fig, ax = plt.subplots()
ax.loglog(npix, tt_ours, "o-", color=SERIES[0], label="union-find (Python)")
ax.loglog(npix, tt_gudhi, "s-", color=SERIES[2], label="gudhi (C++)")
ax.set_xlabel("number of pixels")
ax.set_ylabel("time for $H_0$ and $H_1$ [s]")
ax.legend()
fig.tight_layout()
savefig(fig, "ch12", "b_timing")

save_numbers("ch12", "b02_grf_persistence", {
    "PbGrfN": N, "PbGrfR": R,
    "PbGrfNzero": len(h0), "PbGrfNone": len(h1),
    "PbGrfdBgudhi": max(dB0, dB1), "PbGrfdBripser": dBr,
    "PbGrfMismatch": mism, "PbGrfMonotone": "yes" if same else "no",
    "PbTimeOurs": t_ours, "PbTimeGudhi": t_gudhi,
    "PbTimeOursBig": tt_ours[-1], "PbTimeGudhiBig": tt_gudhi[-1], "PbTimeSlope": slope,
    "PbGrfMaxPers": float(np.max(fin[:, 0] - fin[:, 1])),
    "PbGrfShortFrac": float(np.mean((fin[:, 0] - fin[:, 1]) < 0.25)),
})
