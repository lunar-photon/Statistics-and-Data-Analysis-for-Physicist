"""c01_tiny.py -- the sampling distribution of a topological summary, on a 3 x 3 grid.

Question: if each pixel of a 3 x 3 grid is "hot" independently with probability p, what is the
probability distribution of the number of pieces (beta0), of holes (beta1) and of the Euler
characteristic chi of the hot set?  The means were derived by hand in the text:
    E[chi] = p + 4 q^2 - 4 q^4,   E[beta1] = p^4 q,   E[beta0] = E[chi] + E[beta1],  q = 1 - p.
Computes: the exact distributions by enumerating all 512 configurations (from scratch, with
lib_topo), a Monte Carlo check with 100000 random grids, and the same counts with
scikit-image (label, euler_number) as an independent library.
Writes: figures/ch12/c_tiny.pdf, results/ch12/c01_tiny.tex
"""
import sys
import pathlib
import itertools

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2
from lib_topo import betti_plane, euler_cubical
from skimage import measure

rng = rng_for("ch12", "c01_tiny")
p = 0.5
q = 1 - p

# ---- every configuration, once: (beta0, beta1, chi, number of hot pixels)
configs = []
for bits in itertools.product([0, 1], repeat=9):
    m = np.array(bits, bool).reshape(3, 3)
    b0, b1 = betti_plane(m)
    chi = euler_cubical(m)
    assert chi == b0 - b1                                   # the Euler relation, checked 512 times
    # scikit-image: 8-connected pieces, Euler number with 8-connectivity (connectivity=2)
    assert measure.label(m, connectivity=2).max() == b0
    assert measure.euler_number(np.pad(m, 1), connectivity=2) == chi
    configs.append((b0, b1, chi, m.sum()))
configs = np.array(configs)
w = p ** configs[:, 3] * q ** (9 - configs[:, 3])           # probability of each configuration


def exact_dist(col):
    vals = np.unique(configs[:, col])
    return vals, np.array([w[configs[:, col] == v].sum() for v in vals])


Eb0, Eb1, Echi = (w * configs[:, 0]).sum(), (w * configs[:, 1]).sum(), (w * configs[:, 2]).sum()
hand_chi = p + 4 * q ** 2 - 4 * q ** 4
hand_b1 = p ** 4 * q
Vb0 = (w * configs[:, 0] ** 2).sum() - Eb0 ** 2
Vchi = (w * configs[:, 2] ** 2).sum() - Echi ** 2
Cb0b1 = (w * configs[:, 0] * configs[:, 1]).sum() - Eb0 * Eb1

# ---- Monte Carlo: 100000 random grids
n = 100_000
grids = rng.random((n, 3, 3)) < p
mc = np.array([betti_plane(g) for g in grids])
mc_b0 = mc[:, 0].mean()
mc_b0_se = mc[:, 0].std() / np.sqrt(n)
mc_b1 = mc[:, 1].mean()

# ---- the means against p
ps = np.linspace(0, 1, 101)
qs = 1 - ps
mean_curves = {"chi": ps + 4 * qs ** 2 - 4 * qs ** 4, "b1": ps ** 4 * qs}
mean_curves["b0"] = mean_curves["chi"] + mean_curves["b1"]

setup(8.0, 3.0)
fig, ax = plt.subplots(1, 2)
vals, prob = exact_dist(0)
ax[0].bar(vals - 0.18, prob, 0.36, color=SERIES[0], label=r"exact (512 grids)")
h = np.bincount(mc[:, 0], minlength=vals.max() + 1)[vals] / n
ax[0].bar(vals + 0.18, h, 0.36, color=SERIES[1], label=r"Monte Carlo ($10^5$)")
ax[0].set_xlabel(r"number of pieces $\beta_0$")
ax[0].set_ylabel("probability")
ax[0].set_title(r"$p=1/2$")
ax[0].legend()
ax[1].plot(ps, mean_curves["b0"], color=SERIES[0], label=r"$\mathrm{E}[\beta_0]$")
ax[1].plot(ps, mean_curves["chi"], color=SERIES[2], label=r"$\mathrm{E}[\chi]$")
ax[1].plot(ps, 10 * mean_curves["b1"], color=SERIES[1], label=r"$10\times\mathrm{E}[\beta_1]$")
pp = np.linspace(0.05, 0.95, 10)
for pv in pp:                                               # exact enumeration at a few p
    ww = pv ** configs[:, 3] * (1 - pv) ** (9 - configs[:, 3])
    ax[1].plot(pv, (ww * configs[:, 0]).sum(), "o", color=SERIES[0], ms=3)
    ax[1].plot(pv, 10 * (ww * configs[:, 1]).sum(), "o", color=SERIES[1], ms=3)
ax[1].set_xlabel(r"probability $p$ that a pixel is hot")
ax[1].legend()
fig.tight_layout()
savefig(fig, "ch12", "c_tiny")

save_numbers("ch12", "c01_tiny", {
    "CtEbZero": f"{Eb0:.5f}", "CtEbOne": f"{Eb1:.5f}", "CtEchi": f"{Echi:.5f}",
    "CtHandChi": f"{hand_chi:.5f}", "CtHandbOne": f"{hand_b1:.5f}",
    "CtSdbZero": f"{np.sqrt(Vb0):.3f}", "CtSdChi": f"{np.sqrt(Vchi):.3f}", "CtCovbb": f"{Cb0b1:.4f}",
    "CtPbZeroTwo": f"{exact_dist(0)[1][list(exact_dist(0)[0]).index(2)]:.4f}",
    "CtMaxbZero": int(vals.max()),
    "CtMCbZero": f"{mc_b0:.4f}", "CtMCbZeroSe": f"{mc_b0_se:.4f}", "CtMCbOne": f"{mc_b1:.4f}",
})
print(Eb0, Eb1, Echi, hand_chi, hand_b1, mc_b0, mc_b1, np.sqrt(Vb0), Cb0b1)
