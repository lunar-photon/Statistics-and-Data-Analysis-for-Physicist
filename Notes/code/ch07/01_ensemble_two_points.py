"""01_ensemble_two_points.py -- a random field as an ensemble of maps.

Question: what does "the correlation between the field at x and at x + r" mean,
when the universe gives us only one map?
Computes: M = 3000 independent realisations of a 2-D Gaussian random field with
xi(r) = exp(-r^2 / 2 ell^2), ell = 4 cells.  At a fixed point x0 and three
separations r = 2, 6, 15 cells, it collects the pairs (delta(x0), delta(x0 + r))
across the ensemble, and estimates xi(r) as the ensemble average of the product.
A second point x1 checks homogeneity: the same r gives the same answer anywhere.
Writes: figures/ch07/ensemble_two_points.pdf, results/ch07/01_ensemble_two_points.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_fields import gaussian_field

setup(9.0, 6.0)
rng = rng_for("ch07", "01_ensemble_two_points")

N, L, ell = 128, 128.0, 4.0                       # cells of unit size
P = lambda k: 2 * np.pi * ell**2 * np.exp(-k**2 * ell**2 / 2)   # 2-D transform of the Gaussian xi
xi = lambda r: np.exp(-r**2 / (2 * ell**2))
seps = [2, 6, 15]
x0, x1 = (40, 40), (90, 70)                       # two reference points (row, column)

M, chunk = 3000, 250
a0, a1 = [], []                                   # field values at x0 (+r) and x1 (+r)
keep = None
for _ in range(M // chunk):
    f = gaussian_field(P, N, L, d=2, rng=rng, nsim=chunk)
    if keep is None:
        keep = f[:3].copy()                       # three realisations to show
    a0.append(np.stack([f[:, x0[0], x0[1]]] + [f[:, x0[0], x0[1] + s] for s in seps], 1))
    a1.append(np.stack([f[:, x1[0], x1[1]]] + [f[:, x1[0], x1[1] + s] for s in seps], 1))
a0, a1 = np.concatenate(a0), np.concatenate(a1)

out = {}
fig, ax = plt.subplots(2, 3)
for j in range(3):                                # top row: three members of the ensemble
    im = ax[0, j].imshow(keep[j], cmap="RdBu_r", vmin=-3, vmax=3, origin="lower")
    ax[0, j].plot(x0[1], x0[0], "ko", ms=4)
    for s, c in zip(seps, SERIES):
        ax[0, j].plot(x0[1] + s, x0[0], "o", mfc="none", mec=c, ms=6, mew=1.5)
    ax[0, j].set_title(f"realisation {j + 1}")
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([]); ax[0, j].grid(False)
names = ["Two", "Six", "Fifteen"]
for j, (s, c) in enumerate(zip(seps, SERIES)):   # bottom row: pairs across the ensemble
    u, v = a0[:, 0], a0[:, j + 1]
    ax[1, j].plot(u, v, ".", color=c, ms=1.5, alpha=0.5)
    est = np.mean(u * v)
    est1 = np.mean(a1[:, 0] * a1[:, j + 1])
    ax[1, j].set_title(rf"$r={s}$" + "\n" + rf"ensemble $\langle\delta\delta\rangle={est:.2f}$, $\xi={xi(s):.2f}$")
    ax[1, j].set_xlabel(r"$\delta(\mathbf{x}_0)$")
    ax[1, j].set_ylabel(r"$\delta(\mathbf{x}_0+\mathbf{r})$")
    ax[1, j].set_xlim(-4, 4); ax[1, j].set_ylim(-4, 4); ax[1, j].set_aspect("equal")
    out[f"EnsXi{names[j]}"] = est
    out[f"EnsXiB{names[j]}"] = est1
    out[f"TheoXi{names[j]}"] = xi(s)
out["EnsVar"] = np.var(a0[:, 0])
out["EnsM"] = M
out["EnsSE"] = 1 / np.sqrt(M)                     # rough standard error of an ensemble product
fig.tight_layout()
savefig(fig, "ch07", "ensemble_two_points")
save_numbers("ch07", "01_ensemble_two_points", {f"SevA{k}": v for k, v in out.items()})
