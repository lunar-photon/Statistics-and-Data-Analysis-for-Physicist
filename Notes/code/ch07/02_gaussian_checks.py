"""02_gaussian_checks.py -- what "Gaussian random field" buys us, checked on a simulated map.

Question: if every set of field values is jointly Gaussian, then (i) the pair
(delta(x), delta(x+r)) has elliptical contours set by xi(r); (ii) the mean field
around a hot spot is xi(r)/sigma^2 times the hot-spot value; (iii) the four-point
moment <delta1^2 delta2^2> is fixed by xi alone (Isserlis/Wick): sigma^4 + 2 xi(r)^2.
Do these hold for a Gaussian field, and fail for a non-Gaussian (lognormal) one?
Computes: one 512 x 512 field with xi = exp(-r^2/2 ell^2), ell = 4; the profile
around pixels with delta > 2; the Wick ratio for the Gaussian and for a lognormal
field y = exp(g) - <exp(g)> made from it.
Writes: figures/ch07/gaussian_checks.pdf, results/ch07/02_gaussian_checks.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_fields import gaussian_field

setup(9.6, 3.2)
rng = rng_for("ch07", "02_gaussian_checks")
N, L, ell = 512, 512.0, 4.0
P = lambda k: 2 * np.pi * ell**2 * np.exp(-k**2 * ell**2 / 2)
xi = lambda r: np.exp(-r**2 / (2 * ell**2))
g = gaussian_field(P, N, L, d=2, rng=rng)
g /= g.std()                                       # unit variance, so xi(0) = 1
out = {}

fig, ax = plt.subplots(1, 3)
# (i) contours of the bivariate Gaussian of (delta(x), delta(x+r)) for three separations
t = np.linspace(0, 2 * np.pi, 300)
for s, c in zip([2, 6, 15], SERIES):
    rho = xi(s)
    C = np.array([[1, rho], [rho, 1]])
    Lc = np.linalg.cholesky(C)
    for lev in (1, 2):                             # 1- and 2-sigma ellipses: x^T C^-1 x = lev^2
        e = Lc @ np.vstack([lev * np.cos(t), lev * np.sin(t)])
        ax[0].plot(e[0], e[1], color=c, lw=1.2, label=rf"$r={s}$, $\rho={rho:.2f}$" if lev == 1 else None)
ax[0].set_aspect("equal"); ax[0].set_xlim(-3, 3); ax[0].set_ylim(-3, 3)
ax[0].set_xlabel(r"$\delta(\mathbf{x})/\sigma$"); ax[0].set_ylabel(r"$\delta(\mathbf{x}+\mathbf{r})/\sigma$")
ax[0].legend(loc="upper left", fontsize=7)
ax[0].set_title("joint density: contours")

# (ii) the average field around hot spots
nu = 2.0
sel = g > nu
nubar = g[sel].mean()                              # mean height of the selected pixels
rs = np.arange(0, 16)
prof = [np.roll(g, -s, axis=1)[sel].mean() for s in rs]   # value s cells to the right
ax[1].plot(rs, prof, "o", color=SERIES[0], label="simulation")
rr = np.linspace(0, 15, 200)
theory_line(ax[1], rr, nubar * xi(rr), label=r"$\bar\nu\,\xi(r)/\sigma^2$")
ax[1].set_xlabel(r"$r$ (cells)"); ax[1].set_ylabel(r"mean of $\delta(\mathbf{x}+\mathbf{r})$")
ax[1].set_title(rf"around pixels with $\delta>{nu:.0f}\sigma$")
ax[1].legend()
out["CondNubar"] = nubar
out["CondProfFour"] = prof[4]
out["CondTheoFour"] = nubar * xi(4)
out["HotFrac"] = sel.mean()

# (iii) Wick: <d1^2 d2^2> versus xi(0)^2 + 2 xi(r)^2, Gaussian and lognormal
y = np.exp(0.8 * g); y = (y - y.mean()) / y.std()  # lognormal field, standardised
def four(f, s):
    a, b = f, np.roll(f, -s, axis=1)
    m4 = np.mean(a**2 * b**2)
    x = np.mean(a * b)
    return m4, np.mean(a**2) * np.mean(b**2) + 2 * x**2
ss = np.arange(0, 16)
wg = np.array([four(g, s) for s in ss])
wy = np.array([four(y, s) for s in ss])
ax[2].plot(ss, wg[:, 0] / wg[:, 1], "o", color=SERIES[0], label="Gaussian field")
ax[2].plot(ss, wy[:, 0] / wy[:, 1], "s", color=SERIES[1], label="lognormal field")
ax[2].axhline(1, color="k", ls="--", lw=1.2)
ax[2].set_xlabel(r"$r$ (cells)")
ax[2].set_ylabel(r"$\langle\delta_1^2\delta_2^2\rangle/(\sigma^4+2\xi^2)$")
ax[2].set_title("Wick test")
ax[2].legend()
out["WickGaussZero"] = wg[0, 0] / wg[0, 1]
out["WickGaussSix"] = wg[6, 0] / wg[6, 1]
out["WickLogZero"] = wy[0, 0] / wy[0, 1]
out["WickLogSix"] = wy[6, 0] / wy[6, 1]
fig.tight_layout()
savefig(fig, "ch07", "gaussian_checks")
save_numbers("ch07", "02_gaussian_checks", {f"SevA{k}": v for k, v in out.items()})
