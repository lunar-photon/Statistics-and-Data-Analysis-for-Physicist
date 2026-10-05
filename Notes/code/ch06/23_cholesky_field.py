"""23_cholesky_field.py -- correlated Gaussian vectors and a correlated 1-D field.

Question: how do we draw a Gaussian vector with a prescribed covariance C?
Answer: factor C = L L^T (Cholesky, L lower triangular), draw independent
standard normals z, return mu + L z.  Then Cov(L z) = L Cov(z) L^T = L L^T = C.

Two uses:
  1. the 2x2 example sigma_x = 1, sigma_y = 2, rho = 0.8 (L = [[1, 0], [1.6, 1.2]]);
  2. a one-dimensional Gaussian random field on n = 200 points of [0, 20] with
     exponential correlation C(r) = exp(-|r| / ell), ell = 2 -- a toy version of
     the correlated fields of chapters 7-8.  We check the correlation function
     estimated from M = 4000 realisations against C(r), and show the pitfall
     x = C z (covariance C^2, correlation too long).

Writes: figures/ch06/cholesky_field.pdf, results/ch06/23_cholesky_field.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt

setup(9.0, 3.2)
rng = rng_for("ch06", "23_cholesky_field")
out = {}

# ---------------------------------------------------------------- 1. the 2x2 example
C2 = np.array([[1.0, 1.6], [1.6, 4.0]])
L2 = np.linalg.cholesky(C2)
z = rng.standard_normal((50_000, 2))
xy = z @ L2.T                                        # each row is L z
S2 = np.cov(xy, rowvar=False)
out.update(SixBChLa=L2[1, 0], SixBChLb=L2[1, 1],
           SixBChSxx=S2[0, 0], SixBChSxy=S2[0, 1], SixBChSyy=S2[1, 1],
           SixBChRho=np.corrcoef(xy, rowvar=False)[0, 1])

# ---------------------------------------------------------------- 2. a correlated 1-D field
n, Lbox, ell, M = 200, 20.0, 2.0, 4000
xg = np.linspace(0, Lbox, n)
C = np.exp(-np.abs(xg[:, None] - xg[None, :]) / ell)
L = np.linalg.cholesky(C)
Z = rng.standard_normal((M, n))
F = Z @ L.T                                          # M realisations, each row one field
Fbad = Z @ C.T                                       # the pitfall: multiply by C instead of L


def corr_fn(fields, maxlag):
    """Average of f(x) f(x + r) over positions and realisations, normalised to r = 0."""
    c = np.array([np.mean(fields[:, : n - k] * fields[:, k:]) for k in range(maxlag)])
    return c / c[0]


lags = np.arange(60)
r = lags * (xg[1] - xg[0])
cf, cfbad = corr_fn(F, 60), corr_fn(Fbad, 60)
Semp = np.cov(F, rowvar=False)
out.update(SixBChN=n, SixBChM=M, SixBChEll=ell,
           SixBChMaxDev=np.max(np.abs(Semp - C)),
           SixBChExpDev=np.sqrt(2.0 / M),
           SixBChCfAtEll=cf[np.argmin(np.abs(r - ell))],
           SixBChCfBadAtEll=cfbad[np.argmin(np.abs(r - ell))])

fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.4, 1]})
for i in range(3):
    ax[0].plot(xg, F[i], color=SERIES[i], lw=1.1)
ax[0].plot(xg, Z[0], color="0.7", lw=0.6, zorder=0, label="white noise $z$")
ax[0].set(xlabel=r"position $x$", ylabel=r"field $f(x)$", title=r"realisations of $f=\mathbf{L}z$")
ax[0].set_ylim(-3.6, 3.8)
ax[0].legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
ax[1].plot(r, cf, "o", ms=3, color=SERIES[0], label=r"$\mathbf{L}z$ (correct)")
ax[1].plot(r, cfbad, "s", ms=3, color=SERIES[1], label=r"$\mathbf{C}z$ (wrong)")
theory_line(ax[1], r, np.exp(-r / ell), label=r"$e^{-r/\ell}$")
ax[1].set(xlabel=r"separation $r$", ylabel="correlation", title="correlation function")
ax[1].legend()
fig.tight_layout()
savefig(fig, "ch06", "cholesky_field")
save_numbers("ch06", "23_cholesky_field", out)
print(out)
