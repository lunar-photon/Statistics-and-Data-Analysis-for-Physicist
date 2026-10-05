"""What correlation measures, and what zero correlation does NOT imply.

Question: rho measures the strength of the LINEAR relation between X and Y.  For
X ~ N(0,1) and Y = X^2, Y is completely determined by X, yet Cov(X, Y) = E[X^3] = 0.
We (i) draw bivariate Gaussians with rho = -0.8, 0, 0.5, 0.95 to calibrate the eye, and
(ii) show the Y = X^2 cloud, its sample correlation (about 0), and the conditional mean and
spread of Y in bins of X (which change with X: dependence).
Writes: figures/ch01/correlation_gallery.pdf, figures/ch01/zero_correlation.pdf,
        results/ch01/17_zero_correlation.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "17_zero_correlation")
setup(7.0, 2.2)

# ---- gallery of correlated Gaussians ---------------------------------------------
fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
rhos = [-0.8, 0.0, 0.5, 0.95]
sample_r = []
for ax, r in zip(axes, rhos):
    C = np.array([[1.0, r], [r, 1.0]])
    xy = rng.multivariate_normal([0, 0], C, size=1500)
    sample_r.append(np.corrcoef(xy.T)[0, 1])
    ax.plot(xy[:, 0], xy[:, 1], ".", ms=1.5, color=SERIES[0], alpha=0.5)
    ax.set_title(f"$\\rho={r:g}$")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
axes[0].set_ylabel("$y$")
fig.tight_layout()
savefig(fig, "ch01", "correlation_gallery")

# ---- Y = X^2 ------------------------------------------------------------------------
setup(7.0, 3.0)
N = 200_000
X = rng.normal(size=N)
Y = X**2
r_xy = np.corrcoef(X, Y)[0, 1]
r_x2y = np.corrcoef(X**2, Y)[0, 1]
r_absx_y = np.corrcoef(np.abs(X), Y)[0, 1]

edges = np.linspace(-3, 3, 25)
cent = 0.5 * (edges[1:] + edges[:-1])
idx = np.digitize(X, edges) - 1
ok = (idx >= 0) & (idx < len(cent))
cond_mean = np.array([Y[ok & (idx == i)].mean() for i in range(len(cent))])

fig, axes = plt.subplots(1, 2)
ax = axes[0]
ax.plot(X[:3000], Y[:3000], ".", ms=1.5, color=SERIES[0], alpha=0.5)
ax.plot(cent, cond_mean, "o", ms=3.5, color=SERIES[1], label="$\\mathrm{E}[Y\\,|\\,X]$ in bins")
xx = np.linspace(-3, 3, 200)
theory_line(ax, xx, xx**2, label="$y=x^2$")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-0.5, 9.5)
ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.legend(fontsize=7, loc="upper center")
ax.set_title(f"sample $r(X,Y)={r_xy:.3f}$")

ax = axes[1]
# conditional density of Y for two slices of X: disjoint supports -> dependence
for (lo, hi), c in zip([(-0.3, 0.3), (1.7, 2.3)], SERIES[2:]):
    sel = (X > lo) & (X < hi)
    ax.hist(Y[sel], bins=60, range=(0, 6), density=True, color=c, alpha=0.6,
            label=f"$Y$ given ${lo:g}<X<{hi:g}$")
ax.hist(Y, bins=60, range=(0, 6), density=True, histtype="step", color="k", lw=1.0,
        label="$Y$ (marginal)")
ax.set_yscale("log")
ax.set_xlabel("$y$"); ax.set_ylabel("density")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "ch01", "zero_correlation")

save_numbers("ch01", "17_zero_correlation", {
    "OneBGalRa": f"{sample_r[0]:.3f}", "OneBGalRb": f"{sample_r[1]:.3f}",
    "OneBGalRc": f"{sample_r[2]:.3f}", "OneBGalRd": f"{sample_r[3]:.3f}",
    "OneBZeroRxy": f"{r_xy:.4f}",
    "OneBZeroRxxy": f"{r_x2y:.4f}",
    "OneBZeroRabsxy": f"{r_absx_y:.4f}",
    "OneBZeroN": "2\\times10^{5}",
})
