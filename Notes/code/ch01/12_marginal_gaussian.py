"""Marginal and conditional densities of a correlated 2-D Gaussian, by projection and by slicing.

Question: the marginal p_X(x) is 'the joint with y averaged out'; the conditional p(y|x) is
'the joint on a thin band at fixed x, renormalised'.  Do the simulated projections and slices
agree with the analytic formulas derived in the text?

Model: (X, Y) bivariate normal, mu = (0, 0), sigma_x = 1, sigma_y = 2, rho = 0.7.
  marginal:     X ~ N(0, 1),  Y ~ N(0, 4)
  conditional:  Y | X = x ~ N(rho sigma_y x / sigma_x, sigma_y^2 (1 - rho^2))
Writes: figures/ch01/marginal_gaussian.pdf, results/ch01/12_marginal_gaussian.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch01", "12_marginal_gaussian")
setup(7.0, 3.2)

sx, sy, rho = 1.0, 2.0, 0.7
C = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
N = 200_000
XY = rng.multivariate_normal([0.0, 0.0], C, size=N)
X, Y = XY[:, 0], XY[:, 1]

# thin band at x0
x0, half = 1.0, 0.05
band = np.abs(X - x0) < half
Yb = Y[band]
cond_mean = rho * sy / sx * x0
cond_sd = sy * np.sqrt(1 - rho**2)

fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1.2, 1, 1]})
ax = axes[0]
ax.plot(X[:3000], Y[:3000], ".", ms=1.5, color=SERIES[0], alpha=0.5)
ax.axvspan(x0 - half, x0 + half, color=SERIES[1], alpha=0.35)
ax.set_xlim(-4, 4); ax.set_ylim(-8, 8)
ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.set_title("joint: 3000 draws")

ax = axes[1]
yy = np.linspace(-8, 8, 400)
ax.hist(Y, bins=80, range=(-8, 8), density=True, color=SERIES[0], alpha=0.6,
        label="all $y$ (projection)")
theory_line(ax, yy, stats.norm.pdf(yy, 0, sy), label="$\\mathcal{N}(0,\\sigma_y^2)$")
ax.set_xlabel("$y$"); ax.set_ylabel("density")
ax.set_title("marginal $p_Y(y)$")
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.legend(fontsize=7, loc="upper left")

ax = axes[2]
ax.hist(Yb, bins=40, range=(-8, 8), density=True, color=SERIES[1], alpha=0.6,
        label=f"$y$ in band ({band.sum()} pts)")
theory_line(ax, yy, stats.norm.pdf(yy, cond_mean, cond_sd), label="conditional formula")
ax.set_xlabel("$y$")
ax.set_title(f"conditional $p(y\\,|\\,x={x0:g})$")
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch01", "marginal_gaussian")

save_numbers("ch01", "12_marginal_gaussian", {
    "OneBMargN": "2\\times10^{5}",
    "OneBMargSdY": f"{Y.std():.3f}",
    "OneBMargSdX": f"{X.std():.3f}",
    "OneBBandN": int(band.sum()),
    "OneBCondMeanSim": f"{Yb.mean():.3f}",
    "OneBCondMeanExact": f"{cond_mean:.3f}",
    "OneBCondSdSim": f"{Yb.std():.3f}",
    "OneBCondSdExact": f"{cond_sd:.3f}",
})
