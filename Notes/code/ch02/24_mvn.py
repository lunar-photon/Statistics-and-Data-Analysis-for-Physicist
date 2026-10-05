"""The two-dimensional Gaussian: ellipses, marginals and conditionals, checked by sampling.

Question answered
    We claim that for a bivariate Gaussian (i) the ellipses chi^2 = 2.30 and
    6.18 contain 68.3% and 95.4% of the samples, (ii) the marginal of y is
    N(mu_y, sigma_y^2) whatever rho is, and (iii) the conditional of y at fixed
    x is N(mu_y + rho sigma_y/sigma_x (x - mu_x), sigma_y^2 (1 - rho^2)).
    Do samples agree?

What it computes
    * 20 000 draws x = mu + L z with L the Cholesky factor of C.
    * The fraction of draws inside the two ellipses.
    * The histogram of y in a thin slab |x - x0| < 0.05 (the conditional) and
      the histogram of all y (the marginal), with their predicted Gaussians.

What it writes
    figures/ch02/mvn_ellipses.pdf, results/ch02/24_mvn.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup()
rng = rng_for("ch02", "24_mvn")

mu = np.array([1.0, 2.0])
sx, sy, rho = 1.0, 2.0, 0.8
C = np.array([[sx ** 2, rho * sx * sy], [rho * sx * sy, sy ** 2]])
L = np.linalg.cholesky(C)                   # C = L L^T
N = 20_000
x = mu + rng.standard_normal((N, 2)) @ L.T  # each row: mu + L z

Cinv = np.linalg.inv(C)
r = x - mu
chi2 = np.einsum("ni,ij,nj->n", r, Cinv, r)
lev1, lev2 = stats.chi2.ppf([stats.chi2.cdf(1, 1), stats.chi2.cdf(4, 1)], df=2)
f1, f2 = np.mean(chi2 < lev1), np.mean(chi2 < lev2)

# ellipse from the eigen-decomposition C = R diag(l1,l2) R^T
evals, evecs = np.linalg.eigh(C)
t = np.linspace(0, 2 * np.pi, 400)
circle = np.vstack([np.cos(t), np.sin(t)])

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
ax = axes[0]
ax.plot(x[:4000, 0], x[:4000, 1], ".", ms=1.5, color=SERIES[0], alpha=0.5, label="samples")
for lev, ls, lab in [(lev1, "-", r"$\chi^2=2.30$"), (lev2, "--", r"$\chi^2=6.18$")]:
    ell = mu[:, None] + evecs @ (np.sqrt(lev * evals)[:, None] * circle)
    ax.plot(ell[0], ell[1], color="k", ls=ls, lw=1.4, label=lab)
for k in range(2):  # principal axes, half-length sqrt(eigenvalue) (1-sigma along that axis)
    v = evecs[:, k] * np.sqrt(evals[k])
    ax.annotate("", xy=mu + v, xytext=mu, arrowprops=dict(arrowstyle="->", color=SERIES[2], lw=1.6))
x0 = 2.0
ax.axvspan(x0 - 0.05, x0 + 0.05, color=SERIES[1], alpha=0.35, label=f"slab $x={x0}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_aspect("equal")
ax.set_title(r"(a) $\sigma_x=1$, $\sigma_y=2$, $\rho=0.8$")
ax.legend(loc="upper left", fontsize=7)

ax = axes[1]
slab = np.abs(x[:, 0] - x0) < 0.05
# extra draws so the conditional histogram is well populated
xb = mu + rng.standard_normal((400_000, 2)) @ L.T
yb = xb[np.abs(xb[:, 0] - x0) < 0.05, 1]
m_c = mu[1] + rho * sy / sx * (x0 - mu[0])
s_c = sy * np.sqrt(1 - rho ** 2)
yy = np.linspace(-6, 10, 400)
ax.hist(xb[:, 1], bins=np.linspace(-6, 10, 81), density=True, color=SERIES[0], alpha=0.5,
        label="all $y$ (marginal)")
ax.hist(yb, bins=np.linspace(-6, 10, 81), density=True, color=SERIES[1], alpha=0.6,
        label=f"$y$ in the slab $x\\approx{x0}$")
theory_line(ax, yy, stats.norm.pdf(yy, mu[1], sy), label=None)
theory_line(ax, yy, stats.norm.pdf(yy, m_c, s_c), label="predicted Gaussians")
ax.set_xlabel("$y$")
ax.set_ylabel("density")
ax.set_title("(b) marginal and conditional of $y$")
ax.legend(loc="upper left", fontsize=7)
savefig(fig, "ch02", "mvn_ellipses")

save_numbers("ch02", "24_mvn", {
    "tbMVNfracOne": f"{f1:.4f}",
    "tbMVNfracTwo": f"{f2:.4f}",
    "tbMVNlevOne": f"{lev1:.3f}",
    "tbMVNlevTwo": f"{lev2:.3f}",
    "tbMVNcondMeanTh": f"{m_c:.3f}",
    "tbMVNcondSdTh": f"{s_c:.3f}",
    "tbMVNcondMean": f"{yb.mean():.3f}",
    "tbMVNcondSd": f"{yb.std(ddof=1):.3f}",
    "tbMVNnSlab": int(yb.size),
    "tbMVNmargSd": f"{xb[:, 1].std(ddof=1):.3f}",
    "tbMVNlamOne": f"{evals[1]:.3f}",
    "tbMVNlamTwo": f"{evals[0]:.3f}",
    "tbMVNangle": f"{np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])):.1f}",
    "tbMVNN": N,
})
