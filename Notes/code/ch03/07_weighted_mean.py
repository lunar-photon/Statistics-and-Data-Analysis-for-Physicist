"""07_weighted_mean.py -- how should measurements of different quality be combined?

Question: (i) eight labs measure the same mu with different error bars sigma_i.
Is the inverse-variance weighted mean really better than the plain average, with
standard deviation 1/sqrt(sum 1/sigma_i^2)?  Is chi^2 = sum (x_i - xhat)^2/sigma_i^2
distributed as chi^2 with N-1 degrees of freedom?  (ii) for two correlated
measurements (sigma_1 = 1, sigma_2 = 2), how do the optimal weight w and the
variance of the combination depend on the correlation rho?

Writes: figures/ch03/weighted_mean.pdf, figures/ch03/correlated_combination.pdf,
        results/ch03/07_weighted_mean.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch03", "07_weighted_mean")
setup()
M = 200_000

# ---------- (i) eight labs ----------
mu = 0.0
sig = np.array([0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
N = len(sig)
x = mu + sig * rng.standard_normal((M, N))
w = 1 / sig**2
xw = (x * w).sum(1) / w.sum()
xu = x.mean(1)
chi2 = (((x - xw[:, None]) / sig) ** 2).sum(1)
sd_w_th = 1 / np.sqrt(w.sum())
sd_u_th = np.sqrt((sig**2).sum()) / N

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
b = np.linspace(-2.5, 2.5, 121)
axs[0].hist(xu, bins=b, density=True, histtype="step", lw=1.5, color=SERIES[1], label="plain average")
axs[0].hist(xw, bins=b, density=True, histtype="step", lw=1.5, color=SERIES[0], label="weighted mean")
gg = np.linspace(-2.5, 2.5, 400)
theory_line(axs[0], gg, stats.norm.pdf(gg, 0, sd_w_th), label=r"$\mathcal{N}(0,1/\sum\sigma_i^{-2})$")
axs[0].set_xlabel(r"combined estimate $-\mu$"); axs[0].legend(fontsize=7)
axs[1].hist(chi2, bins=np.linspace(0, 25, 101), density=True, color=SERIES[2], alpha=0.6,
            label=r"$\chi^2$ about $\hat\mu$")
cc = np.linspace(0.01, 25, 400)
theory_line(axs[1], cc, stats.chi2.pdf(cc, N - 1), label=rf"$\chi^2_{{{N-1}}}$")
axs[1].set_xlabel(r"$\chi^2$"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "weighted_mean")

# ---------- (ii) two correlated measurements ----------
s1, s2 = 1.0, 2.0
rhos = np.linspace(-0.9, 0.95, 200)
w_th = (s2**2 - rhos * s1 * s2) / (s1**2 + s2**2 - 2 * rhos * s1 * s2)
V_th = (1 - rhos**2) * s1**2 * s2**2 / (s1**2 + s2**2 - 2 * rhos * s1 * s2)
rho_pts = np.array([-0.8, -0.4, 0.0, 0.3, 0.5, 0.7, 0.9])
V_mc = []
for r in rho_pts:
    C = np.array([[s1**2, r * s1 * s2], [r * s1 * s2, s2**2]])
    y = rng.multivariate_normal([0, 0], C, size=M)
    ww = (s2**2 - r * s1 * s2) / (s1**2 + s2**2 - 2 * r * s1 * s2)
    V_mc.append(np.var(ww * y[:, 0] + (1 - ww) * y[:, 1], ddof=1))
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
axs[0].plot(rhos, w_th, color=SERIES[0]); axs[0].axhline(1, color="0.4", ls=":", lw=0.9)
axs[0].axvline(s1 / s2, color="0.4", ls="--", lw=0.8)
axs[0].set_xlabel(r"correlation $\rho$"); axs[0].set_ylabel(r"weight $w$ of the better measurement")
theory_line(axs[1], rhos, V_th, label="derived")
axs[1].plot(rho_pts, V_mc, "o", color=SERIES[1], label="Monte Carlo")
axs[1].axhline(s1**2, color="0.4", ls=":", lw=0.9)
axs[1].set_xlabel(r"correlation $\rho$"); axs[1].set_ylabel(r"Var of combination"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "correlated_combination")

save_numbers("ch03", "07_weighted_mean", {
    "ThreeAWsdW": xw.std(ddof=1), "ThreeAWsdWth": sd_w_th,
    "ThreeAWsdU": xu.std(ddof=1), "ThreeAWsdUth": sd_u_th,
    "ThreeAWchiMean": chi2.mean(), "ThreeAWchiVar": chi2.var(ddof=1),
    "ThreeAWbest": sig.min(),
    "ThreeACorrVmcNine": V_mc[-1],
    "ThreeACorrVthNine": (1 - 0.81) * 4 / (1 + 4 - 2 * 0.9 * 2),
})
