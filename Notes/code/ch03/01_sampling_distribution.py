"""01_sampling_distribution.py -- what does 'the distribution of an estimator' look like?

Question: we measure the same quantity N times.  If we could repeat the whole
experiment many times, how would each candidate 'answer' (sample mean, median,
mid-range, first reading only) be distributed?  And for a non-Gaussian case
(muon decay times) does the histogram of the estimator match its exact
sampling distribution?

Computes: M imagined repetitions of
  (A) N=10 Gaussian readings of g (true 9.81 m/s^2, sigma 0.05 m/s^2),
  (B) N=20 exponential decay times (tau = 2.197 microseconds),
  (C) n=100 darts thrown at a square with an inscribed circle (estimate of pi).
Writes: figures/ch03/sampling_gauss.pdf, figures/ch03/sampling_lifetime.pdf,
        results/ch03/01_sampling_distribution.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch03", "01_sampling_distribution")
setup()
M = 200_000                      # number of imagined repetitions of the experiment

# ---------- (A) ten Gaussian readings of g ----------
g_true, sigma, N = 9.81, 0.05, 10
x = g_true + sigma * rng.standard_normal((M, N))      # each row = one experiment
est = {
    "sample mean": x.mean(axis=1),
    "sample median": np.median(x, axis=1),
    "mid-range": 0.5 * (x.min(axis=1) + x.max(axis=1)),
    "first reading": x[:, 0],
}
sd = {k: v.std(ddof=1) for k, v in est.items()}

fig, ax = plt.subplots(figsize=(6.2, 3.6))
bins = np.linspace(g_true - 4 * sigma, g_true + 4 * sigma, 121)
for (k, v), c in zip(est.items(), SERIES):
    ax.hist(v, bins=bins, density=True, histtype="step", lw=1.5, color=c,
            label=f"{k}: sd = {sd[k]:.4f}")
grid = np.linspace(bins[0], bins[-1], 400)
theory_line(ax, grid, stats.norm.pdf(grid, g_true, sigma / np.sqrt(N)),
            label=r"$\mathcal{N}(g,\sigma^2/N)$")
ax.axvline(g_true, color="0.4", lw=0.8)
ax.set_xlabel(r"estimate of $g$ from one experiment  [m s$^{-2}$]")
ax.set_ylabel("density over repetitions")
ax.legend(loc="upper left", fontsize=7.5)
savefig(fig, "ch03", "sampling_gauss")

# ---------- (B) twenty muon decay times ----------
tau, Nmu = 2.197, 20
t = rng.exponential(tau, size=(M, Nmu))
tau_hat = t.mean(axis=1)
fig, ax = plt.subplots(figsize=(6.0, 3.4))
b = np.linspace(0.5, 4.5, 121)
ax.hist(tau_hat, bins=b, density=True, color=SERIES[0], alpha=0.55,
        label=r"$\hat\tau$ from $2\times10^5$ simulated experiments")
grid = np.linspace(b[0], b[-1], 400)
theory_line(ax, grid, stats.gamma.pdf(grid, a=Nmu, scale=tau / Nmu),
            label=r"exact: Gamma$(N,\tau/N)$")
ax.plot(grid, stats.norm.pdf(grid, tau, tau / np.sqrt(Nmu)), color=SERIES[1],
        lw=1.2, label=r"CLT: $\mathcal{N}(\tau,\tau^2/N)$")
ax.axvline(tau, color="0.4", lw=0.8)
ax.set_xlabel(r"$\hat\tau$  [$\mu$s]")
ax.set_ylabel("density")
ax.set_ylim(0, 1.2)
ax.legend(fontsize=8, loc="upper right")
savefig(fig, "ch03", "sampling_lifetime")

# ---------- (C) darts: 4 * fraction inside the circle ----------
n_darts = 100
p = np.pi / 4
hits = rng.binomial(n_darts, p, size=M)
pi_hat = 4 * hits / n_darts
se_pi = 4 * np.sqrt(p * (1 - p) / n_darts)

save_numbers("ch03", "01_sampling_distribution", {
    "ThreeAM": f"{M:,}".replace(",", r"\,"),
    "ThreeASdMean": sd["sample mean"],
    "ThreeASdMedian": sd["sample median"],
    "ThreeASdMidrange": sd["mid-range"],
    "ThreeASdFirst": sd["first reading"],
    "ThreeASdTheory": sigma / np.sqrt(N),
    "ThreeAMedianVarRatio": (sd["sample median"] / sd["sample mean"]) ** 2,
    "ThreeATauMean": tau_hat.mean(),
    "ThreeATauSd": tau_hat.std(ddof=1),
    "ThreeATauSkew": stats.skew(tau_hat),
    "ThreeATauSkewTheory": 2 / np.sqrt(Nmu),
    "ThreeAPiMean": pi_hat.mean(),
    "ThreeAPiSd": pi_hat.std(ddof=1),
    "ThreeAPiSeTheory": se_pi,
})
