"""Discrete waiting times and overdispersed counts: geometric and negative binomial.

Question:  (a) if each bunch crossing fires the trigger with probability p,
           is the number of crossings up to and including the first trigger
           geometric, P(k) = p (1-p)^(k-1), with mean 1/p and variance (1-p)/p^2?
           (b) is the number of failures before the r-th success negative
           binomial?  (c) if the Poisson mean itself fluctuates from run to run
           as a gamma variable, is the count negative binomial with variance
           mu + mu^2/kappa > mu?
Computes:  explicit trial-by-trial simulation for (a), (b); a gamma-Poisson
           mixture for (c) with mu = 10, kappa = 5.
Writes:    figures/ch02/07_geometric_negbin.pdf, results/ch02/07_geometric_negbin.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom, poisson
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

setup(6.4, 2.9)
rng = rng_for("ch02", "07_geometric_negbin")

# (a),(b): run Bernoulli trials one after another and record where successes fall
p, r, reps = 0.1, 3, 50_000
first, fails_before_r = np.empty(reps, int), np.empty(reps, int)
for i in range(reps):
    trials = rng.uniform(size=400) <= p            # 400 trials is (much) more than enough here
    s = np.flatnonzero(trials) + 1                 # trial numbers (1-based) of the successes
    first[i] = s[0]                                # geometric: trial of the first success
    fails_before_r[i] = s[r - 1] - r               # negative binomial: failures before r-th success

# (c): gamma-distributed mean, then a Poisson count with that mean
mu, kappa, runs = 10.0, 5.0, 200_000
lam_run = rng.gamma(shape=kappa, scale=mu / kappa, size=runs)   # mean mu, variance mu^2/kappa
N = rng.poisson(lam_run)

fig, (ax1, ax2) = plt.subplots(1, 2)
k = np.arange(1, 41)
ax1.bar(k, np.bincount(first, minlength=41)[1:41] / reps, color=SERIES[0], alpha=0.4, label="first trigger")
theory_line(ax1, k, p * (1 - p) ** (k - 1), label="$p(1-p)^{k-1}$")
ax1.set_xlabel("crossing number $k$"); ax1.set_ylabel("probability"); ax1.legend(fontsize=8)

k = np.arange(0, 36)
ax2.bar(k, np.bincount(N, minlength=36)[:36] / runs, color=SERIES[1], alpha=0.4, label="gamma-Poisson sim.")
# scipy's nbinom(n, q) counts failures before n successes with success prob q
theory_line(ax2, k, nbinom.pmf(k, kappa, kappa / (kappa + mu)), label="neg. binomial")
ax2.plot(k, poisson.pmf(k, mu), "o", color=SERIES[0], ms=3, label="Poisson(10)")
ax2.set_ylim(0, 0.17)
ax2.set_xlabel("counts $n$"); ax2.set_ylabel("probability"); ax2.legend(fontsize=8, loc="upper right")
fig.tight_layout()
savefig(fig, "ch02", "07_geometric_negbin")

save_numbers("ch02", "07_geometric_negbin", {
    "twoaGeoMean": float(first.mean()), "twoaGeoVar": float(first.var(ddof=1)),
    "twoaNBMean": float(fails_before_r.mean()), "twoaNBVar": float(fails_before_r.var(ddof=1)),
    "twoaNBMeanTh": r * (1 - p) / p, "twoaNBVarTh": r * (1 - p) / p**2,
    "twoaMixMean": float(N.mean()), "twoaMixVar": float(N.var(ddof=1)),
    "twoaMixVarTh": mu + mu**2 / kappa,
    "twoaGeoTail": float(np.mean(first > 30)), "twoaGeoTailTh": (1 - p) ** 30,
})
