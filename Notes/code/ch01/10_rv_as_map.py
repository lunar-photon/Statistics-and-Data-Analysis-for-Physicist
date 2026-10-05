"""A random variable is a map from outcomes to numbers.

Question: what does it mean, operationally, that X is a *function* on the sample space,
and that several different random variables can live on the same outcomes?

Computes:
  * Bernoulli(p) built explicitly as X(omega) = 1{omega <= p} with omega ~ Uniform(0,1)
    (the construction on AoS p.27);
  * for 10 coin flips (the outcome omega is the whole sequence), two different maps on the
    SAME outcomes: X = number of heads, Y = length of the longest run of heads;
    the pmf of X is compared with Binomial(10, 1/2).
Writes: figures/ch01/rv_as_map.pdf, results/ch01/10_rv_as_map.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch01", "10_rv_as_map")
setup(7.0, 3.0)

# --- 1. Bernoulli from a uniform outcome omega in [0,1] ------------------------------
p = 0.3
omega = rng.uniform(0.0, 1.0, size=100_000)       # the outcomes
X_bern = (omega <= p).astype(int)                 # the map omega -> {0,1}
bern_frac = X_bern.mean()

# --- 2. Ten fair coin flips: one outcome = one sequence of 10 H/T -------------------
n_exp, n_flip = 200_000, 10
flips = rng.integers(0, 2, size=(n_exp, n_flip))  # 1 = heads; each row is an omega


def longest_run(row):
    best = cur = 0
    for b in row:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


X = flips.sum(axis=1)                             # map 1: number of heads
Y = np.array([longest_run(r) for r in flips])     # map 2: longest run of heads

k = np.arange(n_flip + 1)
pmf_X = np.array([(X == j).mean() for j in k])
pmf_Y = np.array([(Y == j).mean() for j in k])
pmf_binom = stats.binom.pmf(k, n_flip, 0.5)

fig, axes = plt.subplots(1, 2, sharey=True)
ax = axes[0]
ax.bar(k, pmf_X, width=0.7, color=SERIES[0], alpha=0.8, label="simulated $X$")
ax.plot(k, pmf_binom, "ko", ms=4, mfc="none", label="Binomial$(10,1/2)$")
ax.set_xlabel("$X$ = number of heads")
ax.set_ylabel("fraction of outcomes")
ax.legend(loc="upper left")
ax = axes[1]
ax.bar(k, pmf_Y, width=0.7, color=SERIES[1], alpha=0.8, label="simulated $Y$")
ax.set_xlabel("$Y$ = longest run of heads")
ax.legend(loc="upper right")
fig.tight_layout()
savefig(fig, "ch01", "rv_as_map")

save_numbers("ch01", "10_rv_as_map", {
    "OneBBernP": p,
    "OneBBernFrac": round(bern_frac, 4),
    "OneBNOmega": "10^5",
    "OneBNExp": "2\\times10^{5}",
    "OneBMeanX": round(X.mean(), 3),
    "OneBMeanY": round(Y.mean(), 3),
    "OneBPXfive": round(pmf_X[5], 4),
    "OneBPXfiveExact": round(pmf_binom[5], 4),
    "OneBPYthree": round(pmf_Y[3], 4),
})
