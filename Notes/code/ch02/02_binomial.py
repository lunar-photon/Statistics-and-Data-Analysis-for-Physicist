"""Many successive trials: the binomial distribution from counting paths.

Question:  if we repeat a Bernoulli(p) trial n times and count the successes,
           does the histogram of that count match C(n,k) p^k (1-p)^(n-k)?
Computes:  (a) 20000 repetitions of "n trials, count successes" for six (n,p)
           pairs, compared with the pmf; (b) a 3-out-of-4 coincidence trigger
           with per-layer efficiency 0.95, exact and simulated; (c) the sample
           mean and variance for n=20, p=0.2 against np and np(1-p).
Writes:    figures/ch02/02_binomial.pdf, results/ch02/02_binomial.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from common import setup, savefig, save_numbers, rng_for, SERIES

setup(6.4, 4.6)
rng = rng_for("ch02", "02_binomial")
reps = 20_000

def binomial_by_trials(n, p, reps):
    """Run 'reps' experiments; each is n independent Bernoulli(p) trials."""
    trials = rng.uniform(size=(reps, n)) <= p    # True = success
    return trials.sum(axis=1)                     # number of successes per experiment

# left column: p = 0.5 and growing n;  right column: n = 20 and growing p
cases = [(5, 0.5), (10, 0.5), (20, 0.5), (20, 0.1), (20, 0.2), (20, 0.6)]
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
for ax, (n, p) in zip(axes.T.ravel(), cases):
    k = binomial_by_trials(n, p, reps)
    ks = np.arange(0, 21)
    freq = np.bincount(k, minlength=21)[:21] / reps
    ax.bar(ks, freq, width=0.8, color=SERIES[0], alpha=0.45, label="simulated")
    ax.plot(ks, binom.pmf(ks, n, p), "k_", ms=9, mew=1.6, label="pmf")
    ax.text(0.97, 0.9, f"$n={n},\\ p={p}$", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.set_ylim(0, 0.42)
for ax in axes[-1]:
    ax.set_xlabel("number of successes $k$")
for ax in axes[:, 0]:
    ax.set_ylabel("$P(k)$")
axes[0, 0].legend(loc="center right", fontsize=8, bbox_to_anchor=(1.0, 0.45))
fig.tight_layout()
savefig(fig, "ch02", "02_binomial")

# --- (b) coincidence trigger: at least 3 of 4 layers fire ------------------
eps, layers = 0.95, 4
exact = binom.pmf(3, layers, eps) + binom.pmf(4, layers, eps)
hits = binomial_by_trials(layers, eps, 1_000_000)
sim = np.mean(hits >= 3)

# --- (c) moments ------------------------------------------------------------
k = binomial_by_trials(20, 0.2, 200_000)

save_numbers("ch02", "02_binomial", {
    "twoaTrigExact": exact,
    "twoaTrigSim": sim,
    "twoaTrigFour": eps**4,
    "twoaBinMean": float(k.mean()),
    "twoaBinVar": float(k.var(ddof=1)),
})
