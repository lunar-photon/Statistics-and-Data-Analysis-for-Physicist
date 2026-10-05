"""A single trial: the Bernoulli distribution built from a uniform number.

Question:  if a scintillator paddle fires with probability eps for each muon,
           does the long-run fraction of 'fired' outcomes really settle at eps,
           and are the sample mean and variance p and p(1-p)?
Computes:  Bernoulli draws X = [U <= p] from uniforms U (the sample-space
           construction Omega = [0,1]); the running fraction of successes for
           three values of p; the sample mean and variance of 10^6 draws.
Writes:    figures/ch02/01_bernoulli.pdf, results/ch02/01_bernoulli.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES

setup(6.4, 2.8)
rng = rng_for("ch02", "01_bernoulli")

# --- one trial = one uniform number compared with p -----------------------
def bernoulli(p, size):
    u = rng.uniform(0.0, 1.0, size)      # omega, uniform on [0,1]
    return (u <= p).astype(int)          # X(omega) = 1 if omega <= p else 0

# --- running fraction of successes ----------------------------------------
n = 5000
k = np.arange(1, n + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2.2, 1]})
for c, p in zip(SERIES, [0.2, 0.5, 0.9]):
    x = bernoulli(p, n)
    ax1.plot(k, np.cumsum(x) / k, color=c, lw=1.1, label=f"$p={p}$")
    ax1.axhline(p, color="k", ls="--", lw=0.8)
ax1.set_xscale("log")
ax1.set_ylim(0, 1)
ax1.set_xlabel("number of trials $n$")
ax1.set_ylabel("fraction of successes")
ax1.legend(loc="lower right", ncol=3)

# --- pmf of one trial, with the sample frequencies of 10^6 draws -----------
p = 0.9
big = bernoulli(p, 1_000_000)
ax2.bar([0, 1], [1 - p, p], width=0.5, color=SERIES[0], alpha=0.35, label="$p^x(1-p)^{1-x}$")
ax2.plot([0, 1], [np.mean(big == 0), np.mean(big == 1)], "o", color=SERIES[1], label="simulated")
ax2.set_xticks([0, 1])
ax2.set_xlabel("$x$")
ax2.set_ylabel("probability")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="center left", fontsize=8)
fig.tight_layout()
savefig(fig, "ch02", "01_bernoulli")

save_numbers("ch02", "01_bernoulli", {
    "twoaBernP": p,
    "twoaBernMean": float(big.mean()),
    "twoaBernVar": float(big.var()),
    "twoaBernVarTheory": p * (1 - p),
})
