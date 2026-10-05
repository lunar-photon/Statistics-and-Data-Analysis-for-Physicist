"""Expectations by simulation: the lazy statistician, the broken stick, the mean as a minimiser,
and the law of total variance in a two-stage (hierarchical) counting experiment.

Questions and exact answers derived in the text:
  * X ~ Uniform(0,1):  E[e^X] = e - 1;   longer piece of a stick broken at X:  E = 3/4.
  * X ~ Exponential(1): M -> E[(X-M)^2] is minimised at the mean (1), M -> E|X-M| at the median (ln 2).
  * Q ~ Uniform(0,1), K | Q ~ Binomial(n, Q) (pixel with a random efficiency Q):
        E[K] = n/2,  Var[K] = E[Var(K|Q)] + Var(E[K|Q]) = n/6 + n^2/12,
    far larger than the Binomial(n, 1/2) variance n/4.
Writes: figures/ch01/hierarchical_counts.pdf, figures/ch01/mean_minimiser.pdf,
        results/ch01/16_expectation_checks.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch01", "16_expectation_checks")
setup(7.0, 3.0)

N = 1_000_000
U = rng.uniform(size=N)
E_expU = np.mean(np.exp(U))
E_stick = np.mean(np.maximum(U, 1 - U))

# ---- hierarchical counting experiment --------------------------------------------
n = 20
Q = rng.uniform(size=N)
K = rng.binomial(n, Q)
K_plain = rng.binomial(n, 0.5, size=N)
var_exact = n / 6 + n**2 / 12

k = np.arange(n + 1)
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.bar(k - 0.2, np.bincount(K, minlength=n + 1) / N, width=0.4, color=SERIES[0],
       label="$Q\\sim$ Unif(0,1), $K|Q\\sim$ Bin$(20,Q)$")
ax.bar(k + 0.2, np.bincount(K_plain, minlength=n + 1) / N, width=0.4, color=SERIES[1],
       label="$K\\sim$ Bin$(20,1/2)$")
ax.axhline(1 / (n + 1), color="k", ls="--", lw=1.2, label="$1/21$")
ax.set_xlabel("counts $K$"); ax.set_ylabel("probability")
ax.set_ylim(0, 0.24)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "ch01", "hierarchical_counts")

# ---- mean and median as minimisers ---------------------------------------------------
X = rng.exponential(1.0, size=200_000)
M = np.linspace(0, 2.5, 251)
sq = np.array([np.mean((X - m)**2) for m in M])
ab = np.array([np.mean(np.abs(X - m)) for m in M])
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.plot(M, sq, color=SERIES[0], label="$\\mathrm{E}[(X-M)^2]$")
ax.plot(M, ab, color=SERIES[1], label="$\\mathrm{E}|X-M|$")
ax.axvline(1.0, color=SERIES[0], ls=":", lw=1)
ax.axvline(np.log(2), color=SERIES[1], ls=":", lw=1)
ax.set_xlabel("$M$"); ax.set_ylabel("expected loss")
ax.set_ylim(0, 3)
ax.legend(fontsize=8, loc="upper center")
fig.tight_layout()
savefig(fig, "ch01", "mean_minimiser")

save_numbers("ch01", "16_expectation_checks", {
    "OneBExpUSim": f"{E_expU:.4f}", "OneBExpUExact": f"{np.e - 1:.4f}",
    "OneBStickSim": f"{E_stick:.4f}",
    "OneBHierMean": f"{K.mean():.3f}",
    "OneBHierVar": f"{K.var():.2f}", "OneBHierVarExact": f"{var_exact:.2f}",
    "OneBPlainVar": f"{K_plain.var():.3f}",
    "OneBArgminSq": f"{M[np.argmin(sq)]:.2f}",
    "OneBArgminAbs": f"{M[np.argmin(ab)]:.2f}",
    "OneBMinSq": f"{sq.min():.3f}",
})
