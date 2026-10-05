"""Histogram bins: the multinomial distribution and its negative correlations.

Question:  if N events are dropped independently into m histogram bins with
           probabilities p_1..p_m, what is the covariance of two bin counts?
           Theory: Var n_i = N p_i (1-p_i),  Cov(n_i, n_j) = -N p_i p_j.
           And if N itself is Poisson, do the bins become independent?
Computes:  30000 histograms of N=50 events drawn from a falling spectrum in
           m=5 bins; the sample covariance matrix vs theory; the correlation of
           bins 1 and 2 with fixed N and with N ~ Poisson(50).
Writes:    figures/ch02/03_multinomial.pdf, results/ch02/03_multinomial.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES

setup(6.4, 2.9)
rng = rng_for("ch02", "03_multinomial")

# a falling "energy spectrum": bin probabilities proportional to exp(-bin/2)
m, N, reps = 5, 50, 30_000
p = np.exp(-np.arange(m) / 2.0)
p /= p.sum()

def histogram_of_events(N):
    """Drop N events into bins one by one: each event picks bin j with prob p_j."""
    u = rng.uniform(size=N)
    bins = np.searchsorted(np.cumsum(p), u)      # the bin whose cumulative interval contains u
    return np.bincount(bins, minlength=m)

H = np.array([histogram_of_events(N) for _ in range(reps)])        # fixed N
Hp = np.array([histogram_of_events(rng.poisson(N)) for _ in range(reps)])  # N ~ Poisson

C_sim = np.cov(H, rowvar=False)
C_th = N * (np.diag(p) - np.outer(p, p))          # multinomial covariance matrix
r_fixed = np.corrcoef(H[:, 0], H[:, 1])[0, 1]
r_pois = np.corrcoef(Hp[:, 0], Hp[:, 1])[0, 1]
r_th = -np.sqrt(p[0] * p[1] / ((1 - p[0]) * (1 - p[1])))

fig, (ax1, ax2) = plt.subplots(1, 2)
jit = lambda a: a + rng.uniform(-0.3, 0.3, a.size)     # jitter so integer points do not overlap
ax1.plot(jit(H[:3000, 0]), jit(H[:3000, 1]), ".", ms=1.5, color=SERIES[0], alpha=0.5)
ax1.set_xlabel("$n_1$ (fixed $N=50$)")
ax1.set_ylabel("$n_2$")
ax1.set_title(f"correlation {r_fixed:.2f}")
ax2.plot(jit(Hp[:3000, 0]), jit(Hp[:3000, 1]), ".", ms=1.5, color=SERIES[1], alpha=0.5)
ax2.set_xlabel(r"$n_1$ ($N\sim$ Poisson(50))")
ax2.set_ylabel("$n_2$")
ax2.set_title(f"correlation {r_pois:.2f}")
for ax in (ax1, ax2):
    ax.set_xlim(5, 35); ax.set_ylim(0, 25)
fig.tight_layout()
savefig(fig, "ch02", "03_multinomial")

save_numbers("ch02", "03_multinomial", {
    "twoaMultPone": p[0], "twoaMultPtwo": p[1],
    "twoaMultVarOneSim": C_sim[0, 0], "twoaMultVarOneTh": C_th[0, 0],
    "twoaMultCovSim": C_sim[0, 1], "twoaMultCovTh": C_th[0, 1],
    "twoaMultCorrSim": r_fixed, "twoaMultCorrTh": r_th,
    "twoaMultCorrPois": r_pois,
    "twoaMultRowSumVar": float(np.var(H.sum(axis=1))),
})
print(np.round(C_sim, 2)); print(np.round(C_th, 2))
