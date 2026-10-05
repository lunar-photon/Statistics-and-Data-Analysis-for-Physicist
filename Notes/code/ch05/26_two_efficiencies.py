"""26_two_efficiencies.py -- estimating a difference versus comparing models.

Question: a trigger was upgraded.  Before: x1 = 28 of n1 = 40 test pulses fired
it.  After: x2 = 35 of n2 = 40.  Two different questions can be asked:
  (i)  estimation: what is the posterior of tau = p2 - p1, and P(p2 > p1)?
  (ii) comparison: is 'the efficiency is unchanged' (one p) or 'it changed'
       (independent p1, p2) the better model?
Both use flat priors.

Computes the posterior of tau by simulation (Beta posteriors, Wasserman
Example 11.7), P(p2 > p1 | data), and the Bayes factor 'same' vs 'different'
from the exact evidences.

Writes: figures/ch05/two_efficiencies.pdf, results/ch05/26_two_efficiencies.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betaln

setup()
rng = rng_for("ch05", "26_two_efficiencies")
n1, x1, n2, x2 = 40, 28, 40, 35
B = 200_000

p1 = rng.beta(x1 + 1, n1 - x1 + 1, B)          # flat prior -> Beta(x+1, n-x+1) posterior
p2 = rng.beta(x2 + 1, n2 - x2 + 1, B)
tau = p2 - p1
P_up = np.mean(tau > 0)
lo, hi = np.quantile(tau, [0.025, 0.975])

# evidences (the binomial coefficients are common to both models and cancel in the ratio)
lnZ_same = betaln(x1 + x2 + 1, n1 + n2 - x1 - x2 + 1)
lnZ_diff = betaln(x1 + 1, n1 - x1 + 1) + betaln(x2 + 1, n2 - x2 + 1)
B_same = np.exp(lnZ_same - lnZ_diff)

fig, ax = plt.subplots(figsize=(5.4, 2.8))
ax.hist(tau, bins=120, density=True, color=SERIES[0], alpha=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(r"$\tau=p_2-p_1$ (change of efficiency)"); ax.set_ylabel("posterior density")
savefig(fig, "ch05", "two_efficiencies")

save_numbers("ch05", "26_two_efficiencies", {
    "FiveCEffNone": n1, "FiveCEffXone": x1, "FiveCEffNtwo": n2, "FiveCEffXtwo": x2,
    "FiveCEffPup": P_up, "FiveCEffLo": lo, "FiveCEffHi": hi, "FiveCEffMean": tau.mean(),
    "FiveCEffBsame": B_same, "FiveCEffPsame": B_same / (1 + B_same), "FiveCEffDraws": B,
})
print(P_up, lo, hi, B_same)
