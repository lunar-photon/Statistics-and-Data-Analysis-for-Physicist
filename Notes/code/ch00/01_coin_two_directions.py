"""The two directions of probability, on one coin.

Question: a coin is either fair (p = 0.5) or biased (p = 0.8).
  * Prediction (forward):  for each coin, how often does it give k heads in n = 10 tosses?
  * Inference (backward):  we saw k heads; how probable is each coin now?

Computes: the exact binomial probabilities P(k | coin), a simulation of many 10-toss
experiments from each coin (to check the forward numbers), and the posterior
P(biased | k) for a 50/50 prior.
Writes:   figures/ch00/coin_two_directions.pdf, results/ch00/01_coin_two_directions.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from math import comb

import matplotlib.pyplot as plt
import numpy as np
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

N_TOSS = 10          # tosses per experiment
P_FAIR, P_BIAS = 0.5, 0.8
PRIOR_BIAS = 0.5     # prior probability that we were handed the biased coin
N_EXP = 20000        # simulated experiments per coin
K_OBS = 8            # the outcome we actually observed

setup(7.0, 3.0)
rng = rng_for("ch00", "01_coin_two_directions")
k = np.arange(N_TOSS + 1)


def binom_pmf(k, n, p):
    """P(k heads in n tosses | p): the forward (prediction) question."""
    return np.array([comb(n, int(j)) * p**j * (1 - p) ** (n - j) for j in k])


# ---- forward direction: each coin produces a distribution of outcomes ----
pmf_fair, pmf_bias = binom_pmf(k, N_TOSS, P_FAIR), binom_pmf(k, N_TOSS, P_BIAS)
sim_fair = rng.binomial(N_TOSS, P_FAIR, size=N_EXP)      # 20000 experiments of 10 tosses
sim_bias = rng.binomial(N_TOSS, P_BIAS, size=N_EXP)
freq_fair = np.bincount(sim_fair, minlength=N_TOSS + 1) / N_EXP
freq_bias = np.bincount(sim_bias, minlength=N_TOSS + 1) / N_EXP

# ---- backward direction: Bayes on the observed column of the tree ----
num_bias = pmf_bias * PRIOR_BIAS                  # path "biased, then k heads"
num_fair = pmf_fair * (1 - PRIOR_BIAS)            # path "fair, then k heads"
post_bias = num_bias / (num_bias + num_fair)      # normalise over the paths that end at k

fig, (a1, a2) = plt.subplots(1, 2)
w = 0.4
a1.bar(k - w / 2, freq_fair, w, color=SERIES[0], label="fair coin, simulated")
a1.bar(k + w / 2, freq_bias, w, color=SERIES[1], label="biased coin, simulated")
theory_line(a1, k, pmf_fair, label="exact binomial", marker="o", ms=3)
theory_line(a1, k, pmf_bias, label=None, marker="o", ms=3)
a1.axvline(K_OBS, color="0.5", lw=0.8, ls=":")
a1.set(xlabel="number of heads $k$ in 10 tosses", ylabel="$P(k\\mid$coin$)$",
       title="prediction: coin $\\to$ outcomes")
a1.legend(loc="upper left", fontsize=7)
a2.plot(k, post_bias, "o-", color=SERIES[2])
a2.axhline(PRIOR_BIAS, color="0.5", lw=0.8, ls=":")
a2.axvline(K_OBS, color="0.5", lw=0.8, ls=":")
a2.set(xlabel="observed number of heads $k$", ylabel="$P($biased$\\mid k)$",
       title="inference: outcome $\\to$ coin", ylim=(-0.03, 1.03))
savefig(fig, "ch00", "coin_two_directions")

save_numbers("ch00", "01_coin_two_directions", {
    "CoinPkFair": pmf_fair[K_OBS],            # 45/1024
    "CoinPkBias": pmf_bias[K_OBS],
    "CoinSimFair": freq_fair[K_OBS],
    "CoinSimBias": freq_bias[K_OBS],
    "CoinSimErr": np.sqrt(pmf_fair[K_OBS] * (1 - pmf_fair[K_OBS]) / N_EXP),
    "CoinPostBias": post_bias[K_OBS],
    "CoinPostBiasFive": post_bias[5],
    "CoinLR": pmf_bias[K_OBS] / pmf_fair[K_OBS],
    "CoinNexp": N_EXP,
})
