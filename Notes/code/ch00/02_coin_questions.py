"""Three different questions asked of the same coin.

Question: (i) if we estimate p by the fraction of heads, how much does that estimate
scatter from experiment to experiment?  (ii) If the coin were fair, how surprising is
"8 or more heads in 10"? (the p-value)  (iii) Can we get (ii) by brute-force simulation,
and how fast does the simulation converge?
Computes: the sampling distribution of p_hat = k/n for n = 10 and n = 100 (true p = 0.8);
the exact tail P(K >= 8 | fair); a Monte Carlo estimate of the same tail for growing N.
Writes:   figures/ch00/coin_questions.pdf, results/ch00/02_coin_questions.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from math import comb

import matplotlib.pyplot as plt
import numpy as np
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

P_TRUE = 0.8
K_OBS, N_TOSS = 8, 10
setup(7.0, 3.0)
rng = rng_for("ch00", "02_coin_questions")

# (i) sampling distribution of the estimator p_hat = k/n
n_rep = 50000
phat10 = rng.binomial(10, P_TRUE, n_rep) / 10
phat100 = rng.binomial(100, P_TRUE, n_rep) / 100

# (ii) exact p-value: P(K >= 8 | p = 1/2), "this or more extreme in this direction"
p_exact = sum(comb(N_TOSS, j) for j in range(K_OBS, N_TOSS + 1)) / 2**N_TOSS

# (iii) Monte Carlo estimate of the same number: fraction of simulated fair
#       experiments with K >= 8, for N = 10 ... 10^6 experiments
Ns = np.unique(np.logspace(1, 6, 40).astype(int))
fair = rng.binomial(N_TOSS, 0.5, Ns.max())
hits = np.cumsum(fair >= K_OBS)                  # running count of "as or more extreme"
p_mc = hits[Ns - 1] / Ns
sigma_mc = np.sqrt(p_exact * (1 - p_exact) / Ns)   # binomial error of a fraction

fig, (a1, a2) = plt.subplots(1, 2)
a1.hist(phat10, bins=np.arange(-0.05, 1.1, 0.1), density=True, color=SERIES[0], alpha=0.6,
        label="$n=10$")
a1.hist(phat100, bins=np.arange(0.495, 1.0, 0.01), density=True, color=SERIES[1], alpha=0.7,
        label="$n=100$")
a1.axvline(P_TRUE, color="k", ls="--", lw=1.2)
a1.set(xlabel="estimate $\\hat p=k/n$", ylabel="density", xlim=(0.3, 1.05),
       title="spread of $\\hat p$ (true $p=0.8$)")
a1.legend(fontsize=8)
a2.fill_between(Ns, p_exact - sigma_mc, p_exact + sigma_mc, color=SERIES[0], alpha=0.25,
                label="$\\pm\\sqrt{p(1-p)/N}$")
a2.plot(Ns, p_mc, "o", color=SERIES[0], ms=3, label="Monte Carlo")
theory_line(a2, Ns, np.full(Ns.shape, p_exact), label="exact $P(K\\geq8)$")
a2.set(xscale="log", xlabel="number of simulated experiments $N$",
       ylabel="estimated $P(K\\geq 8\\mid$fair$)$", ylim=(0, 0.15),
       title="Monte Carlo: same number by simulation")
a2.legend(fontsize=8)
savefig(fig, "ch00", "coin_questions")

save_numbers("ch00", "02_coin_questions", {
    "QSdTen": phat10.std(), "QSdHundred": phat100.std(),
    "QSdTenTh": np.sqrt(P_TRUE * (1 - P_TRUE) / 10),
    "QSdHundredTh": np.sqrt(P_TRUE * (1 - P_TRUE) / 100),
    "QMeanTen": phat10.mean(),
    "QPexact": p_exact,
    "QPmcMillion": p_mc[-1],
    "QSigmaMillion": sigma_mc[-1],
    "QPmcHundred": p_mc[np.searchsorted(Ns, 100)],
    "QNhundred": Ns[np.searchsorted(Ns, 100)],
})
