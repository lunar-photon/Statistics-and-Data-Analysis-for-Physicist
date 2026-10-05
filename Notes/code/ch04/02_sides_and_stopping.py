"""02_sides_and_stopping.py -- which outcomes count as "this or even more in this direction"?

Question: (i) a coin gives 17 heads in 20 tosses.  What are the one- and two-sided
p-values of H0: fair coin?  (ii) A coin gives 7 heads in 24 tosses.  The p-value
depends on the cloud of imaginary repetitions: was N = 24 fixed in advance, or
did the experimenter stop at the 7th head?  (iii) Cowan's stopping rule "toss
until at least 3 heads and 3 tails": how probable is it to need 20 or more tosses?

Computes: exact binomial and negative-binomial tail sums, each checked by
simulating the stated experimental protocol many times.

Writes: figures/ch04/coin_sides.pdf, figures/ch04/stopping_rules.pdf,
        results/ch04/02_sides_and_stopping.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "02_sides_and_stopping")
setup()
M = 400_000

# ---------- (i) 17 heads in 20 tosses ----------
N1, h1 = 20, 17
p_one = stats.binom.sf(h1 - 1, N1, 0.5)                   # P(n_h >= 17)
p_two = p_one + stats.binom.cdf(N1 - h1, N1, 0.5)          # + P(n_h <= 3), symmetric tail
sim = rng.binomial(N1, 0.5, size=M)
p_two_mc = np.mean(np.abs(sim - N1 / 2) >= abs(h1 - N1 / 2))

# ---------- (ii) 7 heads in 24 tosses, two stopping intentions ----------
N2, z2 = 24, 7
p_fixN = stats.binom.cdf(z2, N2, 0.5)                      # P(z <= 7 | N = 24)
# fixed z: stop at the 7th head.  "As extreme or more": needing N >= 24 tosses,
# i.e. at most 6 heads among the first 23 tosses.
p_fixz = stats.binom.cdf(z2 - 1, N2 - 1, 0.5)
# simulate the fixed-z protocol: number of tosses to get the 7th head = 7 + (tails before it)
Nstop = z2 + rng.negative_binomial(z2, 0.5, size=M)
p_fixz_mc = np.mean(Nstop >= N2)
p_fixN_mc = np.mean(rng.binomial(N2, 0.5, size=M) <= z2)

# ---------- (iii) Cowan: toss until >= 3 heads and >= 3 tails ----------
# needing >= 20 tosses <=> after 19 tosses, heads <= 2 or tails <= 2
p_cowan = 2 * stats.binom.cdf(2, 19, 0.5)
seqs = rng.integers(0, 2, size=(200_000, 60), dtype=np.int8)
h = np.cumsum(seqs, axis=1); t = np.arange(1, 61) - h
done = (h >= 3) & (t >= 3)
nstop = done.argmax(axis=1) + 1
p_cowan_mc = np.mean(nstop >= 20)

# ---------- figure 1: two-sided tail for 17/20 ----------
k = np.arange(0, N1 + 1)
pmf = stats.binom.pmf(k, N1, 0.5)
ext = np.abs(k - 10) >= 7
fig, ax = plt.subplots(figsize=(5.6, 2.8))
ax.bar(k[~ext], pmf[~ext], color=SERIES[0], alpha=0.55, width=0.8, label="fair coin, $N=20$")
ax.bar(k[ext], pmf[ext], color=SERIES[1], width=0.8, label=r"$|n_h-10|\geq 7$")
ax.set_xlabel(r"number of heads $n_h$"); ax.set_ylabel("probability")
ax.set_yscale("log"); ax.set_ylim(5e-7, 3)      # log scale: the thin tails must be visible
ax.set_xticks(range(0, 21, 2)); ax.legend(loc="upper left", fontsize=8)
savefig(fig, "ch04", "coin_sides")

# ---------- figure 2: the two clouds of imaginary repetitions ----------
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
kz = np.arange(0, N2 + 1)
pz = stats.binom.pmf(kz, N2, 0.5)
axs[0].bar(kz, pz, color=np.where(kz <= z2, SERIES[1], SERIES[0]), alpha=0.8, width=0.8)
axs[0].set_title(r"fixed $N=24$: distribution of heads $z$")
axs[0].set_xlabel(r"$z$"); axs[0].set_ylabel("probability")
kn = np.arange(z2, 45)
pn = stats.nbinom.pmf(kn - z2, z2, 0.5)                    # P(N = n) for the 7th head on toss n
axs[1].bar(kn, pn, color=np.where(kn >= N2, SERIES[1], SERIES[0]), alpha=0.8, width=0.8)
axs[1].set_title(r"fixed $z=7$: distribution of tosses $N$")
axs[1].set_xlabel(r"$N$")
fig.tight_layout()
savefig(fig, "ch04", "stopping_rules")

save_numbers("ch04", "02_sides_and_stopping", {
    "FourACoinPone": f"{p_one:.5f}", "FourACoinPtwo": f"{p_two:.4f}", "FourACoinPtwoMC": f"{p_two_mc:.4f}",
    "FourAKrFixN": f"{p_fixN:.4f}", "FourAKrFixNMC": f"{p_fixN_mc:.4f}",
    "FourAKrFixZ": f"{p_fixz:.4f}", "FourAKrFixZMC": f"{p_fixz_mc:.4f}",
    "FourACowStop": p_cowan, "FourACowStopMC": p_cowan_mc,
})
print(p_one, p_two, p_two_mc, p_fixN, p_fixN_mc, p_fixz, p_fixz_mc, p_cowan, p_cowan_mc)
