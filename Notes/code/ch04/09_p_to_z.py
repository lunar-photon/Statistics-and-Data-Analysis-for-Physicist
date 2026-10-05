"""09_p_to_z.py -- translating a p-value into "sigmas", and why 5 sigma.

Question: (i) what one-sided and two-sided p-values correspond to Z = 1..6?
(ii) if a laboratory runs many independent background-only searches, how often
does at least one of them cross 3 sigma, and 5 sigma?  (iii) the owner's
"extraordinary claims" argument in numbers: if only a small fraction of searched
hypotheses are real, what fraction of 3 sigma and 5 sigma "discoveries" are real,
for a test with power 0.5 at each threshold?

Writes: figures/ch04/p_to_z.pdf, results/ch04/09_p_to_z.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "09_p_to_z")
setup()

Z = np.arange(1, 7)
p1 = stats.norm.sf(Z)
p2 = 2 * stats.norm.sf(Z)

# (ii) many independent searches: P(at least one >= Z) = 1 - (1 - p)^N
Nsearch = np.logspace(0, 6, 200)
any3 = 1 - (1 - stats.norm.sf(3)) ** Nsearch
any5 = 1 - (1 - stats.norm.sf(5)) ** Nsearch
N100_3 = 1 - (1 - stats.norm.sf(3)) ** 100
N100_5 = 1 - (1 - stats.norm.sf(5)) ** 100
# Monte Carlo check of the 100-search number at 3 sigma
M = 200_000
zmax = rng.standard_normal((M, 100)).max(1)
N100_3_mc = np.mean(zmax >= 3)

# (iii) fraction of discoveries that are real
pi1 = 1e-3            # one in a thousand searched hypotheses is real
power = 0.5
def purity(p_thr):
    return pi1 * power / (pi1 * power + (1 - pi1) * p_thr)
pur3, pur5 = purity(stats.norm.sf(3)), purity(stats.norm.sf(5))

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
zz = np.linspace(0, 6.5, 300)
axs[0].semilogy(zz, stats.norm.sf(zz), color=SERIES[0], label="one-sided $p=\\Phi(-Z)$")
axs[0].semilogy(zz, 2 * stats.norm.sf(zz), color=SERIES[1], ls="--", label="two-sided $p=2\\Phi(-Z)$")
for z in (3, 5):
    axs[0].axvline(z, color="0.5", lw=0.8, ls=":")
axs[0].set_xlabel("$Z$"); axs[0].set_ylabel("p-value"); axs[0].set_ylim(1e-10, 1); axs[0].legend(fontsize=7)
axs[1].semilogx(Nsearch, any3, color=SERIES[0], label=r"some search $\geq 3\sigma$")
axs[1].semilogx(Nsearch, any5, color=SERIES[1], label=r"some search $\geq 5\sigma$")
axs[1].plot([100], [N100_3_mc], "ko", ms=4, label="simulation")
axs[1].set_xlabel("number of independent searches")
axs[1].set_ylabel("probability"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch04", "p_to_z")

vals = {}
names = ["One", "Two", "Three", "Four", "Five", "Six"]
for z, a, b in zip(names, p1, p2):
    vals[f"FourAPz{z}One"] = a
    vals[f"FourAPz{z}Two"] = b
vals.update({
    "FourAPzAnyThree": f"{N100_3:.3f}", "FourAPzAnyThreeMC": f"{N100_3_mc:.3f}", "FourAPzAnyFive": N100_5,
    "FourAPzPurThree": f"{pur3:.3f}", "FourAPzPurFive": f"{pur5:.4f}",
    "FourAPzZfivepct": f"{stats.norm.isf(0.05):.3f}", "FourAPzZfivepctTwo": f"{stats.norm.isf(0.025):.3f}",
})
save_numbers("ch04", "09_p_to_z", vals)
print(p1, p2, N100_3, N100_3_mc, N100_5, pur3, pur5)
