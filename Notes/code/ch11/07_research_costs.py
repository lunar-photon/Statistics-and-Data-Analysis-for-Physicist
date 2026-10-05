"""07_research_costs.py -- how many mock catalogues does a clustering covariance need?

Question: a survey's covariance matrix is estimated from N_S mock catalogues for a data vector of
N_D bins. How much does the finite number of mocks cost, for our scaled-down runs and for the
published analyses?
Computes: the Hartlap factor (N_S - N_D - 2)/(N_S - 1) that de-biases the inverse covariance, and
the fractional scatter of a parameter variance, eps = sqrt(2 / (N_S - N_D - 4)) (Taylor et al. 2013,
eq. 55), as functions of N_S for a few data-vector sizes; the same two numbers for our acoustic-peak
mocks (1500 mocks, 16 bins), for the 2005 detection (1278 mocks, 20 bins), and the number of mocks a
5% error on the error bar requires (Taylor et al. 2013, eq. 58).
Writes: figures/ch11/research_mocks.pdf, results/ch11/07_research_costs.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES


def hartlap(ns, nd):
    return (ns - nd - 2) / (ns - 1)


def eps(ns, nd):
    """Fractional scatter of a parameter variance from a Hartlap-corrected precision matrix."""
    return np.sqrt(2.0 / (ns - nd - 4))


ours = (1500, 16)        # code/ch11/04_bao_mocks.py: covariance mocks, xi bins
eis = (1278, 20)         # Eisenstein et al. 2005: PTHalos mocks, xi bins
ns = np.logspace(np.log10(30), np.log10(5000), 300)

setup(6.4, 3.0)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
for i, nd in enumerate((16, 40, 100)):
    ok = ns > nd + 6
    axs[0].plot(ns[ok], hartlap(ns[ok], nd), color=SERIES[i], label=fr"$N_D={nd}$")
    axs[1].plot(ns[ok], 0.5 * eps(ns[ok], nd) * 100, color=SERIES[i], label=fr"$N_D={nd}$")
for a in axs:
    a.set_xscale("log")
    a.set_xlabel(r"number of mocks $N_S$")
    for v in (1000, 2045):
        a.axvline(v, color="0.6", ls=":", lw=1.0)
axs[0].plot(*[ours[0]], hartlap(*ours), "o", color="k", ms=4)
axs[0].plot(*[eis[0]], hartlap(*eis), "s", color="k", ms=4, mfc="none")
axs[1].plot(ours[0], 50 * eps(*ours), "o", color="k", ms=4, label="our BAO mocks")
axs[1].plot(eis[0], 50 * eps(*eis), "s", color="k", ms=4, mfc="none", label="Eisenstein 2005")
axs[0].set_ylabel("Hartlap factor")
axs[0].set_ylim(0, 1.02)
axs[1].set_ylabel(r"scatter of the error bar [%]")
axs[1].set_yscale("log")
axs[0].legend(fontsize=7, loc="lower right")
axs[1].legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "ch11", "research_mocks")

save_numbers("ch11", "07_research_costs", {
    "ResOursHartlap": f"{hartlap(*ours):.3f}", "ResOursEps": f"{100 * eps(*ours):.1f}",
    "ResOursSig": f"{50 * eps(*ours):.1f}",
    "ResEisHartlap": f"{hartlap(*eis):.3f}", "ResEisEps": f"{100 * eps(*eis):.1f}",
    "ResEisSig": f"{50 * eps(*eis):.1f}",
    "ResNsFiveSixteen": f"{int(np.ceil(2 / 0.1 ** 2 + 16 + 4))}",
    "ResNsFiveHundred": f"{int(np.ceil(2 / 0.1 ** 2 + 100 + 4))}",
    "ResHartlapHundredK": f"{hartlap(1000, 100):.3f}",
})
