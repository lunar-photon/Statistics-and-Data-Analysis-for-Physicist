"""23_prior_sensitivity.py -- vague priors: harmless for the posterior, decisive for the evidence.

Question: two 'uninformative' priors give practically the same posterior.  Do
they give the same evidence?

Computes
  * Kruschke's coin, z = 65 heads in N = 100: Bayes factor of 'must be fair'
    Beta(500,500) against 'anything goes' with a Beta(1,1) and with a
    Beta(0.01,0.01) prior, and the 95% HDIs of the two posteriors;
  * the same comparison after informing both models with 10% of the data
    (6 heads in 10), Kruschke's remedy;
  * Lambert's binomial sample (15 counts out of 10 trials each) with a
    Beta(a,a) prior: ln evidence and posterior mean and sd as a varies.

Writes: figures/ch05/prior_sensitivity.pdf, results/ch05/23_prior_sensitivity.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import betaln, gammaln

setup()


def ln_ev(z, N, a, b):                     # probability of a particular sequence, Beta(a,b) prior
    return betaln(z + a, N - z + b) - betaln(a, b)


def hdi(dist, mass=0.95):
    """Shortest interval containing `mass` of a unimodal distribution."""
    width = lambda lo: dist.ppf(lo + mass) - dist.ppf(lo)
    lo = optimize.minimize_scalar(width, bounds=(1e-9, 1 - mass - 1e-9), method="bounded").x
    return dist.ppf(lo), dist.ppf(lo + mass)


z, N = 65, 100
bf_unif = np.exp(ln_ev(z, N, 500, 500) - ln_ev(z, N, 1, 1))
bf_hald = np.exp(ln_ev(z, N, 500, 500) - ln_ev(z, N, 0.01, 0.01))
h_unif = hdi(stats.beta(z + 1, N - z + 1))
h_hald = hdi(stats.beta(z + 0.01, N - z + 0.01))

# Kruschke's remedy: inform both models with the same 10% of the data (6 heads in 10)
z6, n10 = 6, 10
bf_inf_unif = np.exp(ln_ev(z - z6, N - n10, 500 + z6, 500 + n10 - z6) - ln_ev(z - z6, N - n10, 1 + z6, 1 + n10 - z6))
bf_inf_hald = np.exp(ln_ev(z - z6, N - n10, 500 + z6, 500 + n10 - z6) - ln_ev(z - z6, N - n10, 0.01 + z6, 0.01 + n10 - z6))

# Lambert: 15 observations, each a count out of n = 10 trials
counts = np.array([3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]); n = 10
S, T = counts.sum(), counts.size * n
ln_binom = np.sum(gammaln(n + 1) - gammaln(counts + 1) - gammaln(n - counts + 1))
a_grid = np.logspace(-3, 3, 200)
lnZ = ln_binom + betaln(S + a_grid, T - S + a_grid) - betaln(a_grid, a_grid)
pm = (S + a_grid) / (T + 2 * a_grid)
psd = np.sqrt(pm * (1 - pm) / (T + 2 * a_grid + 1))

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
th = np.linspace(0.3, 0.7, 400)
for c, a in zip(SERIES, [0.01, 1.0, 10.0]):
    axes[0].plot(th, stats.beta.pdf(th, S + a, T - S + a), color=c, label=rf"Beta$({a:g},{a:g})$ prior")
axes[0].set_xlabel(r"success probability $\theta$"); axes[0].set_ylabel("posterior density")
axes[0].legend(fontsize=7, loc="upper left", frameon=False)
axes[0].set_ylim(0, 14.5)
axes[1].semilogx(a_grid, lnZ, color="k")
for c, a in zip(SERIES, [0.01, 1.0, 10.0]):
    axes[1].plot(a, ln_binom + betaln(S + a, T - S + a) - betaln(a, a), "o", color=c)
axes[1].set_xlabel(r"prior parameter $a$ of Beta$(a,a)$"); axes[1].set_ylabel(r"$\ln p(\mathrm{data})$")
savefig(fig, "ch05", "prior_sensitivity")

lz = lambda a: ln_binom + betaln(S + a, T - S + a) - betaln(a, a)
save_numbers("ch05", "23_prior_sensitivity", {
    "FiveCHalUnifBF": bf_unif, "FiveCHalHaldBF": bf_hald,
    "FiveCInfUnifBF": bf_inf_unif, "FiveCInfHaldBF": bf_inf_hald,
    "FiveCHdiUnifLo": h_unif[0], "FiveCHdiUnifHi": h_unif[1],
    "FiveCHdiHaldLo": h_hald[0], "FiveCHdiHaldHi": h_hald[1],
    "FiveCLamS": int(S), "FiveCLamT": int(T),
    "FiveCLamLnZa": lz(0.01), "FiveCLamLnZb": lz(1.0), "FiveCLamLnZc": lz(10.0),
    "FiveCLamZratio": np.exp(lz(1.0) - lz(0.01)),
    "FiveCLamMeanA": (S + 0.01) / (T + 0.02), "FiveCLamMeanB": (S + 1) / (T + 2),
    "FiveCLamSdA": psd[np.argmin(abs(a_grid - 0.01))], "FiveCLamSdB": psd[np.argmin(abs(a_grid - 1))],
})
print(bf_unif, bf_hald, h_unif, h_hald, lz(0.01), lz(1), lz(10))
