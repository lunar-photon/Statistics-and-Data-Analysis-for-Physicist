"""04_pvalue_uniform.py -- what does a p-value look like as a random variable?

Question: (i) when H0 is true and the test statistic is continuous, is the p-value
uniform on (0,1)?  What happens when H0 is false?  (ii) for a discrete statistic
(Poisson counts, b = 3.2) is P(p <= alpha) <= alpha ("valid", super-uniform)?
(iii) in a population of experiments where most nulls are true, what fraction of
the "significant" results (p < 0.05), and of those with 0.04 < p < 0.05, come
from a true null?  This is why a p-value is not P(H0 | data).

Writes: figures/ch04/pvalue_hist.pdf, figures/ch04/pvalue_population.pdf,
        results/ch04/04_pvalue_uniform.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "04_pvalue_uniform")
setup()
M = 200_000
n, sigma = 10, 1.0

def pvals(mu, size):
    """one-sided p-values of H0: mu = 0 from n Gaussian readings with known sigma"""
    xbar = rng.normal(mu, sigma / np.sqrt(n), size=size)       # the sample mean is exactly N(mu, sigma^2/n)
    return stats.norm.sf(np.sqrt(n) * xbar / sigma)

p0 = pvals(0.0, M)
p1 = pvals(0.3, M)
p2 = pvals(0.8, M)
frac0 = np.mean(p0 < 0.05)
frac1 = np.mean(p1 < 0.05)
frac2 = np.mean(p2 < 0.05)
ks_stat, ks_p = stats.kstest(p0, "uniform")

# ---------- discrete case: Poisson counts ----------
b = 3.2
nn = rng.poisson(b, size=M)
pdisc = stats.poisson.sf(nn - 1, b)                             # P(N >= n_obs)
alphas = np.linspace(0, 1, 1001)
ecdf = np.searchsorted(np.sort(pdisc), alphas, side="right") / M
disc05 = np.mean(pdisc <= 0.05)

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
bins = np.linspace(0, 1, 41)
axs[0].hist(p0, bins=bins, density=True, histtype="step", lw=1.5, color=SERIES[0], label=r"$H_0$ true ($\mu=0$)")
axs[0].hist(p1, bins=bins, density=True, histtype="step", lw=1.5, color=SERIES[1], label=r"$\mu=0.3$")
axs[0].hist(p2, bins=bins, density=True, histtype="step", lw=1.5, color=SERIES[2], label=r"$\mu=0.8$")
axs[0].set_ylim(0, 6); axs[0].set_xlabel("p-value"); axs[0].set_ylabel("density")
axs[0].legend(fontsize=7)
axs[1].plot(alphas, ecdf, color=SERIES[0], lw=1.5, drawstyle="steps-post", label=r"Poisson, $b=3.2$")
theory_line(axs[1], alphas, alphas, label=r"uniform: $P(p\leq\alpha)=\alpha$")
axs[1].set_xlabel(r"$\alpha$"); axs[1].set_ylabel(r"$P(p\leq\alpha\mid H_0)$")
axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch04", "pvalue_hist")

# ---------- (iii) a population of experiments ----------
pi0 = 0.9                                                      # fraction of experiments where H0 is true
mu_eff = 0.5                                                   # effect size when H0 is false
Mpop = 1_000_000
is_null = rng.random(Mpop) < pi0
mu = np.where(is_null, 0.0, mu_eff)
pp = stats.norm.sf(np.sqrt(n) * rng.normal(mu, sigma / np.sqrt(n)) / sigma)
power = np.mean(pp[~is_null] < 0.05)
sig = pp < 0.05
frac_null_sig = np.mean(is_null[sig])
band = (pp > 0.04) & (pp < 0.05)
frac_null_band = np.mean(is_null[band])
# the same, analytically: P(H0 | p in band) = pi0 P(band|H0) / [pi0 P(band|H0) + (1-pi0) P(band|H1)]
zlo, zhi = stats.norm.isf(0.05), stats.norm.isf(0.04)
d = np.sqrt(n) * mu_eff / sigma
PbandH1 = stats.norm.cdf(zhi - d) - stats.norm.cdf(zlo - d)
frac_null_band_th = pi0 * 0.01 / (pi0 * 0.01 + (1 - pi0) * PbandH1)
frac_null_sig_th = pi0 * 0.05 / (pi0 * 0.05 + (1 - pi0) * stats.norm.sf(zlo - d))

fig, ax = plt.subplots(figsize=(5.6, 2.9))
bins = np.linspace(0, 0.2, 41)
ax.hist(pp[is_null], bins=bins, color=SERIES[0], alpha=0.6, label=r"$H_0$ true (90%)")
ax.hist(pp[~is_null], bins=bins, color=SERIES[1], alpha=0.6, label=r"$H_1$ true (10%)")
ax.axvline(0.05, color="k", lw=0.9, ls="--")
ax.set_xlabel("p-value"); ax.set_ylabel("number of experiments"); ax.legend(fontsize=8)
savefig(fig, "ch04", "pvalue_population")

save_numbers("ch04", "04_pvalue_uniform", {
    "FourAPUfracZero": f"{frac0:.4f}", "FourAPUfracOne": f"{frac1:.3f}", "FourAPUfracTwo": f"{frac2:.3f}",
    "FourAPUks": f"{ks_p:.2f}", "FourAPUdisc": f"{disc05:.4f}",
    "FourAPUpower": f"{power:.3f}",
    "FourAPUnullSig": f"{frac_null_sig:.3f}", "FourAPUnullSigTh": f"{frac_null_sig_th:.3f}",
    "FourAPUnullBand": f"{frac_null_band:.3f}", "FourAPUnullBandTh": f"{frac_null_band_th:.3f}",
})
print(frac0, frac1, frac2, ks_p, disc05, power, frac_null_sig, frac_null_sig_th, frac_null_band, frac_null_band_th)
