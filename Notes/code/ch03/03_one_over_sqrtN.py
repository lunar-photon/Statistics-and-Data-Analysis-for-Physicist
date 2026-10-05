"""03_one_over_sqrtN.py -- why do errors shrink like 1/sqrt(N)?  (and when do they not?)

Question: does the variance of the sample mean fall as sigma^2/N for very different
parent distributions (log-log slope -1)?  How fast does the standardised mean become
Gaussian (CLT)?  What happens for a Cauchy (Breit--Wigner) parent with no variance?
How conservative is Chebyshev's bound in the coin example of the text?

Computes: Var(mean) versus N for Uniform, Exponential, Poisson parents (fit of the
slope); the interquartile half-width of the mean for Exponential and Cauchy parents;
histograms of the standardised mean of exponentials for N = 1, 2, 10, 50;
the exact binomial probability P(0.4 <= mean <= 0.6) for n = 84 fair coin flips.
Writes: figures/ch03/var_mean_vs_N.pdf, figures/ch03/clt_exponential.pdf,
        results/ch03/03_one_over_sqrtN.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch03", "03_one_over_sqrtN")
setup()
M = 20_000
Ns = 2 ** np.arange(0, 11)                       # 1, 2, 4, ..., 1024
parents = {
    "Uniform(0,1)": (lambda s: rng.uniform(0, 1, s), 1 / 12),
    "Exponential(1)": (lambda s: rng.exponential(1.0, s), 1.0),
    "Poisson(3)": (lambda s: rng.poisson(3.0, s).astype(float), 3.0),
}
var_of_mean = {k: [] for k in parents}
iqr = {"Exponential(1)": [], "Cauchy": []}
for N in Ns:
    for k, (draw, _) in parents.items():
        m = draw((M, N)).mean(axis=1)
        var_of_mean[k].append(m.var(ddof=1))
        if k == "Exponential(1)":
            q = np.percentile(m, [25, 75]); iqr[k].append(0.5 * (q[1] - q[0]))
    mc = rng.standard_cauchy((M, N)).mean(axis=1)
    q = np.percentile(mc, [25, 75]); iqr["Cauchy"].append(0.5 * (q[1] - q[0]))

slopes = {k: np.polyfit(np.log(Ns), np.log(v), 1)[0] for k, v in var_of_mean.items()}

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))
ax = axs[0]
for (k, v), c in zip(var_of_mean.items(), SERIES):
    ax.loglog(Ns, v, "o", color=c, label=f"{k}: slope {slopes[k]:.3f}")
    theory_line(ax, Ns, parents[k][1] / Ns, label=r"$\sigma^2/N$" if k == "Poisson(3)" else None)
ax.set_xlabel(r"$N$"); ax.set_ylabel(r"Var$(\bar X_N)$ over repetitions")
ax.legend(fontsize=7)
ax = axs[1]
ax.loglog(Ns, iqr["Exponential(1)"], "o", color=SERIES[1], label="Exponential(1) parent")
theory_line(ax, Ns, 0.6745 / np.sqrt(Ns), label=r"$0.674\,\sigma/\sqrt{N}$")
ax.loglog(Ns, iqr["Cauchy"], "s", color=SERIES[4], label="Cauchy parent")
ax.set_xlabel(r"$N$"); ax.set_ylabel(r"half-width of central 50% of $\bar X_N$")
ax.set_ylim(5e-3, 3)
ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "var_mean_vs_N")

# ---------- CLT for exponential parent ----------
fig, axs = plt.subplots(1, 4, figsize=(8.0, 2.4), sharey=True)
z = np.linspace(-4, 4, 400)
for ax, N in zip(axs, [1, 2, 10, 50]):
    m = rng.exponential(1.0, (200_000, N)).mean(axis=1)
    Z = np.sqrt(N) * (m - 1.0) / 1.0
    ax.hist(Z, bins=np.linspace(-4, 6, 101), density=True, color=SERIES[0], alpha=0.6)
    theory_line(ax, z, stats.norm.pdf(z), label=r"$\mathcal{N}(0,1)$")
    ax.set_title(f"$N={N}$"); ax.set_xlabel(r"$Z_N$"); ax.set_xlim(-4, 6)
axs[0].set_ylabel("density"); axs[0].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "clt_exponential")

# ---------- coin example: Chebyshev versus truth ----------
n, p = 84, 0.5
k = np.arange(n + 1)
inside = (k / n >= 0.4 - 1e-12) & (k / n <= 0.6 + 1e-12)
p_exact = stats.binom.pmf(k[inside], n, p).sum()
# smallest n for which the CLT approximation gives >= 0.7:  0.1 sqrt(n)/0.5 >= z_{0.85}
z85 = stats.norm.ppf(0.85)
n_clt = int(np.ceil((z85 * 0.5 / 0.1) ** 2))
# check exact probability at that n
kk = np.arange(n_clt + 1)
ins = (kk / n_clt >= 0.4 - 1e-12) & (kk / n_clt <= 0.6 + 1e-12)
p_exact_nclt = stats.binom.pmf(kk[ins], n_clt, p).sum()

save_numbers("ch03", "03_one_over_sqrtN", {
    "ThreeASlopeU": slopes["Uniform(0,1)"],
    "ThreeASlopeE": slopes["Exponential(1)"],
    "ThreeASlopeP": slopes["Poisson(3)"],
    "ThreeACauchyIQRone": iqr["Cauchy"][0],
    "ThreeACauchyIQRlast": iqr["Cauchy"][-1],
    "ThreeACoinExact": p_exact,
    "ThreeACoinNclt": n_clt,
    "ThreeACoinExactNclt": p_exact_nclt,
    "ThreeAZeightyfive": z85,
})
