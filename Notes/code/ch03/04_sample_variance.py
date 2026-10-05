"""04_sample_variance.py -- why 1/(N-1), and how noisy is a variance estimate?

Question: (i) is the 1/N variance estimator really biased low by (N-1)/N, and the
1/(N-1) estimator unbiased?  (ii) does the variance of S^2 follow
    Var(S^2) = (1/N) [mu_4 - (N-3)/(N-1) sigma^4]
for a Gaussian parent (mu_4 = 3 sigma^4) and an exponential parent (mu_4 = 9 sigma^4)?
(iii) which divisor, N-1, N or N+1, gives the smallest MSE for Gaussian data?

Computes: M repetitions for N = 2..60.
Writes: figures/ch03/variance_bias.pdf, figures/ch03/variance_of_variance.pdf,
        results/ch03/04_sample_variance.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.special import gammaln

rng = rng_for("ch03", "04_sample_variance")
setup()
M = 100_000
Ns = np.array([2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60])

mean_biased, mean_unbiased = [], []
var_S2_gauss, var_S2_exp = [], []
mse = {"N-1": [], "N": [], "N+1": []}
for N in Ns:
    x = rng.standard_normal((M, N))                  # sigma = 1
    ss = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    mean_biased.append((ss / N).mean())
    mean_unbiased.append((ss / (N - 1)).mean())
    var_S2_gauss.append((ss / (N - 1)).var(ddof=1))
    for lab, d in (("N-1", N - 1), ("N", N), ("N+1", N + 1)):
        mse[lab].append(np.mean((ss / d - 1.0) ** 2))
    y = rng.exponential(1.0, (M, N))                 # sigma = 1, mu_4 = 9
    ssy = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    var_S2_exp.append((ssy / (N - 1)).var(ddof=1))
mean_biased, mean_unbiased = np.array(mean_biased), np.array(mean_unbiased)


def var_S2(N, mu4, s4=1.0):
    return (mu4 - (N - 3) / (N - 1) * s4) / N


fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.semilogx(Ns, mean_unbiased, "o", color=SERIES[0], label=r"$S^2$ (divide by $N-1$)")
ax.semilogx(Ns, mean_biased, "s", color=SERIES[1], label=r"$\hat\sigma^2$ (divide by $N$)")
nn = np.geomspace(2, 60, 200)
theory_line(ax, nn, (nn - 1) / nn, label=r"$(N-1)/N$")
ax.axhline(1.0, color="0.3", lw=0.9, ls=":")
ax.set_xlabel(r"$N$"); ax.set_ylabel(r"average estimate / $\sigma^2$")
ax.xaxis.set_minor_formatter(NullFormatter())
ax.legend(fontsize=8)
savefig(fig, "ch03", "variance_bias")

fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.loglog(Ns, var_S2_gauss, "o", color=SERIES[0], label=r"Gaussian parent")
ax.loglog(Ns, var_S2_exp, "s", color=SERIES[1], label=r"Exponential parent")
theory_line(ax, nn, var_S2(nn, 3.0), label=r"$\frac{1}{N}[\mu_4-\frac{N-3}{N-1}\sigma^4]$")
theory_line(ax, nn, var_S2(nn, 9.0), label="_nolegend_")
ax.set_xlabel(r"$N$"); ax.set_ylabel(r"Var$(S^2)$ / $\sigma^4$")
ax.xaxis.set_minor_formatter(NullFormatter())
ax.legend(fontsize=8)
savefig(fig, "ch03", "variance_of_variance")

i5 = int(np.where(Ns == 5)[0][0]); i10 = int(np.where(Ns == 10)[0][0])
N = 5
c4 = np.exp(0.5 * np.log(2 / (N - 1)) + gammaln(N / 2) - gammaln((N - 1) / 2))
xs = rng.standard_normal((M, N))
S = xs.std(axis=1, ddof=1)
save_numbers("ch03", "04_sample_variance", {
    "ThreeAVbiasFive": mean_biased[i5], "ThreeAVunbFive": mean_unbiased[i5],
    "ThreeAVarSGaussTen": var_S2_gauss[i10], "ThreeAVarSGaussTenTh": var_S2(10, 3.0),
    "ThreeAVarSExpTen": var_S2_exp[i10], "ThreeAVarSExpTenTh": var_S2(10, 9.0),
    "ThreeAMseNmOneTen": mse["N-1"][i10], "ThreeAMseNTen": mse["N"][i10],
    "ThreeAMseNpOneTen": mse["N+1"][i10],
    "ThreeACfourFive": c4, "ThreeAMeanSFive": S.mean(),
})
