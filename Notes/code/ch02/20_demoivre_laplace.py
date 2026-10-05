"""How good is the Gaussian approximation to the binomial, and how fast does it improve?

Question answered
    de Moivre and Laplace showed that Binomial(n, p) looks more and more like a
    Gaussian with mean np and variance np(1-p) as n grows.  How close is it for
    the n we meet in practice, and what does the continuity correction buy us?

What it computes
    * Stirling's formula against the exact n! (ratio and the 1 + 1/(12n) correction).
    * Binomial(n, 0.3) pmf against the Gaussian density for n = 10, 40, 200.
    * The largest pointwise error max_k |P(k) - gauss(k)| as a function of n.
    * The Casella-Berger example P(X <= 13) for X ~ Binomial(25, 0.6):
      exact, plain Gaussian, Gaussian with continuity correction.

What it writes
    figures/ch02/demoivre_panels.pdf, figures/ch02/demoivre_errors.pdf,
    results/ch02/20_demoivre_laplace.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special

setup()

# ---------------------------------------------------------------- Stirling
n_st = np.arange(1, 31)
ln_fact = special.gammaln(n_st + 1)                       # ln n!, exact
ln_stirling = n_st * np.log(n_st) - n_st + 0.5 * np.log(2 * np.pi * n_st)
ratio = np.exp(ln_fact - ln_stirling)                      # n! / Stirling

# ---------------------------------------------------------------- pmf vs Gaussian
p = 0.3
ns = [10, 40, 200]
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
for ax, n in zip(axes, ns):
    k = np.arange(0, n + 1)
    pmf = stats.binom.pmf(k, n, p)
    mu, sd = n * p, np.sqrt(n * p * (1 - p))
    lo, hi = int(max(0, mu - 4.5 * sd)), int(min(n, mu + 4.5 * sd)) + 1
    ax.bar(k[lo:hi], pmf[lo:hi], width=0.85, color=SERIES[0], alpha=0.75,
           label="binomial pmf")
    x = np.linspace(lo - 0.5, hi - 0.5, 400)
    theory_line(ax, x, stats.norm.pdf(x, mu, sd), label="Gaussian")
    ax.set_title(f"$n={n}$, $p={p}$")
    ax.set_xlabel("$k$")
    ax.set_xlim(lo - 1, hi)
axes[0].set_ylabel("probability")
fig.subplots_adjust(wspace=0.35)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
savefig(fig, "ch02", "demoivre_panels")

# ---------------------------------------------------------------- error vs n
n_grid = np.unique(np.round(np.logspace(1, 4, 25)).astype(int))
maxerr = []
for n in n_grid:
    k = np.arange(0, n + 1)
    mu, sd = n * p, np.sqrt(n * p * (1 - p))
    maxerr.append(np.max(np.abs(stats.binom.pmf(k, n, p) - stats.norm.pdf(k, mu, sd))))
maxerr = np.array(maxerr)
slope = np.polyfit(np.log(n_grid), np.log(maxerr), 1)[0]

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
ax = axes[0]
ax.plot(n_st, ratio, "o", color=SERIES[0], label=r"$n!\,/\,$Stirling")
theory_line(ax, n_st, 1 + 1 / (12 * n_st), label=r"$1+1/(12n)$")
ax.set_xlabel("$n$")
ax.set_ylabel("ratio")
ax.set_title("(a) Stirling's formula")
ax.legend()
ax = axes[1]
ax.loglog(n_grid, maxerr, "o", color=SERIES[1], label=r"$\max_k|P(k)-\mathrm{gauss}(k)|$")
theory_line(ax, n_grid, maxerr[0] * (n_grid / n_grid[0]) ** -1.0, label=r"$\propto n^{-1}$")
ax.set_xlabel("$n$")
ax.set_ylabel("largest pointwise error")
ax.set_title("(b) de Moivre--Laplace error, $p=0.3$")
ax.legend()
savefig(fig, "ch02", "demoivre_errors")

# ---------------------------------------------------------------- Casella-Berger Example 3.3.2
nCB, pCB = 25, 0.6
muCB, sdCB = nCB * pCB, np.sqrt(nCB * pCB * (1 - pCB))
exact = stats.binom.cdf(13, nCB, pCB)
plain = stats.norm.cdf((13 - muCB) / sdCB)
cc = stats.norm.cdf((13.5 - muCB) / sdCB)

save_numbers("ch02", "20_demoivre_laplace", {
    "tbStirTen": f"{ratio[9]:.5f}",                 # 10!/Stirling(10)
    "tbStirOne": f"{ratio[0]:.4f}",
    "tbDMLslope": f"{slope:.2f}",
    "tbDMLerrTen": f"{maxerr[0]:.4f}",
    "tbDMLerrMax": f"{maxerr[-1]:.1e}".replace("e-0", r"\times10^{-") + "}",
    "tbDMLnMax": int(n_grid[-1]),
    "tbCBexact": f"{exact:.3f}",
    "tbCBplain": f"{plain:.3f}",
    "tbCBcc": f"{cc:.3f}",
    "tbCBsd": f"{sdCB:.3f}",
})
