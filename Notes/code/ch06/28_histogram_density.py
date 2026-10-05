"""28_histogram_density.py -- a histogram of samples approximates the density, with error bars.

Question: if we can only draw samples, how well does their histogram reproduce the
density, and what error bar belongs on each bin?

  1. The harmonic oscillator x(t) = A cos(omega t): histogram x at N = 10^4 random
     times t, uniform over a period, against the arcsine density
     1 / (pi sqrt(A^2 - x^2)).  Bin k holds n_k ~ Binomial(N, p_k) counts,
     p_k = F(b_{k+1}) - F(b_k) with F(x) = 1/2 + arcsin(x/A)/pi, so the density
     estimate n_k / (N Delta) has error sqrt(n_k (1 - n_k/N)) / (N Delta).
     Pulls (n_k - N p_k) / sqrt(N p_k (1 - p_k)) and chi^2 over the bins.
  2. Kruschke's Beta(15, 7) represented by N = 500, 5000, 50000 samples: histogram,
     mode estimate and 95% highest-density interval from the samples.

Writes: figures/ch06/oscillator_histogram.pdf, figures/ch06/beta_samples.pdf,
        results/ch06/28_histogram_density.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

rng = rng_for("ch06", "28_histogram_density")
out = {}

# ---------------------------------------------------------------- 1. oscillator
A, omega, N, K = 1.0, 2 * np.pi, 10_000, 40
t = rng.random(N) * (2 * np.pi / omega)          # random times over one period
x = A * np.cos(omega * t)
edges = np.linspace(-A, A, K + 1)
Delta = edges[1] - edges[0]
n_k, _ = np.histogram(x, bins=edges)
F = lambda x: 0.5 + np.arcsin(np.clip(x / A, -1, 1)) / np.pi
p_k = np.diff(F(edges))
dens = n_k / (N * Delta)
err = np.sqrt(n_k * (1 - n_k / N)) / (N * Delta)
pull = (n_k - N * p_k) / np.sqrt(N * p_k * (1 - p_k))
chi2 = np.sum((n_k - N * p_k) ** 2 / (N * p_k))  # Pearson chi^2, K - 1 degrees of freedom
cen = 0.5 * (edges[1:] + edges[:-1])
f_cen = 1 / (np.pi * np.sqrt(A**2 - cen**2))
out.update(SixBHoN=N, SixBHoK=K, SixBHoChi=chi2, SixBHoDof=K - 1,
           SixBHoPval=stats.chi2.sf(chi2, K - 1),
           SixBHoPullSd=pull.std(ddof=1),
           SixBHoEdgeAvg=p_k[-1] / Delta, SixBHoEdgeCen=f_cen[-1],
           SixBHoMidAvg=p_k[K // 2] / Delta, SixBHoMidCen=f_cen[K // 2],
           SixBHoRelErrMid=np.sqrt((1 - p_k[K // 2]) / (N * p_k[K // 2])),
           SixBHoRelErrEdge=np.sqrt((1 - p_k[-1]) / (N * p_k[-1])))

setup(9.0, 3.2)
fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
ax[0].errorbar(cen, dens, yerr=err, fmt="o", ms=3, color=SERIES[0], label="histogram of $x(t)$, random $t$")
xx = np.linspace(-0.999, 0.999, 600)
theory_line(ax[0], xx, 1 / (np.pi * np.sqrt(1 - xx**2)), label=r"$1/\pi\sqrt{A^2-x^2}$")
ax[0].step(edges, np.r_[p_k / Delta, p_k[-1] / Delta], where="post", color=SERIES[2], lw=1.0,
           label=r"bin average $p_k/\Delta$")
ax[0].set(xlabel=r"position $x/A$", ylabel="density", ylim=(0, 2.6))
ax[0].legend(fontsize=8, loc="upper center")
ax[1].hist(pull, bins=np.linspace(-3.5, 3.5, 15), density=True, color=SERIES[0], alpha=0.6, label=f"{K} bin pulls")
zz = np.linspace(-3.5, 3.5, 200)
theory_line(ax[1], zz, stats.norm.pdf(zz), label=r"$\mathcal{N}(0,1)$")
ax[1].set(xlabel="pull", ylabel="density")
ax[1].legend(fontsize=8)
fig.tight_layout()
savefig(fig, "ch06", "oscillator_histogram")

# ---------------------------------------------------------------- 2. Beta(15, 7) from samples
a, b = 15, 7
beta = stats.beta(a, b)


def hdi_from_samples(s, mass=0.95):
    s = np.sort(s)
    m = int(np.floor(mass * s.size))
    widths = s[m:] - s[: s.size - m]
    i = np.argmin(widths)
    return s[i], s[i + m]


def exact_hdi(mass=0.95):
    def width(lo):
        return beta.ppf(beta.cdf(lo) + mass) - lo
    lo = optimize.minimize_scalar(width, bounds=(0.3, beta.ppf(1 - mass) - 1e-9), method="bounded").x
    return lo, beta.ppf(beta.cdf(lo) + mass)


elo, ehi = exact_hdi()
out.update(SixBKrLoExact=elo, SixBKrHiExact=ehi, SixBKrModeExact=(a - 1) / (a + b - 2))
setup(9.0, 2.8)
fig, ax = plt.subplots(1, 3, sharey=True)
xx = np.linspace(0, 1, 400)
for i, (n, tag) in enumerate([(500, "A"), (5000, "B"), (50000, "C")]):
    s = beta.rvs(size=n, random_state=rng)
    h, e = np.histogram(s, bins=np.linspace(0, 1, 51))
    c = 0.5 * (e[1:] + e[:-1])
    mode = c[np.argmax(h)]
    lo, hi = hdi_from_samples(s)
    out.update({f"SixBKrN{tag}": n, f"SixBKrMode{tag}": mode, f"SixBKrLo{tag}": lo, f"SixBKrHi{tag}": hi})
    ax[i].hist(s, bins=e, density=True, color=SERIES[0], alpha=0.6)
    theory_line(ax[i], xx, beta.pdf(xx), label="Beta(15,7)")
    ax[i].plot([lo, hi], [0.25, 0.25], color=SERIES[1], lw=3, solid_capstyle="butt", label="95% HDI")
    ax[i].set(xlabel=r"$\theta$", title=f"$N={n}$")
ax[0].set(ylabel="density")
ax[0].legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch06", "beta_samples")
save_numbers("ch06", "28_histogram_density", out)
print(out)
