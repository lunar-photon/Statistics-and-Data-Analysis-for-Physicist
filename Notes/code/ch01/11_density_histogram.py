"""Density is probability per unit x: from a histogram of counts to a pdf, and the CDF/quantiles.

Question: what is a probability density operationally, why can it exceed 1, why does it
carry units, and how are the CDF and quantiles read off a sample?

Toy: a calorimeter measures the energy of 50 GeV electrons with Gaussian resolution
sigma = 0.25 GeV.  We histogram N measurements, divide counts by (N * bin width) and watch
the result approach the pdf, whose peak 1/(sigma sqrt(2 pi)) = 1.6 per GeV is larger than 1
(and is 1.6e-3 per MeV: a density carries the inverse units of x).  Then we build the
empirical CDF and compare sample quantiles with the exact ones.
Writes: figures/ch01/density_histogram.pdf, figures/ch01/ecdf_quantiles.pdf,
        results/ch01/11_density_histogram.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch01", "11_density_histogram")
setup(7.0, 3.0)

mu, sigma = 50.0, 0.25                     # GeV
E_all = rng.normal(mu, sigma, size=100_000)
grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
pdf = stats.norm.pdf(grid, mu, sigma)      # per GeV

# --- figure 1: counts (left) and counts/(N*width) (right) ---------------------------
fig, axes = plt.subplots(1, 2)
cases = [(100, 0.2), (1_000, 0.1), (100_000, 0.025)]
for (N, w), c in zip(cases, SERIES):
    E = E_all[:N]
    bins = np.arange(mu - 4 * sigma, mu + 4 * sigma + w / 2, w)
    counts, edges = np.histogram(E, bins=bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    axes[0].step(centres, counts, where="mid", color=c, label=f"$N={N}$, $\\Delta={w}$")
    dens = counts / (N * w)
    axes[1].step(centres, dens, where="mid", color=c, label=f"$N={N}$")
theory_line(axes[1], grid, pdf, label="pdf")
axes[0].set_yscale("log")
axes[0].set_xlabel("$E$ [GeV]")
axes[0].set_ylabel("counts per bin")
axes[0].set_ylim(0.7, 2e5)
axes[0].legend(fontsize=7, loc="upper left")
axes[1].set_xlabel("$E$ [GeV]")
axes[1].set_ylabel("counts / ($N\\Delta$)  [GeV$^{-1}$]")
axes[1].legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "ch01", "density_histogram")

# peak density estimated from the finest histogram
w = 0.025
counts, edges = np.histogram(E_all, bins=np.arange(mu - 4 * sigma, mu + 4 * sigma + w / 2, w))
peak_est = counts.max() / (E_all.size * w)
frac_1sig = np.mean(np.abs(E_all - mu) < sigma)

# --- figure 2: empirical CDF and quantiles ----------------------------------------
N = 1_000
E = np.sort(E_all[:N])
ecdf = np.arange(1, N + 1) / N
fig, axes = plt.subplots(1, 2)
ax = axes[0]
ax.step(E, ecdf, where="post", color=SERIES[0], label=f"empirical CDF, $N={N}$")
theory_line(ax, grid, stats.norm.cdf(grid, mu, sigma), label="$F(E)$")
for q, c in zip([0.25, 0.5, 0.75], SERIES[1:]):
    xq = stats.norm.ppf(q, mu, sigma)
    ax.plot([grid[0], xq], [q, q], color=c, lw=0.9)
    ax.plot([xq, xq], [0, q], color=c, lw=0.9)
ax.set_xlabel("$E$ [GeV]")
ax.set_ylabel("$F(E)=\\mathrm{P}(E'\\leq E)$")
ax.legend(fontsize=7, loc="upper left")
ax = axes[1]
qs = np.linspace(0.01, 0.99, 99)
ax.plot(qs, np.quantile(E_all[:N], qs), color=SERIES[0], label=f"sample quantiles, $N={N}$")
theory_line(ax, qs, stats.norm.ppf(qs, mu, sigma), label="$F^{-1}(q)$")
ax.set_xlabel("$q$")
ax.set_ylabel("$F^{-1}(q)$ [GeV]")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch01", "ecdf_quantiles")

sq = np.quantile(E_all[:N], [0.25, 0.5, 0.75])
eq = stats.norm.ppf([0.25, 0.5, 0.75], mu, sigma)
save_numbers("ch01", "11_density_histogram", {
    "OneBPeakExact": round(1 / (sigma * np.sqrt(2 * np.pi)), 4),
    "OneBPeakEst": round(peak_est, 4),
    "OneBFracOneSig": round(frac_1sig, 4),
    "OneBQoneSample": f"{sq[0]:.3f}", "OneBQoneExact": f"{eq[0]:.3f}",
    "OneBMedSample": f"{sq[1]:.3f}", "OneBMedExact": f"{eq[1]:.3f}",
    "OneBQthreeSample": f"{sq[2]:.3f}", "OneBQthreeExact": f"{eq[2]:.3f}",
})
