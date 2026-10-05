"""Where the central limit theorem fails: Breit-Wigner resonances and Landau energy loss.

Question answered
    The CLT needs a finite variance.  What actually happens to the running mean
    of a quantity drawn from a Breit-Wigner (Cauchy) line shape, and does the
    average of many Landau-distributed energy losses become Gaussian?

What it computes
    * Running mean and running median of invariant masses drawn from a
      relativistic-free Breit-Wigner around the Z mass (M = 91.1876 GeV,
      Gamma = 2.4952 GeV), for three independent streams, compared with a
      Gaussian of the same half width.
    * The distribution of the mean of n = 1000 Breit-Wigner draws: it is the
      same Breit-Wigner, not narrower (the width does not shrink like 1/sqrt n).
    * The mean of n = 25 Landau variables (25 thin layers of gas) against a
      Gaussian with the same median and interquartile range.

What it writes
    figures/ch02/cauchy_running.pdf, figures/ch02/landau_sum.pdf,
    results/ch02/22_cauchy_landau.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup()
rng = rng_for("ch02", "22_cauchy_landau")

M, G = 91.1876, 2.4952          # Z mass and full width (GeV)
gam = G / 2                      # half width at half maximum = Cauchy scale
sig_g = gam / np.sqrt(2 * np.log(2))   # Gaussian with the same HWHM

N = 100_000
nn = np.arange(1, N + 1)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
ax = axes[0]
for s in range(3):
    m_bw = M + gam * rng.standard_cauchy(N)                 # Breit-Wigner masses
    ax.semilogx(nn, np.cumsum(m_bw) / nn, color=SERIES[s], lw=1.0,
                label="Breit--Wigner, running mean" if s == 0 else None)
m_g = M + sig_g * rng.standard_normal(N)
ax.semilogx(nn, np.cumsum(m_g) / nn, color="k", lw=1.2, label="Gaussian, running mean")
# running median of the last Breit-Wigner stream, on a sparse grid
grid = np.unique(np.round(np.logspace(0, 5, 60)).astype(int))
ax.semilogx(grid, [np.median(m_bw[:k]) for k in grid], color=SERIES[3], ls="--",
            lw=1.4, label="Breit--Wigner, running median")
ax.axhline(M, color="0.5", lw=0.8)
lo, hi = M - 6, M + 6                                    # widen if the quoted final mean falls outside
fin = np.mean(m_bw)
ax.set_ylim(min(lo, fin - 2), max(hi, fin + 2))
ax.set_xlabel("number of events $N$")
ax.set_ylabel("estimate of $M$ (GeV)")
ax.set_title("(a) running estimates")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)

# distribution of the mean of n Breit-Wigner draws
n_avg, n_rep = 1000, 20000
means = M + gam * rng.standard_cauchy((n_rep, n_avg)).mean(axis=1)
ax = axes[1]
bins = np.linspace(M - 10, M + 10, 101)
ax.hist(means, bins=bins, density=True, color=SERIES[0], alpha=0.75,
        label=f"mean of $n={n_avg}$ events")
x = np.linspace(M - 10, M + 10, 600)
theory_line(ax, x, stats.cauchy.pdf(x, M, gam), label="single-event Breit--Wigner")
ax.set_xlabel("sample mean (GeV)")
ax.set_ylabel("density")
ax.set_title("(b) the mean does not narrow")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
savefig(fig, "ch02", "cauchy_running")

q25, q75 = np.percentile(means, [25, 75])
iqr_mean = q75 - q25

# ---------------------------------------------------------------- Landau
n_layers, n_rep_l = 25, 200_000
lan = stats.landau.rvs(size=(n_rep_l, n_layers), random_state=rng)
lan_mean = lan.mean(axis=1)
med = np.median(lan_mean)
l25, l75 = np.percentile(lan_mean, [25, 75])
sd_match = (l75 - l25) / (2 * stats.norm.ppf(0.75))
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
ax = axes[0]
x = np.linspace(-4, 20, 800)
ax.plot(x, stats.landau.pdf(x), color=SERIES[0], label="Landau (one layer)")
ax.set_xlabel(r"scaled energy loss $\lambda$")
ax.set_ylabel("density")
ax.set_title("(a) one thin layer")
ax.set_yscale("log")
ax.set_ylim(1e-4, 0.4)
ax.legend()
ax = axes[1]
bins = np.linspace(med - 6, med + 30, 181)
ax.hist(lan_mean, bins=bins, density=True, color=SERIES[0], alpha=0.75,
        label=f"mean of {n_layers} layers")
xx = np.linspace(med - 6, med + 30, 800)
theory_line(ax, xx, stats.norm.pdf(xx, med, sd_match), label="Gaussian, same median and IQR")
ax.set_yscale("log")
ax.set_ylim(1e-5, 1)
ax.set_xlabel(r"mean scaled energy loss")
ax.set_title("(b) the sum keeps its tail")
ax.legend(fontsize=7)
savefig(fig, "ch02", "landau_sum")

tail_frac = np.mean(lan_mean > med + 5 * sd_match)
tail_gauss = stats.norm.sf(5)

save_numbers("ch02", "22_cauchy_landau", {
    "tbBWgam": f"{gam:.4f}",
    "tbBWiqrMean": f"{iqr_mean:.3f}",
    "tbBWiqrTheory": f"{2 * gam:.3f}",
    "tbBWnavg": n_avg,
    "tbBWfinalMean": f"{np.mean(m_bw):.2f}",
    "tbBWfinalOff": f"{abs(np.mean(m_bw) - M):.1f}",     # distance of that running mean from M
    "tbBWfinalMedian": f"{np.median(m_bw):.3f}",
    "tbBWmedSE": f"{np.pi * gam / (2 * np.sqrt(N)):.4f}",
    "tbLanTailFrac": f"{tail_frac:.1e}".replace("e-0", r"\times10^{-") + "}",
    "tbLanTailGauss": f"{tail_gauss:.1e}".replace("e-0", r"\times10^{-") + "}",
    "tbLanNlayers": n_layers,
})
