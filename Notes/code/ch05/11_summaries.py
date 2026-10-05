"""11_summaries.py -- summarising a posterior: mean, median, mode, equal-tailed interval, HDI.

Question: a posterior is a whole curve.  When we must quote a few numbers, which ones,
and how do they behave for a skewed and for a two-peaked posterior?

Computes:
  * for the skewed posterior of the CMB quadrupole band power x = C_2 / C_hat_2
    (inverse-gamma with alpha = beta = 5/2): mean, median, mode, 95% equal-tailed
    interval (ETI) and 95% highest-density interval (HDI) from a grid ("water level")
    and from 10^5 posterior samples (shortest interval containing 95% of them);
  * the same posterior in the variable ln x: its HDI mapped back to x is a different
    interval, while the ETI maps onto itself;
  * the two-peaked coupling posterior of 08_grid_posterior: the 68% HDI is two
    disjoint intervals, and the posterior mean sits where the density is lowest.

Writes: figures/ch05/summaries.pdf, results/ch05/11_summaries.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch05", "11_summaries")
setup()


def hdi_grid(x, p, mass):
    """Water level: lower a horizontal line until the region above it holds `mass`.
    Returns the level W and the list of intervals [lo, hi] where p > W."""
    dx = x[1] - x[0]
    order = np.argsort(p)[::-1]                    # grid points from highest density down
    cum = np.cumsum(p[order]) * dx
    W = p[order][np.searchsorted(cum, mass)]
    inside = p >= W
    edges = np.flatnonzero(np.diff(inside.astype(int)))
    starts = list(x[edges[inside[edges + 1]] + 1]) if inside.any() else []
    ends = list(x[edges[~inside[edges + 1]]])
    if inside[0]:
        starts = [x[0]] + starts
    if inside[-1]:
        ends = ends + [x[-1]]
    return W, list(zip(starts, ends))


def hdi_samples(samples, mass):
    """Shortest interval containing a fraction `mass` of the sorted samples (unimodal case)."""
    s = np.sort(samples)
    k = int(np.floor(mass * s.size))
    widths = s[k:] - s[: s.size - k]
    i = np.argmin(widths)
    return s[i], s[i + k]


# ---------------------------------------------------------------- skewed: quadrupole
nu = 5
post = stats.invgamma(nu / 2, scale=nu / 2)
x = np.linspace(1e-3, 30, 300001)
p = post.pdf(x)
mean, median, mode = post.mean(), post.median(), (nu / 2) / (nu / 2 + 1)
eti = (post.ppf(0.025), post.ppf(0.975))
W, (hdi,) = hdi_grid(x, p, 0.95)
samp = post.rvs(size=100000, random_state=rng)
hdi_s = hdi_samples(samp, 0.95)
# the same posterior in y = ln x: density p(x) x ; its HDI mapped back to x
y = np.log(x)
yy = np.linspace(y[0], y[-1], 300001)
py = post.pdf(np.exp(yy)) * np.exp(yy)
_, (hdi_y,) = hdi_grid(yy, py, 0.95)
hdi_from_y = (np.exp(hdi_y[0]), np.exp(hdi_y[1]))

# ---------------------------------------------------------------- bimodal: coupling posterior
d, sig = 0.6041, 0.1                               # the datum of 08_grid_posterior
g = np.linspace(-1.5, 1.5, 30001)
pg = np.exp(-(d - g**2) ** 2 / (2 * sig**2)); pg /= pg.sum() * (g[1] - g[0])
Wg, ints = hdi_grid(g, pg, 0.68)
mean_g = np.sum(g * pg) * (g[1] - g[0])

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9))
m = x < 6
ax[0].plot(x[m], p[m], color=SERIES[0])
ax[0].fill_between(x[(x > hdi[0]) & (x < hdi[1])], p[(x > hdi[0]) & (x < hdi[1])],
                   color=SERIES[0], alpha=0.2, lw=0)
ax[0].hlines(W, *hdi, color=SERIES[0], lw=1.2)
ax[0].hlines(-0.03, *eti, color=SERIES[1], lw=3)
ax[0].hlines(-0.07, *hdi, color=SERIES[0], lw=3)
for v, c, lab in [(mode, SERIES[2], "mode"), (median, SERIES[3], "median"), (mean, SERIES[4], "mean")]:
    ax[0].axvline(v, color=c, ls="--", lw=1, label=f"{lab} {v:.2f}")
ax[0].plot([], [], color=SERIES[1], lw=3, label="95% ETI")
ax[0].plot([], [], color=SERIES[0], lw=3, label="95% HDI")
ax[0].set_xlim(0, 6); ax[0].set_xlabel(r"$x=C_2/\widehat{C}_2$")
ax[0].set_title("skewed: quadrupole band power", fontsize=8)
ax[0].legend(fontsize=6, loc="upper right")
ax[1].plot(g, pg, color=SERIES[0])
for lo, hi in ints:
    mm = (g > lo) & (g < hi)
    ax[1].fill_between(g[mm], pg[mm], color=SERIES[0], alpha=0.2, lw=0)
    ax[1].hlines(-0.1, lo, hi, color=SERIES[0], lw=3)
ax[1].axhline(Wg, color=INK2, lw=0.6, ls=":")
ax[1].axvline(mean_g, color=SERIES[4], ls="--", lw=1, label=f"mean {mean_g:.2f}")
ax[1].set_xlabel("coupling $g$"); ax[1].set_title("two peaks: 68% HDI is split", fontsize=8)
ax[1].legend(fontsize=6, loc="upper center")
for a in ax:
    a.set_yticks([])
fig.tight_layout()
savefig(fig, "ch05", "summaries")

save_numbers("ch05", "11_summaries", {
    "FiveBSumAboveMode": f"{100 * post.sf(mode):.0f}",
    "FiveBSumMean": mean, "FiveBSumMedian": median, "FiveBSumMode": mode,
    "FiveBSumEtiLo": eti[0], "FiveBSumEtiHi": eti[1],
    "FiveBSumHdiLo": hdi[0], "FiveBSumHdiHi": hdi[1],
    "FiveBSumHdiSLo": hdi_s[0], "FiveBSumHdiSHi": hdi_s[1],
    "FiveBSumHdiYLo": hdi_from_y[0], "FiveBSumHdiYHi": hdi_from_y[1],
    "FiveBSumWater": W,
    "FiveBBiLoA": ints[0][0], "FiveBBiHiA": ints[0][1], "FiveBBiLoB": ints[1][0], "FiveBBiHiB": ints[1][1],
    "FiveBBiMean": f"{mean_g:.3f}",
})
