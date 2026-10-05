"""Every kind of plot the book draws, on one page: plot, loglog, hist, errorbar, imshow, fill_between.

Question:  which few Matplotlib calls produce all of the book's figures, and what does each draw?
Computes:  six small panels from simulated data: a damped oscillation with its theory curve,
           a power law on log axes, a histogram of Gaussian draws against the density, binned
           counts with Poisson error bars, a smooth random image, and a band around a curve.
Writes:    figures/chT0/04_plot_gallery.pdf
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from common import setup, savefig, rng_for, SERIES, theory_line

rng = rng_for("chT0", "04_plot_gallery")
setup(9.0, 5.4)
fig, axes = plt.subplots(2, 3)                    # a 2 x 3 grid of panels; axes[row, column]

# 1. plot: noisy samples of a damped oscillation, and the curve they scatter around
t = np.linspace(0, 10, 60)
truth = np.exp(-t / 4) * np.cos(2 * t)
ax = axes[0, 0]
ax.plot(t, truth + 0.1 * rng.standard_normal(t.size), "o", ms=3, label="data")
theory_line(ax, t, truth)
ax.set(xlabel="time $t$", ylabel="$x(t)$", title="ax.plot")
ax.legend(fontsize=7)

# 2. loglog: a power law is a straight line on logarithmic axes
k = np.logspace(-2, 1, 50)
ax = axes[0, 1]
ax.loglog(k, k**-3, label="$k^{-3}$")
ax.loglog(k, 0.1 * k**-1, label="$0.1\\,k^{-1}$")
ax.set(xlabel="$k$", ylabel="$P(k)$", title="ax.loglog")
ax.legend(fontsize=7)

# 3. hist: 5000 Gaussian draws, normalised to a density, against the exact density
x = rng.standard_normal(5000)
ax = axes[0, 2]
ax.hist(x, bins=40, density=True, alpha=0.5, label="5000 draws")
xx = np.linspace(-4, 4, 200)
theory_line(ax, xx, stats.norm.pdf(xx), label="density")
ax.set(xlabel="$x$", ylabel="density", title="ax.hist")
ax.legend(fontsize=7)

# 4. errorbar: Poisson counts in bins, each with its error bar sqrt(n)
edges = np.linspace(0, 5, 11)
centres = 0.5 * (edges[:-1] + edges[1:])
expected = 200 * np.exp(-centres)
n = rng.poisson(expected)
ax = axes[1, 0]
ax.errorbar(centres, n, yerr=np.sqrt(n), fmt="o", ms=3, capsize=2, label="counts")
theory_line(ax, centres, expected, label="expected")
ax.set(xlabel="energy", ylabel="counts per bin", title="ax.errorbar")
ax.legend(fontsize=7)

# 5. imshow: an image, here white noise smoothed into a random field
field = ndimage.gaussian_filter(rng.standard_normal((128, 128)), sigma=4)
ax = axes[1, 1]
im = ax.imshow(field, cmap="RdBu_r", origin="lower")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set(title="ax.imshow")
ax.grid(False)

# 6. fill_between: a band of plus or minus one standard deviation around a mean
sims = np.cumsum(rng.standard_normal((500, 100)), axis=1)    # 500 random walks of 100 steps
mean, sd = sims.mean(axis=0), sims.std(axis=0)
steps = np.arange(1, 101)
ax = axes[1, 2]
ax.fill_between(steps, mean - sd, mean + sd, alpha=0.3, lw=0, label="$\\pm1\\sigma$ of 500 walks")
ax.plot(steps, mean, label="mean")
theory_line(ax, steps, np.sqrt(steps), label="$\\sqrt{n}$")
ax.set(xlabel="step $n$", ylabel="position", title="ax.fill_between")
ax.legend(fontsize=7)

fig.tight_layout()
savefig(fig, "chT0", "04_plot_gallery")
