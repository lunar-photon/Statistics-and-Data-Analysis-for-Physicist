"""Watch the central limit theorem happen, and measure how fast it happens.

Question answered
    Add up n independent copies of a decidedly non-Gaussian variable (a flat
    uniform, a skewed exponential).  After standardising, how quickly does the
    histogram of the sum become the N(0,1) bell curve, and why is the symmetric
    case so much faster than the skewed one?

What it computes
    * Monte Carlo histograms of Z_n = (S_n - n mu)/(sigma sqrt(n)) for n = 1, 2, 5, 30.
    * The Kolmogorov distance D_n = sup_z |P(Z_n <= z) - Phi(z)|, computed without
      Monte Carlo noise: for exponentials S_n is exactly Gamma(n, 1); for uniforms
      the density of S_n is obtained by repeated numerical convolution (FFT).
    * The measured skewness of Z_n for exponentials against the prediction 2/sqrt(n).

What it writes
    figures/ch02/clt_histograms.pdf, figures/ch02/clt_rate.pdf,
    results/ch02/21_clt_sums.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal

setup()
rng = rng_for("ch02", "21_clt_sums")
NSAMP = 200_000
ns = [1, 2, 5, 30]

# the two parent distributions, with their exact mean and standard deviation
parents = {
    "uniform": (lambda size: rng.uniform(0.0, 1.0, size), 0.5, np.sqrt(1 / 12)),
    "exponential": (lambda size: rng.exponential(1.0, size), 1.0, 1.0),
}

z = np.linspace(-4, 4, 400)
fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.6), sharex=True, sharey=True)
skew_meas = {}
for row, (name, (draw, mu, sd)) in enumerate(parents.items()):
    for col, n in enumerate(ns):
        s = draw((NSAMP, n)).sum(axis=1)                    # the sum of n copies
        zn = (s - n * mu) / (sd * np.sqrt(n))               # standardised sum
        if name == "exponential":
            skew_meas[n] = stats.skew(zn)
        ax = axes[row, col]
        ax.hist(zn, bins=np.linspace(-4, 4, 81), density=True,
                color=SERIES[row], alpha=0.75, label=f"{name} sum")
        theory_line(ax, z, stats.norm.pdf(z), label="$N(0,1)$")
        ax.set_title(f"{name}, $n={n}$")
        ax.set_xlim(-4, 4)
        if row == 1:
            ax.set_xlabel("$z_n$")
    axes[row, 0].set_ylabel("density")
axes[0, 0].set_ylim(0, 0.62)
axes[0, 0].legend(loc="upper left", fontsize=7)
savefig(fig, "ch02", "clt_histograms")

# ---------------------------------------------------------------- exact convergence rate
n_rate = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
zz = np.linspace(-6, 6, 24001)
D_exp = []
for n in n_rate:
    # S_n ~ Gamma(n,1): P(Z_n <= z) = GammaCDF(n + z sqrt(n))
    D_exp.append(np.max(np.abs(stats.gamma.cdf(n + zz * np.sqrt(n), a=n) - stats.norm.cdf(zz))))
D_exp = np.array(D_exp)

dx = 2e-4                                   # grid step for the uniform convolutions
base = np.ones(int(round(1 / dx))) * 1.0    # density of U(0,1) sampled on the grid
dens = base.copy()
D_uni = []
for n in range(1, n_rate.max() + 1):
    if n > 1:
        dens = signal.fftconvolve(dens, base) * dx
    if n in n_rate:
        # U(0,1) is sampled at (i+1/2)dx, so the n-fold sum lives at (m + n/2)dx
        x = (np.arange(dens.size) + 0.5 * n) * dx
        cdf = np.cumsum(dens) * dx - 0.5 * dens * dx   # trapezoidal CDF at the grid points
        zn = (x - n * 0.5) / (np.sqrt(n / 12))
        D_uni.append(np.max(np.abs(cdf - stats.norm.cdf(zn))))
D_uni = np.array(D_uni)

sl_exp = np.polyfit(np.log(n_rate[3:]), np.log(D_exp[3:]), 1)[0]
sl_uni = np.polyfit(np.log(n_rate[3:]), np.log(D_uni[3:]), 1)[0]

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.loglog(n_rate, D_exp, "o", color=SERIES[1], label="exponential (skewed)")
ax.loglog(n_rate, D_uni, "s", color=SERIES[0], label="uniform (symmetric)")
theory_line(ax, n_rate, D_exp[0] * n_rate ** -0.5, label=r"$\propto n^{-1/2}$")
ax.loglog(n_rate, D_uni[0] * n_rate ** -1.0, color="0.35", ls=":", lw=1.4, label=r"$\propto n^{-1}$")
ax.set_xlabel("number of terms $n$")
ax.set_ylabel(r"$D_n=\sup_z|P(Z_n\leq z)-\Phi(z)|$")
ax.legend()
savefig(fig, "ch02", "clt_rate")

# Berry-Esseen constant check for the exponential: E|X-mu|^3 = 12/e - 2 for Exp(1)
rho3 = 12 / np.e - 2
BE30 = 33 / 4 * rho3 / np.sqrt(30)

save_numbers("ch02", "21_clt_sums", {
    "tbCLTskewOne": f"{skew_meas[1]:.2f}",
    "tbCLTskewFive": f"{skew_meas[5]:.2f}",
    "tbCLTskewThirty": f"{skew_meas[30]:.3f}",
    "tbCLTskewThirtyTh": f"{2 / np.sqrt(30):.3f}",
    "tbCLTslopeExp": f"{sl_exp:.2f}",
    "tbCLTslopeUni": f"{sl_uni:.2f}",
    "tbCLTDexpThirtyTwo": f"{D_exp[list(n_rate).index(32)]:.4f}",
    "tbCLTDuniThirtyTwo": f"{D_uni[list(n_rate).index(32)]:.1e}".replace("e-0", r"\times10^{-") + "}",
    "tbCLTDexpOne": f"{D_exp[0]:.3f}",
    "tbCLTBEthirty": f"{BE30:.2f}",
    "tbCLTBEmodern": f"{0.4748 * rho3 / np.sqrt(30):.3f}",   # sharpest known constant, 0.4748
    "tbCLTDexpThirty": f"{np.max(np.abs(stats.gamma.cdf(30 + zz * np.sqrt(30), a=30) - stats.norm.cdf(zz))):.4f}",
    "tbCLTrhothree": f"{rho3:.3f}",
    "tbCLTnsamp": NSAMP,
})
