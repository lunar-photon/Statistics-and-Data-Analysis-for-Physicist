"""Chi-squared from first principles, and why the sample variance has n-1 degrees of freedom.

Question answered
    (1) Is the sum of nu squared standard normals really chi^2_nu, with mean nu
        and variance 2 nu?  Is the quadratic form (x-mu)^T C^{-1} (x-mu) of a
        correlated 3-D Gaussian really chi^2_3?
    (2) For n = 5 Gaussian measurements, (n-1) S^2 / sigma^2 is claimed to be
        chi^2_{n-1}, not chi^2_n.  Which one do the simulations follow, and how
        big is the bias of the 1/n variance estimator?
    (3) Are the sample mean and sample variance independent?  (Yes for a
        Gaussian parent, no for an exponential one.)

What it writes
    figures/ch02/chi2_sums.pdf, figures/ch02/chi2_sampvar.pdf,
    results/ch02/25_chi2_sampvar.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup()
rng = rng_for("ch02", "25_chi2_sampvar")
NREP = 200_000

# ---------------------------------------------------------------- (1) sums of squares
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
ax = axes[0]
mv = {}
for i, nu in enumerate([1, 2, 5, 10]):
    q = (rng.standard_normal((NREP, nu)) ** 2).sum(axis=1)
    mv[nu] = (q.mean(), q.var(ddof=1))
    ax.hist(q, bins=np.linspace(0, 25, 101), density=True, histtype="stepfilled",
            color=SERIES[i], alpha=0.45, label=rf"$\nu={nu}$")
    xx = np.linspace(0.02, 25, 500)
    theory_line(ax, xx, stats.chi2.pdf(xx, nu), label="$\\chi^2_\\nu$ pdf" if i == 3 else None)
ax.set_ylim(0, 0.5)
ax.set_xlabel(r"$q=\sum z_i^2$ (sum over $\nu$ terms)")
ax.set_ylabel("density")
ax.set_title("(a) sums of squared standard normals")
ax.legend()

# correlated 3-D Gaussian quadratic form
C3 = np.array([[1.0, 0.6, 0.3], [0.6, 2.0, -0.5], [0.3, -0.5, 1.5]])
mu3 = np.array([0.5, -1.0, 2.0])
L3 = np.linalg.cholesky(C3)
x3 = mu3 + rng.standard_normal((NREP, 3)) @ L3.T
r3 = x3 - mu3
qf = np.einsum("ni,ij,nj->n", r3, np.linalg.inv(C3), r3)
naive = ((r3 / np.sqrt(np.diag(C3))) ** 2).sum(axis=1)       # ignores the correlations
ax = axes[1]
ax.hist(qf, bins=np.linspace(0, 16, 81), density=True, color=SERIES[0], alpha=0.6,
        label=r"$(\mathbf{x}-\boldsymbol{\mu})^{\mathsf{T}}\mathbf{C}^{-1}(\mathbf{x}-\boldsymbol{\mu})$")
ax.hist(naive, bins=np.linspace(0, 16, 81), density=True, histtype="step", color=SERIES[1], lw=1.4,
        label=r"$\sum_i (x_i-\mu_i)^2/C_{ii}$ (wrong)")
xx = np.linspace(0.02, 16, 400)
theory_line(ax, xx, stats.chi2.pdf(xx, 3), label=r"$\chi^2_3$")
ax.set_xlabel("quadratic form")
ax.set_title("(b) correlated 3-D Gaussian")
ax.legend(fontsize=7)
savefig(fig, "ch02", "chi2_sums")

# ---------------------------------------------------------------- (2) sample variance
n, sigma, mu = 5, 2.0, 10.0
xs = rng.normal(mu, sigma, (NREP, n))
xbar = xs.mean(axis=1)
S2 = xs.var(axis=1, ddof=1)             # 1/(n-1) estimator
S2n = xs.var(axis=1, ddof=0)            # 1/n estimator
w = (n - 1) * S2 / sigma ** 2

# known-mean version: sum (x_i - mu)^2 / sigma^2 has n dof
w_known = ((xs - mu) ** 2).sum(axis=1) / sigma ** 2

# independence of xbar and S^2: Gaussian vs exponential parent
xe = rng.exponential(1.0, (NREP, n))
corr_gauss = np.corrcoef(xbar, S2)[0, 1]
corr_exp = np.corrcoef(xe.mean(axis=1), xe.var(axis=1, ddof=1))[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
ax = axes[0]
bins = np.linspace(0, 20, 101)
ax.hist(w, bins=bins, density=True, color=SERIES[0], alpha=0.6,
        label=r"$(n-1)S^2/\sigma^2$, mean estimated")
ax.hist(w_known, bins=bins, density=True, histtype="step", color=SERIES[1], lw=1.4,
        label=r"$\sum(x_i-\mu)^2/\sigma^2$, true mean")
xx = np.linspace(0.02, 20, 400)
theory_line(ax, xx, stats.chi2.pdf(xx, n - 1), label=rf"$\chi^2_{{{n - 1}}}$")
ax.plot(xx, stats.chi2.pdf(xx, n), color="0.4", ls=":", lw=1.4, label=rf"$\chi^2_{{{n}}}$")
ax.set_xlabel("scaled sum of squares")
ax.set_ylabel("density")
ax.set_title(f"(a) $n={n}$ Gaussian measurements")
ax.legend(fontsize=7)

ax = axes[1]
sub = slice(0, 6000)
ax.plot(xbar[sub], S2[sub], ".", ms=1.5, color=SERIES[0], alpha=0.5, label="Gaussian parent")
ax.set_xlabel(r"sample mean $\bar{x}$")
ax.set_ylabel(r"sample variance $S^2$")
ax.set_title(r"(b) $\bar{x}$ and $S^2$ for a Gaussian parent")
ax.text(0.03, 0.93, f"corr = {corr_gauss:+.3f}", transform=ax.transAxes, fontsize=9)
savefig(fig, "ch02", "chi2_sampvar")

save_numbers("ch02", "25_chi2_sampvar", {
    "tbChiMeanFive": f"{mv[5][0]:.3f}", "tbChiVarFive": f"{mv[5][1]:.3f}",
    "tbChiMeanTen": f"{mv[10][0]:.3f}", "tbChiVarTen": f"{mv[10][1]:.3f}",
    "tbQFmean": f"{qf.mean():.3f}", "tbQFvar": f"{qf.var(ddof=1):.3f}",
    "tbNaiveMean": f"{naive.mean():.3f}", "tbNaiveVar": f"{naive.var(ddof=1):.3f}",
    "tbSVmeanS": f"{S2.mean():.4f}", "tbSVmeanSn": f"{S2n.mean():.4f}",
    "tbSVsigmaSq": f"{sigma ** 2:.1f}", "tbSVbiasTh": f"{(n - 1) / n * sigma ** 2:.4f}",
    "tbSVwMean": f"{w.mean():.3f}", "tbSVwVar": f"{w.var(ddof=1):.3f}",
    "tbSVcorrGauss": f"{corr_gauss:+.4f}", "tbSVcorrExp": f"{corr_exp:+.3f}",
    "tbSVn": n, "tbSVnrep": NREP,
})
