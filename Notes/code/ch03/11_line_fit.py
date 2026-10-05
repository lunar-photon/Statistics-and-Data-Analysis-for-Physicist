"""11_line_fit.py -- least squares as Gaussian maximum likelihood: a straight line, then correlated noise.

Question: (i) fit y = a + b x to N = 10 points with known Gaussian errors.  Do the
normal-equation estimates have the covariance (A^T C^-1 A)^-1, and is chi^2_min distributed
as chi^2 with N - 2 degrees of freedom?  What happens to chi^2_min when the model is wrong?
(ii) if the noise is correlated (exponential correlation between neighbours) but we fit
with the diagonal chi^2, is the slope still unbiased?  Is its quoted error right?  Does the
full chi^2 = r^T C^-1 r do better?

Computes: one data set and its fit; 10^4 experiments for the covariance, chi^2_min and a
          wrong (constant) model; 10^4 correlated-noise experiments fitted both ways.
Writes:   figures/ch03/line_fit.pdf, figures/ch03/line_fit_corr.pdf, results/ch03/11_line_fit.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch03", "11_line_fit")
setup()
a_true, b_true, N, M = 1.0, 0.5, 10, 10_000
x = np.arange(1, N + 1, dtype=float)
sig = 0.4 + 0.08 * x                       # errors grow along the axis
A = np.column_stack([np.ones(N), x])       # design matrix: model = A @ (a, b)


def gls(y, C):
    """generalised least squares: theta = (A^T C^-1 A)^-1 A^T C^-1 y and its covariance"""
    Ci = np.linalg.inv(C)
    U = np.linalg.inv(A.T @ Ci @ A)
    th = U @ A.T @ Ci @ y.T
    r = y - (A @ th).T
    chi2 = np.einsum("...i,ij,...j->...", r, Ci, r)
    return th.T, U, chi2


# ---------- (i) independent errors ----------
C = np.diag(sig**2)
y0 = a_true + b_true * x + sig * rng.standard_normal(N)
th0, U, chi0 = gls(y0, C)
Y = a_true + b_true * x + sig * rng.standard_normal((M, N))
th, _, chi = gls(Y, C)
cov_mc = np.cov(th.T)
p0 = stats.chi2.sf(chi0, N - 2)
# a wrong model: a constant fitted to the same straight-line data
A1 = np.ones((N, 1))
w = 1 / sig**2
c_hat = (Y * w).sum(1) / w.sum()
chi_wrong = (((Y - c_hat[:, None]) / sig) ** 2).sum(1)

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0))
axs[0].errorbar(x, y0, sig, fmt="o", color=SERIES[0], ms=3, capsize=2, label="data")
xx = np.linspace(0, N + 1, 100)
Axx = np.column_stack([np.ones_like(xx), xx])
band = np.sqrt(np.einsum("ij,jk,ik->i", Axx, U, Axx))
axs[0].plot(xx, Axx @ th0, color=SERIES[1], label="ML = least-squares line")
axs[0].fill_between(xx, Axx @ th0 - band, Axx @ th0 + band, color=SERIES[1], alpha=0.2, lw=0)
axs[0].plot(xx, a_true + b_true * xx, color="k", ls="--", lw=1, label="true line")
axs[0].set_xlabel(r"$x$"); axs[0].set_ylabel(r"$y$"); axs[0].legend(fontsize=7)
cc = np.linspace(0, 40, 400)
axs[1].hist(chi, bins=np.linspace(0, 40, 81), density=True, color=SERIES[0], alpha=0.55, label=r"line: $\chi^2_{\min}$")
axs[1].hist(chi_wrong, bins=np.linspace(0, 40, 81), density=True, histtype="step", color=SERIES[3], lw=1.3,
            label=r"constant: $\chi^2_{\min}$")
theory_line(axs[1], cc, stats.chi2.pdf(cc, N - 2), label=rf"$\chi^2_{{{N-2}}}$")
axs[1].set_xlabel(r"$\chi^2_{\min}$"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "line_fit")

# ---------- (ii) correlated noise ----------
ell = 2.0                                   # correlation length in units of the spacing
R = np.exp(-np.abs(x[:, None] - x[None, :]) / ell)
Cc = np.outer(sig, sig) * R
L = np.linalg.cholesky(Cc)
Yc = a_true + b_true * x + (L @ rng.standard_normal((N, M))).T
th_d, U_d, chi_d = gls(Yc, C)               # pretend the errors are independent
th_f, U_f, chi_f = gls(Yc, Cc)              # use the full covariance
# true covariance of the diagonal estimator: B Cc B^T with B = U_d A^T C^-1
B = U_d @ A.T @ np.linalg.inv(C)
V_d_true = B @ Cc @ B.T

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0))
bb = np.linspace(b_true - 0.25, b_true + 0.25, 81)
axs[0].hist(th_d[:, 1], bins=bb, density=True, histtype="step", color=SERIES[3], lw=1.4, label=r"diagonal $\chi^2$")
axs[0].hist(th_f[:, 1], bins=bb, density=True, color=SERIES[0], alpha=0.5, label=r"full $\mathbf{r}^{T}\mathbf{C}^{-1}\mathbf{r}$")
g = np.linspace(bb[0], bb[-1], 300)
axs[0].plot(g, stats.norm.pdf(g, b_true, np.sqrt(U_d[1, 1])), color=SERIES[3], ls=":", lw=1.2, label="quoted (diagonal)")
axs[0].set_xlabel(r"slope $\hat b$"); axs[0].legend(fontsize=6.5)
axs[1].hist(chi_d, bins=np.linspace(0, 30, 61), density=True, histtype="step", color=SERIES[3], lw=1.4, label="diagonal")
axs[1].hist(chi_f, bins=np.linspace(0, 30, 61), density=True, color=SERIES[0], alpha=0.5, label="full")
theory_line(axs[1], cc[cc < 30], stats.chi2.pdf(cc[cc < 30], N - 2), label=rf"$\chi^2_{{{N-2}}}$")
axs[1].set_xlabel(r"$\chi^2_{\min}$"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "line_fit_corr")

save_numbers("ch03", "11_line_fit", {
    "ThreeBLineA": th0[0], "ThreeBLineB": th0[1],
    "ThreeBLineSa": np.sqrt(U[0, 0]), "ThreeBLineSb": np.sqrt(U[1, 1]),
    "ThreeBLineRho": U[0, 1] / np.sqrt(U[0, 0] * U[1, 1]),
    "ThreeBLineChi": chi0, "ThreeBLineP": p0,
    "ThreeBLineSaMC": np.sqrt(cov_mc[0, 0]), "ThreeBLineSbMC": np.sqrt(cov_mc[1, 1]),
    "ThreeBLineRhoMC": cov_mc[0, 1] / np.sqrt(cov_mc[0, 0] * cov_mc[1, 1]),
    "ThreeBLineChiMean": chi.mean(), "ThreeBLineChiVar": chi.var(),
    "ThreeBLineWrongMean": chi_wrong.mean(),
    "ThreeBCorrMeanBd": th_d[:, 1].mean(), "ThreeBCorrMeanBf": th_f[:, 1].mean(),
    "ThreeBCorrQuoted": np.sqrt(U_d[1, 1]), "ThreeBCorrTrueD": np.sqrt(V_d_true[1, 1]),
    "ThreeBCorrSdD": th_d[:, 1].std(ddof=1), "ThreeBCorrSdF": th_f[:, 1].std(ddof=1),
    "ThreeBCorrQuotedF": np.sqrt(U_f[1, 1]),
    "ThreeBCorrChiD": chi_d.mean(), "ThreeBCorrChiF": chi_f.mean(),
})
