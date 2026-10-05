"""05_covariance_noise.py -- how noisy is a covariance or correlation matrix estimated from samples?

Question: (i) if p = 20 quantities are truly independent and we estimate their
correlation matrix from n = 50 samples, how large are the spurious correlations?
(ii) for one pair with true rho = 0.6 and n = 20, what is the sampling distribution
of the sample correlation r, and does Fisher's z = artanh(r) make it Gaussian with
standard deviation 1/sqrt(n-3)?  (iii) is the inverse of a sample covariance matrix
biased high by (n-1)/(n-p-2) (the Hartlap factor)?

Writes: figures/ch03/spurious_correlations.pdf, figures/ch03/corr_fisher_z.pdf,
        figures/ch03/hartlap.pdf, results/ch03/05_covariance_noise.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line, DIV

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch03", "05_covariance_noise")
setup()

# ---------- (i) spurious correlations among independent variables ----------
p, n = 20, 50
X = rng.standard_normal((n, p))                  # rows = samples, columns = variables
R = np.corrcoef(X, rowvar=False)
off = R[np.triu_indices(p, k=1)]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
im = axs[0].imshow(R, cmap=DIV, vmin=-1, vmax=1)
axs[0].set_title(r"sample correlation, true $=\mathbb{1}$"); axs[0].grid(False)
axs[0].set_xticks([0, 9, 19]); axs[0].set_xticklabels([1, 10, 20])
axs[0].set_yticks([0, 9, 19]); axs[0].set_yticklabels([1, 10, 20])
fig.colorbar(im, ax=axs[0], fraction=0.046)
# many repetitions for a smooth histogram of off-diagonal elements
many = []
for _ in range(400):
    Rm = np.corrcoef(rng.standard_normal((n, p)), rowvar=False)
    many.append(Rm[np.triu_indices(p, k=1)])
many = np.concatenate(many)
axs[1].hist(many, bins=np.linspace(-0.6, 0.6, 81), density=True, color=SERIES[0], alpha=0.6,
            label="off-diagonal $r_{ab}$")
rr = np.linspace(-0.6, 0.6, 400)
theory_line(axs[1], rr, stats.norm.pdf(rr, 0, 1 / np.sqrt(n - 1)), label=r"$\mathcal{N}(0,1/(n-1))$")
axs[1].set_xlabel(r"$r_{ab}$  ($n=50$)"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "spurious_correlations")

# ---------- (ii) r and Fisher z for rho = 0.6, n = 20 ----------
rho, n2, M = 0.6, 20, 100_000
L = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
Z = rng.standard_normal((M, n2, 2)) @ L.T
a = Z[..., 0] - Z[..., 0].mean(axis=1, keepdims=True)
b = Z[..., 1] - Z[..., 1].mean(axis=1, keepdims=True)
r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
z = np.arctanh(r)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
axs[0].hist(r, bins=np.linspace(-0.2, 1, 121), density=True, color=SERIES[0], alpha=0.6)
grid = np.linspace(-0.2, 1, 400)
theory_line(axs[0], grid, stats.norm.pdf(grid, rho, (1 - rho**2) / np.sqrt(n2)),
            label=r"$\mathcal{N}(\rho,(1-\rho^2)^2/n)$")
axs[0].axvline(rho, color="0.3", ls=":", lw=0.9)
axs[0].set_xlabel(r"sample correlation $r$ ($\rho=0.6$, $n=20$)"); axs[0].legend(fontsize=7)
axs[1].hist(z, bins=np.linspace(-0.4, 1.8, 121), density=True, color=SERIES[2], alpha=0.6)
gz = np.linspace(-0.4, 1.8, 400)
theory_line(axs[1], gz, stats.norm.pdf(gz, np.arctanh(rho), 1 / np.sqrt(n2 - 3)),
            label=r"$\mathcal{N}(\mathrm{artanh}\,\rho,1/(n-3))$")
axs[1].set_xlabel(r"Fisher $z=\mathrm{artanh}\,r$"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "corr_fisher_z")

# ---------- (iii) bias of the inverse sample covariance ----------
n3 = 30
ps = np.array([1, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 26])
ratio = []
for pp in ps:
    vals = []
    for _ in range(3000):
        Y = rng.standard_normal((n3, pp))
        C = np.atleast_2d(np.cov(Y, rowvar=False))
        vals.append(np.trace(np.linalg.inv(C)) / pp)      # true C = identity
    ratio.append(np.mean(vals))
ratio = np.array(ratio)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(ps, ratio, "o", color=SERIES[0], label=r"simulated $\langle \mathrm{tr}\,\hat{C}^{-1}\rangle/p$")
pg = np.linspace(1, 26, 200)
theory_line(ax, pg, (n3 - 1) / (n3 - pg - 2), label=r"$(n-1)/(n-p-2)$")
ax.set_yscale("log"); ax.set_xlabel(r"number of data bins $p$ ($n=30$ simulations)")
ax.set_ylabel("overestimate of inverse covariance"); ax.legend(fontsize=8)
savefig(fig, "ch03", "hartlap")

i10 = int(np.where(ps == 10)[0][0])
save_numbers("ch03", "05_covariance_noise", {
    "ThreeASpurSd": off.std(ddof=1), "ThreeASpurSdTh": 1 / np.sqrt(n - 1),
    "ThreeASpurMax": np.abs(off).max(), "ThreeANpairs": len(off),
    "ThreeARMean": r.mean(), "ThreeARMeanTh": rho - rho * (1 - rho**2) / (2 * n2),
    "ThreeARSd": r.std(ddof=1), "ThreeARSdTh": (1 - rho**2) / np.sqrt(n2),
    "ThreeAZSd": z.std(ddof=1), "ThreeAZSdTh": 1 / np.sqrt(n2 - 3),
    "ThreeARSkew": stats.skew(r), "ThreeAZSkew": stats.skew(z),
    "ThreeAHartlapTen": ratio[i10], "ThreeAHartlapTenTh": (n3 - 1) / (n3 - 10 - 2),
})
