"""13_coverage.py -- coverage is a property of the procedure, not of one interval.

Question: (i) if 100 physicists each measure a Gaussian mean and quote xbar +- 1.96 sigma/sqrt(n),
how many intervals miss the truth?  What if they use the sample s in place of sigma with the
Normal quantile at small n?  (ii) for a Poisson mean with small counts, what is the true
coverage of the exact (Garwood) central interval, and of the quick n +- sqrt(n) interval,
as a function of the true mean nu?

Computes: a "ladder" of 100 intervals, long-run coverage from 2e5 experiments, the z-vs-t
small-sample undercoverage, and the exact coverage function
C(nu) = sum_n P(n | nu) 1[a(n) <= nu <= b(n)] on a fine nu grid (checked by Monte Carlo).

Writes: figures/ch04/coverage_ladder.pdf, figures/ch04/coverage_poisson.pdf,
        results/ch04/13_coverage.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "13_coverage")
setup()

# ---------- (i) Gaussian mean, sigma known ----------
mu_true, sigma, n = 5.0, 1.0, 10
z95 = stats.norm.isf(0.025)
x = rng.normal(mu_true, sigma, size=(100, n))
xbar = x.mean(1)
half = z95 * sigma / np.sqrt(n)
miss = np.abs(xbar - mu_true) > half

fig, ax = plt.subplots(figsize=(6.0, 3.6))
k = np.arange(100)
ax.vlines(k[~miss], xbar[~miss] - half, xbar[~miss] + half, color=SERIES[0], lw=1.2)
ax.vlines(k[miss], xbar[miss] - half, xbar[miss] + half, color=SERIES[1], lw=2.0,
          label=f"misses the truth ({miss.sum()} of 100)")
ax.plot(k, xbar, ".", color="0.25", ms=2.5)
ax.axhline(mu_true, color="k", lw=1.0, ls="--", label=r"true $\mu$ (fixed)")
ax.set_xlabel("experiment number")
ax.set_ylabel(r"95% interval $\bar x\pm1.96\,\sigma/\sqrt{n}$")
ax.set_xlim(-1, 100)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_ylim((xbar - half).min() - 0.1, (xbar + half).max() + 0.6)
savefig(fig, "ch04", "coverage_ladder")

M = 200_000
xb = rng.normal(mu_true, sigma / np.sqrt(n), size=M)
cov_known = np.mean(np.abs(xb - mu_true) <= half)
cov_known_err = np.sqrt(cov_known * (1 - cov_known) / M)
# 68.27% interval xbar +- sigma/sqrt(n)
cov_one_sigma = np.mean(np.abs(xb - mu_true) <= sigma / np.sqrt(n))

# ---------- sigma estimated by s: z-quantile vs t-quantile ----------
small = {}
for nn in (3, 5, 10, 30):
    xs = rng.normal(mu_true, sigma, size=(M, nn))
    m, s = xs.mean(1), xs.std(1, ddof=1)
    cz = np.mean(np.abs(m - mu_true) <= z95 * s / np.sqrt(nn))
    ct = np.mean(np.abs(m - mu_true) <= stats.t.isf(0.025, nn - 1) * s / np.sqrt(nn))
    small[nn] = (cz, ct)

# ---------- (ii) Poisson mean: exact coverage function ----------
def garwood(nobs, cl):
    a = (1 - cl) / 2
    lo = np.where(nobs == 0, 0.0, 0.5 * stats.chi2.ppf(a, 2 * np.maximum(nobs, 1)))
    hi = 0.5 * stats.chi2.ppf(1 - a, 2 * (nobs + 1))
    return lo, hi

def wald(nobs, cl):
    zq = stats.norm.isf((1 - cl) / 2)
    return nobs - zq * np.sqrt(nobs), nobs + zq * np.sqrt(nobs)

nmax = 80
nn_all = np.arange(nmax + 1)
nu = np.linspace(0.02, 15, 3000)
pmf = stats.poisson.pmf(nn_all[None, :], nu[:, None])

def coverage(method, cl):
    lo, hi = method(nn_all, cl)
    inside = (lo[None, :] <= nu[:, None]) & (nu[:, None] <= hi[None, :])
    return (pmf * inside).sum(1)

cov_g68, cov_g90 = coverage(garwood, 0.6827), coverage(garwood, 0.90)
cov_w68, cov_w90 = coverage(wald, 0.6827), coverage(wald, 0.90)

# Monte Carlo spot check of the exact sum
nu_chk = 2.7
nsim = rng.poisson(nu_chk, size=M)
lo, hi = garwood(nsim, 0.90)
cov_mc = np.mean((lo <= nu_chk) & (nu_chk <= hi))
i_chk = np.argmin(np.abs(nu - nu_chk))
cov_ex = coverage(garwood, 0.90)[i_chk]
lo, hi = wald(nsim, 0.90)
cov_mc_w = np.mean((lo <= nu_chk) & (nu_chk <= hi))

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
for ax, (cg, cw, cl) in zip(axs, ((cov_g68, cov_w68, 0.6827), (cov_g90, cov_w90, 0.90))):
    ax.plot(nu, cg, color=SERIES[0], lw=1.1, label="exact (Garwood) interval")
    ax.plot(nu, cw, color=SERIES[1], lw=1.1, label=r"$n\pm z\sqrt{n}$")
    ax.axhline(cl, color="k", lw=1.0, ls="--", label="nominal")
    ax.set_xlabel(r"true Poisson mean $\nu$")
    ax.set_title(f"nominal {100*cl:.2f}%" if cl < 0.9 else "nominal 90%")
    ax.set_xlim(0, 15)
axs[0].set_ylabel(r"coverage $C(\nu)$")
axs[0].set_ylim(0, 1.02)
axs[1].legend(loc="lower right", fontsize=7.5)
fig.tight_layout()
savefig(fig, "ch04", "coverage_poisson")

save_numbers("ch04", "13_coverage", {
    "FourBLadderMiss": int(miss.sum()), "FourBLadderN": n,
    "FourBCovKnown": f"{cov_known:.4f}", "FourBCovKnownErr": f"{cov_known_err:.4f}",
    "FourBCovOneSigma": f"{cov_one_sigma:.4f}", "FourBSims": M,
    "FourBCovZthree": f"{small[3][0]:.3f}", "FourBCovTthree": f"{small[3][1]:.3f}",
    "FourBCovZfive": f"{small[5][0]:.3f}", "FourBCovTfive": f"{small[5][1]:.3f}",
    "FourBCovZten": f"{small[10][0]:.3f}", "FourBCovTten": f"{small[10][1]:.3f}",
    "FourBCovZthirty": f"{small[30][0]:.3f}", "FourBCovTthirty": f"{small[30][1]:.3f}",
    "FourBGninetyMin": f"{cov_g90.min():.3f}", "FourBGninetyMean": f"{cov_g90.mean():.3f}",
    "FourBGsixtyeightMin": f"{cov_g68.min():.3f}", "FourBGsixtyeightMean": f"{cov_g68.mean():.3f}",
    "FourBWninetyMin": f"{cov_w90.min():.3f}", "FourBWsixtyeightMin": f"{cov_w68.min():.3f}",
    "FourBWninetyAtOne": f"{cov_w90[np.argmin(np.abs(nu - 1.0))]:.3f}",
    "FourBGninetyAtOne": f"{cov_g90[np.argmin(np.abs(nu - 1.0))]:.3f}",
    "FourBNuChk": nu_chk, "FourBCovChkExact": f"{cov_ex:.4f}", "FourBCovChkMC": f"{cov_mc:.4f}",
    "FourBCovChkMCw": f"{cov_mc_w:.4f}",
})
print("ladder misses", miss.sum(), "cov known", cov_known, "one sigma", cov_one_sigma)
print("small n (z, t):", small)
print(f"Garwood 90 min {cov_g90.min():.4f} mean {cov_g90.mean():.4f}; 68 min {cov_g68.min():.4f} mean {cov_g68.mean():.4f}")
print(f"Wald 90 min {cov_w90.min():.4f}; 68 min {cov_w68.min():.4f}; at nu=1: W {cov_w90[np.argmin(np.abs(nu-1))]:.3f} G {cov_g90[np.argmin(np.abs(nu-1))]:.3f}")
print(f"check at nu={nu_chk}: exact {cov_ex:.4f}, MC {cov_mc:.4f}, Wald MC {cov_mc_w:.4f}")
