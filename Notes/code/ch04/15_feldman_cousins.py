"""15_feldman_cousins.py -- limits near a physical boundary: classical, shifted, Bayesian,
flip-flop, Feldman-Cousins and CLs, and the coverage of each.

Question: a Gaussian measurement x of a quantity mu >= 0 (sigma = 1), and a Poisson count n on a
known background b = 3 with signal mu >= 0.  What does each recipe report when the data sit at
or below the boundary, and what fraction of repeated experiments does each recipe's interval
actually cover the true mu?

Computes: (Gaussian) Cowan's numbers for x = -2; 95% upper limits versus x for the classical,
shifted and flat-prior Bayesian recipes; the Feldman-Cousins (likelihood-ratio ordered) belt at
90% and 95% by brute force on a grid; the coverage of every recipe as a function of mu,
including the data-dependent "flip-flop" choice.  (Poisson) Feldman-Cousins Table I at mu = 0.5,
the FC belt for b = 3 and its intervals for n0 = 0..10, the classical and CLs upper limits, the
flat-prior Bayesian limit (equal to CLs for a counting experiment), and their coverage.

Writes: figures/ch04/boundary_limits.pdf, figures/ch04/fc_belts.pdf,
        figures/ch04/fc_coverage.pdf, figures/ch04/fc_poisson_limits.pdf,
        results/ch04/15_feldman_cousins.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize, integrate

setup()
Phi, Phinv = stats.norm.cdf, stats.norm.ppf

# =====================================================================================
# Gaussian measurement, sigma = 1, physical region mu >= 0
# =====================================================================================
x_obs = -2.0
up95 = x_obs + Phinv(0.95)                     # classical (9.39)
up99 = x_obs + Phinv(0.99)
up_shift = max(x_obs, 0) + Phinv(0.95)          # shifted (9.40)

def bayes_up(x, cl):
    """flat prior on mu >= 0: posterior is N(x,1) truncated to mu >= 0 (derivation in text)."""
    return x + Phinv(1 - (1 - cl) * Phi(x))

up_bayes = bayes_up(x_obs, 0.95)

# ---------- the Feldman-Cousins belt by brute force ----------
xg = np.arange(-8.0, 14.0, 0.002)
dx = xg[1] - xg[0]
mu_best = np.maximum(xg, 0.0)

def fc_gauss_belt(mus, cl):
    """for each mu: accept x in decreasing order of R = P(x|mu)/P(x|mu_best) until prob = cl."""
    x1, x2 = np.empty_like(mus), np.empty_like(mus)
    for k, mu in enumerate(mus):
        p = stats.norm.pdf(xg, mu, 1.0)
        lnR = -0.5 * (xg - mu) ** 2 + 0.5 * (xg - mu_best) ** 2
        order = np.argsort(-lnR)
        cum = np.cumsum(p[order]) * dx
        keep = order[: np.searchsorted(cum, cl) + 1]
        x1[k], x2[k] = xg[keep].min(), xg[keep].max()
    return x1, x2

mus = np.arange(0.0, 7.0, 0.005)
fc90_x1, fc90_x2 = fc_gauss_belt(mus, 0.90)
fc95_x1, fc95_x2 = fc_gauss_belt(mus, 0.95)

def fc_interval(x0, x1, x2):
    inside = (x1 <= x0) & (x0 <= x2)
    m = mus[inside]
    return (m.min(), m.max()) if m.size else (np.nan, np.nan)

x_table = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
fc_tab = {x0: fc_interval(x0, fc90_x1, fc90_x2) for x0 in x_table}
# transition: smallest x0 whose interval excludes mu = 0 (FC: 1.28 at 90%)
x_trans = fc90_x2[0]

# ---------- coverage of each recipe as a function of true mu (integrate over x) ----------
z90, z95, zc90 = Phinv(0.90), Phinv(0.95), Phinv(0.95)   # 1.28, 1.645, and central-90 quantile 1.645

def cover_prob(mu, lo_fn, hi_fn):
    lo, hi = lo_fn(xg), hi_fn(xg)
    ok = (lo <= mu) & (mu <= hi)
    return np.sum(stats.norm.pdf(xg, mu, 1.0) * ok) * dx

def ff_lo(x):          # Physicist X: upper limit below 3 sigma (from max(x,0)), central above
    return np.where(x < 3.0, 0.0, x - zc90)

def ff_hi(x):
    return np.where(x < 3.0, np.maximum(x, 0.0) + z90, x + zc90)

mu_cov = np.arange(0.0, 6.0, 0.01)
cov_ff = np.array([cover_prob(m, ff_lo, ff_hi) for m in mu_cov])
# FC coverage: by construction P(x in [x1(mu), x2(mu)] | mu) = cl ; check on the grid
cov_fc = np.array([Phi(fc90_x2[i] - mu) - Phi(fc90_x1[i] - mu) for i, mu in
                   enumerate(mus) if mu < 6.0])
cov_ff_at2 = cover_prob(2.0, ff_lo, ff_hi)
cov_ff_min = cov_ff.min()

# 95% upper-limit recipes: classical, shifted, Bayesian flat prior, FC-95 upper end
x_lim = np.linspace(-4.0, 3.0, 701)
lim_class = x_lim + z95
lim_shift = np.maximum(x_lim, 0) + z95
lim_bayes = bayes_up(x_lim, 0.95)
lim_fc95 = np.array([fc_interval(x0, fc95_x1, fc95_x2)[1] for x0 in x_lim])
zero = lambda x: np.full_like(x, -np.inf)
cov_class = np.array([cover_prob(m, zero, lambda x: x + z95) for m in mu_cov])
cov_shift = np.array([cover_prob(m, zero, lambda x: np.maximum(x, 0) + z95) for m in mu_cov])
cov_bayes = np.array([cover_prob(m, zero, lambda x: bayes_up(x, 0.95)) for m in mu_cov])
i0 = 0
cov_bayes_min = cov_bayes.min()
mu_bayes_min = mu_cov[np.argmin(cov_bayes)]

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axs[0]
ax.plot(x_lim, lim_class, color=SERIES[0], label="classical")
ax.plot(x_lim, lim_shift, color=SERIES[1], ls="--", label="shifted")
ax.plot(x_lim, lim_bayes, color=SERIES[2], label="Bayesian, flat prior")
ax.plot(x_lim, lim_fc95, color=SERIES[6], ls="-.", label="Feldman-Cousins")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(r"observed $x$ (units of $\sigma$)")
ax.set_ylabel(r"95% upper limit on $\mu$")
ax.set_ylim(-2.5, 5)
ax.legend(fontsize=7.5, loc="upper left")
ax = axs[1]
ax.plot(mu_cov, cov_class, color=SERIES[0], label="classical")
ax.plot(mu_cov, cov_shift, color=SERIES[1], ls="--", label="shifted")
ax.plot(mu_cov, cov_bayes, color=SERIES[2], label="Bayesian, flat prior")
ax.axhline(0.95, color="k", lw=0.9, ls=":")
ax.set_xlabel(r"true $\mu$")
ax.set_ylabel("coverage of the upper limit")
ax.set_ylim(0.9, 1.005)
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
savefig(fig, "ch04", "boundary_limits")

# =====================================================================================
# Poisson count with known background b = 3
# =====================================================================================
b = 3.0
CL = 0.90
nmax = 80
nn = np.arange(nmax + 1)

def fc_poisson_sets(mu_grid, b, cl=CL):
    """acceptance sets [n1, n2] for each mu, ranked by R = P(n|mu)/P(n|mu_best)."""
    mub = np.maximum(nn - b, 0.0)
    pbest = stats.poisson.pmf(nn, mub + b)
    n1 = np.empty(mu_grid.size, int); n2 = np.empty(mu_grid.size, int)
    for k, mu in enumerate(mu_grid):
        p = stats.poisson.pmf(nn, mu + b)
        order = np.argsort(-(p / pbest), kind="stable")
        cum = np.cumsum(p[order])
        keep = order[: np.searchsorted(cum, cl) + 1]
        n1[k], n2[k] = keep.min(), keep.max()
    return n1, n2

# Table I of FC (mu = 0.5)
mu05 = 0.5
p05 = stats.poisson.pmf(nn[:11], mu05 + b)
mub = np.maximum(nn[:11] - b, 0.0)
pb05 = stats.poisson.pmf(nn[:11], mub + b)
R05 = p05 / pb05
rank = np.empty(11, int); rank[np.argsort(-R05)] = np.arange(1, 12)
n1_05, n2_05 = fc_poisson_sets(np.array([mu05]), b)
acc05 = stats.poisson.pmf(np.arange(n1_05[0], n2_05[0] + 1), mu05 + b).sum()

mu_p = np.arange(0.0, 45.0, 0.005)
n1, n2 = fc_poisson_sets(mu_p, b)

def fc_poisson_interval(n0, n1=n1, n2=n2, grid=mu_p):
    inside = (n1 <= n0) & (n0 <= n2)
    m = grid[inside]
    return m.min(), m.max()

fc_int = [fc_poisson_interval(k) for k in range(11)]

# classical one-sided upper limit: P(n <= n0 | mu + b) = 1 - CL   (Cowan 9.49-9.50)
def classical_up(n0, b, cl=CL):
    return 0.5 * stats.chi2.ppf(cl, 2 * (n0 + 1)) - b

def cls_up(n0, b, cl=CL):
    f = lambda mu: stats.poisson.cdf(n0, mu + b) / stats.poisson.cdf(n0, b) - (1 - cl)
    return optimize.brentq(f, 0.0, 100.0)

def bayes_poisson_up(n0, b, cl=CL):
    """flat prior mu >= 0: integrate the likelihood (mu+b)^n0 e^-(mu+b) numerically."""
    L = lambda mu: stats.poisson.pmf(n0, mu + b)
    tot = integrate.quad(L, 0, np.inf)[0]
    f = lambda u: integrate.quad(L, 0, u)[0] / tot - cl
    return optimize.brentq(f, 1e-9, 100.0)

cls_tab = [cls_up(k, b) for k in range(11)]
bay_tab = [bayes_poisson_up(k, b) for k in range(11)]
cls_bayes_maxdiff = max(abs(c - y) for c, y in zip(cls_tab, bay_tab))
cl_tab = [classical_up(k, b) for k in range(11)]
central_up_n0 = 0.5 * stats.chi2.ppf(0.95, 2) - b        # upper end of central 90% at n0 = 0

# coverage versus true mu for FC, classical UL, CLs
mu_c = np.arange(0.0, 10.0, 0.01)
fc_lo = np.array([fc_poisson_interval(k)[0] for k in nn[:40]])
fc_hi = np.array([fc_poisson_interval(k)[1] for k in nn[:40]])
cl_hi = np.array([classical_up(k, b) for k in nn[:40]])
cls_hi = np.array([cls_up(k, b) for k in nn[:40]])
pm = stats.poisson.pmf(nn[None, :40], mu_c[:, None] + b)
cov_fcp = (pm * ((fc_lo[None, :] <= mu_c[:, None]) & (mu_c[:, None] <= fc_hi[None, :]))).sum(1)
cov_clp = (pm * (mu_c[:, None] <= cl_hi[None, :])).sum(1)
cov_clsp = (pm * (mu_c[:, None] <= cls_hi[None, :])).sum(1)

# n0 = 0 and 3 upper limits versus background
bs = np.arange(0.0, 8.001, 0.02)
mu_small = np.arange(0.0, 9.0, 0.005)
fc_up_b = {0: [], 3: []}
for bb in bs:
    a1, a2 = fc_poisson_sets(mu_small, bb)
    for n0 in (0, 3):
        inside = (a1 <= n0) & (n0 <= a2)
        fc_up_b[n0].append(mu_small[inside].max())
fc_up_b_raw = {k: np.array(v) for k, v in fc_up_b.items()}
# FC's fix of the discreteness pathology: force mu2(b) to be non-increasing in b by lengthening
# intervals, i.e. replace mu2(b) by the largest value found at any larger background.
fc_up_b = {k: np.maximum.accumulate(v[::-1])[::-1] for k, v in fc_up_b_raw.items()}
i_b3 = np.argmin(np.abs(bs - b))

# ---------- figures ----------
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.3))
ax = axs[0]
sel = mus <= 5.0
ax.fill_betweenx(mus[sel], fc90_x1[sel], fc90_x2[sel], color=SERIES[0], alpha=0.18, lw=0,
                 label="Feldman-Cousins 90%")
ax.plot(fc90_x1[sel], mus[sel], color=SERIES[0], lw=1.0)
ax.plot(fc90_x2[sel], mus[sel], color=SERIES[0], lw=1.0)
# flip-flop belt: for each mu, the x values whose quoted interval contains mu
ff_x1 = np.array([xg[(ff_lo(xg) <= m) & (m <= ff_hi(xg))].min() for m in mus[sel]])
ff_x2 = np.array([xg[(ff_lo(xg) <= m) & (m <= ff_hi(xg))].max() for m in mus[sel]])
ax.plot(ff_x1, mus[sel], color=SERIES[1], lw=1.1, ls="--", label="flip-flop")
ax.plot(ff_x2, mus[sel], color=SERIES[1], lw=1.1, ls="--")
ax.set_xlim(-2, 5); ax.set_ylim(0, 5)
ax.set_xlabel(r"measured $x$"); ax.set_ylabel(r"true $\mu$")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("Gaussian, $\\mu\\geq0$", fontsize=9)
ax = axs[1]
for mu in np.arange(0.0, 12.01, 0.5):
    i = np.argmin(np.abs(mu_p - mu))
    ax.plot([n1[i] - 0.3, n2[i] + 0.3], [mu, mu], color=SERIES[0], lw=2.0, alpha=0.55,
            solid_capstyle="butt")
lo0, hi0 = fc_int[0]
ax.plot([0, 0], [lo0, hi0], color=SERIES[1], lw=4, solid_capstyle="butt",
        label=rf"$n_0=0$: [0, {hi0:.2f}]")
lo6, hi6 = fc_int[6]
ax.plot([6, 6], [lo6, hi6], color=SERIES[2], lw=4, solid_capstyle="butt",
        label=rf"$n_0=6$: [{lo6:.2f}, {hi6:.2f}]")
ax.set_xlim(-0.8, 20); ax.set_ylim(0, 12.5)
ax.set_xlabel(r"observed count $n$"); ax.set_ylabel(r"signal mean $\mu$")
ax.set_xticks(range(0, 21, 2))
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("Poisson, $b=3$", fontsize=9)
fig.tight_layout()
savefig(fig, "ch04", "fc_belts")

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.1))
ax = axs[0]
ax.plot(mu_cov, cov_ff, color=SERIES[1], label="flip-flop")
ax.plot(mus[mus < 6.0], cov_fc, color=SERIES[0], label="Feldman-Cousins")
ax.axhline(0.9, color="k", lw=0.9, ls=":")
ax.set_xlabel(r"true $\mu$"); ax.set_ylabel("coverage")
ax.set_ylim(0.8, 1.0); ax.legend(fontsize=7.5, loc="lower right")
ax.set_title("Gaussian, nominal 90%", fontsize=9)
ax = axs[1]
ax.plot(mu_c, cov_fcp, color=SERIES[0], lw=1.0, label="Feldman-Cousins")
ax.plot(mu_c, cov_clp, color=SERIES[1], lw=1.0, ls="--", label="classical upper limit")
ax.plot(mu_c, cov_clsp, color=SERIES[2], lw=1.0, label=r"CL$_s$ upper limit")
ax.axhline(0.9, color="k", lw=0.9, ls=":")
ax.set_xlabel(r"true signal $\mu$"); ax.set_ylim(0.8, 1.005)
ax.legend(fontsize=7.5, loc="lower right")
ax.set_title("Poisson, $b=3$, nominal 90%", fontsize=9)
fig.tight_layout()
savefig(fig, "ch04", "fc_coverage")

fig, ax = plt.subplots(figsize=(4.6, 3.2))
for n0, ls in ((0, "-"), (3, "--")):
    ax.plot(bs, classical_up(n0, bs), color=SERIES[1], ls=ls, label=rf"classical, $n_0={n0}$")
    ax.plot(bs, fc_up_b[n0], color=SERIES[0], ls=ls, label=rf"Feldman-Cousins, $n_0={n0}$")
    ax.plot(bs, [cls_up(n0, bb) for bb in bs], color=SERIES[2], ls=ls,
            label=rf"CL$_s$ = flat-prior Bayes, $n_0={n0}$")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(r"known background $b$"); ax.set_ylabel(r"90% upper limit on $\mu$"); ax.set_xlim(0, 7)
ax.set_ylim(-3, 8)
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
savefig(fig, "ch04", "fc_poisson_limits")

# ---------- numbers ----------
names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
vals = {
    "FourBBdUpNinetyFive": f"{up95:.3f}", "FourBBdUpNinetyNine": f"{up99:.3f}",
    "FourBBdUpShift": f"{up_shift:.3f}", "FourBBdUpBayes": f"{up_bayes:.2f}",
    "FourBBdUpFCNinetyFive": f"{fc_interval(x_obs, fc95_x1, fc95_x2)[1]:.2f}",
    "FourBBdCovBayesMin": f"{cov_bayes_min:.3f}", "FourBBdMuBayesMin": f"{mu_bayes_min:.2f}",
    "FourBBdCovShiftZero": f"{cov_shift[0]:.3f}",
    "FourBFfCovTwo": f"{cov_ff_at2:.3f}", "FourBFfCovMin": f"{cov_ff_min:.3f}",
    "FourBFcCovMin": f"{cov_fc.min():.4f}", "FourBFcCovMax": f"{cov_fc.max():.4f}",
    "FourBFcTrans": f"{x_trans:.2f}",
    "FourBPtAccLo": int(n1_05[0]), "FourBPtAccHi": int(n2_05[0]), "FourBPtAcc": f"{acc05:.3f}",
    "FourBPtCentralUpZero": f"{central_up_n0:.3f}",
    "FourBPtCovFCMin": f"{cov_fcp.min():.3f}", "FourBPtCovClMin": f"{cov_clp.min():.3f}",
    "FourBPtCovCLsMin": f"{cov_clsp.min():.3f}",
    "FourBPtCovCLsZero": f"{cov_clsp[0]:.3f}", "FourBPtCovFCZero": f"{cov_fcp[0]:.3f}",
    "FourBPtCovClZero": f"{cov_clp[0]:.3f}",
    "FourBPtCLsBayesDiff": float(f"{cls_bayes_maxdiff:.1e}"),
    "FourBPtFCUpZeroBzero": f"{fc_up_b[0][0]:.2f}", "FourBPtFCUpZeroBeight": f"{fc_up_b[0][-1]:.2f}",
    "FourBPtFCZeroHiFixed": f"{fc_up_b[0][i_b3]:.2f}", "FourBPtFCZeroHiRaw": f"{fc_up_b_raw[0][i_b3]:.2f}",
}
for x0, nm in zip(x_table, ["MinusTwo", "MinusOne", "Zero", "One", "Two", "Three"]):
    vals[f"FourBFcG{nm}Lo"] = f"{fc_tab[x0][0]:.2f}"
    vals[f"FourBFcG{nm}Hi"] = f"{fc_tab[x0][1]:.2f}"
for k in range(11):
    vals[f"FourBPtR{names[k]}"] = f"{R05[k]:.3f}"
    vals[f"FourBPtP{names[k]}"] = f"{p05[k]:.3f}"
    vals[f"FourBPtRank{names[k]}"] = int(rank[k])
    vals[f"FourBPtFC{names[k]}Lo"] = f"{fc_int[k][0]:.2f}"
    vals[f"FourBPtFC{names[k]}Hi"] = f"{fc_int[k][1]:.2f}"
    vals[f"FourBPtCl{names[k]}"] = f"{cl_tab[k]:.2f}"
    vals[f"FourBPtCLs{names[k]}"] = f"{cls_tab[k]:.2f}"
save_numbers("ch04", "15_feldman_cousins", vals)

print(f"x=-2: classical95 {up95:.3f}, 99 {up99:.3f}, shifted {up_shift:.3f}, bayes {up_bayes:.3f}")
print("FC gaussian 90% intervals:", {k: tuple(round(v, 2) for v in iv) for k, iv in fc_tab.items()},
      "transition", x_trans)
print(f"flip-flop coverage at mu=2: {cov_ff_at2:.4f}, min {cov_ff_min:.4f}; FC cov {cov_fc.min():.4f}..{cov_fc.max():.4f}")
print(f"bayes 95% UL coverage min {cov_bayes_min:.4f} at mu={mu_bayes_min}; shifted at 0 {cov_shift[0]:.4f}")
print("Table I R:", np.round(R05, 3), "rank", rank, "acc", n1_05, n2_05, acc05)
for k in range(11):
    print(f"n0={k}: FC [{fc_int[k][0]:.2f},{fc_int[k][1]:.2f}] classical UL {cl_tab[k]:.2f} "
          f"CLs {cls_tab[k]:.2f} Bayes {bay_tab[k]:.2f}")
print(f"coverage min: FC {cov_fcp.min():.3f} classical {cov_clp.min():.3f} CLs {cov_clsp.min():.3f}")
print("FC n0=0 UL vs b (fixed):", np.round(fc_up_b[0][::25], 2), "raw at b=3", fc_up_b_raw[0][i_b3])
