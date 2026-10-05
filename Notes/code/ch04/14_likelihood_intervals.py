"""14_likelihood_intervals.py -- intervals from the likelihood function, and how well they cover.

Question: (i) for n = 5 exponential lifetimes, the rule "ln L drops by 1/2" gives an asymmetric
interval (Cowan Fig. 9.6).  What is its true coverage, compared with the exact chi-square
interval and the symmetric tau_hat +- tau_hat/sqrt(n)?  (ii) For a correlation coefficient r = 0.5
from n = 20 pairs (Cowan Sec. 9.5), how do the naive Gaussian interval and Fisher's z interval
compare, and which one covers?  (iii) For a two-parameter straight-line fit, what fraction of
repeated experiments put the truth inside the Delta chi^2 = 1 and Delta chi^2 = 2.30 contours?

Computes: the Delta lnL = 1/2 interval for one n = 5 sample, the exact coverage of that rule
(it depends only on the pivot tau_hat/tau ~ Gamma(n, 1/n)), a Monte Carlo check, the Fisher-z
numbers and their coverage by simulation, and the 2-D contour coverage by simulation.

Writes: figures/ch04/likelihood_interval.pdf, figures/ch04/ellipse_coverage.pdf,
        results/ch04/14_likelihood_intervals.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

rng = rng_for("ch04", "14_likelihood_intervals")
setup()

# ---------- (i) exponential lifetimes: Delta lnL = 1/2 ----------
n_small, tau_true = 5, 1.0
t = rng.exponential(tau_true, size=n_small)
tau_hat = t.mean()

def dlnl(u, n):
    """ln L(tau) - ln L_max written in u = tau_hat / tau (derivation in the text)."""
    return n * (np.log(u) - u + 1.0)

def u_roots(n, drop=0.5):
    """the two values of u = tau_hat/tau where ln L has dropped by `drop`."""
    f = lambda u: dlnl(u, n) + drop
    return optimize.brentq(f, 1e-6, 1.0), optimize.brentq(f, 1.0, 50.0)

u_lo, u_hi = u_roots(n_small)                      # u_lo < 1 < u_hi
lik_lo, lik_hi = tau_hat / u_hi, tau_hat / u_lo    # tau = tau_hat/u
# exact central 68.27% interval from the pivot 2 n tau_hat / tau ~ chi2_{2n}
cl = stats.norm.cdf(1) - stats.norm.cdf(-1)
a = (1 - cl) / 2
ex_lo = 2 * n_small * tau_hat / stats.chi2.ppf(1 - a, 2 * n_small)
ex_hi = 2 * n_small * tau_hat / stats.chi2.ppf(a, 2 * n_small)

def coverage_lik(n):
    """P(u_lo <= U <= u_hi) with U = tau_hat/tau ~ Gamma(shape n, scale 1/n): exact."""
    lo, hi = u_roots(n)
    g = stats.gamma(n, scale=1.0 / n)
    return g.cdf(hi) - g.cdf(lo)

def coverage_sym(n):
    """tau_hat (1 -+ 1/sqrt n) covers tau  <=>  1/(1+1/sqrt n) <= U <= 1/(1-1/sqrt n)."""
    g = stats.gamma(n, scale=1.0 / n)
    lo = 1 / (1 + 1 / np.sqrt(n))
    hi = 1 / (1 - 1 / np.sqrt(n)) if n > 1 else np.inf
    return g.cdf(hi) - g.cdf(lo)

ns = np.array([1, 2, 3, 5, 8, 12, 20, 35, 50, 100, 200])
cov_lik = np.array([coverage_lik(n) for n in ns])
cov_sym = np.array([coverage_sym(n) for n in ns])

# how the misses split between the two sides at n = 5 (a central interval has 15.9% on each)
g5 = stats.gamma(n_small, scale=1.0 / n_small)
miss_lik_above, miss_lik_below = g5.sf(u_hi), g5.cdf(u_lo)       # interval above / below tau
s_lo, s_hi = 1 / (1 + 1 / np.sqrt(n_small)), 1 / (1 - 1 / np.sqrt(n_small))
miss_sym_above, miss_sym_below = g5.sf(s_hi), g5.cdf(s_lo)

# Monte Carlo check at n = 5
M = 200_000
th = rng.exponential(tau_true, size=(M, n_small)).mean(1)
cov_mc = np.mean((th / u_hi <= tau_true) & (tau_true <= th / u_lo))

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
tau = np.linspace(0.25, 3.0, 600)
ll = dlnl(tau_hat / tau, n_small)
ax = axs[0]
ax.plot(tau, ll, color=SERIES[0])
ax.axhline(-0.5, color="k", lw=0.9, ls=":")
ax.axvline(tau_hat, color="0.5", lw=0.8)
ax.plot([lik_lo, lik_hi], [-0.5, -0.5], "o", color=SERIES[1], ms=4)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$\ln L(\tau)-\ln L_{\max}$")
ax.set_ylim(-2.5, 0.2)
ax.set_title(rf"$n={n_small}$: $\hat\tau={tau_hat:.2f}$, interval [{lik_lo:.2f}, {lik_hi:.2f}]",
             fontsize=9)
ax = axs[1]
ax.semilogx(ns, cov_lik, "o-", color=SERIES[0], label=r"$\Delta\ln L=\frac{1}{2}$ interval")
ax.semilogx(ns, cov_sym, "s-", color=SERIES[1], label=r"$\hat\tau\pm\hat\tau/\sqrt{n}$")
ax.axhline(cl, color="k", lw=1.0, ls="--", label="nominal 68.3%")
ax.set_xlabel(r"sample size $n$")
ax.set_ylabel("coverage")
ax.set_ylim(0.45, 0.75)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
savefig(fig, "ch04", "likelihood_interval")

# ---------- (ii) Fisher z for a correlation coefficient ----------
n_r, r_obs = 20, 0.5
sig_r = (1 - r_obs**2) / np.sqrt(n_r)
z_obs = np.arctanh(r_obs)
sig_z = 1 / np.sqrt(n_r - 3)
q68, q99 = 1.0, stats.norm.isf(0.005)
naive99 = (r_obs - q99 * sig_r, r_obs + q99 * sig_r)
zeta99 = (z_obs - q99 * sig_z, z_obs + q99 * sig_z)
rho99 = tuple(np.tanh(zeta99))
p_naive_zero = stats.norm.sf(r_obs / sig_r)          # "confidence level for a lower limit of zero"
p_fisher_zero = stats.norm.sf(z_obs / sig_z)
# exact: under rho = 0, t = r sqrt(n-2)/sqrt(1-r^2) follows Student t with n-2 dof
t_obs = r_obs * np.sqrt(n_r - 2) / np.sqrt(1 - r_obs**2)
p_exact_zero = stats.t.sf(t_obs, n_r - 2)

def corr_coverage(rho, n, M=100_000):
    cov = [[1, rho], [rho, 1]]
    xy = rng.multivariate_normal([0, 0], cov, size=(M, n))
    x, y = xy[..., 0], xy[..., 1]
    x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
    r = (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))
    naive = np.abs(r - rho) <= q99 * (1 - r**2) / np.sqrt(n)
    fz = np.abs(np.arctanh(r) - np.arctanh(rho)) <= q99 / np.sqrt(n - 3)
    return naive.mean(), fz.mean()

cn5, cf5 = corr_coverage(0.5, n_r)
cn8, cf8 = corr_coverage(0.8, n_r)

# ---------- (iii) two-parameter straight-line fit ----------
xk = np.linspace(0, 1, 10)
sig_y = 0.3
A = np.vstack([np.ones_like(xk), xk]).T
Cinv = A.T @ A / sig_y**2                              # Fisher matrix = inverse covariance
C = np.linalg.inv(Cinv)
theta_true = np.array([1.0, 2.0])
Mfit = 200_000
Y = theta_true @ A.T + sig_y * rng.standard_normal((Mfit, xk.size))
est = Y @ A @ C.T / sig_y**2                           # least-squares estimates, (A^T A)^-1 A^T y
d = est - theta_true
Q = np.einsum("ij,jk,ik->i", d, Cinv, d)               # Delta chi^2 at the true point
cov_q1 = np.mean(Q <= 1.0)
cov_q230 = np.mean(Q <= stats.chi2.ppf(cl, 2))
cov_slope_alone = np.mean(np.abs(d[:, 1]) <= np.sqrt(C[1, 1]))
q230 = stats.chi2.ppf(cl, 2)

fig, ax = plt.subplots(figsize=(4.6, 3.8))
sel = slice(0, 400)
ax.plot(est[sel, 0], est[sel, 1], ".", color="0.6", ms=2.5, label="estimates, 400 experiments")
w, V = np.linalg.eigh(C)
ang = np.linspace(0, 2 * np.pi, 400)
circ = np.vstack([np.cos(ang), np.sin(ang)])
for q, col, lab in ((1.0, SERIES[0], r"$\Delta\chi^2=1$"), (q230, SERIES[1], r"$\Delta\chi^2=2.30$")):
    ell = theta_true[:, None] + V @ (np.sqrt(q * w)[:, None] * circ)
    ax.plot(ell[0], ell[1], color=col, lw=1.6, label=lab)
ax.plot(*theta_true, "k+", ms=10, mew=1.6)
ax.set_xlabel(r"intercept $\theta_1$")
ax.set_ylabel(r"slope $\theta_2$")
ax.legend(fontsize=8, loc="upper right")
savefig(fig, "ch04", "ellipse_coverage")

save_numbers("ch04", "14_likelihood_intervals", {
    "FourBLiN": n_small, "FourBLiTauHat": f"{tau_hat:.2f}",
    "FourBLiLo": f"{lik_lo:.2f}", "FourBLiHi": f"{lik_hi:.2f}",
    "FourBLiMinus": f"{tau_hat - lik_lo:.2f}", "FourBLiPlus": f"{lik_hi - tau_hat:.2f}",
    "FourBLiUlo": f"{u_lo:.3f}", "FourBLiUhi": f"{u_hi:.3f}",
    "FourBLiExLo": f"{ex_lo:.2f}", "FourBLiExHi": f"{ex_hi:.2f}",
    "FourBLiCovFive": f"{coverage_lik(5):.3f}", "FourBLiCovFiveMC": f"{cov_mc:.3f}",
    "FourBLiCovOne": f"{coverage_lik(1):.3f}", "FourBLiCovFifty": f"{coverage_lik(50):.3f}",
    "FourBSymCovFive": f"{coverage_sym(5):.3f}",
    "FourBLiMissAbove": f"{miss_lik_above:.3f}", "FourBLiMissBelow": f"{miss_lik_below:.3f}",
    "FourBSymMissAbove": f"{miss_sym_above:.3f}", "FourBSymMissBelow": f"{miss_sym_below:.3f}", "FourBSymCovFifty": f"{coverage_sym(50):.3f}",
    "FourBFzSigR": f"{sig_r:.3f}", "FourBFzZ": f"{z_obs:.3f}", "FourBFzSigZ": f"{sig_z:.3f}",
    "FourBFzNaiveLo": f"{naive99[0]:.3f}", "FourBFzNaiveHi": f"{naive99[1]:.3f}",
    "FourBFzZetaLo": f"{zeta99[0]:.3f}", "FourBFzZetaHi": f"{zeta99[1]:.3f}",
    "FourBFzRhoLo": f"{rho99[0]:.3f}", "FourBFzRhoHi": f"{rho99[1]:.3f}",
    "FourBFzPnaive": f"{100 * p_naive_zero:.2f}", "FourBFzPfisher": f"{100 * p_fisher_zero:.1f}",
    "FourBFzPexact": f"{100 * p_exact_zero:.1f}", "FourBFzT": f"{t_obs:.2f}",
    "FourBFzCovNaiveFive": f"{cn5:.3f}", "FourBFzCovFisherFive": f"{cf5:.3f}",
    "FourBFzCovNaiveEight": f"{cn8:.3f}", "FourBFzCovFisherEight": f"{cf8:.3f}",
    "FourBElCovOne": f"{cov_q1:.3f}", "FourBElCovTwoThirty": f"{cov_q230:.3f}",
    "FourBElCovSlope": f"{cov_slope_alone:.3f}", "FourBElQ": f"{q230:.2f}",
    "FourBElRho": f"{C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]):.2f}",
})
print(f"n=5 sample: tau_hat={tau_hat:.3f} lik [{lik_lo:.3f},{lik_hi:.3f}] exact [{ex_lo:.3f},{ex_hi:.3f}]")
print("coverage lik:", dict(zip(ns, np.round(cov_lik, 4))), "MC n=5", cov_mc)
print("coverage sym:", dict(zip(ns, np.round(cov_sym, 4))))
print(f"Fisher z: sig_r={sig_r:.3f} naive99 {naive99} z={z_obs:.3f} sz={sig_z:.3f} zeta99 {zeta99} rho99 {rho99}")
print(f"  P(zero) naive {p_naive_zero:.4f} fisher {p_fisher_zero:.4f}; coverage rho=.5 {cn5:.3f},{cf5:.3f}; rho=.8 {cn8:.3f},{cf8:.3f}")
print(f"ellipse coverage q<=1 {cov_q1:.4f}, q<=2.30 {cov_q230:.4f}, slope alone {cov_slope_alone:.4f}")
