"""02_score_cloud.py -- the Fisher matrix is the covariance of the score, and the Gaussian formula is right.

Question: the Fisher matrix has two definitions, the covariance of the score vector and minus the
average Hessian of ln L, and for Gaussian data a closed form
    F_ij = mu_,i^T C^-1 mu_,j + (1/2) Tr[C^-1 C_,i C^-1 C_,j].
Do simulated data sets agree with all three?

Computes:
 (a) n = 20 draws from N(mu, sigma^2), theta = (mu, sigma): the score of 20000 data sets, its mean
     and covariance, the average of minus the Hessian, and F = n diag(1/sigma^2, 2/sigma^2);
 (b) a correlated time series: d_i = A sin(t_i) + noise with covariance
     C_ij = s^2 exp(-|t_i - t_j| / lc) + sn^2 delta_ij, theta = (A, lc): one parameter in the mean, one in the
     covariance; the score covariance of 20000 simulated series against the closed form;
 (c) x ~ N(theta, theta^2): mean and variance both carry theta, F = 3/theta^2 per point.
Writes: figures/ch10/score_cloud.pdf, figures/ch10/score_gauss.pdf, results/ch10/02_score_cloud.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES

setup()
rng = rng_for("ch10", "02_score_cloud")
NSIM = 20000
nums = {"TenBScNsim": NSIM}


def ellipse(ax, mean, cov, k2, **kw):
    t = np.linspace(0, 2 * np.pi, 400)
    L = np.linalg.cholesky(cov)
    e = np.asarray(mean)[:, None] + np.sqrt(k2) * L @ np.vstack([np.cos(t), np.sin(t)])
    ax.plot(e[0], e[1], **kw)


# ---------------------------------------------------------------- (a) Gaussian mean and width
n, mu, sig = 20, 0.0, 1.0
x = rng.normal(mu, sig, (NSIM, n))
r = x - mu
S = np.stack([r.sum(1) / sig**2, -n / sig + (r**2).sum(1) / sig**3], 1)      # the score at the truth
negH = np.empty((NSIM, 2, 2))                                                # minus the Hessian
negH[:, 0, 0] = n / sig**2
negH[:, 0, 1] = negH[:, 1, 0] = 2 * r.sum(1) / sig**3
negH[:, 1, 1] = -n / sig**2 + 3 * (r**2).sum(1) / sig**4
F_a = n * np.diag([1 / sig**2, 2 / sig**2])
covS = np.cov(S.T)
mH = negH.mean(0)
nums.update({
    "TenBScAmeanMu": f"{S[:, 0].mean():+.3f}", "TenBScAmeanSig": f"{S[:, 1].mean():+.3f}",
    "TenBScAcovMuMu": f"{covS[0, 0]:.2f}", "TenBScAcovSigSig": f"{covS[1, 1]:.2f}",
    "TenBScAcovMuSig": f"{covS[0, 1]:+.2f}",
    "TenBScAhMuMu": f"{mH[0, 0]:.2f}", "TenBScAhSigSig": f"{mH[1, 1]:.2f}", "TenBScAhMuSig": f"{mH[0, 1]:+.2f}",
    "TenBScAn": n,
})

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
ax = axs[0]
ax.scatter(S[:3000, 0], S[:3000, 1], s=3, color=SERIES[0], alpha=0.35, lw=0)
ellipse(ax, [0, 0], F_a, 1.0, color="k", lw=1.3, label=r"covariance $\mathbf{F}$ (1$\sigma$)")
ellipse(ax, [0, 0], F_a, 4.0, color="k", lw=1.0, ls="--", label=r"$2\sigma$")
ax.plot(0, 0, "k+", ms=9)
ax.set_xlabel(r"$S_\mu=\partial_\mu\ln\mathcal{L}$")
ax.set_ylabel(r"$S_\sigma=\partial_\sigma\ln\mathcal{L}$")
ax.set_title(r"(a) score of 20 draws from $\mathcal{N}(\mu,\sigma^2)$")
ax.legend(loc="upper left", fontsize=7.5)
ax.set_aspect("equal", "datalim")

# ---------------------------------------------------------------- (b) correlated time series
t = np.linspace(0, 4 * np.pi, 40)
A0, lc0, s0, sn0 = 1.0, 1.5, 0.8, 0.5
dt = np.abs(t[:, None] - t[None, :])


def model(A, lc):
    m = A * np.sin(t)
    K = s0**2 * np.exp(-dt / lc)
    return m, K + sn0**2 * np.eye(len(t))


m0, C0 = model(A0, lc0)
dm = [np.sin(t), np.zeros_like(t)]                              # d mu / dA, d mu / dlc
dC = [np.zeros_like(C0), s0**2 * np.exp(-dt / lc0) * dt / lc0**2]  # d C / dA, d C / dlc
Ci = np.linalg.inv(C0)
F_b = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        F_b[i, j] = dm[i] @ Ci @ dm[j] + 0.5 * np.trace(Ci @ dC[i] @ Ci @ dC[j])
# simulate and form the score:  S_i = mu_,i^T C^-1 r - (1/2) Tr[C^-1 C_,i] + (1/2) r^T C^-1 C_,i C^-1 r
Lc = np.linalg.cholesky(C0)
rr = (Lc @ rng.standard_normal((len(t), NSIM))).T
w = rr @ Ci                                   # rows: C^-1 r
Sb = np.empty((NSIM, 2))
for i in range(2):
    Sb[:, i] = w @ dm[i] - 0.5 * np.trace(Ci @ dC[i]) + 0.5 * np.einsum("ki,ij,kj->k", w, dC[i], w)
covSb = np.cov(Sb.T)
nums.update({
    "TenBScBFAA": f"{F_b[0, 0]:.2f}", "TenBScBFLL": f"{F_b[1, 1]:.2f}", "TenBScBFAL": f"{F_b[0, 1]:.2f}",
    "TenBScBcAA": f"{covSb[0, 0]:.2f}", "TenBScBcLL": f"{covSb[1, 1]:.2f}", "TenBScBcAL": f"{covSb[0, 1]:+.2f}",
    "TenBScBmA": f"{Sb[:, 0].mean():+.3f}", "TenBScBmL": f"{Sb[:, 1].mean():+.3f}",
    "TenBScBsA": f"{1 / np.sqrt(F_b[0, 0]):.3f}", "TenBScBsL": f"{1 / np.sqrt(F_b[1, 1]):.2f}",
    "TenBScBnpts": len(t), "TenBScBerr": f"{100 * np.sqrt(2 / NSIM):.1f}",
})

ax = axs[1]
ax.scatter(Sb[:3000, 0], Sb[:3000, 1], s=3, color=SERIES[1], alpha=0.35, lw=0)
ellipse(ax, [0, 0], F_b, 1.0, color="k", lw=1.3)
ellipse(ax, [0, 0], F_b, 4.0, color="k", lw=1.0, ls="--")
ax.plot(0, 0, "k+", ms=9)
ax.set_xlabel(r"$S_A$")
ax.set_ylabel(r"$S_{\ell_c}$")
ax.set_title(r"(b) correlated series, $A$ in $\mu$, $\ell_c$ in $\mathbf{C}$")
savefig(fig, "ch10", "score_cloud")

# a second figure: one simulated series and the Fisher contributions per point
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
d1 = m0 + rr[0]
axs[0].plot(t, m0, color="k", ls="--", lw=1.2, label=r"$\mu_i=A\sin t_i$")
axs[0].plot(t, d1, "o-", ms=3, color=SERIES[1], lw=0.8, label="one simulated series")
axs[0].set_xlabel("$t$"); axs[0].set_ylabel("$d_i$"); axs[0].legend(fontsize=7.5)
# how does F_lc grow with the number of samples in the same window?  and F_A?
npts = np.array([10, 20, 40, 80, 160, 320])
FA, FL = [], []
for npp in npts:
    tt = np.linspace(0, 4 * np.pi, npp)
    dd = np.abs(tt[:, None] - tt[None, :])
    C = s0**2 * np.exp(-dd / lc0) + sn0**2 * np.eye(npp)
    Cin = np.linalg.inv(C)
    dCl = s0**2 * np.exp(-dd / lc0) * dd / lc0**2
    FA.append(np.sin(tt) @ Cin @ np.sin(tt))
    FL.append(0.5 * np.trace(Cin @ dCl @ Cin @ dCl))
axs[1].loglog(npts, 1 / np.sqrt(FA), "o-", color=SERIES[0], label=r"$1/\sqrt{F_{AA}}$")
axs[1].loglog(npts, 1 / np.sqrt(FL), "s-", color=SERIES[2], label=r"$1/\sqrt{F_{\ell_c\ell_c}}$")
axs[1].loglog(npts, (1 / np.sqrt(FA[0])) * np.sqrt(npts[0] / npts), color="k", ls=":", lw=1,
              label=r"$\propto n^{-1/2}$")
axs[1].set_xlabel("points in the same window $n$"); axs[1].set_ylabel("conditional error")
axs[1].legend(fontsize=7.5)
fig.subplots_adjust(wspace=0.42)
savefig(fig, "ch10", "score_gauss")
nums.update({"TenBScSatA": f"{1 / np.sqrt(FA[-1]):.3f}", "TenBScSatL": f"{1 / np.sqrt(FL[-1]):.2f}",
             "TenBScSatAten": f"{1 / np.sqrt(FA[0]):.3f}", "TenBScSatLten": f"{1 / np.sqrt(FL[0]):.2f}"})

# ---------------------------------------------------------------- (c) N(theta, theta^2)
th0, nn = 2.0, 10
xc = rng.normal(th0, th0, (10 * NSIM, nn))
# per point: ln f = -ln theta - (x - theta)^2 / (2 theta^2);  d/dtheta = -1/theta + (x-theta)/theta^2 + (x-theta)^2/theta^3
Sc = (-1 / th0 + (xc - th0) / th0**2 + (xc - th0) ** 2 / th0**3).sum(1)
nums.update({"TenBScCvar": f"{Sc.var():.3f}", "TenBScCth": f"{3 * nn / th0**2:.3f}", "TenBScCn": nn,
             "TenBScCtheta": f"{th0:g}", "TenBScCnsim": 10 * NSIM})
save_numbers("ch10", "02_score_cloud", nums)
print("a: covS", covS, "mH", mH, "\nb: F", F_b, "cov", covSb, "\nc:", Sc.var(), 3 * nn / th0**2)
print("saturation", FA, FL)
