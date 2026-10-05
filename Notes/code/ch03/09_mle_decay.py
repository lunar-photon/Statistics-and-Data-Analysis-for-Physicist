"""09_mle_decay.py -- does the curvature of ln L give the right error bar?

Question: fit the mean lifetime tau to n = 50 exponential decay times by maximum likelihood.
Is the error read off the curvature of ln L (and off Delta ln L = 1/2) the same as the true
scatter of the MLE over many repetitions of the experiment?

Computes: one 'real' experiment: tau_hat, curvature error tau_hat/sqrt(n), the asymmetric
          Delta ln L = 1/2 interval, ln L(tau) and its parabola;  then 10^4 simulated
          experiments: the spread of tau_hat, the mean curvature error, the coverage of both
          intervals, the exact Gamma sampling distribution, and the bias of lambda_hat = 1/tau_hat.
Writes:   figures/ch03/mle_decay_lnL.pdf, figures/ch03/mle_decay_scatter.pdf,
          results/ch03/09_mle_decay.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

rng = rng_for("ch03", "09_mle_decay")
setup()
tau_true, n, M = 1.0, 50, 10_000


def lnL(tau, t):
    """log-likelihood of mean life tau for decay times t (constants dropped)."""
    return -len(t) * np.log(tau) - t.sum() / tau


# ---------- one experiment ----------
t = rng.exponential(tau_true, n)
tau_hat = t.mean()                                   # analytic MLE
tau_num = optimize.minimize_scalar(lambda s: -lnL(s, t), bounds=(0.1, 5), method="bounded").x
sig_curv = tau_hat / np.sqrt(n)                      # 1/sqrt(-d2 lnL/dtau2) at tau_hat
lmax = lnL(tau_hat, t)
f = lambda s: lnL(s, t) - lmax + 0.5                 # zero where ln L has dropped by 1/2
lo = optimize.brentq(f, 0.3 * tau_hat, tau_hat)
hi = optimize.brentq(f, tau_hat, 3 * tau_hat)

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ss = np.linspace(0.6 * tau_hat, 1.6 * tau_hat, 400)
ax.plot(ss, lnL(ss, t) - lmax, color=SERIES[0], label=r"$\ln\mathcal{L}(\tau)-\ln\mathcal{L}_{\max}$")
theory_line(ax, ss, -0.5 * ((ss - tau_hat) / sig_curv) ** 2, label="parabola, curvature at max")
ax.axhline(-0.5, color=SERIES[1], lw=1, ls=":")
for v in (lo, hi):
    ax.axvline(v, color=SERIES[1], lw=0.8, ls=":")
ax.set_ylim(-3, 0.3)
ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\Delta\ln\mathcal{L}$")
ax.legend(fontsize=7, loc="lower center")
fig.tight_layout()
savefig(fig, "ch03", "mle_decay_lnL")

# ---------- 10^4 experiments ----------
T = rng.exponential(tau_true, (M, n))
th = T.mean(1)                                        # MLE of each experiment
sc = th / np.sqrt(n)                                  # its curvature error
sd_mc = th.std(ddof=1)
cover_curv = np.mean(np.abs(th - tau_true) <= sc)
# Delta lnL = 1/2 interval: solve n[ln(tau/th) + th/tau - 1] = 1/2 on both sides, per experiment
# in units u = tau/th the equation is u-independent: ln u + 1/u - 1 = 1/(2n)
g = lambda u: np.log(u) + 1 / u - 1 - 0.5 / n
ulo = optimize.brentq(g, 0.3, 1.0); uhi = optimize.brentq(g, 1.0, 3.0)
cover_dl = np.mean((ulo * th <= tau_true) & (tau_true <= uhi * th))
lam_hat = 1 / th
lam_mean = lam_hat.mean()

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
b = np.linspace(0.5, 1.6, 90)
axs[0].hist(th, bins=b, density=True, color=SERIES[0], alpha=0.55, label=r"$\hat\tau$, $10^4$ experiments")
xx = np.linspace(0.5, 1.6, 400)
theory_line(axs[0], xx, stats.norm.pdf(xx, tau_true, tau_true / np.sqrt(n)), label=r"$\mathcal{N}(\tau,\tau^2/n)$")
axs[0].plot(xx, stats.gamma.pdf(xx, a=n, scale=tau_true / n), color=SERIES[1], lw=1.2, label="exact Gamma")
axs[0].set_xlabel(r"$\hat\tau$"); axs[0].legend(fontsize=7)
axs[1].scatter(th[:2000], sc[:2000], s=2, color=SERIES[2], alpha=0.5)
axs[1].axhline(sd_mc, color=SERIES[1], lw=1.2, label=r"true spread of $\hat\tau$")
axs[1].set_xlabel(r"$\hat\tau$"); axs[1].set_ylabel(r"curvature error $\hat\tau/\sqrt{n}$")
axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "mle_decay_scatter")

save_numbers("ch03", "09_mle_decay", {
    "ThreeBDecN": n,
    "ThreeBDecTauHat": tau_hat,
    "ThreeBDecTauNum": tau_num,
    "ThreeBDecSigCurv": sig_curv,
    "ThreeBDecLo": tau_hat - lo,
    "ThreeBDecHi": hi - tau_hat,
    "ThreeBDecSdTh": tau_true / np.sqrt(n),
    "ThreeBDecSdMC": sd_mc,
    "ThreeBDecMeanTau": th.mean(),
    "ThreeBDecMeanCurv": sc.mean(),
    "ThreeBDecCoverCurv": cover_curv,
    "ThreeBDecCoverDl": cover_dl,
    "ThreeBDecLamMean": lam_mean,
    "ThreeBDecLamTh": n / (n - 1) / tau_true,
})
