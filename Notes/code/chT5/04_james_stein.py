"""04_james_stein.py -- estimating many means at once: Stein's paradox in numbers.

Question: k independent readings X_i ~ N(theta_i, 1). Maximum likelihood estimates each theta_i by
X_i, with total mean squared error k. The James-Stein estimator (1 - (k-2)/|X|^2) X shrinks all of
them toward zero. Is its total error really below k for every theta when k >= 3, and how does
the gain depend on k and on the spread |theta| of the true means? What happens to one component
that is far from the others? Does the positive-part version do better still?

Computes: total risk against |theta| for k = 3, 5, 10, 50 (exact series for E[1/|X|^2] with
          Monte Carlo points); empirical Bayes: true means drawn from N(0, tau^2), estimated
          shrinkage factor against the ideal 1/(1+tau^2); risk of a single outlying component.
Writes:   figures/chT5/james_stein.pdf, figures/chT5/js_outlier.pdf, results/chT5/04_james_stein.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("chT5", "04_james_stein")
setup()


def inv_mean_ncx2(k, lam, jmax=400):
    """E[1/|X|^2] for |X|^2 noncentral chi^2 with k dof and noncentrality lam = |theta|^2:
    a Poisson(lam/2) mixture of central chi^2_{k+2j}, each with E[1/chi^2_m] = 1/(m-2)."""
    j = np.arange(jmax)
    return np.sum(stats.poisson.pmf(j[None, :], np.atleast_1d(lam)[:, None] / 2) / (k - 2 + 2 * j), axis=1)


def js(X, positive=False):
    c = 1 - (X.shape[1] - 2) / np.sum(X**2, axis=1, keepdims=True)
    return (np.maximum(c, 0) if positive else c) * X


M = 20_000
norms = np.linspace(0, 8, 81)
ks = [3, 5, 10, 50]
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1))
mc_pts = {}
for k, col in zip(ks, SERIES):
    R = k - (k - 2)**2 * inv_mean_ncx2(k, norms**2)
    axs[0].plot(norms, R / k, color=col, label=rf"$k={k}$")
    pts = []
    for r in [0.0, 2.0, 4.0, 6.0]:
        theta = np.zeros(k); theta[0] = r            # risk depends on theta only through |theta|
        X = theta + rng.standard_normal((M, k))
        pts.append(np.mean(np.sum((js(X) - theta)**2, 1)) / k)
    axs[0].plot([0, 2, 4, 6], pts, "o", color=col, ms=3.5)
    mc_pts[k] = pts
    assert abs(pts[0] - 2 / k) < 0.05                 # at theta = 0: risk 2, i.e. 2/k per component
axs[0].axhline(1, color="k", ls="--", lw=1.2, label="ML: $R=k$")
axs[0].set_xlabel(r"spread of the true means $|\theta|$"); axs[0].set_ylabel(r"total risk $/\,k$")
axs[0].legend(fontsize=7, loc="lower right"); axs[0].set_ylim(0, 1.1)

# risk against k at fixed spread per component, |theta|^2 = k * tau^2
kk = np.arange(1, 41)
for tau, col in [(0.5, SERIES[0]), (1.0, SERIES[1]), (2.0, SERIES[2])]:
    Rk = np.array([k - (k - 2)**2 * inv_mean_ncx2(k, k * tau**2)[0] if k >= 3 else k for k in kk])
    axs[1].plot(kk, Rk / kk, "-", color=col, label=rf"$|\theta|^2/k={tau**2:g}$")
    axs[1].plot(kk, np.full(kk.shape, 1 / (1 + 1 / tau**2)), ":", color=col, lw=1)
axs[1].axhline(1, color="k", ls="--", lw=1.2)
axs[1].set_xlabel(r"number of means $k$"); axs[1].set_ylabel(r"total risk $/\,k$")
axs[1].legend(fontsize=7, loc="lower right"); axs[1].set_ylim(0, 1.1)
fig.tight_layout()
savefig(fig, "chT5", "james_stein")

# ---------- empirical Bayes: theta_i ~ N(0, tau^2) ----------
k_eb, tau_eb = 20, 1.0
theta = tau_eb * rng.standard_normal((M, k_eb))
X = theta + rng.standard_normal((M, k_eb))
shrink_hat = (k_eb - 2) / np.sum(X**2, 1)               # estimates 1/(1+tau^2)
risk_ml = np.mean(np.sum((X - theta)**2, 1))
risk_js = np.mean(np.sum((js(X) - theta)**2, 1))
risk_jsp = np.mean(np.sum((js(X, True) - theta)**2, 1))
risk_bayes = k_eb * tau_eb**2 / (1 + tau_eb**2)          # posterior mean with the true tau

# ---------- one component far from the rest ----------
k_o = 10
far = np.linspace(0, 12, 49)
r_far, r_rest = [], []
for f in far:
    th = np.zeros(k_o); th[0] = f
    Xo = th + rng.standard_normal((M, k_o))
    E = (js(Xo) - th)**2
    r_far.append(E[:, 0].mean()); r_rest.append(E[:, 1:].sum(1).mean() / (k_o - 1))
r_far, r_rest = np.array(r_far), np.array(r_rest)
imax = int(np.argmax(r_far))
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.plot(far, r_far, color=SERIES[1], label="outlying component")
ax.plot(far, r_rest, color=SERIES[0], label="each of the other nine")
ax.axhline(1, color="k", ls="--", lw=1.2, label="ML")
ax.set_xlabel(r"true value $\theta_1$ of the outlying component (others $0$)")
ax.set_ylabel("mean squared error"); ax.legend(fontsize=7, loc="center right")
fig.tight_layout()
savefig(fig, "chT5", "js_outlier")

save_numbers("chT5", "04_james_stein", {
    "TFiveAJsM": M,
    "TFiveAJsTenZero": f"{mc_pts[10][0] * 10:.3f}", "TFiveAJsTenFour": f"{mc_pts[10][2] * 10:.2f}",
    "TFiveAJsFiftyFour": f"{mc_pts[50][2] * 50:.1f}",
    "TFiveAJsThreeFour": f"{mc_pts[3][2] * 3:.3f}",
    "TFiveAEbK": k_eb, "TFiveAEbShrinkMean": f"{shrink_hat.mean():.3f}",
    "TFiveAEbShrinkSd": f"{shrink_hat.std():.3f}",
    "TFiveAEbRiskML": f"{risk_ml:.2f}", "TFiveAEbRiskJS": f"{risk_js:.2f}",
    "TFiveAEbRiskJSP": f"{risk_jsp:.2f}", "TFiveAEbRiskBayes": f"{risk_bayes:.2f}",
    "TFiveAOutK": k_o, "TFiveAOutMax": f"{r_far[imax]:.2f}", "TFiveAOutAt": f"{far[imax]:.2f}",
    "TFiveAOutRest": f"{r_rest[imax]:.3f}", "TFiveAOutTotal": f"{r_far[imax] + (k_o - 1) * r_rest[imax]:.2f}",
})
