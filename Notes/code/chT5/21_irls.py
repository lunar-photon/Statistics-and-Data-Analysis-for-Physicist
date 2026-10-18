"""21_irls.py -- iteratively reweighted least squares: a robust mean and a decay curve.

Question: many estimators solve sum_i w_i(theta) r_i(theta) = 0 with weights that depend
on theta.  Freeze the weights, solve the weighted least-squares problem, recompute the
weights, repeat (IRLS).  Does it reach the right answer, and how fast?

  A. The Huber M-estimator of location for the eleven numbers of Casella & Berger,
     Example 10.2.5, for several tuning constants k: IRLS with weights min(1, k/|r|),
     checked against direct minimisation of sum rho_k(x_i - theta); the contraction
     factor at the fixed point against the formula (share of weight on clipped points).
  B. Poisson regression: counts of a decaying source in 20 time bins,
     lambda_i = exp(a + b t_i), lifetime tau = -1/b.  IRLS (= Newton = Fisher scoring
     for this model) against scipy's general minimiser; and, over many toy data sets,
     the lifetime from IRLS against a straight-line fit to log(counts).

Writes: figures/chT5/irls_huber.pdf, figures/chT5/irls_poisson.pdf, results/chT5/21_irls.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize

rng = rng_for("chT5", "21_irls")
out = {}
setup()

# ---------------------------------------------------------------- A. Huber by IRLS
x = np.array([-1.28, -0.96, -0.46, -0.44, -0.26, -0.21, -0.063, 0.39, 3.0, 6.0, 9.0])
cb_k = [0, 1, 2, 3, 4, 5, 6, 8, 10]                         # Casella & Berger, Table 10.2.3
cb_est = [-0.21, 0.03, -0.04, 0.29, 0.41, 0.52, 0.87, 0.97, 1.33]


def rho(r, k):
    a = np.abs(r)
    return np.where(a <= k, 0.5 * r**2, k * a - 0.5 * k**2)


def huber_irls(x, k, theta=None, iters=200, keep=False):
    theta = np.median(x) if theta is None else theta
    path = [theta]
    for _ in range(iters):
        r = x - theta
        w = np.minimum(1.0, k / np.maximum(np.abs(r), 1e-300))
        new = np.sum(w * x) / np.sum(w)
        path.append(new)
        if abs(new - theta) < 1e-15:
            break
        theta = new
    return (theta, np.array(path)) if keep else theta


ks = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10])
est = []
for k in ks:
    t_irls = huber_irls(x, k)
    t_dir = minimize_scalar(lambda t: np.sum(rho(x - t, k)), bounds=(-2, 10), method="bounded",
                            options=dict(xatol=1e-12)).x
    assert abs(t_irls - t_dir) < 1e-6, (k, t_irls, t_dir)
    est.append(t_irls)
est = np.array(est)
print("Huber by IRLS:", dict(zip(ks, np.round(est, 3))))
for k in [1, 2, 3, 4, 5, 6, 8, 10]:
    i = list(ks).index(k)
    out["TFiveBHub" + ["One", "Two", "Three", "Four", "Five", "Six", "Eight", "Ten"][[1, 2, 3, 4, 5, 6, 8, 10].index(k)]] = est[i]
out.update(TFiveBHubMean=x.mean(), TFiveBHubMedian=np.median(x))

# contraction at k = 1.5: g'(theta_hat) = sum_{clipped} w_i / sum_i w_i
k0 = 1.5
th0, path = huber_irls(x, k0, keep=True)
r = x - th0
w = np.minimum(1.0, k0 / np.abs(r))
clip = np.abs(r) > k0
gp_formula = np.sum(w[clip]) / np.sum(w)
e = np.abs(path - th0)
j = int(np.argmax(e < 1e-6))                     # well inside the linear regime
gp_meas = e[j] / e[j - 1]
assert abs(gp_meas - gp_formula) < 1e-3 * gp_formula, (gp_meas, gp_formula)
n_it = int(np.argmax(e < 1e-10))
out.update(TFiveBHubKzero=k0, TFiveBHubThetaKzero=th0, TFiveBHubGpFormula=gp_formula,
           TFiveBHubGpMeas=gp_meas, TFiveBHubNclip=int(clip.sum()), TFiveBHubIters=n_it,
           TFiveBHubPathZero=path[0], TFiveBHubPathOne=path[1], TFiveBHubPathTwo=path[2])
print(f"k=1.5: theta={th0:.6f}, g' formula {gp_formula:.5f}, measured {gp_meas:.5f}, {n_it} iterations")

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
order = np.argsort(x)
ax[0].bar(np.arange(len(x)), w[order], color=[SERIES[1] if c else SERIES[0] for c in clip[order]])
ax[0].set_xticks(np.arange(len(x)))
ax[0].set_xticklabels([f"{v:g}" for v in x[order]], rotation=60, fontsize=8)
ax[0].set_ylabel("weight $w_i$ at the fixed point")
ax[0].set_title(r"Huber, $k=%.1f$: $\hat\theta=%.3f$" % (k0, th0))
ax[1].plot(ks, est, "o-", color=SERIES[0], label="minimum of $\\sum\\rho_k$ (IRLS)")
ax[1].plot(cb_k, cb_est, "x", color=SERIES[1], ms=7, label="Casella & Berger, Table 10.2.3")
ax[1].axhline(x.mean(), color="0.4", ls=":", lw=1)
ax[1].axhline(np.median(x), color="0.4", ls=":", lw=1)
ax[1].text(10.2, x.mean(), "mean", va="center", fontsize=8)
ax[1].text(10.2, np.median(x), "median", va="center", fontsize=8)
ax[1].set_xlabel("tuning constant $k$")
ax[1].set_ylabel(r"$\hat\theta_k$")
ax[1].legend(loc="upper left", fontsize=8)
fig.tight_layout()
savefig(fig, "chT5", "irls_huber")

# ---------------------------------------------------------------- B. Poisson regression of a decay curve
t = np.arange(20.0)                       # bin centres (time units)
X = np.column_stack([np.ones_like(t), t])
tau0, lam00 = 4.0, 50.0
beta0 = np.array([np.log(lam00), -1.0 / tau0])


def poisson_irls(y, iters=50, keep=False):
    z0 = np.log(y + 0.5)                  # starting point: straight line through log counts
    beta = np.linalg.lstsq(X, z0, rcond=None)[0]
    path = [beta.copy()]
    for _ in range(iters):
        eta = X @ beta
        lam = np.exp(eta)
        z = eta + (y - lam) / lam         # working response
        W = lam                           # working weights
        beta_new = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
        path.append(beta_new.copy())
        if np.max(np.abs(beta_new - beta)) < 1e-14:
            beta = beta_new
            break
        beta = beta_new
    return (beta, np.array(path)) if keep else beta


y = rng.poisson(np.exp(X @ beta0)).astype(float)
b_irls, bpath = poisson_irls(y, keep=True)
nll = lambda bb: np.sum(np.exp(X @ bb) - y * (X @ bb))
b_sci = minimize(nll, np.array([np.log(y[0] + 1), -0.1]), method="BFGS", options=dict(gtol=1e-10)).x
assert np.max(np.abs(b_irls - b_sci)) < 1e-5, (b_irls, b_sci)
errs = np.max(np.abs(bpath - b_irls), 1)
nz = np.sum(y == 0)
pos = y > 0
b_log = np.linalg.lstsq(X[pos], np.log(y[pos]), rcond=None)[0]
F = X.T @ (np.exp(X @ b_irls)[:, None] * X)                  # Fisher matrix at the estimate
sig_b = np.sqrt(np.linalg.inv(F)[1, 1])
out.update(TFiveBPoiTau=tau0, TFiveBPoiLamZero=lam00, TFiveBPoiNzero=int(nz),
           TFiveBPoiTauIRLS=-1 / b_irls[1], TFiveBPoiTauErr=sig_b / b_irls[1] ** 2,
           TFiveBPoiTauLog=-1 / b_log[1], TFiveBPoiIters=int(np.argmax(errs < 1e-12)),
           TFiveBPoiErrOne=errs[1], TFiveBPoiErrTwo=errs[2], TFiveBPoiErrThree=errs[3])
print("decay curve: tau IRLS", -1 / b_irls[1], "log-fit", -1 / b_log[1], "errors per iteration", errs[:6])

Mt = 5000
Y = rng.poisson(np.exp(X @ beta0), size=(Mt, len(t))).astype(float)
tau_ml, tau_log, tau_wlog = np.empty(Mt), np.empty(Mt), np.empty(Mt)
for m in range(Mt):
    ym = Y[m]
    tau_ml[m] = -1 / poisson_irls(ym)[1]
    p = ym > 0
    tau_log[m] = -1 / np.linalg.lstsq(X[p], np.log(ym[p]), rcond=None)[0][1]
    sw = np.sqrt(ym[p])                   # weights y_i: Var(log y) ~ 1/lambda ~ 1/y
    tau_wlog[m] = -1 / np.linalg.lstsq(X[p] * sw[:, None], np.log(ym[p]) * sw, rcond=None)[0][1]
out.update(TFiveBPoiNtoys=Mt, TFiveBPoiMeanML=tau_ml.mean(), TFiveBPoiSdML=tau_ml.std(),
           TFiveBPoiMeanLog=tau_log.mean(), TFiveBPoiSdLog=tau_log.std(),
           TFiveBPoiMeanWlog=tau_wlog.mean(), TFiveBPoiSdWlog=tau_wlog.std(),
           TFiveBPoiMCerr=tau_ml.std() / np.sqrt(Mt))
print(f"toys: tau ML {tau_ml.mean():.3f}+-{tau_ml.std():.3f}, log-LS {tau_log.mean():.3f}+-{tau_log.std():.3f}, "
      f"weighted log-LS {tau_wlog.mean():.3f}+-{tau_wlog.std():.3f}")
assert abs(tau_ml.mean() - tau0) < abs(tau_log.mean() - tau0)

save_numbers("chT5", "21_irls", out)

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
ax[0].semilogy(t[pos], y[pos], "o", color="k", ms=4, label="counts")
ax[0].plot(t[~pos], np.full(nz, 0.3), "v", color="k", ms=5, mfc="none", label="empty bins (drawn at 0.3)")
tt = np.linspace(0, 19, 200)
ax[0].plot(tt, np.exp(b_irls[0] + b_irls[1] * tt), color=SERIES[0], label=r"Poisson ML by IRLS, $\hat\tau=%.2f$" % (-1 / b_irls[1]))
ax[0].plot(tt, np.exp(b_log[0] + b_log[1] * tt), "--", color=SERIES[1], label=r"line through $\ln y_i$, $\hat\tau=%.2f$" % (-1 / b_log[1]))
ax[0].set_xlabel("time bin $t_i$")
ax[0].set_ylabel("counts $y_i$")
ax[0].set_ylim(0.2, 120)
ax[0].legend(fontsize=8, loc="upper right")
bins = np.linspace(2, 9, 71)
ax[1].hist(tau_ml, bins, histtype="step", color=SERIES[0], lw=1.5, label="Poisson ML (IRLS)")
ax[1].hist(tau_log, bins, histtype="step", color=SERIES[1], lw=1.5, label=r"line through $\ln y_i$")
ax[1].hist(tau_wlog, bins, histtype="step", color=SERIES[2], lw=1.5, label=r"same, weights $y_i$")
ax[1].axvline(tau0, color="k", ls="--", lw=1)
ax[1].set_xlabel(r"$\hat\tau$")
ax[1].set_ylabel("toy data sets")
ax[1].legend(fontsize=8)
fig.tight_layout()
savefig(fig, "chT5", "irls_poisson")
