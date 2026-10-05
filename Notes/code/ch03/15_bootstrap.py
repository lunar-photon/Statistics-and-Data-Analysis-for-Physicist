"""15_bootstrap.py -- the bootstrap: a Monte Carlo estimate of the sampling distribution from one data set.

Question: we have ONE data set of n = 50 decay times.  Resampling it with replacement (the
nonparametric bootstrap), or simulating from the fitted model (the parametric bootstrap),
gives replicas of any estimator.  Does the spread of the replicas reproduce the true
standard error, which we can compute here because we know the truth?  Does it work for the
median, which has no simple error formula?  Where does it fail?  (Uniform(0, theta) and its
maximum: a bootstrap replica equals the observed maximum with probability 1-(1-1/n)^n ~ 0.632.)

Computes: true sampling distributions (10^4 fresh experiments), nonparametric and parametric
          bootstrap distributions (B = 10^4) of tau_hat and of the median; the Normal,
          pivotal and percentile 95% intervals; the uniform-maximum failure.
Writes:   figures/ch03/bootstrap.pdf, figures/ch03/bootstrap_uniform.pdf, results/ch03/15_bootstrap.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch03", "15_bootstrap")
setup()
tau_true, n, B = 1.0, 50, 10_000

t = rng.exponential(tau_true, n)                       # the one data set we have
tau_hat, med_hat = t.mean(), np.median(t)

# the truth (available only because this is a simulation)
T = rng.exponential(tau_true, (B, n))
true_tau, true_med = T.mean(1), np.median(T, 1)

# nonparametric bootstrap: draw n points WITH replacement from the data
idx = rng.integers(0, n, (B, n))
Tb = t[idx]
np_tau, np_med = Tb.mean(1), np.median(Tb, 1)
# parametric bootstrap: simulate from the fitted model f(t; tau_hat)
Tp = rng.exponential(tau_hat, (B, n))
p_tau, p_med = Tp.mean(1), np.median(Tp, 1)

se = lambda a: a.std(ddof=1)
q = lambda a, p: np.quantile(a, p)
ci_norm = (med_hat - 1.96 * se(np_med), med_hat + 1.96 * se(np_med))
ci_piv = (2 * med_hat - q(np_med, 0.975), 2 * med_hat - q(np_med, 0.025))
ci_pct = (q(np_med, 0.025), q(np_med, 0.975))

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0))
for ax, (tr, nb, pb, est, lab) in zip(axs, [(true_tau, np_tau, p_tau, tau_hat, r"$\hat\tau$ (mean)"),
                                            (true_med, np_med, p_med, med_hat, r"median")]):
    lo, hi = np.quantile(np.concatenate([tr, nb, pb]), [0.001, 0.999])
    bins = np.linspace(lo, hi, 70)
    ax.hist(tr - tr.mean(), bins=bins - tr.mean(), density=True, color="0.6", alpha=0.6, label="true (fresh experiments)")
    ax.hist(nb - est, bins=bins - est, density=True, histtype="step", color=SERIES[0], lw=1.4, label="nonparametric bootstrap")
    ax.hist(pb - pb.mean(), bins=bins - pb.mean(), density=True, histtype="step", color=SERIES[1], lw=1.2, label="parametric bootstrap")
    ax.set_xlabel(f"{lab} minus its centre")
axs[0].legend(fontsize=6.5)
fig.tight_layout()
savefig(fig, "ch03", "bootstrap")

# ---------- where it fails: the maximum of Uniform(0, theta) ----------
m = 50
u = rng.uniform(0, 1, m)
umax = u.max()
ub = u[rng.integers(0, m, (B, m))].max(1)
frac_equal = np.mean(ub == umax)
true_max = rng.uniform(0, 1, (B, m)).max(1)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
bins = np.linspace(0.85, 1.0, 61)
ax.hist(true_max, bins=bins, density=True, color="0.6", alpha=0.6, label=r"true: density $m\,x^{m-1}$")
ax.hist(ub, bins=bins, density=True, histtype="step", color=SERIES[0], lw=1.4, label="nonparametric bootstrap")
ax.axvline(umax, color=SERIES[1], lw=1, ls=":", label="observed maximum")
ax.set_xlabel(r"$\max x_i$"); ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch03", "bootstrap_uniform")

# how lucky was our data set?  The nonparametric bootstrap error of the mean is s/sqrt(n) (divisor n);
# its relative size s/(xbar sqrt n), evaluated on each of the B fresh experiments, scatters like this:
rel_fresh = T.std(1) / (T.mean(1) * np.sqrt(n))

save_numbers("ch03", "15_bootstrap", {
    "ThreeBBootRelOne": se(np_tau) / tau_hat,
    "ThreeBBootRelMean": rel_fresh.mean(), "ThreeBBootRelSd": rel_fresh.std(ddof=1),
    "ThreeBBootTauHat": tau_hat, "ThreeBBootMed": med_hat,
    "ThreeBBootSeTauTrue": se(true_tau), "ThreeBBootSeTauNP": se(np_tau), "ThreeBBootSeTauP": se(p_tau),
    "ThreeBBootSeTauCurv": tau_hat / np.sqrt(n),
    "ThreeBBootSeMedTrue": se(true_med), "ThreeBBootSeMedNP": se(np_med), "ThreeBBootSeMedP": se(p_med),
    "ThreeBBootNormLo": ci_norm[0], "ThreeBBootNormHi": ci_norm[1],
    "ThreeBBootPivLo": ci_piv[0], "ThreeBBootPivHi": ci_piv[1],
    "ThreeBBootPctLo": ci_pct[0], "ThreeBBootPctHi": ci_pct[1],
    "ThreeBBootMedTrueVal": np.log(2) * tau_true,
    "ThreeBBootFrac": frac_equal, "ThreeBBootFracTh": 1 - (1 - 1 / m) ** m, "ThreeBBootB": B,
})
