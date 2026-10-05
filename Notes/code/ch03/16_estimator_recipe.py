"""16_estimator_recipe.py -- from a model to an estimator: Rao-Blackwell, Bayes loss, and the CMB C_ell.

Question: (i) does conditioning a crude unbiased estimator on a sufficient statistic really shrink
its variance (Poisson: estimate exp(-lambda) from n counts)?  (ii) do the posterior mean, median
and mode minimise the expected squared, absolute and 0-1 loss?  (iii) for a full-sky Gaussian
field, is C_hat = sum_m |a_lm|^2/(2l+1) unbiased with variance 2 C^2/(2l+1), the Cramer-Rao bound
(cosmic variance)?

Computes: 20000 Poisson experiments with the crude and Rao-Blackwellised estimators; expected
          losses of three point estimates for a skewed (Gamma) posterior; 20000 simulated
          skies at l = 2 and l = 30 with the MLE, its mean, variance and the score at the MLE.
Writes:   figures/ch03/estimator_recipe.pdf, results/ch03/16_estimator_recipe.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch03", "16_estimator_recipe")
setup()
out = {}

# ---------- (i) Rao-Blackwell: estimate g(lambda) = exp(-lambda) from n Poisson counts ----------
lam, n, M = 1.5, 10, 20000
X = rng.poisson(lam, size=(M, n))
T_crude = (X[:, 0] == 0).astype(float)            # unbiased: E 1{X_1 = 0} = exp(-lambda)
S = X.sum(axis=1)                                  # sufficient statistic
T_rb = ((n - 1) / n) ** S                          # E[T_crude | S] = ((n-1)/n)^S
g = np.exp(-lam)
out["RBTruth"] = g
out["RBCrudeMean"], out["RBCrudeVar"] = T_crude.mean(), T_crude.var()
out["RBRBMean"], out["RBRBVar"] = T_rb.mean(), T_rb.var()
out["RBRatio"] = T_rb.var() / T_crude.var()
out["RBCR"] = lam * np.exp(-2 * lam) / n           # Cramer-Rao bound for g: g'^2 / I_n
out["RBCrudeVarTh"] = g * (1 - g)                   # variance of a 0/1 variable
out["RBRBVarTh"] = np.exp(-2 * lam) * (np.exp(lam / n) - 1)
out["RBn"], out["RBlam"] = n, lam

# ---------- (ii) Bayes estimators: minimise the expected loss under a skewed posterior ----------
post = rng.gamma(3.0, 1.0, size=400000)            # a posterior sample, Gamma(3,1)
grid = np.linspace(0.05, 8, 800)
sq = np.array([np.mean((post - a) ** 2) for a in grid])
ab = np.array([np.mean(np.abs(post - a)) for a in grid])
from scipy.stats import gamma as gamma_dist
h = 0.05                                           # 0-1 loss with a tolerance h: P(|theta-a| > h),
z1 = 1 - 2 * h * gamma_dist.pdf(grid, 3.0)         # evaluated with the exact posterior density
out["BayesSqArg"], out["BayesAbsArg"], out["BayesZOArg"] = grid[sq.argmin()], grid[ab.argmin()], grid[z1.argmin()]
out["BayesMean"], out["BayesMedian"], out["BayesMode"] = post.mean(), np.median(post), 2.0

# ---------- (iii) full-sky C_ell: a_lm Gaussian, variance C ----------
def sky(ell, C=1.0, M=20000):
    a = rng.normal(0, np.sqrt(C), size=(M, 2 * ell + 1))    # 2l+1 independent real degrees of freedom
    return (a ** 2).sum(axis=1)                             # sum_m |a_lm|^2
res = {}
for ell in (2, 30):
    Ssum = sky(ell)
    Chat = Ssum / (2 * ell + 1)
    w = {2: "Two", 30: "Thirty"}[ell]
    out[f"CMBMean{w}"], out[f"CMBVar{w}"] = Chat.mean(), Chat.var()
    out[f"CMBBound{w}"] = 2 / (2 * ell + 1)
    out[f"CMBRelErr{w}"] = Chat.std()
    res[ell] = Chat
# Fisher information from the score at the truth: S = -(2l+1)/(2C) + sum|a|^2/(2C^2), Var S = (2l+1)/(2C^2)
ell = 30; Ssum = sky(ell); score = -(2 * ell + 1) / 2 + Ssum / 2
out["CMBScoreVarThirty"], out["CMBFishThirty"] = score.var(), (2 * ell + 1) / 2
out["CMBScoreMeanThirty"] = score.mean()

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
bins = np.linspace(0, 0.6, 60)
ax[0].hist(T_rb, bins=np.linspace(0, 0.9, 40), color=SERIES[1], alpha=.8, density=True, label="Rao--Blackwell")
ax[0].axvline(g, color="k", ls="--", lw=.8)
ax[0].set_xlabel(r"$\hat g$ (crude estimator: 0 or 1)"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=6)
ax[1].plot(grid, sq / sq.max(), color=SERIES[0], label="squared")
ax[1].plot(grid, ab / ab.max(), color=SERIES[1], label="absolute")
ax[1].plot(grid, z1, color=SERIES[2], label="0-1")
ax[1].set_xlabel("point estimate $a$"); ax[1].set_ylabel("expected loss (scaled)"); ax[1].legend(fontsize=6)
x = np.linspace(0, 2.2, 300)
for ell_, c in zip((2, 30), SERIES):
    ax[2].hist(res[ell_], bins=60, density=True, histtype="step", color=c, label=rf"$\ell={ell_}$")
ax[2].set_xlabel(r"$\hat C_\ell/C_\ell$"); ax[2].set_xlim(0, 3); ax[2].legend(fontsize=6)
fig.tight_layout(); savefig(fig, "ch03", "estimator_recipe")

save_numbers("ch03", "16_estimator_recipe", {"ThreeB" + k: v for k, v in out.items()})
print({k: round(float(v), 4) for k, v in out.items()})
