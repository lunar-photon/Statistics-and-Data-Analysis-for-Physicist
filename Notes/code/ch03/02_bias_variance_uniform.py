"""02_bias_variance_uniform.py -- can a biased estimator beat an unbiased one?

Question: X_1..X_n ~ Uniform(0, theta).  Three estimators of theta:
  T1 = 2*mean (unbiased),  T2 = max (biased low),  T3 = (n+1)/n * max (unbiased).
Which has the smallest mean squared error, and do the simulated bias, variance
and MSE agree with the formulas derived in the text?

Computes: sampling distributions at n = 10 and MSE versus n (M repetitions each).
Writes: figures/ch03/uniform_estimators.pdf, figures/ch03/uniform_mse.pdf,
        results/ch03/02_bias_variance_uniform.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch03", "02_bias_variance_uniform")
setup()
theta, M = 1.0, 200_000


def estimators(x):
    n = x.shape[1]
    mx = x.max(axis=1)
    return {"T1": 2 * x.mean(axis=1), "T2": mx, "T3": (n + 1) / n * mx}


def theory(n):
    """bias, variance, MSE of T1, T2, T3 (derived in the text), theta = 1."""
    return {
        "T1": (0.0, theta**2 / (3 * n)),
        "T2": (-theta / (n + 1), n * theta**2 / ((n + 1) ** 2 * (n + 2))),
        "T3": (0.0, theta**2 / (n * (n + 2))),
    }


# ---- sampling distributions at n = 10 ----
n = 10
x = rng.uniform(0, theta, size=(M, n))
T = estimators(x)
labels = {"T1": r"$T_1=2\bar X$", "T2": r"$T_2=\max X_i$", "T3": r"$T_3=\frac{n+1}{n}\max X_i$"}
fig, ax = plt.subplots(figsize=(6.0, 3.4))
bins = np.linspace(0.4, 1.6, 121)
for (k, v), c in zip(T.items(), SERIES):
    ax.hist(v, bins=bins, density=True, histtype="step", lw=1.5, color=c, label=labels[k])
ax.axvline(theta, color="0.3", lw=0.9, ls=":")
ax.set_xlabel(r"estimate of $\theta$ (true $\theta=1$), $n=10$")
ax.set_ylabel("density")
ax.legend(fontsize=8)
savefig(fig, "ch03", "uniform_estimators")

sim = {k: (v.mean() - theta, v.var(ddof=1), np.mean((v - theta) ** 2)) for k, v in T.items()}
th = theory(n)

# ---- MSE versus n ----
ns = np.array([2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200])
mse = {k: [] for k in T}
for m in ns:
    xx = rng.uniform(0, theta, size=(40_000, m))
    for k, v in estimators(xx).items():
        mse[k].append(np.mean((v - theta) ** 2))
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for (k, v), c in zip(mse.items(), SERIES):
    ax.loglog(ns, v, "o", color=c, label=labels[k] + " (simulated)")
nn = np.geomspace(2, 200, 200)
for k in T:
    b, var = theory(nn)[k]
    theory_line(ax, nn, b**2 + var, label="derived MSE" if k == "T1" else None)
ax.set_xlabel(r"sample size $n$")
ax.set_ylabel(r"MSE $=\mathbb{E}(T-\theta)^2$")
ax.legend(fontsize=8)
savefig(fig, "ch03", "uniform_mse")

save_numbers("ch03", "02_bias_variance_uniform", {
    "ThreeAUOneBias": sim["T1"][0], "ThreeAUTwoBias": sim["T2"][0], "ThreeAUThreeBias": sim["T3"][0],
    "ThreeAUOneMSE": sim["T1"][2], "ThreeAUTwoMSE": sim["T2"][2], "ThreeAUThreeMSE": sim["T3"][2],
    "ThreeAUOneMSEth": th["T1"][1], "ThreeAUTwoMSEth": th["T2"][0] ** 2 + th["T2"][1],
    "ThreeAUThreeMSEth": th["T3"][1], "ThreeAUTwoBiasth": th["T2"][0],
})
