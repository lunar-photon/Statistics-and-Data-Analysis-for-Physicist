"""29_variance_reduction.py -- the same integral with fewer draws: antithetic, control, stratified.

Question: sigma / sqrt(N) has two factors.  We cannot beat sqrt(N) with
independent draws, but we can shrink sigma.  Test case I = int_0^1 e^x dx = e - 1.

  * plain:        mean of e^{U};                      Var per draw (e^2 - 1)/2 - (e - 1)^2
  * antithetic:   mean of [e^U + e^{1-U}] / 2 over N/2 pairs (same N evaluations)
  * control:      mean of e^U - beta (U - 1/2), beta = Cov(e^U, U) / Var(U) = 6 (3 - e)
  * stratified:   one draw in each of N equal strata [(k + u)/N]

Each estimator uses N function evaluations; we repeat 4000 times at each N and
record the standard deviation of the estimate.

Writes: figures/ch06/variance_reduction.pdf, results/ch06/29_variance_reduction.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt

setup(5.4, 3.4)
rng = rng_for("ch06", "29_variance_reduction")
I = np.e - 1
R = 4000


def plain(n):
    return np.exp(rng.random((R, n))).mean(axis=1)


def antithetic(n):
    u = rng.random((R, n // 2))
    return (0.5 * (np.exp(u) + np.exp(1 - u))).mean(axis=1)


def control(n):
    u = rng.random((R, n))
    y = np.exp(u)
    beta = 6 * (3 - np.e)                         # Cov(e^U, U) / Var(U) = [(3 - e)/2] / (1/12), exact
    return (y - beta * (u - 0.5)).mean(axis=1)


def stratified(n):
    u = (np.arange(n)[None, :] + rng.random((R, n))) / n
    return np.exp(u).mean(axis=1)


methods = {"plain": plain, "antithetic": antithetic, "control variate": control, "stratified": stratified}
Ns = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
sd = {k: np.array([f(n).std() for n in Ns]) for k, f in methods.items()}

var_plain = (np.e**2 - 1) / 2 - I**2
cov_anti = np.e - I**2
var_anti_pair = 0.5 * (var_plain + cov_anti)       # variance of one pair average
var_ctrl = var_plain - 12 * ((3 - np.e) / 2) ** 2   # Var(Y) - Cov^2/Var(U)
out = {"SixBVrI": I, "SixBVrVarPlain": var_plain, "SixBVrCovAnti": cov_anti,
       "SixBVrVarAntiPair": var_anti_pair, "SixBVrVarAntiPerEval": 2 * var_anti_pair, "SixBVrVarCtrl": var_ctrl,
       "SixBVrBeta": 6 * (3 - np.e), "SixBVrN": 1024, "SixBVrR": R}
k = -1
for name, tag in [("plain", "Plain"), ("antithetic", "Anti"), ("control variate", "Ctrl"), ("stratified", "Strat")]:
    out[f"SixBVrSd{tag}"] = sd[name][k]
    out[f"SixBVrGain{tag}"] = (sd["plain"][k] / sd[name][k]) ** 2
out["SixBVrStratSlope"] = np.polyfit(np.log(Ns), np.log(sd["stratified"]), 1)[0]

fig, ax = plt.subplots()
for i, (name, s) in enumerate(sd.items()):
    ax.loglog(Ns, s, "o-", ms=3.5, lw=1.1, color=SERIES[i], label=name)
theory_line(ax, Ns, np.sqrt(var_plain / Ns), label=r"$\sigma/\sqrt{N}$ (plain)")
ax.set(xlabel=r"function evaluations $N$", ylabel=r"standard deviation of $\hat I$",
       title=r"$\int_0^1 e^x\,dx$")
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "ch06", "variance_reduction")
save_numbers("ch06", "29_variance_reduction", out)
print(out)
