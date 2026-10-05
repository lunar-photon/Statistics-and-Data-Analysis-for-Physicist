"""Reading a chain: autocorrelation, integrated autocorrelation time, ESS, thinning, burn-in.

Question 1: can we estimate tau = 1 + 2 sum_k rho_k from the chain itself?  Calibrate on the
AR(1) chain x_{t+1} = rho x_t + sqrt(1-rho^2) z (exact tau = (1+rho)/(1-rho) = 19 for rho = 0.9):
the naive sum over all lags, Kruschke's stop-at-ACF<0.05 rule, and Sokal's window M >= 5 tau(M).
Question 2: what do the ACFs of Kruschke's three coin chains look like?
Question 3: does thinning a chain by k improve the estimate of the mean?  (theory for AR(1))
Question 4: a 10-D Gaussian chain started far away: what does burn-in look like in log p?
Writes figures/ch09/acf_tau.pdf, figures/ch09/thin_burnin.pdf, results/ch09/16_autocorr.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_mcmc import metropolis, acf, tau_int

rng = rng_for("ch09", "16_autocorr")
nums = {}


def ar1(rho, n, rng):
    x = np.empty(n)
    x[0] = rng.standard_normal()
    z = rng.standard_normal(n) * np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + z[t]
    return x


# ---- 1. calibration on AR(1) -----------------------------------------------------------------------
rho, N, R = 0.9, 10_000, 300
tau_exact = (1 + rho) / (1 - rho)
naive, krus, sokal = [], [], []
for r in range(R):
    x = ar1(rho, N, rng)
    rh = acf(x)
    naive.append(2 * rh.sum() - 1)                     # all lags: identically zero up to rounding
    k_stop = np.argmax(rh < 0.05)
    krus.append(1 + 2 * rh[1:k_stop].sum())
    sokal.append(tau_int(x)[0])
naive, krus, sokal = map(np.array, (naive, krus, sokal))
nums.update({"NineBAcRho": rho, "NineBAcN": f"{N:,}".replace(",", r"\,"), "NineBAcR": R,
             "NineBAcTauExact": tau_exact,
             "NineBAcNaiveMax": f"{np.abs(naive).max():.1e}".replace("e-", r"\times10^{-") + "}",
             "NineBAcKrMean": krus.mean(), "NineBAcKrSd": krus.std(),
             "NineBAcSoMean": sokal.mean(), "NineBAcSoSd": sokal.std()})
# spread of the chain mean over replicates versus sigma^2 tau / N
means = np.array([ar1(rho, N, rng).mean() for _ in range(R)])
nums["NineBAcVarMeanMeas"] = means.var() * N
nums["NineBAcVarMeanPred"] = tau_exact

# ---- 2. ACFs of the coin chains (Kruschke Fig. 7.4 settings) ----------------------------------------
z, Nf = 14, 20
logp_coin = lambda x: (z * np.log(x[0]) + (Nf - z) * np.log1p(-x[0])) if 0 < x[0] < 1 else -np.inf
coin = {}
for sd in (0.02, 0.2, 2.0):
    ch, _, a = metropolis(logp_coin, [0.7], 50_000, sd, rng)
    coin[sd] = ch[1:, 0]

setup(7.0, 2.9)
fig, (a1, a2) = plt.subplots(1, 2)
a1.hist(krus, bins=30, histtype="step", color=SERIES[1], lw=1.3, label="stop at ACF $<0.05$")
a1.hist(sokal, bins=30, histtype="step", color=SERIES[0], lw=1.3, label="window $M\\geq5\\tau(M)$")
a1.axvline(tau_exact, color="k", ls="--", lw=1.2, label=r"exact $(1+\rho)/(1-\rho)$")
a1.set_xlabel(r"estimated $\tau$ from one chain of $10^4$ steps")
a1.set_ylabel("chains")
a1.set_title(r"(a) AR(1), $\rho=0.9$", fontsize=9, loc="left")
a1.legend(fontsize=7)
lags = np.arange(0, 61)
for sd, c in zip((0.02, 0.2, 2.0), SERIES):
    t, M = tau_int(coin[sd])
    a2.plot(lags, acf(coin[sd], 60), "-o", ms=2, lw=0.9, color=c,
            label=f"sd {sd:g}: $\\tau={t:.0f}$")
a2.axhline(0, color="0.5", lw=0.6)
a2.set_xlabel("lag $k$")
a2.set_ylabel(r"$\hat\rho_k$")
a2.set_title("(b) the coin chains of Kruschke", fontsize=9, loc="left")
a2.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "acf_tau")

# ---- 3. thinning --------------------------------------------------------------------------------------
ks = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50])
R3 = 400
full_var = np.empty(R3)
thin_var = np.empty((R3, ks.size))
for r in range(R3):
    x = ar1(rho, N, rng)
    for j, k in enumerate(ks):
        thin_var[r, j] = x[::k].mean()
ratio_meas = thin_var.var(axis=0) / thin_var[:, 0].var()
ratio_th = ks * (1 + rho**ks) / (1 - rho**ks) / tau_exact
nums["NineBAcThinTwenty"] = ratio_th[ks == 20][0]
nums["NineBAcThinTwentyMeas"] = ratio_meas[ks == 20][0]

# ---- 4. burn-in in 10 dimensions ----------------------------------------------------------------------
D = 10
logp10 = lambda x: -0.5 * x @ x
x0 = np.full(D, 10.0)
ch, lp, a = metropolis(logp10, x0, 20_000, 2.38 / np.sqrt(D), rng)
eq = lp[10_000:]
burn = int(np.argmax(lp > np.median(eq)))                # first time log p reaches the equilibrium median
nums.update({"NineBAcDim": D, "NineBAcBurn": burn, "NineBAcLpStart": lp[0],
             "NineBAcLpEqMean": eq.mean(), "NineBAcLpEqSd": eq.std(),
             "NineBAcLpMax": lp.max(),
             "NineBAcDunkFrac": np.mean(eq - lp.max() < np.log(0.1))})
save_numbers("ch09", "16_autocorr", nums)

setup(7.0, 2.9)
fig, (b1, b2) = plt.subplots(1, 2)
b1.plot(ks, ratio_meas, "o", color=SERIES[0], label="400 AR(1) chains")
theory_line(b1, ks, ratio_th, label=r"$k(1+\rho^k)/[(1-\rho^k)\tau]$")
b1.set_xscale("log")
b1.set_xlabel("keep every $k$-th step")
b1.set_ylabel(r"Var(mean, thinned) / Var(mean, full)")
b1.set_title("(a) thinning throws information away", fontsize=9, loc="left")
b1.legend(fontsize=7)
b2.plot(lp[:3000], color=SERIES[1], lw=0.6)
b2.axhline(-D / 2, color="k", ls="--", lw=1.0, label=r"equilibrium mean $-D/2$")
b2.axhline(0, color="0.5", ls=":", lw=1.0, label="the peak, $\\ln p=0$")
b2.axvline(burn, color=SERIES[2], lw=1.0, label=f"burn-in ends ({burn} steps)")
b2.set_ylim(-80, 5)
b2.set_xlabel("step")
b2.set_ylabel(r"$\ln p(\theta_t)$ (up to a constant)")
b2.set_title(f"(b) burn-in of a {D}-D chain started far away", fontsize=9, loc="left")
b2.legend(fontsize=7, loc="lower right")
fig.tight_layout()
savefig(fig, "ch09", "thin_burnin")
