"""Several chains from wildly different starts, and the Gelman--Rubin statistic R-hat.

Question 1: four Metropolis chains for the two-bump density of script 11, started at -30, -10,
+10 and +30.  Do they end up sampling the same distribution, and how does R-hat (computed on
the second half of the steps so far) fall towards 1?
Question 2: the same four starts on a target whose bumps are far apart (at -6 and +6) with small
steps.  Every chain looks healthy on its own; does R-hat see that they disagree?  And what if all
four chains start in the same bump?
Question 3: for chains that ARE in equilibrium, how far above 1 is R-hat?  Compare with the
prediction R-hat - 1 ~ (1 + 1/M) tau / (2 N).
Question 4: does our tau agree with emcee's estimator on the same chain?  And how does the
burn-in rule "drop the start while p/p_max < 0.1" behave in D dimensions (typical ln p sits D/2
below the peak)?
Writes figures/ch09/multichain.pdf, figures/ch09/rhat_traps.pdf, results/ch09/17_multichain.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import emcee
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_mcmc import metropolis, gelman_rubin, tau_int

rng = rng_for("ch09", "17_multichain")
nums = {}


def two_bumps(m1, s1, m2, s2, w1=0.3):
    """log density of w1 N(m1, s1^2) + (1-w1) N(m2, s2^2), and the density itself."""
    def logf(x):
        a = np.log(w1 / s1) - 0.5 * ((x[0] - m1) / s1) ** 2
        b = np.log((1 - w1) / s2) - 0.5 * ((x[0] - m2) / s2) ** 2
        return np.logaddexp(a, b)
    def f(x):
        g = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        return w1 * g(m1, s1) + (1 - w1) * g(m2, s2)
    return logf, f


def rhat_history(chains, checkpoints):
    """R-hat after n steps, using the second half (steps n/2..n) of each chain."""
    return np.array([gelman_rubin(chains[:, n // 2:n]) for n in checkpoints])


# ---- 1. well-mixed case ----------------------------------------------------------------------------
logf, f = two_bumps(-2.0, 0.6, 2.0, 0.8)
starts = [-30.0, -10.0, 10.0, 30.0]
n = 20_000
ch1 = np.array([metropolis(logf, [x0], n, 1.5, rng)[0][:, 0] for x0 in starts])
checks = np.unique(np.geomspace(20, n, 60).astype(int))
rh1 = rhat_history(ch1, checks)
first_ok = checks[np.argmax(rh1 < 1.01)]
nums["NineBMcN"] = f"{n:,}".replace(",", r"\,")
nums["NineBMcRhatFinal"] = gelman_rubin(ch1[:, n // 2:])
nums["NineBMcRhatSplit"] = gelman_rubin(ch1[:, n // 2:], split=True)
nums["NineBMcFirstOk"] = int(first_ok)
nums["NineBMcRhatHundred"] = rh1[np.argmin(np.abs(checks - 100))]
means = ch1[:, n // 2:].mean(axis=1)
nums["NineBMcMeanLo"], nums["NineBMcMeanHi"] = means.min(), means.max()
nums["NineBMcMeanExact"] = 0.3 * -2.0 + 0.7 * 2.0

# ---- 2. traps: far-apart bumps, small steps --------------------------------------------------------
logf_far, f_far = two_bumps(-6.0, 0.6, 6.0, 0.8)
n2 = 20_000
ch_far = np.array([metropolis(logf_far, [x0], n2, 0.5, rng)[0][:, 0]
                   for x0 in (-30.0, -10.0, 10.0, 30.0)])
ch_same = np.array([metropolis(logf_far, [x0], n2, 0.5, rng)[0][:, 0]
                    for x0 in (8.0, 12.0, 20.0, 30.0)])
nums["NineBMcFarRhat"] = gelman_rubin(ch_far[:, n2 // 2:])
nums["NineBMcSameRhat"] = f"{gelman_rubin(ch_same[:, n2 // 2:]):.3f}"
nums["NineBMcSameMean"] = ch_same[:, n2 // 2:].mean()
nums["NineBMcFarMeanExact"] = 0.3 * -6.0 + 0.7 * 6.0
nums["NineBMcFarSingleTau"] = max(tau_int(c[n2 // 2:])[0] for c in ch_far)

# ---- 3. calibration: chains started in equilibrium --------------------------------------------------
# Standard Gaussian target, step 2.4: tau ~ 4.4.  M = 4 chains of N steps, many replicates.
logn = lambda x: -0.5 * x[0] ** 2
M, Ncal, R = 4, 2000, 200
rh_cal, taus = [], []
for r in range(R):
    chs = np.array([metropolis(logn, [rng.standard_normal()], Ncal, 2.4, rng)[0][1:, 0]
                    for _ in range(M)])
    rh_cal.append(gelman_rubin(chs))
    taus.append(np.mean([tau_int(c)[0] for c in chs]))
rh_cal = np.array(rh_cal)
tau_bar = np.mean(taus)
nums["NineBMcCalM"], nums["NineBMcCalN"], nums["NineBMcCalR"] = M, Ncal, R
nums["NineBMcCalTau"] = tau_bar
nums["NineBMcCalExcess"] = np.mean(rh_cal) - 1
nums["NineBMcCalPred"] = ((1 + 1 / M) * tau_bar - 1) / (2 * Ncal)   # keeps the (N-1)/N term
nums["NineBMcCalExcessErr"] = np.std(rh_cal - 1, ddof=1) / np.sqrt(R)

# ---- 4a. cross-check of tau against emcee ------------------------------------------------------------
x = metropolis(logn, [0.0], 200_000, 2.4, rng)[0][1:, 0]
nums["NineBMcTauOurs"] = tau_int(x)[0]
nums["NineBMcTauEmcee"] = float(emcee.autocorr.integrated_time(x, c=5, quiet=True)[0])

# ---- 4b. the p/p_max < 0.1 burn-in rule in D dimensions -------------------------------------------------
# In equilibrium -2 ln(p/p_max) ~ chi^2_D for a Gaussian, so P(p/p_max >= 0.1) = P(chi^2_D <= 2 ln 10).
for D, tag in ((2, "Two"), (10, "Ten"), (50, "Fifty")):
    nums[f"NineBMcNearPeak{tag}"] = stats.chi2(D).cdf(2 * np.log(10))
save_numbers("ch09", "17_multichain", nums)

# ---- figure 1: four chains, histograms, R-hat ------------------------------------------------------------
setup(7.2, 2.8)
fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1.6, 1, 1.1]})
a1, a2, a3 = axes
for c, x0, col in zip(ch1, starts, SERIES):
    a1.plot(np.arange(301), c[:301], color=col, lw=0.7, label=f"start {x0:+g}")
a1.set_xlabel("step $t$"); a1.set_ylabel("$x_t$")
a1.set_title("(a) the first 300 steps", fontsize=9, loc="left")
a1.legend(fontsize=6.5, ncol=2, loc="upper right")
xx = np.linspace(-5, 5, 400)
for c, col in zip(ch1, SERIES):
    a2.hist(c[n // 2:], bins=60, range=(-5, 5), density=True, histtype="step", color=col, lw=1.0)
theory_line(a2, xx, f(xx), label="target")
a2.set_xlabel("$x$"); a2.set_title("(b) second halves", fontsize=9, loc="left")
a2.legend(fontsize=7)
a3.loglog(checks, rh1 - 1, color=SERIES[0], lw=1.2)
a3.axhline(0.01, color="k", ls=":", lw=0.9)
a3.text(checks[0] * 1.2, 0.013, r"$\hat R=1.01$", fontsize=7.5)
a3.set_xlabel("steps per chain $n$"); a3.set_ylabel(r"$\hat R-1$")
a3.set_title(r"(c) $\hat R$ on steps $n/2..n$", fontsize=9, loc="left")
fig.tight_layout()
savefig(fig, "ch09", "multichain")

# ---- figure 2: the two traps ------------------------------------------------------------------------------
setup(7.2, 2.7)
fig, (b1, b2) = plt.subplots(1, 2, sharey=True)
tt = np.arange(n2 + 1)
for c, col in zip(ch_far, SERIES):
    b1.plot(tt[::10], c[::10], color=col, lw=0.5)
b1.set_title(rf"(a) starts on both sides: $\hat R={nums['NineBMcFarRhat']:.1f}$", fontsize=9,
             loc="left")
for c, col in zip(ch_same, SERIES):
    b2.plot(tt[::10], c[::10], color=col, lw=0.5)
b2.set_title(rf"(b) all start on the right: $\hat R={nums['NineBMcSameRhat']}$", fontsize=9,
             loc="left")
for b in (b1, b2):
    b.axhline(-6, color="0.6", ls=":", lw=0.8); b.axhline(6, color="0.6", ls=":", lw=0.8)
    b.set_xlabel("step $t$")
    b.set_ylim(-12, 32)
b1.set_ylabel("$x_t$")
fig.tight_layout()
savefig(fig, "ch09", "rhat_traps")
