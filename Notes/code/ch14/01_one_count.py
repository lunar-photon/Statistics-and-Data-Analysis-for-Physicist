"""One count, many questions.

Question: a counting experiment expects b = 4 background events in a mass window and
records n = 11. Which object answers each of the questions we can ask about that one
number: the forward probability, the estimate and its error bar, the p-value, the
posterior, the evidence for a signal, a simulation check, and a forecast of how much
more data a discovery would need?

Computes: the Poisson pmf under the background, the tail p-value (exact and by
simulation), the maximum-likelihood signal with its curvature error and its
Delta ln L = 1/2 interval, the flat-prior posterior for s >= 0 with a 68% central
credible interval, the Bayes factor for "signal uniform in [0, 20]" against "no
signal", and the Asimov significance as a function of the exposure.

Writes: figures/ch14/one_count.pdf, results/ch14/01_one_count.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
from scipy import stats, optimize, integrate
import matplotlib.pyplot as plt

b, n = 4.0, 11          # known background, observed count
s_max = 20.0            # upper end of the uniform signal prior under H1

# --- forward: the pmf of N if only background is present -------------------------
k = np.arange(0, 21)
pmf = stats.poisson.pmf(k, b)

# --- test: P(N >= n | b), exact and by simulation ---------------------------------
p_exact = stats.poisson.sf(n - 1, b)
Z = stats.norm.isf(p_exact)
rng = rng_for("ch14", "01_one_count")
Nsim = 2_000_000
draws = rng.poisson(b, Nsim)
p_sim = np.mean(draws >= n)
p_sim_err = np.sqrt(p_sim * (1 - p_sim) / Nsim)

# --- estimate: likelihood in the signal s, mu = s + b -----------------------------
def lnL(s):
    mu = s + b
    return n * np.log(mu) - mu

s_hat = n - b
sig_curv = np.sqrt(n)                       # (-d2 lnL/ds2)^(-1/2) = sqrt(n) at s_hat
f = lambda s: lnL(s_hat) - lnL(s) - 0.5
lo = optimize.brentq(f, -b + 1e-9, s_hat)
hi = optimize.brentq(f, s_hat, 60)

# --- posterior with a flat prior on s >= 0 ----------------------------------------
sg = np.linspace(0, 40, 40001)
post = np.exp(lnL(sg) - lnL(s_hat))
post /= integrate.trapezoid(post, sg)
cdf = integrate.cumulative_trapezoid(post, sg, initial=0)
q16, q50, q84 = np.interp([0.16, 0.5, 0.84], cdf, sg)
post_mean = integrate.trapezoid(sg * post, sg)

# --- evidence: H0 (s = 0) against H1 (s uniform on [0, s_max]) ----------------------
Z0 = stats.poisson.pmf(n, b)
Z1 = integrate.quad(lambda s: stats.poisson.pmf(n, s + b), 0, s_max)[0] / s_max
B10 = Z1 / Z0
LRmax = np.exp(lnL(s_hat) - lnL(0.0))     # no prior can give B10 above L(s_hat)/L(0)

# --- forecast: Asimov significance as the exposure grows by a factor kexp ----------
def ZA(kexp):
    s, bb = kexp * s_hat, kexp * b
    return np.sqrt(2 * ((s + bb) * np.log(1 + s / bb) - s))

ZA1 = ZA(1.0)
k5 = (5.0 / ZA1) ** 2
k3 = (3.0 / ZA1) ** 2

save_numbers("ch14", "01_one_count", {
    "ThOcB": "4", "ThOcN": "11",
    "ThOcPexact": f"{p_exact:.5f}", "ThOcZ": f"{Z:.2f}",
    "ThOcPsim": f"{p_sim:.5f}", "ThOcPsimErr": f"{p_sim_err:.5f}", "ThOcNsim": r"2\times10^{6}",
    "ThOcShat": f"{s_hat:.0f}", "ThOcSigCurv": f"{sig_curv:.2f}",
    "ThOcLikLo": f"{lo:.2f}", "ThOcLikHi": f"{hi:.2f}",
    "ThOcPostMed": f"{q50:.2f}", "ThOcPostLo": f"{q16:.2f}", "ThOcPostHi": f"{q84:.2f}",
    "ThOcPostMean": f"{post_mean:.2f}",
    "ThOcZzero": f"{Z0:.5f}", "ThOcZone": f"{Z1:.5f}", "ThOcBten": f"{B10:.1f}",
    "ThOcLRmax": f"{LRmax:.0f}", "ThOcSmax": "20",
    "ThOcZA": f"{ZA1:.2f}", "ThOcKfive": f"{k5:.1f}", "ThOcKthree": f"{k3:.2f}",
})

# --- figure: four questions, four objects ------------------------------------------
setup(7.0, 5.2)
fig, ax = plt.subplots(2, 2)
a = ax[0, 0]
a.bar(k, pmf, color=SERIES[0], alpha=0.35, width=0.8, label=r"$P(N\mid b)$")
a.bar(k[k >= n], pmf[k >= n], color=SERIES[1], width=0.8, label=r"tail $N\geq n$")
a.axvline(n, color="k", lw=0.8)
a.set_xlabel("count $N$"); a.set_ylabel("probability")
a.set_title(r"(a) test: the tail beyond $n=11$")
a.set_yscale("log"); a.set_ylim(1e-6, 0.4); a.legend()

a = ax[0, 1]
sl = np.linspace(-2, 25, 600)
a.plot(sl, np.exp(lnL(sl) - lnL(s_hat)), color=SERIES[0], label=r"$\mathcal{L}(s)/\mathcal{L}_{\max}$")
theory_line(a, sl, np.exp(-0.5 * ((sl - s_hat) / sig_curv) ** 2), label="curvature Gaussian")
a.axhline(np.exp(-0.5), color=SERIES[2], lw=0.8)
a.axvspan(lo, hi, color=SERIES[2], alpha=0.12, label=r"$\Delta\ln\mathcal{L}=1/2$")
a.set_xlabel("signal $s$"); a.set_title("(b) estimate: the likelihood")
a.legend(loc="upper right", fontsize=8)

a = ax[1, 0]
a.plot(sg, post, color=SERIES[0], label=r"$p(s\mid n)$, flat prior $s\geq 0$")
m = (sg >= q16) & (sg <= q84)
a.fill_between(sg[m], post[m], color=SERIES[0], alpha=0.25, label="68% central")
a.set_xlim(0, 25); a.set_xlabel("signal $s$"); a.set_ylabel("density")
a.set_title("(c) inference: the posterior"); a.legend(fontsize=8)

a = ax[1, 1]
kk = np.linspace(0.2, 5, 200)
a.plot(kk, ZA(kk), color=SERIES[0], label="Asimov $Z_A$")
for zz, c in ((3, SERIES[3]), (5, SERIES[1])):
    a.axhline(zz, color=c, lw=0.8, ls=":")
a.axvline(k5, color=SERIES[1], lw=0.8)
a.set_xlabel("exposure / present exposure"); a.set_ylabel("expected significance")
a.set_title("(d) forecast: how much more data"); a.legend(loc="upper left")
fig.tight_layout()
savefig(fig, "ch14", "one_count")
