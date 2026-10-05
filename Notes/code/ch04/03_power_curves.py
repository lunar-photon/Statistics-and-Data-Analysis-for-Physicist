"""03_power_curves.py -- how often does a test catch an effect that is really there?

Question: (i) for n Gaussian readings with known sigma, the one-sided test
"reject H0: mu <= 0 when sqrt(n) xbar / sigma > z_alpha" has power
beta(mu) = 1 - Phi(sqrt(n)(c - mu)/sigma).  Does a simulation agree, and how does
the curve sharpen with n?  (ii) the two-sided Wald test with estimated standard
error: does the approximate power formula (Wasserman 10.6) hold?  (iii) a
counting experiment with b = 3.2 expected background events: how large must the
signal be before a 3 sigma or 5 sigma threshold is usually crossed?

Writes: figures/ch04/power_curves.pdf, figures/ch04/power_counting.pdf,
        results/ch04/03_power_curves.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "03_power_curves")
setup()
M = 20_000
alpha, sigma = 0.05, 1.0
z_a = stats.norm.isf(alpha)
z_a2 = stats.norm.isf(alpha / 2)

# ---------- (i) one-sided z-test ----------
mus = np.linspace(-0.6, 1.2, 37)
ns = [4, 16, 64]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
size_mc = {}
for j, n in enumerate(ns):
    c = sigma * z_a / np.sqrt(n)
    th = 1 - stats.norm.cdf(np.sqrt(n) * (c - mus) / sigma)
    # the sample mean of n N(mu, sigma^2) readings is N(mu, sigma^2/n): simulate the readings anyway
    mc = [np.mean(rng.normal(m, sigma, size=(M, n)).mean(1) > c) for m in mus[::3]]
    axs[0].plot(mus, th, color=SERIES[j], label=f"$n={n}$")
    axs[0].plot(mus[::3], mc, "o", color=SERIES[j], ms=3)
    size_mc[n] = np.mean(rng.normal(0.0, sigma, size=(M, n)).mean(1) > c)
axs[0].axhline(alpha, color="0.4", ls=":", lw=0.9)
axs[0].axvline(0, color="0.4", lw=0.8)
axs[0].set_xlabel(r"true mean $\mu$"); axs[0].set_ylabel(r"power $\beta(\mu)$")
axs[0].set_title(r"one-sided, $H_0:\mu\leq 0$"); axs[0].legend(fontsize=7)

# ---------- (ii) two-sided Wald test with estimated se ----------
n2 = 25
th_wald = lambda m: (1 - stats.norm.cdf((0 - m) / (sigma / np.sqrt(n2)) + z_a2)
                     + stats.norm.cdf((0 - m) / (sigma / np.sqrt(n2)) - z_a2))
mus2 = np.linspace(-1.0, 1.0, 41)
mc2 = []
for m in mus2[::4]:
    x = rng.normal(m, sigma, size=(M, n2))
    W = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n2))
    mc2.append(np.mean(np.abs(W) > z_a2))
theory_line(axs[1], mus2, th_wald(mus2), label="approximate formula")
axs[1].plot(mus2[::4], mc2, "o", color=SERIES[0], ms=3, label="simulation")
axs[1].axhline(alpha, color="0.4", ls=":", lw=0.9)
axs[1].set_xlabel(r"true mean $\mu$"); axs[1].set_title(r"two-sided Wald, $n=25$")
axs[1].legend(fontsize=7); axs[1].set_ylim(axs[0].get_ylim())
fig.tight_layout()
savefig(fig, "ch04", "power_curves")

# sample size for power 0.8 at mu = 0.5 (one-sided, alpha = 0.05)
z_b = stats.norm.isf(0.2)
n_req = ((z_a + z_b) * sigma / 0.5) ** 2
n_req_int = int(np.ceil(n_req))
c_req = sigma * z_a / np.sqrt(n_req_int)
pow_req_mc = np.mean(rng.normal(0.5, sigma, size=(M, n_req_int)).mean(1) > c_req)

# ---------- (iii) counting experiment b = 3.2 ----------
b = 3.2
def n_crit(p_thr):
    """smallest n with P(N >= n | b) <= p_thr"""
    n = 0
    while stats.poisson.sf(n - 1, b) > p_thr:
        n += 1
    return n
p3, p5 = stats.norm.sf(3), stats.norm.sf(5)
n3, n5 = n_crit(p3), n_crit(p5)
s = np.linspace(0, 40, 201)
pow3 = stats.poisson.sf(n3 - 1, b + s)
pow5 = stats.poisson.sf(n5 - 1, b + s)
s50_3 = s[np.argmax(pow3 >= 0.5)]
s50_5 = s[np.argmax(pow5 >= 0.5)]
s_chk = 20.0
pow5_mc = np.mean(rng.poisson(b + s_chk, size=M) >= n5)
fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.plot(s, pow3, color=SERIES[0], label=rf"$3\sigma$: reject if $n\geq {n3}$")
ax.plot(s, pow5, color=SERIES[1], label=rf"$5\sigma$: reject if $n\geq {n5}$")
ax.plot([s_chk], [pow5_mc], "ko", ms=4, label="simulation")
ax.axhline(0.5, color="0.4", ls=":", lw=0.9)
ax.set_xlabel(r"expected signal events $s$  (background $b=3.2$)")
ax.set_ylabel("power"); ax.legend(fontsize=8)
savefig(fig, "ch04", "power_counting")

save_numbers("ch04", "03_power_curves", {
    "FourAPowSizeFour": f"{size_mc[4]:.4f}", "FourAPowSizeSixtyfour": f"{size_mc[64]:.4f}",
    "FourAPowNreq": f"{n_req:.1f}", "FourAPowNreqInt": n_req_int, "FourAPowReqMC": f"{pow_req_mc:.3f}",
    "FourAPowWaldSize": f"{mc2[len(mc2)//2]:.4f}",
    "FourAPowNthree": n3, "FourAPowNfive": n5,
    "FourAPowSfiftyThree": f"{s50_3:.1f}", "FourAPowSfiftyFive": f"{s50_5:.1f}",
    "FourAPowFiveAtTwenty": f"{stats.poisson.sf(n5 - 1, b + s_chk):.3f}",
    "FourAPowFiveAtTwentyMC": f"{pow5_mc:.3f}",
})
print(size_mc, n_req, n_req_int, pow_req_mc, mc2[len(mc2)//2], n3, n5, s50_3, s50_5, pow5_mc)
