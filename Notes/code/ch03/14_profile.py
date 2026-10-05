"""14_profile.py -- a nuisance parameter: profile likelihood for a signal on top of a background.

Question: a counting experiment sees n_on events in the signal region, expected s + b, and
n_off events in a background-only control region, expected k b (k = 1, the same exposure).
We want the signal s; the background b is a nuisance.  What does the profile likelihood
L_p(s) = max_b L(s, b) look like, what interval does Delta ln L_p = 1/2 give, and how does it
compare with pretending b is known (fixing b at its estimate)?  Which interval covers the
true s in 68% of repeated experiments?

Computes: for the observed (n_on, n_off) = (20, 8): the 2-D ln L, the profile path b_hat(s),
          profile and conditional ln L curves and intervals; then 10^4 simulated experiments
          with s = 12, b = 8 for the coverage of both intervals.
Writes:   figures/ch03/profile.pdf, results/ch03/14_profile.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

rng = rng_for("ch03", "14_profile")
setup()
k = 1.0
n_on, n_off = 20, 8
s_true, b_true, M = 12.0, 8.0, 10_000


def lnL(s, b, non, noff):
    return non * np.log(s + b) - (s + b) + noff * np.log(k * b) - k * b


def bhat_of_s(s, non, noff):
    """maximise over b at fixed s: (1+k) b^2 + ((1+k) s - non - noff) b - noff s = 0"""
    A = 1 + k; B = (1 + k) * s - non - noff; C = -noff * s
    return (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)


def intervals(non, noff):
    b0 = noff / k; s0 = non - b0                      # the global MLE
    Lmax = lnL(s0, b0, non, noff)
    prof = lambda s: lnL(s, bhat_of_s(s, non, noff), non, noff) - Lmax + 0.5
    cond = lambda s: lnL(s, b0, non, noff) - Lmax + 0.5
    lo_s = -b0 + 1e-9                                  # s + b must stay positive
    out = []
    for f in (prof, cond):
        lo = optimize.brentq(f, max(lo_s, s0 - 20 * np.sqrt(non + 1)), s0)
        hi = optimize.brentq(f, s0, s0 + 20 * np.sqrt(non + 1))
        out.append((lo, hi))
    return s0, b0, out


s0, b0, ((plo, phi), (clo, chi)) = intervals(n_on, n_off)

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
S, B = np.meshgrid(np.linspace(0, 26, 241), np.linspace(3, 14, 241))
Z = lnL(S, B, n_on, n_off) - lnL(s0, b0, n_on, n_off)
cs = axs[0].contour(S, B, Z, levels=[-4.5, -2.0, -0.5], colors=[SERIES[0]], linewidths=[0.6, 0.9, 1.4], linestyles="solid")
ss = np.linspace(0.5, 26, 300)
axs[0].plot(ss, bhat_of_s(ss, n_on, n_off), color=SERIES[1], lw=1.4, label=r"profile path $\hat{\hat b}(s)$")
axs[0].axhline(b0, color=SERIES[2], lw=1.0, ls="--", label=r"fixed $b=\hat b$")
axs[0].plot(s0, b0, "o", color="k", ms=4)
axs[0].set_xlabel(r"signal $s$"); axs[0].set_ylabel(r"background $b$"); axs[0].legend(fontsize=7, loc="upper right")
lp = lnL(ss, bhat_of_s(ss, n_on, n_off), n_on, n_off) - lnL(s0, b0, n_on, n_off)
lc = lnL(ss, b0, n_on, n_off) - lnL(s0, b0, n_on, n_off)
axs[1].plot(ss, lp, color=SERIES[1], label=r"profile $\ln\mathcal{L}_p(s)$")
axs[1].plot(ss, lc, color=SERIES[2], ls="--", label=r"conditional, $b=\hat b$")
axs[1].axhline(-0.5, color="k", lw=0.7, ls=":")
axs[1].set_ylim(-3, 0.3); axs[1].set_xlabel(r"signal $s$"); axs[1].set_ylabel(r"$\Delta\ln\mathcal{L}$")
axs[1].legend(fontsize=7, loc="lower center")
fig.tight_layout()
savefig(fig, "ch03", "profile")

# ---------- coverage ----------
hit_p = hit_c = 0; used = 0
for _ in range(M):
    non = rng.poisson(s_true + b_true); noff = rng.poisson(k * b_true)
    if noff == 0 or non - noff / k <= -noff / k + 1e-6:
        continue
    try:
        _, _, ((a1, a2), (c1, c2)) = intervals(non, noff)
    except ValueError:
        continue
    used += 1
    hit_p += a1 <= s_true <= a2
    hit_c += c1 <= s_true <= c2

save_numbers("ch03", "14_profile", {
    "ThreeBProfNon": n_on, "ThreeBProfNoff": n_off, "ThreeBProfK": int(k),
    "ThreeBProfS": s0, "ThreeBProfB": b0,
    "ThreeBProfLo": s0 - plo, "ThreeBProfHi": phi - s0,
    "ThreeBCondLo": s0 - clo, "ThreeBCondHi": chi - s0,
    "ThreeBProfGauss": np.sqrt(n_on + n_off / k**2), "ThreeBCondGauss": np.sqrt(n_on),
    "ThreeBProfCover": hit_p / used, "ThreeBCondCover": hit_c / used, "ThreeBProfUsed": used,
})
