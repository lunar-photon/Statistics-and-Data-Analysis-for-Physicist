"""Waiting times of a Poisson process: exponential gaps and gamma (Erlang) arrival times.

Question:  (a) if gaps between events are drawn as T = -ln(U)/lam (inverse
           transform of a uniform U), are the counts in a window Poisson(lam*t)?
           (b) is the time W_n of the n-th event Gamma(n, 1/lam), density
           lam^n w^(n-1) e^(-lam w)/(n-1)!?   (c) is the exponential memoryless?
Computes:  200000 gaps at lam = 2 /s -> event times -> counts in 1 s windows;
           W_n for n = 1, 3, 10 from sums of gaps; the conditional survival
           P(T > t+s | T > t) against P(T > s).
Writes:    figures/ch02/06_waiting_times.pdf, results/ch02/06_waiting_times.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

setup(6.4, 2.9)
rng = rng_for("ch02", "06_waiting_times")
lam = 2.0

# (a) gaps by inverse transform, then the event times are the running sums
gaps = -np.log(rng.uniform(size=200_000)) / lam
times = np.cumsum(gaps)
Tmax = np.floor(times[-1])
counts = np.bincount(np.floor(times[times < Tmax]).astype(int), minlength=int(Tmax))  # events per 1 s window

# (b) arrival time of the n-th event: sum of n independent gaps
def erlang_pdf(w, n):
    return lam**n * w**(n - 1) * np.exp(-lam * w) / factorial(n - 1)

fig, (ax1, ax2) = plt.subplots(1, 2)
w = np.linspace(1e-3, 9, 400)
for c, n in zip(SERIES, [1, 3, 10]):
    Wn = gaps[: (len(gaps) // n) * n].reshape(-1, n).sum(axis=1)
    ax1.hist(Wn, bins=np.linspace(0, 9, 61), density=True, color=c, alpha=0.4, label=f"$W_{{{n}}}$")
    ax1.plot(w, erlang_pdf(w, n), "k--", lw=1.2)
ax1.set_xlabel("waiting time $w$ [s]"); ax1.set_ylabel("density")
ax1.set_ylim(0, 2.05); ax1.legend()

k = np.arange(0, 11)
ax2.bar(k, np.bincount(counts, minlength=11)[:11] / counts.size, color=SERIES[0], alpha=0.4,
        label="counts per 1 s, from gaps")
theory_line(ax2, k, poisson.pmf(k, lam), label="Poisson(2)", marker="_", ms=9, mew=1.6)
ax2.set_xticks(range(0, 11, 2)); ax2.set_ylim(0, 0.36)
ax2.set_xlabel("counts in 1 s"); ax2.set_ylabel("probability"); ax2.legend(fontsize=8, loc="upper right")
fig.tight_layout()
savefig(fig, "ch02", "06_waiting_times")

# (c) memorylessness:  P(T > t + s | T > t)  vs  P(T > s),  t = 1 s, s = 0.5 s
t0, s0 = 1.0, 0.5
cond = np.mean(gaps[gaps > t0] > t0 + s0)
uncond = np.mean(gaps > s0)

W10 = gaps[: (len(gaps) // 10) * 10].reshape(-1, 10).sum(axis=1)
save_numbers("ch02", "06_waiting_times", {
    "twoaWTGapMean": float(gaps.mean()), "twoaWTGapVar": float(gaps.var(ddof=1)),
    "twoaWTCountMean": float(counts.mean()), "twoaWTCountVar": float(counts.var(ddof=1)),
    "twoaWTWtenMean": float(W10.mean()), "twoaWTWtenVar": float(W10.var(ddof=1)),
    "twoaWTCond": cond, "twoaWTUncond": uncond, "twoaWTMemTh": np.exp(-lam * s0),
    "twoaMuonSurv": np.exp(-5.0 / 2.197),
})
