"""The Poisson process from its postulates: constant rate, no memory.

Question:  if time is cut into tiny slices of width h and each slice holds an
           event with probability lam*h, independently of all other slices,
           is the count in a window of length T Poisson(lam*T)?  Is a thinned
           count (detector efficiency eps) Poisson(eps*lam*T)?  Is the sum of
           two independent sources Poisson with the summed mean?  And does
           the master equation dP_n/dt = -lam P_n + lam P_(n-1) integrate to
           the Poisson pmf?
Computes:  20000 windows of T = 5 s at lam = 2 /s with h = 1 ms; thinning with
           eps = 0.3; a second source of rate 1 /s; Euler integration of the
           master equation for n = 0..4.
Writes:    figures/ch02/05_poisson_process.pdf, results/ch02/05_poisson_process.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

setup(6.4, 2.9)
rng = rng_for("ch02", "05_poisson_process")
lam, T, h, eps, lam2 = 2.0, 5.0, 1e-3, 0.3, 1.0
slices = int(round(T / h))          # 5000 slices per window
windows, chunk = 20_000, 1000

N = np.empty(windows, int); Ndet = np.empty(windows, int); Nsum = np.empty(windows, int)
for s in range(0, windows, chunk):
    ev = rng.uniform(size=(chunk, slices)) < lam * h          # the postulate, slice by slice
    det = ev & (rng.uniform(size=(chunk, slices)) < eps)      # each event survives with prob eps
    ev2 = rng.uniform(size=(chunk, slices)) < lam2 * h        # an independent second source
    N[s:s + chunk] = ev.sum(1)
    Ndet[s:s + chunk] = det.sum(1)
    Nsum[s:s + chunk] = ev.sum(1) + ev2.sum(1)

# Euler integration of the master equation, P_n(0) = delta_{n0}
dt, tmax, nmax = 1e-3, 4.0, 5
t = np.arange(0, tmax + dt / 2, dt)
P = np.zeros((t.size, nmax)); P[0, 0] = 1.0
for i in range(1, t.size):
    prev = P[i - 1]
    gain = np.concatenate([[0.0], prev[:-1]])          # P_{n-1}, with P_{-1} = 0
    P[i] = prev + dt * lam * (gain - prev)

fig, (ax1, ax2) = plt.subplots(1, 2)
k = np.arange(0, 25)
ax1.bar(k, np.bincount(N, minlength=25)[:25] / windows, color=SERIES[0], alpha=0.4, label=r"all events")
ax1.bar(k, np.bincount(Ndet, minlength=25)[:25] / windows, color=SERIES[1], alpha=0.4, label=r"detected ($\epsilon=0.3$)")
theory_line(ax1, k, poisson.pmf(k, lam * T), label="Poisson(10), Poisson(3)", marker="_", ms=8, mew=1.5)
ax1.plot(k, poisson.pmf(k, eps * lam * T), color="k", ls="--", lw=1.4, marker="_", ms=8, mew=1.5)
ax1.set_xlabel("counts in $T=5$ s"); ax1.set_ylabel("probability")
ax1.legend(fontsize=7.5)
for n in range(nmax):
    ax2.plot(t, P[:, n], color=SERIES[n], lw=2.2, alpha=0.6, label=f"$n={n}$")
for n in range(nmax):                                   # analytic Poisson pmf, black dashed
    ax2.plot(t, np.exp(-lam * t) * (lam * t) ** n / factorial(n), "k--", lw=0.8)
ax2.set_xlabel("$t$ [s]"); ax2.set_ylabel("$P_n(t)$")
ax2.legend(fontsize=7.5, ncol=2)
fig.tight_layout()
savefig(fig, "ch02", "05_poisson_process")

maxerr = max(np.max(np.abs(P[:, n] - np.exp(-lam * t) * (lam * t) ** n / factorial(n))) for n in range(nmax))
save_numbers("ch02", "05_poisson_process", {
    "twoaPPMean": float(N.mean()), "twoaPPVar": float(N.var(ddof=1)),
    "twoaPPDetMean": float(Ndet.mean()), "twoaPPDetVar": float(Ndet.var(ddof=1)),
    "twoaPPSumMean": float(Nsum.mean()), "twoaPPSumVar": float(Nsum.var(ddof=1)),
    "twoaPPZeroSim": float(np.mean(N == 0)), "twoaPPZeroTh": poisson.pmf(0, lam * T),
    "twoaPPMaxErr": maxerr,
    "twoaPPZeroCount": int(np.sum(N == 0)),                       # empty windows in this run
    "twoaPPZeroExpect": f"{windows * poisson.pmf(0, lam * T):.1f}",  # expected number of them
})
