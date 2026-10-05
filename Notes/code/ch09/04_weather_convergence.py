"""Forgetting where you started: a three-state weather chain from wildly different beginnings.

Question: sunny/cloudy/rainy weather moves by the transition matrix P below.  Five forecasters
start from very different beliefs about day 0.  Do their forecasts for day t agree in the end,
and how fast does the disagreement die?
Computes p_t = p_0 P^t exactly, the total-variation distance to the stationary distribution pi,
its log-slope (compared with ln|lambda_2|), the Doeblin bound (1-delta)^t, and a Monte Carlo
check with independent simulated weather histories.  Also the two-state AoS weather chain.
Writes figures/ch09/weather_convergence.pdf and results/ch09/04_weather_convergence.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_markov import stationary, evolve, tv, doeblin, step_many

P = np.array([[0.6, 0.2, 0.2],      # sunny  -> sunny, cloudy, rainy
              [0.3, 0.7, 0.0],      # cloudy -> ...
              [0.6, 0.0, 0.4]])     # rainy  -> ...
pi = stationary(P)
lam = np.sort(np.linalg.eigvals(P).real)[::-1]
delta = doeblin(P)

starts = {"certain sun": [1, 0, 0], "certain cloud": [0, 1, 0], "certain rain": [0, 0, 1],
          "uniform": [1 / 3, 1 / 3, 1 / 3], "rain-heavy": [0.05, 0.05, 0.9]}
T = 40
paths = {k: evolve(np.array(v, float), P, T) for k, v in starts.items()}
dist = {k: tv(v, pi) for k, v in paths.items()}

# slope of ln TV over the late part of the curve, for the certain-rain start
t = np.arange(T + 1)
sl = slice(10, 31)
slope = np.polyfit(t[sl], np.log(dist["certain rain"][sl]), 1)[0]

# Monte Carlo: simulated weather histories, all starting with rain
rng = rng_for("ch09", "04_weather_convergence")
W = 200_000
s = np.full(W, 2)
mc_tv = [1.0 - pi[2]]
for _ in range(T):
    s = step_many(s, P, rng)
    mc_tv.append(tv(np.bincount(s, minlength=3) / W, pi))
mc_tv = np.array(mc_tv)
floor = np.sqrt(2 / (np.pi * W)) * np.sqrt(pi * (1 - pi)).sum() / 2   # E TV of a pure-noise histogram

setup(7.0, 3.1)
fig, (ax1, ax2) = plt.subplots(1, 2)
for c, (k, v) in zip(SERIES, paths.items()):
    ax1.plot(t[:16], v[:16, 2], "-o", ms=2.5, color=c, label=k)
ax1.axhline(pi[2], color="k", ls="--", lw=1.2)
ax1.set_xlabel("day $t$")
ax1.set_ylabel(r"$p_t(\mathrm{rain})$")
ax1.set_title(r"(a) forecasts of rain, five starts")
ax1.legend(fontsize=7, loc="upper right")
for c, (k, v) in zip(SERIES, dist.items()):
    ax2.semilogy(t, np.maximum(v, 1e-18), "-", color=c, lw=1.2)
ax2.semilogy(t[1:], mc_tv[1:], ".", color="0.35", ms=3, label="simulated histories (rain start)")
theory_line(ax2, t, 0.6 * lam[1] ** t, label=r"$\frac{3}{5}(0.6)^t$")
ax2.semilogy(t, (1 - delta) ** t, ":", color="k", lw=1.4, label=r"Doeblin bound $0.7^t$")
ax2.axhline(floor, color="0.6", lw=0.8)
ax2.set_ylim(1e-10, 2)
ax2.set_xlabel("day $t$")
ax2.set_ylabel(r"$\|p_t-\pi\|_{\rm TV}$")
ax2.set_title("(b) distance to $\\pi$, log scale")
ax2.legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "ch09", "weather_convergence")

# the two-state chain of Wasserman Example 23.8: sunny -> (0.4, 0.6), cloudy -> (0.8, 0.2)
P2 = np.array([[0.4, 0.6], [0.8, 0.2]])
p2 = evolve(np.array([1.0, 0.0]), P2, 6)

save_numbers("ch09", "04_weather_convergence", {
    "NineAWLamTwo": round(lam[1], 6), "NineAWLamThree": round(lam[2], 6),
    "NineAWSlope": round(slope, 4), "NineAWLnLam": round(np.log(lam[1]), 4),
    "NineAWDelta": round(delta, 3),
    "NineAWTVTen": f"{dist['certain rain'][10]:.4f}", "NineAWTVTenPred": f"{0.6 * 0.6**10:.4f}",
    "NineAWTVTwenty": f"{dist['certain rain'][20]:.2e}".replace("e-0", r"\times10^{-") + "}",
    "NineAWMCFloor": f"{floor:.1e}".replace("e-0", r"\times10^{-") + "}",
    "NineAWWalkers": f"{W:,}".replace(",", r"\,"),
    "NineAWTwoSunThree": round(p2[3][0], 4), "NineAWTwoSunSix": round(p2[6][0], 4),
})
np.set_printoptions(precision=5, suppress=True)
print("pi", pi, "lam", lam, "delta", delta, "slope", slope, np.log(lam[1]))
print("tv rain", dist["certain rain"][:12])
print("2-state", p2[:, 0])
