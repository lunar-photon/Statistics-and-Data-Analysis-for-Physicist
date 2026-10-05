"""The island-hopping politician as a Markov chain.

Question: seven islands theta = 1..7 with relative populations 1..7.  Each day the politician
proposes a neighbour by a fair coin and goes there for sure if it is more populous, otherwise
with probability P_proposed / P_current (islands 0 and 8 have no people).  Where is he on day t,
and where does he spend his time in the long run?
Computes the 7x7 transition matrix, the exact distribution p_t = p_1 P^(t-1) from a start on
island 4, an ensemble of independent politicians, and the time fractions of one long trip.
Writes figures/ch09/island_trajectory.pdf, figures/ch09/island_distributions.pdf and
results/ch09/03_island_hopping.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES
from lib_markov import stationary, evolve, tv, step_many, walk

K = 7
pop = np.arange(1, K + 1, dtype=float)          # relative populations P(theta) = theta

# --- transition matrix: propose a neighbour (prob 1/2 each), accept with min(1, ratio) ------
P = np.zeros((K, K))
for i in range(K):
    for j in (i - 1, i + 1):
        if 0 <= j < K:                          # islands outside 1..7 have population 0: never accepted
            P[i, j] = 0.5 * min(1.0, pop[j] / pop[i])
    P[i, i] = 1.0 - P[i].sum()                  # rejected proposals: stay put

rng = rng_for("ch09", "03_island_hopping")
start = np.zeros(K); start[3] = 1.0              # day 1 on island 4
days = 99
p = evolve(start, P, days - 1)                   # p[s] is the distribution on day s+1

# ensemble of independent politicians: histogram of where they are on chosen days
W = 20_000
states = np.full(W, 3)
snap = {}
for day in range(2, days + 1):
    states = step_many(states, P, rng)
    if day in (2, 3, 5, 13, 99):
        snap[day] = np.bincount(states, minlength=K) / W

# one long trip: fraction of days on each island
n_days = 100_000
trip = walk(P, 3, n_days - 1, rng)
frac = np.bincount(trip, minlength=K) / n_days
pi = stationary(P)

# --- figure 1: one trajectory (log time, as in Kruschke Fig. 7.2) and its time fractions ----
setup(7.0, 3.0)
fig, (ax1, ax2) = plt.subplots(1, 2)
tt = np.arange(1, 501)
ax1.plot(trip[:500] + 1, tt, color=SERIES[0], lw=0.7, marker="o", ms=1.5)
ax1.set_yscale("log")
ax1.set_xlabel(r"island $\theta$")
ax1.set_ylabel("day")
ax1.set_title("(a) the first 500 days of one trip")
ax2.bar(np.arange(1, K + 1), frac, color=SERIES[0], alpha=0.6, label=f"fraction of {n_days:,} days")
ax2.plot(np.arange(1, K + 1), pop / pop.sum(), "k_", ms=18, mew=1.6, label=r"$\theta/28$")
ax2.set_xlabel(r"island $\theta$")
ax2.set_ylabel("fraction of days")
ax2.set_title("(b) time spent tracks the population")
ax2.legend(loc="upper left", fontsize=8)
fig.tight_layout()
savefig(fig, "ch09", "island_trajectory")

# --- figure 2: exact p_t (bars) versus the ensemble (dots) ----------------------------------
setup(7.0, 2.4)
fig, axes = plt.subplots(1, 5, sharey=False)
for ax, day in zip(axes, sorted(snap)):
    ax.bar(np.arange(1, K + 1), p[day - 1], color=SERIES[0], alpha=0.55)
    ax.plot(np.arange(1, K + 1), snap[day], "o", color=SERIES[1], ms=3)
    ax.set_title(f"day {day}")
    ax.set_xticks([1, 4, 7])
    ax.set_xlabel(r"$\theta$")
axes[0].set_ylabel(r"$p_t(\theta)$")
fig.tight_layout()
savefig(fig, "ch09", "island_distributions")

save_numbers("ch09", "03_island_hopping", {
    "NineAIslTwoThree": round(p[1][2], 4), "NineAIslTwoFour": round(p[1][3], 4),
    "NineAIslTwoFive": round(p[1][4], 4),
    "NineAIslThreeTwo": round(p[2][1], 4), "NineAIslThreeThree": round(p[2][2], 4),
    "NineAIslThreeFour": round(p[2][3], 4), "NineAIslThreeFive": round(p[2][4], 4),
    "NineAIslThreeSix": round(p[2][5], 4),
    "NineAIslTVThirteen": round(float(tv(p[12], pi)), 3),
    "NineAIslTVNinetyNine": f"{float(tv(p[98], pi)):.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}",
    "NineAIslMaxDev": round(float(np.max(np.abs(frac - pi))), 4),
    "NineAIslDays": f"{n_days:,}".replace(",", r"\,"),
    "NineAIslWalkers": f"{W:,}".replace(",", r"\,"),
    "NineAIslLamTwo": round(float(np.sort(np.abs(np.linalg.eigvals(P)))[-2]), 3),
})
np.set_printoptions(precision=4, suppress=True)
print(P); print("p2", p[1]); print("p3", p[2]); print("pi", pi); print("frac", frac)
print("TV day13", tv(p[12], pi), "day99", tv(p[98], pi))
print("eig", np.sort(np.abs(np.linalg.eigvals(P)))[::-1])
