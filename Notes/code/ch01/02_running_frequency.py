"""Probability as a long-run relative frequency.

Question: what does "P(heads) = p" mean operationally, and how fast does the
running proportion of heads settle down to p?

Computes: the running proportion of heads in n = 1000 flips of a coin with
p = 0.3 and with p = 0.03 (AoS Ch.1, exercise 21), four independent runs each,
and the band p +- 2 sqrt(p(1-p)/n) that the proportion typically stays inside.
Writes: figures/ch01/1a_running_frequency.pdf, results/ch01/1a_running_frequency.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt

n_max, runs = 1000, 4
n = np.arange(1, n_max + 1)
setup(8.0, 3.4)
fig, axes = plt.subplots(1, 2, sharey=False)
finals = {}
for ax, p, tag in zip(axes, (0.3, 0.03), ("A", "B")):
    rng = rng_for("ch01", "1a_running_frequency", stream=int(1000 * p))
    ends = []
    for r in range(runs):
        flips = rng.random(n_max) < p          # True = heads, with probability p
        running = np.cumsum(flips) / n         # proportion of heads after n flips
        ax.plot(n, running, color=SERIES[r], lw=1.0, alpha=0.9)
        ends.append(running[-1])
    band = 2 * np.sqrt(p * (1 - p) / n)
    theory_line(ax, n, np.full_like(n, p, dtype=float), label=f"$p={p}$")
    ax.fill_between(n, p - band, p + band, color="0.85", zorder=0,
                    label=r"$p\pm2\sqrt{p(1-p)/n}$")
    ax.set_xscale("log")
    ax.set_xlabel("number of flips $n$")
    ax.set_ylabel("running proportion of heads")
    ax.set_ylim(-0.02, min(1.0, 6 * p + 0.1))
    ax.set_title(f"$p = {p}$, {runs} independent runs")
    ax.legend(loc="upper right")
    finals[tag] = ends
fig.tight_layout()
savefig(fig, "ch01", "1a_running_frequency")

save_numbers("ch01", "1a_running_frequency", {
    "OneARunFinalA": f"{finals['A'][0]:.3f}", "OneARunFinalAtwo": f"{finals['A'][1]:.3f}",
    "OneARunFinalB": f"{finals['B'][0]:.3f}", "OneARunFinalBtwo": f"{finals['B'][1]:.3f}",
    "OneARunBandA": f"{2*np.sqrt(0.3*0.7/1000):.3f}",
    "OneARunBandB": f"{2*np.sqrt(0.03*0.97/1000):.3f}",
    # how many of the runs end inside the band p +- 2 sqrt(p(1-p)/1000)
    "OneARunInsideA": int(np.sum(np.abs(np.array(finals['A']) - 0.3) <= 2 * np.sqrt(0.3 * 0.7 / 1000))),
    "OneARunInsideB": int(np.sum(np.abs(np.array(finals['B']) - 0.03) <= 2 * np.sqrt(0.03 * 0.97 / 1000))),
    "OneARunRuns": runs,
})
print(finals)
