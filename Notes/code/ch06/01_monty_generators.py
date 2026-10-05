"""Monty Hall played with a good and with a bad source of random numbers.

Question: a simulation of Monty Hall must reproduce, for each hypothesis about
the car, the probability that Monty opens door C.  Does it matter which
pseudorandom generator supplies the coin Monty tosses when he has a choice?

Computes: N games in which we always pick door A.  Each game uses two integer
draws: car = x % 3 and Monty's coin = x % 2.  The integers come either from
numpy's PCG64 (through rng_for) or from the textbook LCG
x -> (1664525 x + 1013904223) mod 2^32, whose lowest bit alternates.
For each source: P(Monty opens C | H) for H = car behind A, B, C, and the
fraction of the games with "Monty opened C" in which the car was behind A, B, C.
Writes: figures/ch06/monty_generators.pdf, results/ch06/01_monty_generators.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_prng import lcg_ints_vec

N = 300_000
A_NR, C_NR, M32 = 1664525, 1013904223, 2**32


def play(ints):
    """ints has 2N entries; game k uses ints[2k] for the car and ints[2k+1] for the coin."""
    car = ints[0::2] % 3                       # 0 = A, 1 = B, 2 = C
    coin = ints[1::2] % 2                      # Monty's tie-break when the car is behind A
    # We picked A. Monty opens a door that is neither ours nor the car's.
    opened = np.where(car == 0, np.where(coin == 1, 2, 1),   # free choice: B or C
                      np.where(car == 1, 2, 1))              # forced choice
    lik = np.array([np.mean(opened[car == h] == 2) for h in range(3)])   # P(open C | H)
    sel = opened == 2
    post = np.array([np.mean(car[sel] == h) for h in range(3)])          # P(H | open C)
    switch = np.mean(car != 0)
    return lik, post, switch


rng = rng_for("ch06", "01_monty_generators")
good = rng.integers(0, 2**32, size=2 * N, dtype=np.int64)
bad1 = lcg_ints_vec(A_NR, C_NR, M32, 2026, 2 * N)     # seed 2026
bad2 = lcg_ints_vec(A_NR, C_NR, M32, 2027, 2 * N)     # seed 2027

res = {name: play(x) for name, x in [("good", good), ("bad1", bad1), ("bad2", bad2)]}
coinbits1 = bad1[1::2] % 2
coinbits2 = bad2[1::2] % 2
print("coin bits, seed 2026:", coinbits1[:8], " seed 2027:", coinbits2[:8])
for k, (lik, post, sw) in res.items():
    print(k, "P(open C|H) =", lik.round(4), " P(H|open C) =", post.round(4), " switch wins", round(sw, 4))

# ---------------- figure ----------------
setup(6.4, 2.8)
fig, axes = plt.subplots(1, 2, sharey=True)
labels = ["$H_A$", "$H_B$", "$H_C$"]
xs = np.arange(3)
w = 0.26
names = [("good", "PCG64"), ("bad1", "LCG, seed 2026"), ("bad2", "LCG, seed 2027")]
for j, (k, lab) in enumerate(names):
    lik, post, _ = res[k]
    axes[0].bar(xs + (j - 1) * w, lik, w, color=SERIES[j], label=lab)
    axes[1].bar(xs + (j - 1) * w, post, w, color=SERIES[j], label=lab)
for x0, v in zip(xs, [0.5, 1, 0]):
    axes[0].hlines(v, x0 - 1.6 * w, x0 + 1.6 * w, colors="k", linestyles="--", lw=1.2)
for x0, v in zip(xs, [1 / 3, 2 / 3, 0]):
    axes[1].hlines(v, x0 - 1.6 * w, x0 + 1.6 * w, colors="k", linestyles="--", lw=1.2)
axes[0].set_title(r"$P(\mathrm{Monty\ opens\ }C\mid H)$")
axes[1].set_title(r"$P(H\mid \mathrm{Monty\ opened\ }C)$, by counting")
for ax in axes:
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1.08)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
savefig(fig, "ch06", "monty_generators")

g, b1, b2 = res["good"], res["bad1"], res["bad2"]
save_numbers("ch06", "01_monty_generators", {
    "SixAMontyN": f"{N:,}".replace(",", "{,}"),
    "SixAMontyGoodLikA": f"{g[0][0]:.3f}", "SixAMontyGoodLikB": f"{g[0][1]:.3f}",
    "SixAMontyGoodPostA": f"{g[1][0]:.3f}", "SixAMontyGoodPostB": f"{g[1][1]:.3f}",
    "SixAMontyBadOneLikA": f"{b1[0][0]:.3f}", "SixAMontyBadOnePostA": f"{b1[1][0]:.3f}",
    "SixAMontyBadOnePostB": f"{b1[1][1]:.3f}",
    "SixAMontyBadTwoLikA": f"{b2[0][0]:.3f}", "SixAMontyBadTwoPostB": f"{b2[1][1]:.3f}",
    "SixAMontyGoodSwitch": f"{g[2]:.3f}", "SixAMontyBadOneSwitch": f"{b1[2]:.3f}",
    "SixAMontyBadTwoSwitch": f"{b2[2]:.3f}",
})
