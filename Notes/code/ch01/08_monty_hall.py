"""Monty Hall and the three-card problem: Bayes' theorem as selection of simulated events.

Question: after Monty opens an empty door, is the prize more likely behind our
door or behind the other closed door? After seeing a green face of a randomly
drawn card, how likely is the hidden face to be green? (AoS Ch.1, exercises 10, 12.)

Computes: for many simulated games, the prize door, our choice (always door 1)
and the door Monty opens (chosen at random when he has a choice). We keep only
the games in which Monty opened door 3 (conditioning = filtering) and count
where the prize is. Same for the three cards: keep the draws that show green,
count how often the hidden side is green. Also the running win rate of the
"stay" and "switch" strategies over all games.
Writes: figures/ch01/1a_monty_hall.pdf, results/ch01/1a_monty_hall.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "1a_monty_hall")
N = 300_000

# ---------------- Monty Hall ----------------
prize = rng.integers(1, 4, size=N)            # door hiding the prize, uniform on {1,2,3}
coin = rng.random(N) < 0.5                     # Monty's tie-break when prize is behind door 1
# we always pick door 1. Monty opens a door that is neither ours nor the prize door.
opened = np.where(prize == 1, np.where(coin, 2, 3),   # free choice between 2 and 3
                  np.where(prize == 2, 3, 2))          # forced choice
sel = opened == 3                              # the observation "Monty opened door 3"
post = np.array([np.mean(prize[sel] == d) for d in (1, 2, 3)])
p_open3 = sel.mean()

stay_wins = prize == 1
switch_wins = ~stay_wins                       # the other closed door hides the prize
n = np.arange(1, N + 1)
run_stay = np.cumsum(stay_wins) / n
run_switch = np.cumsum(switch_wins) / n

# ---------------- three cards ----------------
# card 0: green/green, card 1: red/red, card 2: green/red. Faces: 1 = green, 0 = red
faces = np.array([[1, 1], [0, 0], [1, 0]])
card = rng.integers(0, 3, size=N)
side = rng.integers(0, 2, size=N)              # which face we happen to see
seen = faces[card, side]
hidden = faces[card, 1 - side]
green_seen = seen == 1
p_hidden_green = hidden[green_seen].mean()

# ---------------- figure ----------------
idx = np.unique(np.logspace(0, np.log10(N), 1500).astype(int)) - 1
setup(8.0, 3.3)
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
ax1.plot(n[idx], run_switch[idx], color=SERIES[0], label="switch")
ax1.plot(n[idx], run_stay[idx], color=SERIES[1], label="stay")
theory_line(ax1, n[idx], np.full(idx.size, 2 / 3), label="exact 2/3 and 1/3")
ax1.plot(n[idx], np.full(idx.size, 1 / 3), color="k", ls="--", lw=1.4, zorder=5)
ax1.set_xscale("log")
ax1.set_ylim(0, 1)
ax1.set_xlabel("number of simulated games")
ax1.set_ylabel("fraction of games won")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title("Monty Hall: running win rate")

ax2.bar([1, 2, 3], post, color=[SERIES[1], SERIES[0], SERIES[2]], width=0.6)
ax2.plot([0.7, 1.3], [1 / 3, 1 / 3], "k--", lw=1.4)
ax2.plot([1.7, 2.3], [2 / 3, 2 / 3], "k--", lw=1.4)
ax2.set_xticks([1, 2, 3], ["door 1\n(ours)", "door 2", "door 3\n(opened)"])
ax2.set_ylim(0, 1)
ax2.set_ylabel("fraction of selected games")
ax2.set_title("prize location, games with door 3 opened")
fig.tight_layout()
savefig(fig, "ch01", "1a_monty_hall")

save_numbers("ch01", "1a_monty_hall", {
    "OneAMontyN": N,
    "OneAMontyNsel": int(sel.sum()),
    "OneAMontyPostOne": f"{post[0]:.4f}",
    "OneAMontyPostTwo": f"{post[1]:.4f}",
    "OneAMontyPostThree": f"{post[2]:.4f}",
    "OneAMontyOpenThree": f"{p_open3:.4f}",
    "OneAMontySwitch": f"{run_switch[-1]:.4f}",
    "OneAMontyStay": f"{run_stay[-1]:.4f}",
    "OneACardsNgreen": int(green_seen.sum()),
    "OneACardsHiddenGreen": f"{p_hidden_green:.4f}",
})
print(post, p_open3, run_switch[-1], p_hidden_green)
