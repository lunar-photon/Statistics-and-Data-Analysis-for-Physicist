"""Events are boolean masks over simulated outcomes.

Question: if we simulate an experiment many times, how do the set operations on
events (complement, union, intersection) look in code, and do the identities
of set algebra (De Morgan, inclusion-exclusion) hold for the simulated counts?

Computes: N rolls of two fair dice; events A = "first die even" and
B = "sum >= 9" as boolean arrays; checks De Morgan outcome by outcome and
inclusion-exclusion on the counts; compares frequencies with exact values.
Writes: figures/ch01/1a_events_masks.pdf, results/ch01/1a_events_masks.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rng = rng_for("ch01", "1a_events_masks")
N = 100_000

# --- one row per repetition of the experiment: the outcome omega = (d1, d2)
d1 = rng.integers(1, 7, size=N)
d2 = rng.integers(1, 7, size=N)

# --- events are boolean arrays: A[i] is True when outcome i lies in A
A = (d1 % 2 == 0)            # first die even
B = (d1 + d2 >= 9)           # sum at least 9

# --- set operations are logical operations on the masks
A_and_B = A & B              # intersection
A_or_B = A | B               # union
not_A = ~A                   # complement

# De Morgan: (A u B)^c == A^c n B^c must hold outcome by outcome
demorgan_violations = np.count_nonzero(~A_or_B != (~A & ~B))

# inclusion-exclusion on counts: |A u B| = |A| + |B| - |A n B|  (exact identity)
ie_gap = A_or_B.sum() - (A.sum() + B.sum() - A_and_B.sum())

# --- exact probabilities by counting the 36 equally likely outcomes
grid1, grid2 = np.meshgrid(np.arange(1, 7), np.arange(1, 7), indexing="ij")
Aex = (grid1 % 2 == 0)
Bex = (grid1 + grid2 >= 9)
pA, pB, pAB = Aex.mean(), Bex.mean(), (Aex & Bex).mean()
pAorB = (Aex | Bex).mean()

save_numbers("ch01", "1a_events_masks", {
    "OneAMaskN": N,
    "OneAMaskFreqA": A.mean(), "OneAMaskFreqB": B.mean(),
    "OneAMaskFreqAB": A_and_B.mean(), "OneAMaskFreqAorB": A_or_B.mean(),
    "OneAMaskExA": f"{pA:.4f}", "OneAMaskExB": f"{pB:.4f}",
    "OneAMaskExAB": f"{pAB:.4f}", "OneAMaskExAorB": f"{pAorB:.4f}",
    "OneAMaskDeMorgan": int(demorgan_violations), "OneAMaskIEGap": int(ie_gap),
})

# --- figure: left, the 36-point sample space with A and B marked;
#             right, the simulated relative frequency of each outcome
setup(8.0, 3.6)
fig, (ax0, ax1) = plt.subplots(1, 2)
for i in range(1, 7):
    for j in range(1, 7):
        inA, inB = (i % 2 == 0), (i + j >= 9)
        colour = "white"
        if inA and inB:
            colour = SERIES[2]
        elif inA:
            colour = SERIES[0]
        elif inB:
            colour = SERIES[1]
        ax0.add_patch(Rectangle((i - 0.45, j - 0.45), 0.9, 0.9, facecolor=colour,
                                edgecolor="#52514e", lw=0.6, alpha=0.75))
ax0.set_xlim(0.4, 6.6); ax0.set_ylim(0.4, 6.6); ax0.set_aspect("equal")
ax0.set_xticks(range(1, 7)); ax0.set_yticks(range(1, 7))
ax0.set_xlabel("first die $d_1$"); ax0.set_ylabel("second die $d_2$")
ax0.grid(False)
ax0.set_title("sample space: 36 outcomes")
handles = [Rectangle((0, 0), 1, 1, facecolor=SERIES[k], alpha=0.75) for k in (0, 1, 2)]
ax0.legend(handles, [r"$A$ only", r"$B$ only", r"$A\cap B$"], loc="upper left",
           bbox_to_anchor=(1.0, 1.0), fontsize=8)

counts = np.zeros((6, 6))
np.add.at(counts, (d1 - 1, d2 - 1), 1)
im = ax1.imshow((counts / N).T, origin="lower", extent=(0.5, 6.5, 0.5, 6.5),
                cmap="Blues", vmin=0, vmax=2 / 36)
ax1.set_xticks(range(1, 7)); ax1.set_yticks(range(1, 7)); ax1.grid(False)
ax1.set_xlabel("first die $d_1$"); ax1.set_ylabel("second die $d_2$")
ax1.set_title("relative frequency, $N=" + f"{N:,}".replace(",", "{,}") + "$")
ax1.set_aspect("equal")
from mpl_toolkits.axes_grid1 import make_axes_locatable
cax = make_axes_locatable(ax1).append_axes("right", size="4%", pad=0.1)
cb = fig.colorbar(im, cax=cax)
cb.ax.axhline(1 / 36, color="k", ls="--", lw=1)
cb.set_label("fraction of rolls (dashed: 1/36)")
fig.tight_layout()
savefig(fig, "ch01", "1a_events_masks")
print("De Morgan violations:", demorgan_violations, " IE gap:", ie_gap)
