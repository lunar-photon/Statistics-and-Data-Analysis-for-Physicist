"""b01_hand_examples.py -- the 5x5 landscape worked by hand, checked by the code and by gudhi.

Question: when the water level is lowered over a small grid of heights, which islands and
lakes appear, when do they merge or fill, and does the union-find code find the same pairs
as the hand calculation?
Computes: the islands (H0) and lakes (H1) of the superlevel sets of a 5x5 field, with
lib_persist and with gudhi's CubicalComplex.
Writes: figures/ch12/b_grid_levels.pdf, figures/ch12/b_grid_diagram.pdf
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from common import setup, savefig, SERIES, INK2
from lib_persist import islands, lakes

F = np.array([[0, 0, 0, 0, 0],
              [0, 9, 4, 8, 0],
              [0, 5, 1, 6, 0],
              [0, 3, 7, 2, 0],
              [0, 0, 0, 0, 0]], float)

h0, h1 = islands(F), lakes(F)
print("islands (birth, death):\n", h0)
print("lakes   (birth, death):\n", h1)

try:
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=-F)   # sublevel of -F = superlevel of F
    print("gudhi (dim, (birth, death)) in -F:", cc.persistence())
except ImportError:
    print("gudhi not installed here")

# ---------------------------------------------------------------- panels: land at six levels
setup(7.2, 4.6)
levels = [9, 7, 6, 5, 4, 1]
titles = [r"$\nu=9$: island A born", r"$\nu=7$: island C born", r"$\nu=6$: C joins B, C dies",
          r"$\nu=5$: B joins A, B dies", r"$\nu=4$: a lake is enclosed", r"$\nu=1$: the lake fills"]
fig, axes = plt.subplots(2, 3)
cmap = ListedColormap(["#ffffff", "#c8dcf2"])
for ax, nu, t in zip(axes.flat, levels, titles):
    ax.imshow(F >= nu, cmap=cmap, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{int(F[i, j])}", ha="center", va="center", fontsize=9,
                    color="black" if F[i, j] >= nu else INK2)
    ax.set_xticks(np.arange(-.5, 5, 1))
    ax.set_yticks(np.arange(-.5, 5, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.grid(True, color="#9a9a9a", lw=0.6)
    ax.set_title(t, fontsize=9)
fig.tight_layout()
savefig(fig, "ch12", "b_grid_levels")

# ---------------------------------------------------------------- barcode and diagram
setup(7.0, 2.9)
fig, (a1, a2) = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.2, 1]))
bars = [(9, -0.5, "A", SERIES[0]), (8, 5, "B", SERIES[0]), (7, 6, "C", SERIES[0]),
        (4, 1, "lake", SERIES[1])]
for y, (b, d, lab, c) in enumerate(bars):
    a1.plot([b, d], [y, y], color=c, lw=4, solid_capstyle="butt")
    a1.text(b + 0.25, y, lab, va="center", ha="right", fontsize=9)
a1.annotate("", xy=(-0.6, 0), xytext=(0.2, 0), arrowprops=dict(arrowstyle="->", color=SERIES[0]))
a1.set_xlim(10, -1)
a1.set_ylim(-0.7, 3.7)
a1.set_yticks([])
a1.set_xlabel(r"level $\nu$ (decreasing $\to$)")
a1.set_title("barcode", fontsize=10)
a2.plot([0, 10], [0, 10], color=INK2, lw=0.8, ls=":")
a2.scatter(h0[np.isfinite(h0[:, 1]), 0], h0[np.isfinite(h0[:, 1]), 1], color=SERIES[0], zorder=3,
           label=r"islands ($H_0$)")
a2.scatter([9], [-0.5], color=SERIES[0], marker="^", zorder=3)
a2.scatter(h1[:, 0], h1[:, 1], color=SERIES[1], marker="s", zorder=3, label=r"lakes ($H_1$)")
for (b, d), lab in zip([(8, 5), (7, 6), (9, -0.5), (4, 1)], ["B", "C", r"A ($d=-\infty$)", "lake"]):
    a2.text(b - 0.3, d + 0.35, lab, fontsize=8, ha="right")
a2.set_xlim(0, 10)
a2.set_ylim(-1, 10)
a2.set_xlabel(r"birth level $b$")
a2.set_ylabel(r"death level $d$")
a2.set_title("persistence diagram", fontsize=10)
a2.legend(loc="upper left")
fig.tight_layout()
savefig(fig, "ch12", "b_grid_diagram")
