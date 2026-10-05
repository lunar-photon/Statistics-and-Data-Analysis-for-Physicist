"""Error bars on error bars: the wrong answer to "how do I propagate the error?".

Question: what does it look like to *not* propagate uncertainty, and instead decorate every
error bar with error bars of its own? Each point has an error bar; each end of that bar gets a
smaller error bar, and so on for three levels: the picture never converges to a number.
Writes: figures/ch03/error_bars_on_error_bars.pdf
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, INK

import matplotlib.pyplot as plt

setup(5.0, 3.4)

x = [1.0, 2.0, 3.0, 4.0]
y = [3.0, 3.8, 4.9, 4.9]
err = [1.2, 1.6, 1.3, 1.0]
CAP = 0.12


def bar(ax, xc, yc, e, level, dx):
    """Draw an error bar of half-length e at (xc, yc); recurse onto both of its ends."""
    lw = 1.6 * 0.7 ** level
    ax.plot([xc, xc], [yc - e, yc + e], color=INK, lw=lw, solid_capstyle="round")
    cap = CAP * 0.65 ** level
    for yy in (yc - e, yc + e):
        ax.plot([xc - cap, xc + cap], [yy, yy], color=INK, lw=lw)
        if level < 3:
            bar(ax, xc + dx, yy, e * 0.32, level + 1, dx * 0.55)


with plt.xkcd(scale=0.6, length=120, randomness=2):
    fig, ax = plt.subplots()
    ax.plot(x, y, color=INK, lw=1.6, zorder=3)
    ax.plot(x, y, "o", color=INK, ms=7, zorder=4)
    for xi, yi, ei in zip(x, y, err):
        bar(ax, xi, yi, ei, 0, 0.14)
    ax.set_xlim(0.4, 4.8); ax.set_ylim(0.6, 7.2)
    ax.set_xticks([1, 2, 3, 4]); ax.set_yticks([])
    ax.set_xticklabels([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    savefig(fig, "ch03", "error_bars_on_error_bars")
