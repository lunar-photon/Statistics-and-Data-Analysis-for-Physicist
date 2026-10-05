"""06_shrinking_sequence.py -- the sample space shrinking evidence by evidence.

Question: a coin is either fair (P(heads) = 0.5) or biased (P(heads) = 0.8),
prior 1/2 each.  We see H, H, T, H.  What survives of the sample space after
each toss, and what does the rescaled survivor (the new prior) look like?

Drawn as boxes (lib_area.box_diagram).
Top row: the ORIGINAL sample space.  After k tosses, the evidence "the first k
tosses" is a box that covers the fraction P(first k tosses | coin) of each
coin's strip, so its pieces have areas P(coin and first k tosses).  The box
only shrinks.
Bottom row: the surviving box rescaled to area 1.  Its strips are as wide as
the posterior after k tosses, which is the prior for toss k+1; the box drawn in
it is the evidence of toss k+1.

Writes: figures/ch05/shrinking_sequence.pdf, results/ch05/06_shrinking_sequence.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers
from lib_area import box_diagram

import numpy as np
import matplotlib.pyplot as plt

setup()

p_heads = np.array([0.8, 0.5])        # biased, fair
prior0 = np.array([0.5, 0.5])
seq = ["H", "H", "T", "H"]
like = lambda t: p_heads if t == "H" else 1 - p_heads

K = len(seq)
fig, axes = plt.subplots(2, K + 1, figsize=(7.6, 3.4))
surv = np.ones(2)                      # P(tosses so far | coin)
post = prior0.copy()
posts = [post.copy()]
names = ["", ""]                      # the strips are named in the first column only
for k in range(K + 1):
    so_far = "".join(seq[:k])
    # ---- top: the original sample space, with the box of the tosses so far ----
    ax = axes[0, k]
    box_diagram(ax, prior0, surv, names, so_far or "nothing yet", fontsize=6.5,
                numbers=False, show_priors=False)
    ax.set_title("start" if k == 0 else "after " + so_far, fontsize=8)
    if k == 0:
        for x, name in [(0.375, "biased"), (1.125, "fair")]:
            ax.text(x, 0.5, name, ha="center", va="center", fontsize=7)
    ax.text(0.75, -0.06, f"area {np.dot(prior0, surv):.3f}", ha="center", va="top", fontsize=7)
    # ---- bottom: the survivor rescaled to area 1 = the prior for the next toss ----
    ax = axes[1, k]
    nxt = like(seq[k]) if k < K else np.zeros(2)
    box_diagram(ax, post, nxt, names, f"next: {seq[k]}" if k < K else "", fontsize=6.5,
                numbers=False, show_priors=False)
    ax.set_title(f"P(biased) = {post[0]:.3f}", fontsize=8)
    if k == 0:
        for x, name in [(0.375, "biased"), (1.125, "fair")]:
            ax.text(x, 0.5, name, ha="center", va="center", fontsize=7)
    if k < K:
        surv = surv * like(seq[k])
        post = post * like(seq[k]); post /= post.sum()
        posts.append(post.copy())
fig.subplots_adjust(wspace=0.08, hspace=0.35)
savefig(fig, "ch05", "shrinking_sequence")

save_numbers("ch05", "06_shrinking_sequence", {
    "FiveASeqPa": posts[1][0], "FiveASeqPb": posts[2][0],
    "FiveASeqPc": posts[3][0], "FiveASeqPd": posts[4][0],
    "FiveASeqArea": float(np.dot(prior0, surv)),
})
