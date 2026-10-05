"""05_coin_grid.py -- the bias of a coin, updated toss by toss on a grid.

Question: a coin has an unknown probability theta of heads.  We keep a grid of
candidate values theta_j and a credibility for each.  How does the credibility
move as the tosses arrive one at a time?  Does the order of the tosses matter?

Computes:
  * toss-by-toss posteriors on a 101-point grid for a coin with theta = 0.3,
    from a flat prior and from a triangular prior peaked at 0.5 (panels);
  * order invariance: the posterior after the 50 tosses, updated in 1000 random
    orders and in one batch, largest difference;
  * the triangular-prior grid examples z=1 of N=4 and z=10 of N=40 (posterior modes);
  * fair (0.5) versus biased (0.7): log10 posterior odds along 20 simulated toss
    sequences from each coin, and the mean drift per toss (a Kullback-Leibler divergence).

Writes: figures/ch05/coin_grid_panels.pdf, figures/ch05/coin_logodds.pdf,
        results/ch05/05_coin_grid.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, theory_line

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch05", "05_coin_grid")
setup()

theta = np.linspace(0, 1, 101)
flat = np.full_like(theta, 1 / theta.size)
tri = np.minimum(theta, 1 - theta); tri /= tri.sum()      # triangular prior, zero at 0 and 1


def update(prior, y):
    """One toss y (1 = heads, 0 = tails): multiply by the Bernoulli likelihood, renormalise."""
    post = prior * (theta if y == 1 else 1 - theta)
    return post / post.sum()


THETA_TRUE, N = 0.3, 50
tosses = (rng.uniform(size=N) < THETA_TRUE).astype(int)

show = [0, 1, 2, 3, 5, 10, 20, 50]
fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.6), sharex=True)
pf, pt = flat.copy(), tri.copy()
k = 0
for n in range(N + 1):
    if n > 0:
        pf, pt = update(pf, tosses[n - 1]), update(pt, tosses[n - 1])
    if n in show:
        ax = axes.flat[k]; k += 1
        ax.plot(theta, pf, color=SERIES[0], label="flat prior")
        ax.plot(theta, pt, color=SERIES[1], ls="--", label="triangular prior")
        ax.axvline(THETA_TRUE, color=INK2, lw=0.6, ls=":")
        ax.set_title(f"N={n}, heads={tosses[:n].sum()}", fontsize=8)
        ax.set_yticks([])
axes.flat[0].legend(fontsize=6, loc="upper left")
for ax in axes[1]:
    ax.set_xlabel(r"$\theta$")
fig.tight_layout()
savefig(fig, "ch05", "coin_grid_panels")

# ---------- order invariance ----------
batch = flat * theta ** tosses.sum() * (1 - theta) ** (N - tosses.sum())
batch /= batch.sum()
maxdiff = 0.0
for _ in range(1000):
    p = flat.copy()
    for y in rng.permutation(tosses):
        p = update(p, y)
    maxdiff = max(maxdiff, np.abs(p - batch).max())
post_mean = np.sum(theta * pf)

# ---------- the triangular-prior examples (fine grid of 1001 values) ----------
th2 = np.linspace(0, 1, 1001)
tri2 = np.minimum(th2, 1 - th2); tri2 /= tri2.sum()
def mode(z, n):
    p = tri2 * th2 ** z * (1 - th2) ** (n - z)
    return th2[np.argmax(p)]
mode4, mode40 = mode(1, 4), mode(10, 40)

# ---------- fair versus biased: log odds along toss sequences ----------
T0, T1, M, NT = 0.5, 0.7, 20, 200
lr_h, lr_t = np.log10(T1 / T0), np.log10((1 - T1) / (1 - T0))   # log10 evidence per head / tail
fig, ax = plt.subplots(figsize=(5.8, 3.2))
for true, col in [(T1, SERIES[0]), (T0, SERIES[1])]:
    y = rng.uniform(size=(M, NT)) < true
    lo = np.cumsum(np.where(y, lr_h, lr_t), axis=1)                  # prior odds 1: log10 = 0
    lo = np.hstack([np.zeros((M, 1)), lo])
    for row in lo:
        ax.plot(np.arange(NT + 1), row, color=col, lw=0.6, alpha=0.6)
nn = np.arange(NT + 1)
kl1 = T1 * np.log10(T1 / T0) + (1 - T1) * np.log10((1 - T1) / (1 - T0))
kl0 = T0 * np.log10(T0 / T1) + (1 - T0) * np.log10((1 - T0) / (1 - T1))
theory_line(ax, nn, kl1 * nn, label="mean drift")
ax.plot(nn, -kl0 * nn, color="k", ls="--", lw=1.4)
ax.axhline(2, color=INK2, lw=0.6, ls=":"); ax.axhline(-2, color=INK2, lw=0.6, ls=":")
ax.set_xlabel("number of tosses"); ax.set_ylabel(r"$\log_{10}$ posterior odds, biased : fair")
ax.plot([], [], color=SERIES[0], label="tosses of the biased coin")
ax.plot([], [], color=SERIES[1], label="tosses of the fair coin")
ax.legend(fontsize=7, loc="upper left")
savefig(fig, "ch05", "coin_logodds")

save_numbers("ch05", "05_coin_grid", {
    "FiveACoinHeads": int(tosses.sum()), "FiveACoinN": N,
    "FiveACoinMaxDiff": maxdiff, "FiveACoinPostMean": post_mean,
    "FiveACoinModeFour": mode4, "FiveACoinModeForty": mode40,
    "FiveACoinLRh": lr_h, "FiveACoinLRt": lr_t,
    "FiveACoinKLone": kl1, "FiveACoinKLzero": kl0,
    "FiveACoinTossesHundred": 2 / kl1,
})
