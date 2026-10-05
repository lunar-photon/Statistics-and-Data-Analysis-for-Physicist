"""02_credibility.py -- Bayesian inference as reallocation of credibility.

Question: a fixed amount of credibility (total 1) is spread over a list of
possibilities.  How does an observation move it around?

Computes:
  (a) which trigger subsystem fired a recorded event (muon, calorimeter, track):
      prior -> after "an isolated high-pT muon was reconstructed" -> after
      "the muon trigger was masked in this run" (a likelihood of zero);
  (b) four candidate ball sizes 1,2,3,4 and three noisy measured diameters
      1.77, 2.23, 2.70 with Gaussian spread 1.17 (noisy data only nudge credibility).

Writes: figures/ch05/credibility_trigger.pdf, figures/ch05/credibility_balls.pdf,
        results/ch05/02_credibility.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES, INK2

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup()
plt.rcParams["axes.axisbelow"] = True


def normalise(w):
    return w / w.sum()


# ---------------- (a) which subsystem fired? ----------------
names = ["muon", "calo", "track"]
prior = np.array([0.25, 0.60, 0.15])        # share of L1 accepts from each trigger path
like_mu = np.array([0.70, 0.05, 0.20])      # P(isolated high-pT muon offline | path)
post1 = normalise(prior * like_mu)
like_mask = np.array([0.0, 1.0, 1.0])       # P(run log says 'muon trigger masked' | path)
post2 = normalise(post1 * like_mask)

fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.3), sharey=True)
panels = [(prior, "prior", SERIES[0]), (like_mu, r"$\times$ P(muon | path)", INK2),
          (post1, "posterior 1", SERIES[0]), (like_mask, r"$\times$ P(masked | path)", INK2),
          (post2, "posterior 2", SERIES[0])]
for ax, (vals, title, col) in zip(axes, panels):
    ax.bar(range(3), vals, color=col, width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 1.15)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
axes[0].set_ylabel("credibility / likelihood")
fig.tight_layout()
savefig(fig, "ch05", "credibility_trigger")

# ---------------- (b) noisy data: which ball size? ----------------
sizes = np.arange(1, 5)
d = np.array([1.77, 2.23, 2.70])
SIG = 1.17
b_prior = np.full(4, 0.25)
b_like = np.prod(stats.norm.pdf(d[:, None], sizes[None, :], SIG), axis=0)
b_post = normalise(b_prior * b_like)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 4.0), sharex=True)
x = np.linspace(-1.5, 6.5, 400)
a1.bar(sizes, b_prior, width=0.12, color=SERIES[0], label="prior credibility")
for s in sizes:
    a1.plot(x, stats.norm.pdf(x, s, SIG), color=INK2, lw=0.9)
a1.set_ylim(0, 1.0)
a1.set_ylabel("prior")
a1.legend(loc="upper right")
a2.bar(sizes, b_post, width=0.12, color=SERIES[0], label="posterior credibility")
a2.plot(d, np.zeros_like(d), "o", color=SERIES[1], ms=6, clip_on=False, label="measured diameters")
for s, v in zip(sizes, b_post):
    a2.text(s, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
a2.set_ylim(0, 1.0)
a2.set_ylabel("posterior")
a2.set_xlabel("ball size (bars) and measured diameter (curves, dots)")
a2.legend(loc="upper right")
fig.tight_layout()
savefig(fig, "ch05", "credibility_balls")

save_numbers("ch05", "02_credibility", {
    "FiveABallPa": b_post[0], "FiveABallPb": b_post[1],
    "FiveABallPc": b_post[2], "FiveABallPd": b_post[3],
    "FiveABallLRbc": b_like[1] / b_like[2],
})
