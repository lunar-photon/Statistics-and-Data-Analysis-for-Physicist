"""Particle identification: inverting the tree with Bayes' theorem.

Question: a beam contains pions, kaons and protons with known fractions. A
Cherenkov/RICH detector tags a track as "kaon" with a probability that depends
on the true species. Given a kaon tag, what is the probability that the track
really is a kaon (the purity of the tagged sample)? And how does that purity
depend on how rare kaons are in the beam (the base-rate effect)?

Computes: exact purity from Bayes' theorem; a simulation in which we generate
tracks, tag them, keep only the tagged ones and count the true species; the
purity as a function of the kaon fraction (exact curve plus simulation points).
Writes: figures/ch01/1a_particle_id.pdf, results/ch01/1a_particle_id.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt

species = np.array(["pi", "K", "p"])
prior = np.array([0.80, 0.15, 0.05])          # beam composition P(species)
p_tag = np.array([0.05, 0.90, 0.10])          # P(tagged as K | species)

# --- exact, by the law of total probability and Bayes' theorem
joint = prior * p_tag                          # the three tree paths ending in "tag K"
p_tagK = joint.sum()
posterior = joint / p_tagK                     # P(species | tag K)

# --- simulation: conditioning = selecting the tagged tracks
rng = rng_for("ch01", "1a_particle_id")
N = 1_000_000
true = rng.choice(3, size=N, p=prior)
tagged = rng.random(N) < p_tag[true]
mc_post = np.bincount(true[tagged], minlength=3) / tagged.sum()


def purity(fK, eff=0.90, mis_pi=0.05, mis_p=0.10, fp=0.05):
    """Purity of the K-tagged sample when the kaon fraction is fK (proton fraction fixed)."""
    fpi = 1.0 - fK - fp
    return eff * fK / (eff * fK + mis_pi * fpi + mis_p * fp)


fK_grid = np.linspace(0.001, 0.5, 400)
fK_pts = np.array([0.01, 0.03, 0.1, 0.2, 0.4])
mc_pts = []
for fK in fK_pts:
    pr = np.array([1 - fK - 0.05, fK, 0.05])
    t = rng.choice(3, size=200_000, p=pr)
    tg = rng.random(t.size) < p_tag[t]
    mc_pts.append(np.mean(t[tg] == 1))

setup(6.0, 3.6)
fig, ax = plt.subplots()
theory_line(ax, fK_grid, purity(fK_grid), label="Bayes' theorem")
ax.plot(fK_pts, mc_pts, "o", color=SERIES[1], label="simulation: fraction of tagged that are K")
ax.axhline(0.90, color=SERIES[0], lw=1.0, label=r"efficiency $P(\mathrm{tag}\,K\mid K)=0.90$")
ax.set_xscale("log")
ax.set_xlabel(r"kaon fraction in the beam $P(K)$")
ax.set_ylabel(r"purity $P(K\mid \mathrm{tag}\,K)$")
ax.set_ylim(0, 1)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
savefig(fig, "ch01", "1a_particle_id")

save_numbers("ch01", "1a_particle_id", {
    "OneAPidTagK": f"{p_tagK:.4f}",
    "OneAPidPostK": f"{posterior[1]:.4f}", "OneAPidPostPi": f"{posterior[0]:.4f}",
    "OneAPidPostP": f"{posterior[2]:.4f}",
    "OneAPidMCK": f"{mc_post[1]:.4f}", "OneAPidMCPi": f"{mc_post[0]:.4f}",
    "OneAPidMCP": f"{mc_post[2]:.4f}",
    "OneAPidNtag": int(tagged.sum()),
    "OneAPidPurityOnePct": f"{purity(0.01):.3f}",
})
print(p_tagK, posterior, mc_post, purity(0.01))
