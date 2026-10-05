"""21_coin_factories.py -- which factory minted this coin?  Evidence and Bayes factors.

Question: a coin shows z = 6 heads in N = 9 tosses.  It came from one of two
factories: a tail-biased one whose coins have bias theta ~ Beta(3.5, 8.5) and a
head-biased one with theta ~ Beta(8.5, 3.5).  How probable is each factory?
Each factory is a *model*: a likelihood (Bernoulli tosses) together with a prior
on theta.  Its evidence p(D | m) is the likelihood averaged over that prior.

Computes
  * the exact evidences B(z+a, N-z+b) / B(a, b) and a grid check (1001 thetas),
  * the Bayes factor and the posterior factory probabilities for 50/50 priors,
  * the model-averaged posterior mean of theta,
  * the Occam example: "must be fair" Beta(500,500) against "anything goes"
    Beta(1,1), for every z in N = 20 tosses.

Writes: figures/ch05/factories.pdf, figures/ch05/fair_vs_anything.pdf,
        results/ch05/21_coin_factories.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import betaln

setup()


def log_evidence(z, N, a, b):
    """ln p(D|m) for a particular sequence of z heads in N tosses, prior Beta(a, b)."""
    return betaln(z + a, N - z + b) - betaln(a, b)


z, N = 6, 9
facs = {"tail-biased factory": (3.5, 8.5), "head-biased factory": (8.5, 3.5)}
lnZ = {k: log_evidence(z, N, *ab) for k, ab in facs.items()}
Z1, Z2 = np.exp(list(lnZ.values()))
BF12 = Z1 / Z2
post1 = BF12 / (1 + BF12)                       # 50/50 prior odds

# model averaging: posterior of theta within each factory, weighted by p(m | D)
means = [(z + a) / (N + a + b) for a, b in facs.values()]
mean_avg = post1 * means[0] + (1 - post1) * means[1]

# grid check: the evidence is the area under likelihood x prior
theta = np.linspace(0, 1, 1001)
like = theta**z * (1 - theta)**(N - z)          # probability of the observed sequence
Zgrid = [np.trapezoid(like * stats.beta.pdf(theta, *ab), theta) for ab in facs.values()]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, (name, ab), Zg, c in zip(axes, facs.items(), Zgrid, SERIES):
    prior = stats.beta.pdf(theta, *ab)
    prod = like * prior
    ax.plot(theta, prior, color=c, lw=1.4, label=r"prior $p(\theta\,|\,m)$")
    ax.plot(theta, like / like.max() * 3.0, color="k", ls="--", lw=1.1,
            label="likelihood (rescaled)")
    ax.fill_between(theta, 0, prod / 0.0005 * 0.5, color=c, alpha=0.35,
                    label=r"likelihood$\times$prior ($\times 10^3$)")
    ax.set_title(f"{name}:  $p(D\\,|\\,m)={Zg:.6f}$", fontsize=9)
    ax.set_xlabel(r"coin bias $\theta$")
axes[0].set_ylabel("density")
axes[0].legend(fontsize=7, loc="upper right")
savefig(fig, "ch05", "factories")

# Occam: must-be-fair against anything-goes, N = 20
N2 = 20
zz = np.arange(N2 + 1)
lnB = log_evidence(zz, N2, 500, 500) - log_evidence(zz, N2, 1, 1)
fig, ax = plt.subplots(figsize=(5.8, 2.8))
ax.bar(zz, lnB, color=[SERIES[0] if v > 0 else SERIES[1] for v in lnB])
ax.axhline(0, color="k", lw=0.7)
ax.set_xlabel(r"number of heads $z$ in $N=20$ tosses")
ax.set_ylabel(r"$\ln B$ (fair vs anything)")
ax.set_xticks(zz[::2])
savefig(fig, "ch05", "fair_vs_anything")

save_numbers("ch05", "21_coin_factories", {
    "FiveCFacZone": Z1, "FiveCFacZtwo": Z2,
    "FiveCFacZgridOne": Zgrid[0], "FiveCFacZgridTwo": Zgrid[1],
    "FiveCFacBF": BF12, "FiveCFacInvBF": 1 / BF12,
    "FiveCFacPostOne": 100 * post1, "FiveCFacPostTwo": 100 * (1 - post1),
    "FiveCFairFifteen": np.exp(lnB[15]), "FiveCFairEleven": np.exp(lnB[11]),
    "FiveCFairTen": np.exp(lnB[10]),
    "FiveCFacMeanOne": means[0], "FiveCFacMeanTwo": means[1], "FiveCFacMeanAvg": mean_avg,
    "FiveCFairZmin": int(zz[lnB > 0].min()), "FiveCFairZmax": int(zz[lnB > 0].max()),
})
