"""Maximum entropy: the least committal distribution that respects what we know.

Question answered
    (1) Sivia's die: we are told only that the average roll is 4.5.  Which
        probabilities p_1..p_6 should we assign?
    (2) Among all distributions with the same variance, which one has the
        largest entropy?  (Answer: the Gaussian.)
    (3) Sivia's kangaroo problem: which variational function picks the
        "independent" answer x = 1/9?

What it computes
    * The MaxEnt die p_i = exp(lam*i)/Z with lam solved from sum i p_i = 4.5,
      and the entropies of many random die distributions obeying the same
      constraint (all must be below the MaxEnt value).
    * Differential entropies of unit-variance uniform, triangular, Laplace,
      logistic and Gaussian densities.
    * The maximisers of four candidate functions for the kangaroo table.

What it writes
    figures/ch02/maxent_die.pdf, results/ch02/23_maxent.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats

setup()
rng = rng_for("ch02", "23_maxent")

# ---------------------------------------------------------------- the die
faces = np.arange(1, 7)
target = 4.5


def die_mean(lam):
    w = np.exp(lam * faces)
    return np.sum(faces * w) / np.sum(w)


lam = optimize.brentq(lambda l: die_mean(l) - target, -5, 5)
p_me = np.exp(lam * faces)
p_me /= p_me.sum()
S_me = -np.sum(p_me * np.log(p_me))

# random distributions on the 6-simplex with mean 4.5 (accept within 0.005)
cand = rng.dirichlet(np.ones(6), size=4_000_000)
keep = cand[np.abs(cand @ faces - target) < 0.005]
S_rand = -np.sum(np.where(keep > 0, keep * np.log(keep), 0.0), axis=1)


def maxent_entropy(m):
    """Entropy of the MaxEnt die whose mean is exactly m."""
    l = optimize.brentq(lambda l: die_mean(l) - m, -5, 5)
    q = np.exp(l * faces); q /= q.sum()
    return -np.sum(q * np.log(q))


# compare every candidate with the MaxEnt die of *its own* mean (the window has width 0.01)
deficit = np.array([maxent_entropy(m) for m in keep @ faces]) - S_rand

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
ax = axes[0]
ax.bar(faces, p_me, color=SERIES[0], width=0.7, label="MaxEnt, mean 4.5")
ax.plot(faces, np.full(6, 1 / 6), "k--", lw=1.2, label="uniform, mean 3.5")
ax.set_xlabel("number of dots $i$")
ax.set_ylabel("$p_i$")
ax.set_title("(a) Sivia's die")
ax.legend(loc="upper left")
ax = axes[1]
ax.hist(S_rand, bins=60, color=SERIES[1], alpha=0.8,
        label=f"{len(keep)} random dice with mean 4.5")
ax.axvline(S_me, color="k", ls="--", lw=1.4, label="MaxEnt value")
ax.set_xlabel(r"entropy $S=-\sum_i p_i\ln p_i$")
ax.set_ylabel("number of candidates")
ax.set_title("(b) random dice with mean 4.5")
ax.legend(loc="upper left", fontsize=8)
savefig(fig, "ch02", "maxent_die")

# ---------------------------------------------------------------- unit-variance entropies
dists = {
    "uniform": stats.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
    "triangular": stats.triang(c=0.5, loc=-np.sqrt(6), scale=2 * np.sqrt(6)),
    "Laplace": stats.laplace(scale=1 / np.sqrt(2)),
    "logistic": stats.logistic(scale=np.sqrt(3) / np.pi),
    "Gaussian": stats.norm(),
}
ent = {k: float(d.entropy()) for k, d in dists.items()}
for k, d in dists.items():
    assert abs(d.var() - 1) < 1e-9, k

# ---------------------------------------------------------------- kangaroos
# p1 = x (blue & left), p2 = p3 = 1/3 - x, p4 = 1/3 + x, with 0 <= x <= 1/3
def table(x):
    return np.array([x, 1 / 3 - x, 1 / 3 - x, 1 / 3 + x])


funcs = {
    "entropy": lambda p: -np.sum(p * np.log(p)),
    "minus sum p^2": lambda p: -np.sum(p ** 2),
    "sum log p": lambda p: np.sum(np.log(p)),
    "sum sqrt p": lambda p: np.sum(np.sqrt(p)),
}
xopt = {}
for k, f in funcs.items():
    r = optimize.minimize_scalar(lambda x: -f(table(x)), bounds=(1e-9, 1 / 3 - 1e-9),
                                 method="bounded", options={"xatol": 1e-10})
    xopt[k] = r.x

save_numbers("ch02", "23_maxent", {
    "tbMElam": f"{lam:.4f}",
    "tbMEpOne": f"{p_me[0]:.4f}", "tbMEpTwo": f"{p_me[1]:.4f}", "tbMEpThree": f"{p_me[2]:.4f}",
    "tbMEpFour": f"{p_me[3]:.4f}", "tbMEpFive": f"{p_me[4]:.4f}", "tbMEpSix": f"{p_me[5]:.4f}",
    "tbMESmax": f"{S_me:.4f}",
    "tbMESrandMax": f"{S_rand.max():.4f}",
    "tbMESrandMaxMean": f"{(keep @ faces)[np.argmax(S_rand)]:.4f}",   # mean of that candidate
    "tbMEminDeficit": f"{deficit.min():.1e}".replace("e-0", r"\times10^{-") + "}",
    "tbMEnKeep": int(len(keep)),
    "tbMESuniformDie": f"{np.log(6):.4f}",
    "tbEntUniform": f"{ent['uniform']:.4f}",
    "tbEntTriang": f"{ent['triangular']:.4f}",
    "tbEntLaplace": f"{ent['Laplace']:.4f}",
    "tbEntLogistic": f"{ent['logistic']:.4f}",
    "tbEntGauss": f"{ent['Gaussian']:.4f}",
    "tbKangEnt": f"{xopt['entropy']:.4f}",
    "tbKangSq": f"{xopt['minus sum p^2']:.4f}",
    "tbKangLog": f"{xopt['sum log p']:.4f}",
    "tbKangSqrt": f"{xopt['sum sqrt p']:.4f}",
})
