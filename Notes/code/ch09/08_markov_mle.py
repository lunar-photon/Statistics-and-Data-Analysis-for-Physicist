"""Reading the transition matrix off one observed history (maximum likelihood).

Question: given n consecutive days of sunny/cloudy/rainy weather, what is our best estimate of
the transition matrix, and how uncertain is it?  The MLE is p_hat_ij = n_ij / n_i (counts of
i -> j transitions over visits to i), with standard error sqrt(p_ij (1 - p_ij) / n_i).
We check the estimate and its error bar on repeated simulated histories.
Writes figures/ch09/markov_mle.pdf and results/ch09/08_markov_mle.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_markov import walk

P = np.array([[0.6, 0.2, 0.2], [0.3, 0.7, 0.0], [0.6, 0.0, 0.4]])
rng = rng_for("ch09", "08_markov_mle")
n, REP = 1000, 2000


def mle(x):
    counts = np.zeros((3, 3))
    np.add.at(counts, (x[:-1], x[1:]), 1)
    ni = counts.sum(axis=1)
    return counts / np.maximum(ni, 1)[:, None], ni


est, se, zs = [], [], []
for r in range(REP):
    x = walk(P, 0, n, rng)
    Ph, ni = mle(x)
    est.append(Ph)
    s = np.sqrt(Ph * (1 - Ph) / np.maximum(ni, 1)[:, None])
    se.append(s)
    zs.append((Ph[2, 0] - P[2, 0]) / np.sqrt(P[2, 0] * (1 - P[2, 0]) / ni[2]))
est, se, zs = np.array(est), np.array(se), np.array(zs)
first, first_se = est[0], se[0]
_, ni0 = mle(walk(P, 0, n, rng_for("ch09", "08_markov_mle")))

setup(6.0, 3.0)
fig, ax = plt.subplots()
ax.hist(zs, bins=np.linspace(-4, 4, 41), density=True, color=SERIES[0], alpha=0.6,
        label=f"{REP} histories")
zz = np.linspace(-4, 4, 300)
theory_line(ax, zz, np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi), label=r"$\mathcal{N}(0,1)$")
ax.set_xlabel("standardised error")
ax.set_ylabel("density")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
savefig(fig, "ch09", "markov_mle")

vals = {"NineAMLEn": n, "NineAMLERep": REP,
        "NineAMLEzSd": round(float(zs.std(ddof=1)), 3), "NineAMLEzMean": round(float(zs.mean()), 3)}
names = "SCR"
for i in range(3):
    for j in range(3):
        vals[f"NineAMLE{names[i]}{names[j]}"] = round(first[i, j], 3)
        vals[f"NineAMLEse{names[i]}{names[j]}"] = round(first_se[i, j], 3)
for i in range(3):
    vals[f"NineAMLEn{names[i]}"] = int(ni0[i])
save_numbers("ch09", "08_markov_mle", vals)
print(first.round(3)); print(first_se.round(3)); print(ni0, zs.mean(), zs.std())
