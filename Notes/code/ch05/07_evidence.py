"""07_evidence.py -- the denominator P(E) as a prediction of the data.

Question: before the tosses, what did each belief about the coin predict for
the number of heads z in N = 10 tosses?  The denominator of Bayes' theorem,
P(z) = sum_j P(z | theta_j) P(theta_j), is exactly that prediction (the prior
predictive distribution).  It sums to one over all possible data sets, and its
value at the observed z says how well the whole model anticipated the data.

Computes P(z) for three priors on a 101-point grid of theta: flat, triangular
peaked at 1/2, and 'the coin is fair' (all weight at theta = 0.5); checks the
sums; evaluates them at the observed z = 9.

Writes: figures/ch05/prior_predictive.pdf, results/ch05/07_evidence.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup()
plt.rcParams["axes.axisbelow"] = True

N, Z_OBS = 10, 9
theta = np.linspace(0, 1, 101)
priors = {
    "flat": np.full_like(theta, 1 / theta.size),
    "triangular": (lambda t: t / t.sum())(np.minimum(theta, 1 - theta)),
    "fair only": (theta == 0.5).astype(float),
}
z = np.arange(N + 1)
like = stats.binom.pmf(z[:, None], N, theta[None, :])      # P(z | theta_j), rows z, columns j
pred = {k: like @ p for k, p in priors.items()}           # sum over paths theta_j

fig, ax = plt.subplots(figsize=(5.8, 3.0))
w = 0.27
for i, (k, p) in enumerate(pred.items()):
    ax.bar(z + (i - 1) * w, p, width=w, color=SERIES[i], label=k)
ax.axvline(Z_OBS, color="k", ls=":", lw=0.8)
ax.set_xlabel("number of heads $z$ in 10 tosses"); ax.set_ylabel("$P(z)$ before the tosses")
ax.set_xticks(z); ax.legend(fontsize=8)
savefig(fig, "ch05", "prior_predictive")

save_numbers("ch05", "07_evidence", {
    "FiveAEvFlat": pred["flat"][Z_OBS], "FiveAEvTri": pred["triangular"][Z_OBS],
    "FiveAEvFair": pred["fair only"][Z_OBS],
    "FiveAEvSumFlat": pred["flat"].sum(), "FiveAEvSumTri": pred["triangular"].sum(),
    "FiveAEvRatio": pred["flat"][Z_OBS] / pred["fair only"][Z_OBS],
})
