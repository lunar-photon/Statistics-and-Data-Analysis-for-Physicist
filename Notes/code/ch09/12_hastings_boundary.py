"""Asymmetric proposals need the Hastings factor: a rate parameter that must stay positive.

Question: a Poisson process gives n = 2 events in unit exposure; with a flat prior the posterior
of the rate is f(lam) ∝ lam^2 e^{-lam} on lam > 0 (a Gamma(3, 1) density, mean 3).  Five
samplers try to draw from it:
  A  Gaussian random walk, a proposal below zero is rejected (the chain stays)    -- correct
  B  Gaussian random walk, redraw until the proposal is positive, no correction   -- biased
  C  as B, with the Hastings factor Phi(x/s)/Phi(y/s) of the truncated proposal   -- correct
  D  multiplicative (log-normal) proposal y = x e^{sZ}, Hastings factor y/x      -- correct
  E  as D without the factor: samples lam^1 e^{-lam}, a Gamma(2, 1)              -- biased
Writes figures/ch09/hastings_boundary.pdf and results/ch09/12_hastings_boundary.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_mcmc import mh

K = 3.0                                                     # Gamma shape: n + 1 with n = 2 events


def logp(x):
    lam = x[0]
    return (K - 1) * np.log(lam) - lam if lam > 0 else -np.inf


s = 2.0
gauss = lambda x, rng: x + s * rng.standard_normal(1)


def redraw(x, rng):
    """Gaussian step, repeated until it lands on lam > 0 (a truncated Gaussian proposal)."""
    while True:
        y = x + s * rng.standard_normal(1)
        if y[0] > 0:
            return y


lognorm = lambda x, rng: x * np.exp(s * 0.5 * rng.standard_normal(1))   # width s/2 in ln(lam)
zero_q = lambda a, b: 0.0                                               # pretend q is symmetric
trunc_q = lambda a, b: -stats.norm.logcdf(a[0] / s)                     # log q(b|a) up to b-terms that cancel
logn_q = lambda a, b: -np.log(b[0])                                     # log-normal: q(b|a) ∝ 1/b


rng = rng_for("ch09", "12_hastings_boundary")
n = 200_000
runs = {
    "A": mh(logp, [3.0], n, gauss, zero_q, rng),
    "B": mh(logp, [3.0], n, redraw, zero_q, rng),
    "C": mh(logp, [3.0], n, redraw, trunc_q, rng),
    "D": mh(logp, [3.0], n, lognorm, logn_q, rng),
    "E": mh(logp, [3.0], n, lognorm, zero_q, rng),
}
burn = 2000
draws = {k: v[0][burn:, 0] for k, v in runs.items()}
exact_low = stats.gamma(K).cdf(0.5)
nums = {"NineBHbN": f"{n:,}".replace(",", r"\,"), "NineBHbStep": s,
        "NineBHbLowExact": exact_low}
for k, d in draws.items():
    nums[f"NineBHbMean{k}"] = d.mean()
    nums[f"NineBHbLow{k}"] = np.mean(d < 0.5)
    nums[f"NineBHbAcc{k}"] = runs[k][2]
save_numbers("ch09", "12_hastings_boundary", nums)

setup(7.0, 2.9)
fig, (a1, a2) = plt.subplots(1, 2, sharey=True)
xx = np.linspace(1e-3, 10, 500)
for ax, keys, title in ((a1, "ABC", "(a) Gaussian steps near the wall at 0"),
                        (a2, "DE", "(b) multiplicative steps")):
    for k, c in zip(keys, SERIES):
        lab = {"A": "A: stay if $y<0$", "B": "B: redraw, no factor", "C": "C: redraw, with factor",
               "D": "D: with factor $y/x$", "E": "E: no factor"}[k]
        ax.hist(draws[k], bins=100, range=(0, 10), density=True, histtype="step", lw=1.3,
                color=c, label=lab)
    theory_line(ax, xx, stats.gamma(K).pdf(xx), label=r"$\lambda^2e^{-\lambda}/2$")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel(r"rate $\lambda$")
    ax.legend(fontsize=7)
a1.set_ylabel("density")
fig.tight_layout()
savefig(fig, "ch09", "hastings_boundary")
