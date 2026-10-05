"""14_priors.py -- "flat" depends on the parametrisation; Jeffreys' prior does not.

Question: a flat prior is meant to say "I know nothing".  If we know nothing about p we
also know nothing about logit(p) or p^2 -- but a prior cannot be flat in all of them.
How big is the effect, and what does it do to a posterior from few data?

Computes:
  * 10^5 draws p ~ U(0,1) transformed to psi = ln(p/(1-p)): histogram against the
    density e^psi/(1+e^psi)^2 (Wasserman 11.6);
  * a positive scale parameter lambda in [0.01, 100]: the prior mass per decade for a
    prior flat in lambda and for one flat in ln lambda (Jeffreys 1/lambda);
  * 1 success in 10 trials: posterior mean and 95% upper limit for theta under the
    flat Beta(1,1), Jeffreys Beta(1/2,1/2), and the flat-in-logit (Haldane) Beta(0,0)
    priors; the same for 100 in 1000.

Writes: figures/ch05/priors_param.pdf, results/ch05/14_priors.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch05", "14_priors")
setup()
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.7))

# ---------------------------------------------------------------- flat p seen in logit
p = rng.uniform(size=100000)
psi = np.log(p / (1 - p))
ax[0].hist(psi, bins=120, range=(-8, 8), density=True, color=SERIES[0], alpha=0.6)
s = np.linspace(-8, 8, 400)
theory_line(ax[0], s, np.exp(s) / (1 + np.exp(s)) ** 2, label=r"$e^\psi/(1+e^\psi)^2$")
ax[0].set_xlabel(r"$\psi=\ln\frac{p}{1-p}$"); ax[0].set_title(r"$p\sim U(0,1)$, seen in $\psi$", fontsize=8)
ax[0].legend(fontsize=6); ax[0].set_yticks([])
frac_central = np.mean(np.abs(psi) < 2)

# ---------------------------------------------------------------- scale parameter: mass per decade
edges = 10.0 ** np.arange(-2, 3)          # 0.01, 0.1, 1, 10, 100
lo, hi = edges[0], edges[-1]
flat = np.diff(edges) / (hi - lo)
logflat = np.diff(np.log(edges)) / np.log(hi / lo)
xpos = np.arange(4)
ax[1].bar(xpos - 0.2, flat, 0.4, color=SERIES[1], label=r"flat in $\lambda$")
ax[1].bar(xpos + 0.2, logflat, 0.4, color=SERIES[0], label=r"flat in $\ln\lambda$")
ax[1].set_xticks(xpos, ["0.01-0.1", "0.1-1", "1-10", "10-100"], fontsize=7)
ax[1].set_ylabel("prior probability"); ax[1].set_title(r"scale $\lambda$: mass per decade", fontsize=8)
ax[1].legend(fontsize=6)

# ---------------------------------------------------------------- three priors, few data
th = np.linspace(1e-4, 0.6, 2000)
res = {}
for k, (name, a, b) in enumerate([("flat", 1, 1), ("Jeffreys", 0.5, 0.5), ("Haldane", 0, 0)]):
    for z, N in [(1, 10), (100, 1000)]:
        post = stats.beta(a + z, b + N - z)
        res[(name, N)] = (post.mean(), post.ppf(0.95))
    ax[2].plot(th, stats.beta(a + 1, b + 9).pdf(th), color=SERIES[k],
               label=f"{name} Beta({a:g},{b:g})")
ax[2].set_xlabel(r"$\theta$"); ax[2].set_title("posterior after 1 success in 10", fontsize=8)
ax[2].legend(fontsize=6); ax[2].set_yticks([])
fig.tight_layout()
savefig(fig, "ch05", "priors_param")

vals = {"FiveBPrFracCentral": frac_central,
        "FiveBPrFlatTop": flat[-1], "FiveBPrFlatBottom": flat[0], "FiveBPrLogDecade": logflat[0]}
for name, tag in [("flat", "Flat"), ("Jeffreys", "Jef"), ("Haldane", "Hal")]:
    for N, ntag in [(10, "Ten"), (1000, "Thou")]:
        m, u = res[(name, N)]
        vals[f"FiveBPr{tag}Mean{ntag}"] = m
        vals[f"FiveBPr{tag}Up{ntag}"] = u
save_numbers("ch05", "14_priors", vals)
