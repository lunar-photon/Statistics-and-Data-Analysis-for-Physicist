"""04_bdt_posterior.py -- is this event signal or background, given its BDT score?

Question: a classifier (a boosted decision tree, BDT) gives each event a score
x in [0, 1].  Signal scores follow f_S = Beta(5, 2), background scores f_B = Beta(2, 5).
Given one event's score, what is P(signal | x)?  How does the answer depend on the
signal fraction pi (the prior)?  And how does it differ from the purity of the
sample x > c selected by a cut?

Computes:
  * the likelihood ratio f_S(x)/f_B(x) = (x/(1-x))^3 and the posterior curves for
    pi = 0.5, 0.01, 1e-4;
  * the cut-based purity P(S | x > c) for pi = 0.01;
  * a Monte Carlo check: 10 million events with pi = 0.01, the signal fraction among
    events with x in a narrow bin around 0.9 against the formula.

Writes: figures/ch05/bdt_posterior.pdf, results/ch05/04_bdt_posterior.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch05", "04_bdt_posterior")
setup()

fS, fB = stats.beta(5, 2), stats.beta(2, 5)


def post_signal(x, pi):
    """P(S | x) from Bayes' theorem with densities as likelihoods."""
    num = pi * fS.pdf(x)
    return num / (num + (1 - pi) * fB.pdf(x))


def purity_cut(c, pi):
    """P(S | x > c): the evidence is the event 'x > c', its likelihoods are efficiencies."""
    num = pi * fS.sf(c)
    return num / (num + (1 - pi) * fB.sf(c))


x = np.linspace(0.001, 0.999, 500)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
a1.plot(x, fS.pdf(x), color=SERIES[0], label=r"signal $f_S(x)$")
a1.plot(x, fB.pdf(x), color=SERIES[1], label=r"background $f_B(x)$")
a1.set_xlabel("BDT score $x$"); a1.set_ylabel("density"); a1.legend()
for i, pi in enumerate([0.5, 0.01, 1e-4]):
    a2.plot(x, post_signal(x, pi), color=SERIES[i], label=rf"$\pi={pi:g}$")
a2.plot(x, purity_cut(x, 0.01), color=SERIES[1], ls=":", label=r"$P(S\,|\,x>c)$, $\pi=0.01$")
a2.set_xlabel("score $x$ (or cut $c$)"); a2.set_ylabel(r"$P(S\,|\,\mathrm{evidence})$")
a2.set_ylim(0, 1.02); a2.legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch05", "bdt_posterior")

# ---------- Monte Carlo check: Bayes' theorem by selection ----------
N, PI = 10_000_000, 0.01
is_sig = rng.uniform(size=N) < PI
score = np.where(is_sig, fS.rvs(size=N, random_state=rng), fB.rvs(size=N, random_state=rng))
sel = np.abs(score - 0.9) < 0.005
frac_mc = is_sig[sel].mean()
frac_err = np.sqrt(frac_mc * (1 - frac_mc) / sel.sum())

# the value of x at which P(S|x) = 1/2 for each prior: 3 logit(x) = -logit(pi)
x_half = {pi: 1 / (1 + (pi / (1 - pi)) ** (1 / 3)) for pi in [0.5, 0.01, 1e-4]}

save_numbers("ch05", "04_bdt_posterior", {
    "FiveABdtPostNine": post_signal(0.9, 0.01), "FiveABdtPostEight": post_signal(0.8, 0.01),
    "FiveABdtPostNineTiny": post_signal(0.9, 1e-4),
    "FiveABdtEffS": fS.sf(0.8), "FiveABdtEffB": fB.sf(0.8),
    "FiveABdtCutLR": fS.sf(0.8) / fB.sf(0.8), "FiveABdtCutPur": purity_cut(0.8, 0.01),
    "FiveABdtMC": frac_mc, "FiveABdtMCerr": frac_err, "FiveABdtNsel": int(sel.sum()),
    "FiveABdtHalfOne": x_half[0.01], "FiveABdtHalfTiny": x_half[1e-4],
})
