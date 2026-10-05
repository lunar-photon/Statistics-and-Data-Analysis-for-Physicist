"""16_sampling.py -- sampling the posterior and histogramming the samples.

Question: if we can draw theta_1, ..., theta_B from p(theta | d), a histogram of the draws
approximates the posterior density.  What else do the draws give us for free?

Computes:
  * functions of a parameter (Wasserman 11.3-11.4): 10 heads in 14 tosses, flat prior,
    p ~ Beta(11, 5); the log-odds psi = ln(p/(1-p)) of each draw, histogram against the
    exact density h(psi) obtained by the change of variables;
  * posterior predictive: (i) two more heads after 10 in 14 (exact B(13,5)/B(11,5));
    (ii) 1 diseased in a sample of 10 with a flat prior: the number of cases in the next
    sample of 10, and its 80% predictive interval, against the plug-in binomial p = 0.1;
  * a difference of two rates (Wasserman Example 11.7): 18 of 40 and 28 of 40 successes,
    flat priors: tau = p2 - p1 by subtracting draws; P(tau > 0) and a 95% interval;
  * the same idea for a classical oscillator: x = A cos(omega t) at uniformly random times
    has the arcsine density 1/(pi sqrt(A^2 - x^2)).

Writes: figures/ch05/sampling.pdf, results/ch05/16_sampling.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special

rng = rng_for("ch05", "16_sampling")
setup()
B = 200000
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))

# ---------------------------------------------------------------- function of a parameter
s, n = 10, 14
p = rng.beta(s + 1, n - s + 1, B)
psi = np.log(p / (1 - p))
ps = np.linspace(-2.5, 4.5, 400)
e = np.exp(ps)
h = np.exp(-special.betaln(s + 1, n - s + 1)) * (e / (1 + e)) ** (s + 1) * (1 / (1 + e)) ** (n - s + 1)
a = ax[0, 0]
a.hist(psi, bins=100, range=(-2.5, 4.5), density=True, color=SERIES[0], alpha=0.6, label="draws")
theory_line(a, ps, h, label=r"exact $h(\psi\,|\,x^n)$")
a.set_xlabel(r"log-odds $\psi=\ln\frac{p}{1-p}$"); a.set_yticks([]); a.legend(fontsize=6)
a.set_title("a function of the parameter", fontsize=8)

# ---------------------------------------------------------------- predictive
two_heads = np.mean((rng.uniform(size=B) < p) & (rng.uniform(size=B) < p))
exact2 = np.exp(special.betaln(13, 5) - special.betaln(11, 5))
th = rng.beta(1 + 1, 9 + 1, B)                       # 1 of 10, flat prior
xnew = rng.binomial(10, th)
k = np.arange(11)
pred = np.bincount(xnew, minlength=11) / B
plug = stats.binom(10, 0.1).pmf(k)
cdf = np.cumsum(pred)
lo80, hi80 = int(np.searchsorted(cdf, 0.1)), int(np.searchsorted(cdf, 0.9))
cdfp = np.cumsum(plug)
hi80p = int(np.searchsorted(cdfp, 0.9))
a = ax[0, 1]
a.bar(k - 0.2, pred, 0.4, color=SERIES[0], label="posterior predictive")
a.bar(k + 0.2, plug, 0.4, color=SERIES[1], label=r"plug-in, $\theta=0.1$")
a.set_xlabel("cases in the next sample of 10"); a.legend(fontsize=6)
a.set_title("predicting new data", fontsize=8)

# ---------------------------------------------------------------- difference of two rates
x1, n1, x2, n2 = 18, 40, 28, 40
p1 = rng.beta(x1 + 1, n1 - x1 + 1, B)
p2 = rng.beta(x2 + 1, n2 - x2 + 1, B)
tau = p2 - p1
a = ax[1, 0]
a.hist(tau, bins=100, density=True, color=SERIES[2], alpha=0.6)
a.axvline(0, color="k", lw=0.8)
a.set_xlabel(r"$\tau=p_2-p_1$"); a.set_yticks([])
a.set_title("a difference, by subtracting draws", fontsize=8)
p_pos = np.mean(tau > 0)
t_lo, t_hi = np.quantile(tau, [0.025, 0.975])

# ---------------------------------------------------------------- oscillator
A = 1.0
tt = rng.uniform(0, 2 * np.pi, B)                    # omega t uniform over a period
x = A * np.cos(tt)
xs = np.linspace(-0.995, 0.995, 400)
a = ax[1, 1]
a.hist(x, bins=80, range=(-1, 1), density=True, color=SERIES[3], alpha=0.6, label="$x(t)$, random $t$")
theory_line(a, xs, 1 / (np.pi * np.sqrt(A**2 - xs**2)), label=r"$1/\pi\sqrt{A^2-x^2}$")
a.set_xlabel("position $x$"); a.set_yticks([]); a.legend(fontsize=6, loc="upper center")
a.set_title("the same idea: an oscillator", fontsize=8)
fig.tight_layout()
savefig(fig, "ch05", "sampling")

save_numbers("ch05", "16_sampling", {
    "FiveBSaTwoHeads": two_heads, "FiveBSaTwoHeadsExact": exact2,
    "FiveBSaPredLo": lo80, "FiveBSaPredHi": hi80, "FiveBSaPlugHi": hi80p,
    "FiveBSaPredZero": pred[0], "FiveBSaPlugZero": plug[0],
    "FiveBSaPpos": p_pos, "FiveBSaTauLo": t_lo, "FiveBSaTauHi": t_hi, "FiveBSaB": B,
})
