"""08_likelihood_shapes.py -- what does a likelihood look like, and is it a density in theta?

Question: the likelihood L(theta) = p(data | theta) is the density of the data read as a
function of theta.  What shapes does it take for simple models, and does it integrate to one
over theta (it does not)?  How does its area change when we merely rename the parameter?

Computes: (a) Bernoulli likelihood, n = 20 trials, S = 12 successes, and its area over p;
          (b) Uniform(0, theta) likelihood for ten data points (a kink at max x_i);
          (c) the same exponential-lifetime likelihood as a function of the mean life tau
              and of the decay rate lambda = 1/tau: two different areas.
Writes:   figures/ch03/likelihood_shapes.pdf, results/ch03/08_likelihood_shapes.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate

rng = rng_for("ch03", "08_likelihood_shapes")
setup()

# (a) Bernoulli: L(p) = p^S (1-p)^(n-S); its area is the Beta function B(S+1, n-S+1)
n, S = 20, 12
p = np.linspace(0, 1, 801)
Lp = p**S * (1 - p) ** (n - S)
area_bern = special.beta(S + 1, n - S + 1)            # = S!(n-S)!/(n+1)!
area_bern_num = integrate.quad(lambda q: q**S * (1 - q) ** (n - S), 0, 1)[0]
Lmax_bern = (S / n) ** S * (1 - S / n) ** (n - S)

# (b) Uniform(0, theta): L = theta^-m for theta >= max x, else 0
m = 10
x = rng.uniform(0, 1.0, m)
xmax = x.max()
th = np.linspace(0.6, 1.6, 1001)
Lu = np.where(th >= xmax, th ** (-m), 0.0)

# (c) exponential lifetimes: L(tau) = tau^-k exp(-T/tau);  in lambda: lambda^k exp(-T lambda)
k = 5
t = rng.exponential(1.0, k)
T = t.sum()
area_tau = special.gamma(k - 1) / T ** (k - 1)         # int_0^inf tau^-k e^{-T/tau} dtau
area_lam = special.gamma(k + 1) / T ** (k + 1)         # int_0^inf lam^k e^{-T lam} dlam
area_tau_num = integrate.quad(lambda s: s ** (-k) * np.exp(-T / s), 1e-6, np.inf)[0]

fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.8))
axs[0].plot(p, Lp / Lmax_bern, color=SERIES[0])
axs[0].axvline(S / n, color=SERIES[1], ls=":", lw=1)
axs[0].set_xlabel(r"$p$"); axs[0].set_ylabel(r"$\mathcal{L}(p)/\mathcal{L}_{\max}$")
axs[0].set_title(r"Bernoulli, $n=20$, $S=12$")

axs[1].plot(th, Lu / xmax ** (-m), color=SERIES[0])
axs[1].plot(x, np.full(m, 0.05), "|", color=SERIES[2], ms=10, mew=1.2)
axs[1].set_xlabel(r"$\theta$"); axs[1].set_title(r"Uniform$(0,\theta)$, ten points")

tt = np.linspace(0.05, 6, 800)
Ltau = tt ** (-k) * np.exp(-T / tt)
lam = 1 / tt
Llam = lam**k * np.exp(-T * lam)
axs[2].plot(tt, Ltau / Ltau.max(), color=SERIES[0], label=r"vs $\tau$")
ll = np.linspace(0.02, 4, 800)
axs[2].plot(ll, (ll**k * np.exp(-T * ll)) / Llam.max(), color=SERIES[1], ls="-", label=r"vs $\lambda=1/\tau$")
axs[2].set_xlabel(r"$\tau$ or $\lambda$"); axs[2].legend(fontsize=7)
axs[2].set_title(r"lifetimes, $k=5$")
fig.tight_layout()
savefig(fig, "ch03", "likelihood_shapes")

save_numbers("ch03", "08_likelihood_shapes", {
    "ThreeBBernArea": area_bern,
    "ThreeBBernAreaNum": area_bern_num,
    "ThreeBBernLmax": Lmax_bern,
    "ThreeBBernRatio": area_bern / Lmax_bern,
    "ThreeBUnifMax": xmax,
    "ThreeBLifeT": T,
    "ThreeBAreaTau": area_tau,
    "ThreeBAreaTauNum": area_tau_num,
    "ThreeBAreaLam": area_lam,
    "ThreeBAreaRatio": area_tau / area_lam,
})
