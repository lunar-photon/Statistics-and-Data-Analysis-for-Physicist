"""08_grid_posterior.py -- a posterior density for a continuous parameter, on a grid.

Question: data = prediction(lambda) + Gaussian noise.  What does the posterior
density p(lambda | d) look like, and how do we compute it safely on a computer?

Computes:
  (a) one measurement d of a rate that depends on the square of a coupling,
      mu(g) = g^2, with noise sigma: likelihood and posterior for g on a grid
      (flat prior); the two peaks at +-sqrt(d) and the mass on each side;
  (b) N = 2000 measurements of an exponential decay d_i = A exp(-t_i/tau) + noise:
      the posterior for the lifetime tau on a grid, computed (i) as a raw product of
      densities (it overflows to infinity, and underflows to zero when the same
      data are quoted in per cent) and (ii) as a sum of logs with the maximum
      subtracted before exponentiating (it works); posterior mean and width.

Writes: figures/ch05/grid_coupling.pdf, figures/ch05/grid_lifetime.pdf,
        results/ch05/08_grid_posterior.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, theory_line

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch05", "08_grid_posterior")
setup()

# ---------------------------------------------------------------- (a) coupling
G_TRUE, SIG = 0.8, 0.1
d = G_TRUE**2 + SIG * rng.standard_normal()          # the one measurement
g = np.linspace(-1.5, 1.5, 3001)
dg = g[1] - g[0]
prior = np.ones_like(g) / (g[-1] - g[0])             # flat on [-1.5, 1.5]
like = np.exp(-(d - g**2) ** 2 / (2 * SIG**2)) / np.sqrt(2 * np.pi * SIG**2)
evid = np.sum(like * prior) * dg                     # p(d) = int p(d|g) p(g) dg
post = like * prior / evid
mass_pos = np.sum(post[g > 0]) * dg
peak = g[g > 0][np.argmax(post[g > 0])]

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
ax[0].plot(g, g**2, color=SERIES[0])
ax[0].axhspan(d - SIG, d + SIG, color=SERIES[1], alpha=0.25, lw=0)
ax[0].axhline(d, color=SERIES[1], lw=1)
ax[0].set_xlabel(r"coupling $g$"); ax[0].set_ylabel(r"predicted rate $\mu(g)=g^2$")
ax[0].set_title("forward model and datum", fontsize=9)
ax[1].plot(g, like, color=SERIES[2])
ax[1].set_xlabel(r"$g$"); ax[1].set_title(r"likelihood $p(d\,|\,g)$", fontsize=9)
ax[2].plot(g, post, color=SERIES[0])
ax[2].fill_between(g, post, where=g > 0, color=SERIES[0], alpha=0.25, lw=0)
ax[2].set_xlabel(r"$g$"); ax[2].set_title(r"posterior $p(g\,|\,d)$, flat prior", fontsize=9)
for a in ax[1:]:
    a.set_yticks([])
fig.tight_layout()
savefig(fig, "ch05", "grid_coupling")

# ---------------------------------------------------------------- (b) lifetime
A, TAU_TRUE, N, SIGD = 1.0, 2.0, 2000, 0.05
t = np.sort(rng.uniform(0, 6, N))
data = A * np.exp(-t / TAU_TRUE) + SIGD * rng.standard_normal(N)
tau = np.linspace(1.9, 2.1, 801)
dtau = tau[1] - tau[0]
model = A * np.exp(-t[None, :] / tau[:, None])                  # (n_tau, N)
dens = np.exp(-(data - model) ** 2 / (2 * SIGD**2)) / np.sqrt(2 * np.pi * SIGD**2)
with np.errstate(over="ignore", under="ignore"):
    raw_max = np.prod(dens, axis=1).max()                       # overflows to inf
    # the same data quoted in per cent: every density is 100 times smaller
    raw_max_pct = np.prod(dens / 100.0, axis=1).max()           # underflows to 0
logL = np.sum(-(data - model) ** 2 / (2 * SIGD**2) - 0.5 * np.log(2 * np.pi * SIGD**2), axis=1)
logL_max = logL.max()
w = np.exp(logL - logL_max)                                     # flat prior on tau
post_tau = w / (w.sum() * dtau)
mean_tau = np.sum(tau * post_tau) * dtau
sd_tau = np.sqrt(np.sum((tau - mean_tau) ** 2 * post_tau) * dtau)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].plot(t, data, ".", ms=1.2, color=SERIES[0], alpha=0.5, label="data")
tt = np.linspace(0, 6, 200)
theory_line(ax[0], tt, A * np.exp(-tt / TAU_TRUE), label=r"true $e^{-t/\tau}$")
ax[0].set_xlabel("time $t$"); ax[0].set_ylabel("$d_i$"); ax[0].legend(fontsize=7)
ax[1].plot(tau, post_tau, color=SERIES[0])
ax[1].axvline(TAU_TRUE, color=INK2, ls=":", lw=0.8)
ax[1].set_xlabel(r"lifetime $\tau$"); ax[1].set_ylabel(r"$p(\tau\,|\,\mathbf{d})$")
fig.tight_layout()
savefig(fig, "ch05", "grid_lifetime")

save_numbers("ch05", "08_grid_posterior", {
    "FiveBGd": d, "FiveBGsig": SIG, "FiveBGtrue": G_TRUE, "FiveBGpeak": peak,
    "FiveBGsqrtd": np.sqrt(d), "FiveBGmasspos": mass_pos, "FiveBGevid": evid,
    "FiveBTauN": N, "FiveBTauRawMax": r"\infty" if np.isinf(raw_max) else f"{raw_max:.3g}",
    "FiveBTauRawMaxPct": f"{raw_max_pct:.3g}", "FiveBTauLogLmax": logL_max,
    "FiveBTauMean": mean_tau, "FiveBTauSd": sd_tau, "FiveBTauSigd": SIGD,
})
