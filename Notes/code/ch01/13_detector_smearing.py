"""Marginalising a nuisance variable: the measured spectrum of a smeared falling energy spectrum.

Question: a calorimeter never reports the true energy E, only E_meas = E + noise.  The joint
density of (E, E_meas) is p(E) * p(E_meas | E).  The distribution we actually histogram is the
marginal of E_meas, with the unobserved true energy averaged out.  Does the closed form
      p(x) = (1/beta) exp(sigma^2/(2 beta^2) - x/beta) Phi((x - sigma^2/beta)/sigma)
(derived in the text) match the simulated histogram?

Model: true E ~ Exponential(mean beta = 20 GeV); resolution Gaussian with sigma = 5 GeV.
Writes: figures/ch01/detector_smearing.pdf, results/ch01/13_detector_smearing.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

rng = rng_for("ch01", "13_detector_smearing")
setup(7.0, 3.0)

beta, sigma = 20.0, 5.0
N = 500_000
E_true = rng.exponential(beta, size=N)                 # the nuisance: never observed
E_meas = E_true + rng.normal(0.0, sigma, size=N)       # what the detector reports


def p_meas(x):
    """Marginal density of the measured energy (true energy integrated out)."""
    return (1 / beta) * np.exp(sigma**2 / (2 * beta**2) - x / beta) * \
        stats.norm.cdf((x - sigma**2 / beta) / sigma)


x = np.linspace(-20, 120, 700)
fig, axes = plt.subplots(1, 2)
ax = axes[0]
ax.plot(E_true[:4000], E_meas[:4000], ".", ms=1.5, color=SERIES[0], alpha=0.4)
ax.plot([0, 120], [0, 120], color="k", lw=0.8, ls=":")
ax.set_xlim(-5, 120); ax.set_ylim(-20, 120)
ax.set_xlabel("true $E$ [GeV]  (never observed)")
ax.set_ylabel("measured $E_{\\rm meas}$ [GeV]")
ax.set_title("joint $p(E, E_{\\rm meas})$")

ax = axes[1]
ax.hist(E_true, bins=140, range=(-20, 120), density=True, histtype="step",
        color=SERIES[1], label="true $E$")
ax.hist(E_meas, bins=140, range=(-20, 120), density=True, color=SERIES[0], alpha=0.5,
        label="measured $E_{\\rm meas}$")
theory_line(ax, x, p_meas(x), label="marginal formula")
ax.set_yscale("log"); ax.set_ylim(1e-4, 0.08)
ax.set_xlabel("energy [GeV]"); ax.set_ylabel("density [GeV$^{-1}$]")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "ch01", "detector_smearing")

# checks: mean beta, variance beta^2 + sigma^2, fraction of negative measured energies,
# and a chi-square of the histogram against N * (integral of the marginal over each bin)
frac_neg_exact = quad(p_meas, -np.inf, 0)[0]
counts, edges = np.histogram(E_meas, bins=140, range=(-20, 120))
expected = np.array([N * quad(p_meas, a, b)[0] for a, b in zip(edges[:-1], edges[1:])])
mask = expected > 20
chi2 = np.sum((counts[mask] - expected[mask])**2 / expected[mask])
ndf = int(mask.sum())

save_numbers("ch01", "13_detector_smearing", {
    "OneBSmearN": "5\\times10^{5}",
    "OneBSmearMeanSim": f"{E_meas.mean():.2f}",
    "OneBSmearVarSim": f"{E_meas.var():.1f}",
    "OneBSmearVarExact": f"{beta**2 + sigma**2:.1f}",
    "OneBSmearNegSim": f"{np.mean(E_meas < 0):.4f}",
    "OneBSmearNegExact": f"{frac_neg_exact:.4f}",
    "OneBSmearChi": f"{chi2:.0f}",
    "OneBSmearNdf": ndf,
})
