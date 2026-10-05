"""10_three_instruments.py -- one posterior calculation, three instruments.

Question: the rule "posterior = likelihood x prior, normalised" is the same everywhere;
what changes between a collider, a CMB map and a gravitational-wave detector is only the
likelihood.  What do the posteriors look like in each case?

Computes:
  (a) collider counts with negligible background: posterior for the expected signal
      count s after n = 0, 1, 3 observed events (flat prior: Gamma(n+1, 1)), and the
      95% credible upper limits;
  (b) CMB: posterior for the band power C_ell from the 2 ell + 1 measured a_lm, with the
      prior 1/C_ell: an inverse-gamma in C_ell / C_hat; mode, mean and 68% interval for
      ell = 2, 10, 50;
  (c) GW: amplitude A of a known template in coloured Gaussian noise with a PSD:
      posterior mean (d|h0)/(h0|h0) and width 1/sqrt(h0|h0), checked on 4000 noise
      realisations.

Writes: figures/ch05/three_instruments.pdf, results/ch05/10_three_instruments.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch05", "10_three_instruments")
setup()
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.7))

# ---------------------------------------------------------------- (a) counts
s = np.linspace(0, 12, 1200)
ul = {}
for k, n in enumerate([0, 1, 3]):
    post = stats.gamma(n + 1)                      # flat prior on s >= 0
    ul[n] = post.ppf(0.95)
    ax[0].plot(s, post.pdf(s), color=SERIES[k], label=f"$n={n}$")
    ax[0].axvline(ul[n], color=SERIES[k], ls=":", lw=0.9)
ax[0].set_xlabel("expected signal count $s$"); ax[0].set_title("collider: counts", fontsize=8)
ax[0].legend(fontsize=7)

# ---------------------------------------------------------------- (b) CMB band power
x = np.linspace(0.02, 5, 2000)                     # x = C_ell / C_hat
cmb = {}
for k, ell in enumerate([2, 10, 50]):
    nu = 2 * ell + 1
    post = stats.invgamma(nu / 2, scale=nu / 2)    # prior 1/C, likelihood C^{-nu/2} e^{-nu Chat/2C}
    cmb[ell] = dict(mode=(nu / 2) / (nu / 2 + 1), mean=post.mean(), lo=post.ppf(0.16), hi=post.ppf(0.84))
    ax[1].plot(x, post.pdf(x), color=SERIES[k], label=rf"$\ell={ell}$")
ax[1].set_xlabel(r"$C_\ell/\widehat{C}_\ell$"); ax[1].set_title("CMB: band power", fontsize=8)
ax[1].set_xlim(0, 4); ax[1].legend(fontsize=7)

# ---------------------------------------------------------------- (c) GW amplitude
f = np.linspace(20, 500, 4000); df = f[1] - f[0]
f0 = 150.0
Sn = (f0 / f) ** 4 + 1 + (f / f0) ** 2             # PSD shape (arbitrary units)
h0 = f ** (-7 / 6) * np.exp(1j * 2 * np.pi * f * 0.01)
inner = lambda a, b: 4 * np.real(np.sum(a * np.conj(b) / Sn)) * df
h0 = h0 * 10 / np.sqrt(inner(h0, h0))              # normalise: (h0|h0) = 100, SNR 10 at A = 1
hh = inner(h0, h0)


def noise(size=None):
    """Complex Gaussian noise with Re, Im variance S_n/(4 df): then (n|n) is chi^2 and (n|h) ~ N(0,(h|h))."""
    shape = (f.size,) if size is None else (size, f.size)
    sd = np.sqrt(Sn / (4 * df))
    return sd * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


A_TRUE = 1.0
d = A_TRUE * h0 + noise()
A_hat = inner(d, h0) / hh
A_sd = 1 / np.sqrt(hh)
A = np.linspace(0.4, 1.6, 600)
logL = np.array([-0.5 * inner(d - a * h0, d - a * h0) for a in A])
postA = np.exp(logL - logL.max()); postA /= postA.sum() * (A[1] - A[0])
ax[2].plot(A, postA, color=SERIES[0], label="grid")
theory_line(ax[2], A, stats.norm(A_hat, A_sd).pdf(A), label=r"$\mathcal{N}(\hat A,1/(h_0|h_0))$")
ax[2].axvline(A_TRUE, color=INK2, ls=":", lw=0.8)
ax[2].set_xlabel("amplitude $A$"); ax[2].set_title("GW: template amplitude", fontsize=8)
ax[2].legend(fontsize=6, loc="upper left")
for a in ax:
    a.set_yticks([])
fig.tight_layout()
savefig(fig, "ch05", "three_instruments")

# calibration: the spread of A_hat over many noise realisations equals the posterior width
nn = noise(4000)
A_hats = A_TRUE + (4 * np.real(np.sum(nn * np.conj(h0)[None, :] / Sn, axis=1)) * df) / hh
spread = A_hats.std()

save_numbers("ch05", "10_three_instruments", {
    "FiveBULzero": ul[0], "FiveBULone": ul[1], "FiveBULthree": ul[3],
    "FiveBCmbModeTwo": cmb[2]["mode"], "FiveBCmbMeanTwo": cmb[2]["mean"],
    "FiveBCmbLoTwo": cmb[2]["lo"], "FiveBCmbHiTwo": cmb[2]["hi"],
    "FiveBCmbLoFifty": cmb[50]["lo"], "FiveBCmbHiFifty": cmb[50]["hi"],
    "FiveBCmbModeFifty": cmb[50]["mode"],
    "FiveBGwAhat": A_hat, "FiveBGwAsd": A_sd, "FiveBGwSpread": spread, "FiveBGwHH": hh,
})
