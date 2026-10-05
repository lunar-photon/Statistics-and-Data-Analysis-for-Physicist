"""23_cl_distribution.py -- the sampling distribution of C_hat_l and the likelihood of C_l.

Question: if the sky is Gaussian and isotropic with spectrum C_l, how is the full-sky
estimate C_hat_l = (1/(2l+1)) sum_m |a_lm|^2 distributed?  And, read the other way,
given the C_hat_l of our one sky, which values of C_l could have produced it?
Computes: the scaled chi^2_{2l+1} density of C_hat_l / C_l for several l, its Gaussian
approximation, and the likelihood L(C_l) for an observed C_hat_l; quoted probabilities.
Writes: figures/ch07/chat_pdf.pdf, results/ch07/23_cl_distribution.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from common import setup, savefig, save_numbers, SERIES

setup()
LS = [2, 5, 20, 100]

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
x = np.linspace(1e-4, 3.0, 1200)
for k, l in enumerate(LS):
    nu = 2 * l + 1
    pdf = nu * stats.chi2.pdf(nu * x, nu)             # density of y = C_hat/C when nu*y ~ chi2_nu
    ax[0].plot(x, pdf, color=SERIES[k], label=rf"$\ell={l}$")
    if l in (5, 100):
        g = stats.norm.pdf(x, 1.0, np.sqrt(2.0 / nu))
        ax[0].plot(x, g, color=SERIES[k], ls=":", lw=1.1)
ax[0].set_xlabel(r"$\widehat C_\ell/C_\ell$")
ax[0].set_ylabel("probability density")
ax[0].set_title(r"(a) prediction: $\widehat C_\ell$ given $C_\ell$")
ax[0].legend(loc="upper right")
ax[0].set_ylim(0, 6.0)

# likelihood of C_l given C_hat_l:  L(C) ∝ C^{-nu/2} exp(-nu C_hat / (2C)); plot against C / C_hat
r = np.linspace(0.05, 6.0, 1500)
for k, l in enumerate(LS):
    nu = 2 * l + 1
    lnL = -0.5 * nu * (1.0 / r + np.log(r))
    ax[1].plot(r, np.exp(lnL - lnL.max()), color=SERIES[k], label=rf"$\ell={l}$")
ax[1].set_xlabel(r"candidate $C_\ell$ in units of the observed $\widehat C_\ell$")
ax[1].set_ylabel("likelihood (peak = 1)")
ax[1].set_title(r"(b) inference: which $C_\ell$ made our $\widehat C_\ell$?")
ax[1].legend(loc="upper right")
fig.tight_layout()
savefig(fig, "ch07", "chat_pdf")

# ---------------------------------------------------------------- quoted numbers
nu2 = 5
p_below_mean = stats.chi2.cdf(nu2, nu2)               # P(C_hat_2 < C_2)
p_fifth = stats.chi2.cdf(nu2 * 0.2, nu2)              # P(C_hat_2 < 0.2 C_2)
p_fifth30 = stats.chi2.cdf(61 * 0.8, 61)              # P(C_hat_30 < 0.8 C_30)


def dlnl_interval(nu):
    """C/C_hat where -2 ln L rises by 1 from its minimum at C = C_hat."""
    f = lambda r: nu * (1.0 / r + np.log(r) - 1.0) - 1.0
    return optimize.brentq(f, 1e-3, 1.0), optimize.brentq(f, 1.0, 100.0)

lo2, hi2 = dlnl_interval(5)
lo100, hi100 = dlnl_interval(201)
save_numbers("ch07", "23_cl_distribution", {
    "SBpBelowMean": p_below_mean,
    "SBpFifth": p_fifth,
    "SBpThirty": p_fifth30,
    "SBfracTwo": np.sqrt(2 / 5),
    "SBfracThirty": np.sqrt(2 / 61),
    "SBfracThousand": np.sqrt(2 / 2001),
    "SBfracBinned": 100 * np.sqrt(2 / (2001 * 50)),
    "SBskewTwo": np.sqrt(8 / 5),
    "SBloTwo": f"{lo2:.2f}", "SBhiTwo": f"{hi2:.2f}",
    "SBloHund": f"{lo100:.2f}", "SBhiHund": f"{hi100:.2f}",
})
