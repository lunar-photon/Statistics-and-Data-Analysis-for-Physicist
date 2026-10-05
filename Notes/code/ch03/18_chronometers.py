"""18_chronometers.py -- the closed-form estimator of H0 from cosmic-chronometer H(z) points.

Question: cosmic chronometers measure the expansion rate H(z_i) directly. In flat LCDM,
H(z) = H0 E(z; Om) with E = sqrt(Om (1+z)^3 + 1 - Om), so H0 enters linearly. At fixed Om the
chi^2 is a parabola in H0 and its minimum is the weighted ratio
    H0_hat = sum H_i E_i / s_i^2  /  sum E_i^2 / s_i^2 ,   Var H0_hat = 1 / sum E_i^2 / s_i^2 .
Is it unbiased, and is its scatter the Cramer-Rao bound? What happens when Om is profiled?
Computes: N_CC = 30 mock H(z) points, z uniform in 0.07 < z < 2, 8% errors, from H0 = 70 and
Om = 0.3; the closed-form estimate and its error for one survey; the mean and scatter of the
estimate over N_MOCK surveys at fixed Om against H0 and the bound; and the profile estimate
(Om minimised numerically, H0 analytically) with its scatter.
Writes: figures/ch03/chronometers.pdf, results/ch03/18_chronometers.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES, INK2

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

H0_TRUE, OM_TRUE = 70.0, 0.3
N_CC, N_MOCK, FRAC_ERR = 30, 4000, 0.08
rng = rng_for("ch03", "18_chronometers")


def E(z, om):
    return np.sqrt(om * (1 + z) ** 3 + 1 - om)


def mock():
    z = np.sort(rng.uniform(0.07, 2.0, N_CC))
    s = FRAC_ERR * H0_TRUE * E(z, OM_TRUE)          # errors fixed by the design, not by the data
    return z, H0_TRUE * E(z, OM_TRUE) + s * rng.standard_normal(N_CC), s


def h0_hat(z, H, s, om):
    """Closed-form minimum of chi^2 over H0 at fixed Om, and its standard deviation."""
    e = E(z, om)
    see = np.sum(e**2 / s**2)
    return np.sum(H * e / s**2) / see, 1 / np.sqrt(see)


def profile_chi2(om, z, H, s):
    h, _ = h0_hat(z, H, s, om)
    return np.sum((H - h * E(z, om)) ** 2 / s**2)


def profile_fit(z, H, s):
    om = minimize_scalar(profile_chi2, bounds=(0.05, 0.95), args=(z, H, s), method="bounded").x
    return h0_hat(z, H, s, om)[0], om


# one survey
z1, H1, s1 = mock()
h1, sig1 = h0_hat(z1, H1, s1, OM_TRUE)
hp1, omp1 = profile_fit(z1, H1, s1)

# many surveys (same design: new redshifts and noise each time)
fixed, prof, bound = [], [], []
for _ in range(N_MOCK):
    z, H, s = mock()
    h, sg = h0_hat(z, H, s, OM_TRUE)
    fixed.append(h); bound.append(sg)
    prof.append(profile_fit(z, H, s))
fixed, bound, prof = np.array(fixed), np.array(bound), np.array(prof)

nums = {
    "ThreeBCCN": N_CC, "ThreeBCCNmock": N_MOCK, "ThreeBCCErr": f"{100 * FRAC_ERR:.0f}",
    "ThreeBCCHone": f"{h1:.2f}", "ThreeBCCSigone": f"{sig1:.2f}",
    "ThreeBCCHprofone": f"{hp1:.2f}", "ThreeBCCOmprofone": f"{omp1:.3f}",
    "ThreeBCCMean": f"{fixed.mean():.2f}", "ThreeBCCMeanErr": f"{fixed.std(ddof=1) / np.sqrt(N_MOCK):.2f}",
    "ThreeBCCSd": f"{fixed.std(ddof=1):.3f}", "ThreeBCCBound": f"{np.sqrt(np.mean(bound**2)):.3f}",
    "ThreeBCCProfMean": f"{prof[:, 0].mean():.2f}", "ThreeBCCProfSd": f"{prof[:, 0].std(ddof=1):.2f}",
    "ThreeBCCOmSd": f"{prof[:, 1].std(ddof=1):.3f}",
}
save_numbers("ch03", "18_chronometers", nums)
for k, v in nums.items():
    print(k, v)

# figure: one survey with the fitted curve, and the chi^2 parabola in H0
setup(9.0, 3.4)
fig, (a, b) = plt.subplots(1, 2)
zz = np.linspace(0, 2.1, 200)
a.errorbar(z1, H1, s1, fmt="o", color=SERIES[0], ms=3, lw=0.8, label="mock $H(z_i)$")
theory_line(a, zz, H0_TRUE * E(zz, OM_TRUE), label="truth")
a.plot(zz, h1 * E(zz, OM_TRUE), color=SERIES[1], label=r"$\hat H_0\,E(z;\Omega_m=0.3)$")
a.set_xlabel("$z$"); a.set_ylabel(r"$H(z)$ [km/s/Mpc]"); a.legend(fontsize=7)
hh = np.linspace(h1 - 4 * sig1, h1 + 4 * sig1, 200)
chi = np.array([np.sum((H1 - x * E(z1, OM_TRUE)) ** 2 / s1**2) for x in hh])
b.plot(hh, chi - chi.min(), color=SERIES[0], label=r"$\chi^2(H_0)-\chi^2_{\min}$")
b.axhline(1, color=INK2, lw=0.8, ls=":")
b.axvline(h1, color=SERIES[1], lw=1.0, label=r"$\hat H_0$")
b.axvspan(h1 - sig1, h1 + sig1, color=SERIES[1], alpha=0.15, label=r"$\pm1/\sqrt{\sum E_i^2/\sigma_i^2}$")
b.set_xlabel(r"$H_0$ [km/s/Mpc]"); b.set_ylabel(r"$\Delta\chi^2$"); b.set_ylim(0, 6); b.legend(fontsize=7)
savefig(fig, "ch03", "chronometers")
