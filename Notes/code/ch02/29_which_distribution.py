"""Which distribution when: three recognitions checked by simulation, and why the tails decide.

Question answered
    (1) A Z-boson peak: each measured mass is a Breit-Wigner mass (M, Gamma) plus a
        Gaussian resolution error (sigma).  The sum has the Voigt density.  Its core
        looks Gaussian, but what fraction of events lies more than 10 GeV from M,
        compared with a Gaussian of the same FWHM and with the Cauchy tail estimate
        2 gamma / (pi Delta)?
    (2) A matched filter in Gaussian noise returns two quadratures, each N(0,1).
        Is the squared SNR |rho|^2 an exponential (chi^2_2), so that the chance of
        noise exceeding rho* is exp(-rho*^2/2)?
    (3) A fit with nu = 100 degrees of freedom returns chi^2 = 130.  What is the
        exact tail probability, and how wrong is the Gaussian approximation?
    (4) Four symmetric laws with the same mean and variance: how different are their
        tails at 3 and 5 standard deviations?

What it computes
    Monte Carlo draws for (1) and (2), the Voigt density by scipy.special.voigt_profile, exact tail probabilities from scipy for (2)-(4).

What it writes
    figures/ch02/which_distribution.pdf, results/ch02/29_which_distribution.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import voigt_profile

setup()
rng = rng_for("ch02", "29_which_distribution")
NREP = 200_000

# ---------------------------------------------------------------- (1) resolution-smeared resonance
M, Gam, sig = 91.1876, 2.4952, 2.0           # GeV: Z mass, full width, detector resolution
gam = Gam / 2                                # Cauchy half-width
m = M + gam * rng.standard_cauchy(NREP) + rng.normal(0.0, sig, NREP)
Delta = 10.0
out_sim = np.mean(np.abs(m - M) > Delta)
uu = np.linspace(-Delta, Delta, 200_001)
core = np.trapezoid(voigt_profile(uu, sig, gam), uu)
out_voigt = 1.0 - core
out_tail = 2 * gam / (np.pi * Delta)         # Cauchy tail estimate, valid for Delta >> sigma, gamma
# FWHM of the Voigt density, found numerically, and the Gaussian with the same FWHM
grid = np.linspace(0, 10, 200_001)
vp = voigt_profile(grid, sig, gam)
fwhm = 2 * grid[np.argmin(np.abs(vp - vp[0] / 2))]
sig_eq = fwhm / (2 * np.sqrt(2 * np.log(2)))
out_gauss = 2 * stats.norm.sf(Delta / sig_eq)

# ---------------------------------------------------------------- (2) matched-filter SNR
rho = rng.standard_normal((NREP, 2))            # two quadratures of pure noise
rho2 = (rho ** 2).sum(axis=1)
rstar = 3.0
fa_sim = np.mean(rho2 > rstar ** 2)
fa_th = np.exp(-rstar ** 2 / 2)
fa_eight = np.exp(-8.0 ** 2 / 2)

# ---------------------------------------------------------------- (3) chi^2 of a fit
nu, chi2obs = 100, 130.0
p_exact = stats.chi2.sf(chi2obs, nu)
z_gauss = (chi2obs - nu) / np.sqrt(2 * nu)
p_gauss = stats.norm.sf(z_gauss)

# ---------------------------------------------------------------- (4) tails at equal variance
def two_sided(dist, k):
    return 2 * dist.sf(k)

laws = {
    "Gaussian": stats.norm(),
    "Laplace": stats.laplace(scale=1 / np.sqrt(2)),          # var = 2 b^2 = 1
    "t5": stats.t(5, scale=np.sqrt(3 / 5)),                   # var = nu/(nu-2) * s^2 = 1
    "t3": stats.t(3, scale=np.sqrt(1 / 3)),
}

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
ax = axes[0]
k = np.linspace(0, 6, 400)
labels = {"Gaussian": "Gaussian", "Laplace": "Laplace", "t5": "Student $t_5$", "t3": "Student $t_3$"}
for i, (name, d) in enumerate(laws.items()):
    if name == "Gaussian":
        theory_line(ax, k, two_sided(d, k), label=labels[name])
    else:
        ax.plot(k, two_sided(d, k), color=SERIES[i - 1], lw=1.5, label=labels[name])
ax.set_yscale("log")
ax.set_ylim(1e-9, 1.5)
ax.set_xlabel("$k$ (standard deviations)")
ax.set_ylabel(r"$\Pr(|X-\mu|>k\sigma)$")
ax.set_title("(a) same mean and variance, different tails")
ax.legend(fontsize=7)

ax = axes[1]
bins = np.linspace(M - 25, M + 25, 201)
ax.hist(m, bins=bins, density=True, color=SERIES[0], alpha=0.5, label="simulated masses")
xx = np.linspace(M - 25, M + 25, 1000)
theory_line(ax, xx, voigt_profile(xx - M, sig, gam), label="Voigt (BW $*$ Gaussian)")
ax.plot(xx, stats.norm.pdf(xx, M, sig_eq), color=SERIES[1], lw=1.4, label="Gaussian, same FWHM")
ax.plot(xx, stats.cauchy.pdf(xx, M, gam), color=SERIES[2], lw=1.2, ls=":", label="Breit--Wigner alone")
ax.set_yscale("log")
ax.set_ylim(1e-5, 0.3)
ax.set_xlabel("invariant mass $m$ (GeV)")
ax.set_ylabel("density")
ax.set_title("(b) a Z peak with detector resolution")
ax.legend(fontsize=7, loc="upper left")
savefig(fig, "ch02", "which_distribution")

# ---------------------------------------------------------------- numbers
num = {}
num["tbWsig"] = f"{sig:.1f}"
num["tbWDelta"] = f"{Delta:.0f}"
num["tbWfwhm"] = f"{fwhm:.2f}"
num["tbWsigEq"] = f"{sig_eq:.2f}"
num["tbWoutSim"] = f"{out_sim:.4f}"
num["tbWoutVoigt"] = f"{out_voigt:.4f}"
num["tbWoutTail"] = f"{out_tail:.4f}"
mm, ee = f"{out_gauss:.1e}".split("e")
num["tbWoutGauss"] = rf"{mm}\times10^{{{int(ee)}}}"
num["tbWrstar"] = f"{rstar:.0f}"
num["tbWfaSim"] = f"{fa_sim:.4f}"
num["tbWfaTh"] = f"{fa_th:.4f}"
mant, ex = f"{fa_eight:.2e}".split("e")
num["tbWfaEight"] = rf"{mant}\times10^{{{int(ex)}}}"
num["tbWpExact"] = f"{p_exact:.4f}"
num["tbWzGauss"] = f"{z_gauss:.2f}"
num["tbWpGauss"] = f"{p_gauss:.4f}"
for name, d in laws.items():
    key = {"Gaussian": "Gauss", "Laplace": "Lap", "t5": "Tfive", "t3": "Tthree"}[name]
    num[f"tbWthree{key}"] = f"{two_sided(d, 3.0):.4f}"
    v5 = two_sided(d, 5.0)
    mant, ex = f"{v5:.1e}".split("e")
    num[f"tbWfive{key}"] = rf"{mant}\times10^{{{int(ex)}}}"
num["tbWratioTthree"] = f"{two_sided(laws['t3'], 5.0) / two_sided(laws['Gaussian'], 5.0):.0f}"
save_numbers("ch02", "29_which_distribution", num)

for kk, vv in num.items():
    print(f"{kk:18s} {vv}")
