"""03_luminosity.py -- how many collisions two crossing bunches give, and how that number is calibrated.

Question: the expected number of events is N = sigma x (integrated luminosity). Where does the
luminosity formula L = n_b f N1 N2 / (4 pi sigma_x sigma_y) come from, how many proton-proton
collisions happen in one bunch crossing, and how does a van der Meer scan measure the beam
overlap that the formula needs?

Computes:
  (a) the overlap integral of two Gaussian bunches done numerically on a grid, against the
      closed form, at zero and at nonzero beam separation, for the LHC design parameters;
  (b) the peak luminosity and the mean number of inelastic collisions per crossing (pile-up);
  (c) a toy van der Meer scan: Poisson counts at 25 beam separations, a Gaussian fitted to them,
      and the luminosity recovered from the fitted width, compared with the true value.
Writes: figures/chT4/vdm_scan.pdf, results/chT4/03_luminosity.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

setup()
rng = rng_for("chT4", "03_luminosity")

# LHC design values (Evans and Bryant 2008): protons per bunch, bunches, revolution frequency,
# r.m.s. transverse beam size at the collision point, geometric reduction factor from the crossing angle
N1 = N2 = 1.15e11
NB, FREV = 2808, 11245.0             # bunches per beam, revolutions per second
SIG = 16.7e-4                         # 16.7 micrometres, in cm
F_GEOM = 0.836
SIG_INEL = 78.1e-27                   # inelastic pp cross section at 13 TeV, 78.1 mb, in cm^2


# ---- (a) overlap integral on a grid: n1(x, y) n2(x - d, y), each bunch a normalised 2-D Gaussian
def bunch(x, y, sx, sy):
    return np.exp(-x**2 / (2 * sx**2) - y**2 / (2 * sy**2)) / (2 * np.pi * sx * sy)


g = np.linspace(-8 * SIG, 8 * SIG, 1601)
X, Y = np.meshgrid(g, g, indexing="ij")
dA = (g[1] - g[0])**2
overlap = {}
for d in (0.0, SIG, 2 * SIG):
    num = np.sum(bunch(X, Y, SIG, SIG) * bunch(X - d, Y, SIG, SIG)) * dA
    exact = np.exp(-d**2 / (4 * SIG**2)) / (4 * np.pi * SIG**2)
    overlap[d] = (num, exact)
rel_err = max(abs(n / e - 1) for n, e in overlap.values())

# ---- (b) luminosity and pile-up
L_head_on = NB * FREV * N1 * N2 / (4 * np.pi * SIG**2)
L_design = F_GEOM * L_head_on
mu_pile = SIG_INEL * L_design / (NB * FREV)
rate_inel = SIG_INEL * L_design

# ---- (c) toy van der Meer scan in x (y treated the same way): one bunch pair, rate counted
# for T seconds at each separation. The visible rate is sigma_vis x (luminosity of this pair).
SIG_VIS = 60e-27                      # a luminometer that sees 60 mb of the inelastic 78 mb
SIGX_TRUE = SIG                       # single-beam size; the overlap width is Sigma = sqrt(2) sigma
Sigma_true = np.sqrt(2) * SIGX_TRUE
L_pair_peak = FREV * N1 * N2 / (2 * np.pi * Sigma_true * Sigma_true)    # head-on, one bunch pair
T_point = 0.2                         # seconds per scan step
d = np.linspace(-5, 5, 25) * Sigma_true


def rate_curve(dx, r0, Sig):
    return r0 * np.exp(-dx**2 / (2 * Sig**2))


r_true = SIG_VIS * L_pair_peak * np.exp(-d**2 / (2 * Sigma_true**2))
counts = rng.poisson(r_true * T_point)
rate = counts / T_point
err = np.sqrt(np.maximum(counts, 1)) / T_point
popt, pcov = curve_fit(rate_curve, d, rate, p0=[rate.max(), Sigma_true], sigma=err, absolute_sigma=True)
r0_fit, Sig_fit = popt
dSig = np.sqrt(pcov[1, 1])
# calibration: L_pair = f N1 N2 / (2 pi Sigma_x Sigma_y); with Sigma_y = Sigma_x (round beams),
# sigma_vis = R0 / L_pair, and later L = R / sigma_vis for any rate R measured in physics running
L_pair_fit = FREV * N1 * N2 / (2 * np.pi * Sig_fit * Sig_fit)
sig_vis_fit = r0_fit / L_pair_fit

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
xs = np.linspace(d.min(), d.max(), 400)
ax[0].errorbar(d * 1e4, rate / 1e3, yerr=err / 1e3, fmt="o", color="k", ms=3, lw=0.8, label="toy scan")
theory_line(ax[0], xs * 1e4, rate_curve(xs, *popt) / 1e3, label="fitted Gaussian")
ax[0].set_xlabel(r"beam separation $\Delta x$ [$\mu$m]"); ax[0].set_ylabel("visible rate [kHz]")
ax[0].set_title("(a) a van der Meer scan")
ax[0].legend(fontsize=8)
dd = np.linspace(0, 4, 200) * SIG
ax[1].plot(dd * 1e4, np.exp(-dd**2 / (4 * SIG**2)), color=SERIES[0], label=r"$e^{-\Delta^2/4\sigma^2}$")
for k, (n, e) in overlap.items():
    ax[1].plot(k * 1e4, n * 4 * np.pi * SIG**2, "o", color=SERIES[1], ms=6)
ax[1].plot([], [], "o", color=SERIES[1], label="grid integral")
ax[1].set_xlabel(r"separation $\Delta$ [$\mu$m]"); ax[1].set_ylabel("overlap / head-on overlap")
ax[1].set_title("(b) the overlap falls as a Gaussian")
ax[1].legend(fontsize=8)
fig.tight_layout()
savefig(fig, "chT4", "vdm_scan")

save_numbers("chT4", "03_luminosity", {
    "TflOverlapErr": rel_err, "TflLheadOn": L_head_on, "TflLdesign": L_design,
    "TflPileup": mu_pile, "TflRateInel": rate_inel,
    "TflSigmaTrue": Sigma_true * 1e4, "TflSigmaFit": Sig_fit * 1e4, "TflSigmaErr": dSig * 1e4,
    "TflSigmaRelErr": 100 * dSig / Sig_fit,
    "TflSigVisFit": sig_vis_fit / 1e-27, "TflSigVisTrue": SIG_VIS / 1e-27,
    "TflPeakRate": r0_fit / 1e3, "TflNpoints": len(d), "TflTpoint": T_point,
})
print(f"overlap grid vs exact, worst relative error {rel_err:.2e}")
print(f"L head-on {L_head_on:.3e}, with F {L_design:.3e} cm^-2 s^-1; pile-up {mu_pile:.1f}; inelastic rate {rate_inel:.3e}/s")
print(f"vdM: Sigma true {Sigma_true*1e4:.3f} um, fit {Sig_fit*1e4:.3f} +- {dSig*1e4:.3f} um; sigma_vis {sig_vis_fit/1e-27:.2f} mb (true 60)")
