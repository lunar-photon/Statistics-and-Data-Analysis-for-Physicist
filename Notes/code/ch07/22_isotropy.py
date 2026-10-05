"""22_isotropy.py -- what statistical isotropy does to the a_lm.

Question: if no direction on the sky is special, why are the a_lm uncorrelated, with a
variance that depends on l but not on m?  And what does a field WITH a preferred axis
look like in harmonic space?
Computes: (1) the angular correlation function C(theta) = sum (2l+1)/(4 pi) C_l P_l(cos theta)
of the fiducial model and of five single skies (their full-sky pair average);
(2) <|a_lm|^2>/C_l against m, and the l,l+1 correlation, from simulated isotropic skies
and from skies multiplied by (1 + A cos theta), against the exact predictions.
Writes: figures/ch07/ctheta.pdf, figures/ch07/isotropy_test.pdf, results/ch07/22_isotropy.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES
from camb_fiducial import load_fiducial
import lib_harmonic as lh

setup()
ell, cl = load_fiducial()
rng = rng_for("ch07", "22_isotropy")

# ---------------------------------------------------------------- (1) C(theta)
LMAX = 2000
th = np.radians(np.linspace(0, 12, 400))
th_wide = np.radians(np.linspace(0, 180, 721))
ls = np.arange(LMAX + 1)


def ctheta(c, theta):
    """C(theta) = sum_l (2l+1)/(4 pi) c_l P_l(cos theta), by direct Legendre sum."""
    x = np.cos(theta)
    P = np.array([eval_legendre(l, x) for l in ls])          # (LMAX+1, ntheta)
    return ((2 * ls + 1) / (4 * np.pi) * c[: LMAX + 1]) @ P

Cth = ctheta(cl, th)
Cwide = ctheta(cl, th_wide)
var0 = np.sum((2 * ls + 1) * cl[: LMAX + 1]) / (4 * np.pi)
var_lo = np.sum((2 * ls[:31] + 1) * cl[:31]) / (4 * np.pi)

# single skies: the full-sky pair average of T(n)T(n') at separation theta equals the
# Legendre sum of that sky's own C_hat_l (exact on the full sky)
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
for k in range(5):
    chat = lh.alm2cl(lh.synalm(cl, LMAX, rng), LMAX)
    ax[1].plot(np.degrees(th_wide), ctheta(chat, th_wide), color=SERIES[k], lw=1.0, alpha=0.9)
ax[0].plot(np.degrees(th), Cth, color=SERIES[0])
ax[0].axhline(0, color="0.5", lw=0.6)
ax[0].set_xlabel(r"separation $\vartheta$ [deg]")
ax[0].set_ylabel(r"$C(\vartheta)$ [$\mu$K$^2$]")
ax[0].set_title("(a) fiducial model, small separations")
theory_line(ax[1], np.degrees(th_wide), Cwide, label="ensemble $C(\\vartheta)$")
ax[1].axhline(0, color="0.5", lw=0.6)
ax[1].set_xlim(10, 180)
ax[1].set_ylim(-400, 400)
ax[1].set_xlabel(r"separation $\vartheta$ [deg]")
ax[1].set_title("(b) five single skies, large separations")
ax[1].legend(loc="upper center")
fig.tight_layout()
savefig(fig, "ch07", "ctheta")

# ---------------------------------------------------------------- (2) isotropy test
L0, LM2, NSIM, AMP = 20, 42, 6000, 0.5
lvec, mvec = lh.getlm(LM2)
i_l = lh.index(L0, np.arange(L0 + 1), LM2)
i_lp = lh.index(L0 + 1, np.arange(L0 + 1), LM2)
pow_iso = np.zeros(L0 + 1); pow_mod = np.zeros(L0 + 1)
x_iso = np.zeros(L0 + 1); x_mod = np.zeros(L0 + 1)
for s in range(NSIM):
    a = lh.synalm(cl, LM2, rng)
    b = lh.modulate_dipole(a, LM2, AMP)
    pow_iso += np.abs(a[i_l]) ** 2;  pow_mod += np.abs(b[i_l]) ** 2
    x_iso += np.real(a[i_lp] * np.conj(a[i_l]));  x_mod += np.real(b[i_lp] * np.conj(b[i_l]))
m = np.arange(L0 + 1)
C0, Cm, Cp = cl[L0], cl[L0 - 1], cl[L0 + 1]
pow_iso /= NSIM * C0; pow_mod /= NSIM * C0
norm = np.sqrt(cl[L0] * cl[L0 + 1])
x_iso /= NSIM * norm; x_mod /= NSIM * norm
pow_th = 1 + AMP ** 2 * (lh.ccoef(L0, m) ** 2 * Cm + lh.ccoef(L0 + 1, m) ** 2 * Cp) / C0
x_th = AMP * lh.ccoef(L0 + 1, m) * (C0 + Cp) / norm
# statistical error of each point (m>0): |a|^2/C is exponential with sd 1 -> 1/sqrt(N)
err_pow = 1 / np.sqrt(NSIM)

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
ax[0].plot(m, pow_iso, "o", color=SERIES[0], label="isotropic skies")
ax[0].plot(m, pow_mod, "s", color=SERIES[1], label=r"modulated, $A=0.5$")
theory_line(ax[0], m, np.ones_like(m, dtype=float), label=None)
theory_line(ax[0], m, pow_th, label="exact prediction")
ax[0].set_xlabel(r"$m$"); ax[0].set_ylabel(r"$\langle|a_{20,m}|^2\rangle/C_{20}$")
ax[0].set_title(r"(a) power in each $m$, $\ell=20$")
ax[0].legend(loc="upper right", fontsize=8)
ax[1].plot(m, x_iso, "o", color=SERIES[0], label="isotropic skies")
ax[1].plot(m, x_mod, "s", color=SERIES[1], label=r"modulated, $A=0.5$")
theory_line(ax[1], m, np.zeros_like(m, dtype=float), label=None)
theory_line(ax[1], m, x_th, label="exact prediction")
ax[1].set_xlabel(r"$m$"); ax[1].set_ylabel(r"$\mathrm{Re}\langle a_{21,m}a^*_{20,m}\rangle/\sqrt{C_{20}C_{21}}$")
ax[1].set_title(r"(b) correlation of $\ell=20$ with $\ell=21$")
fig.tight_layout()
savefig(fig, "ch07", "isotropy_test")

save_numbers("ch07", "22_isotropy", {
    "SBvarZero": f"{var0:.0f}",
    "SBrmsZero": np.sqrt(var0),
    "SBfracLowL": 100 * var_lo / var0,
    "SBisoNsim": NSIM,
    "SBisoAmp": AMP,
    "SBisoPowDevMax": np.max(np.abs(pow_iso[1:] - 1)),
    "SBisoErr": err_pow,
    "SBisoXmax": np.max(np.abs(x_iso)),
    "SBmodPowZero": pow_th[0],
    "SBmodXzero": x_th[0],
})
