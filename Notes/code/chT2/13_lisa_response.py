"""How strongly does a two-arm detector respond to a wave, on average and at high frequency?

Question: where do the numbers 4/5, 1/5, 3/20 and 3/10 of the sensitivity-curve literature come
from, and why does LISA's response fall above the transfer frequency f* = c/(2 pi L)?
Computes: (1) by quadrature over the sky and the polarisation angle, the averages
<F_+^2>, <F_x^2>, <F_+ F_x> for a right-angle and a 60-degree Michelson, the inclination average
of a circular binary, and the angular integral of its energy flux; (2) a sky map of
F_+^2 + F_x^2 for a right-angle detector; (3) the frequency-dependent response of a 60-degree
Michelson with round-trip arms, built from the light-travel-time integral along each arm
(one-way transfer factor sinc[(f/2f*)(1 -+ k.a)]), averaged over sky and polarisation, against
the fit R(f) = (3/10)/(1 + 0.6 (f/f*)^2) of Robson, Cornish & Liu (2019).
Writes: figures/chT2/antenna_pattern.pdf, figures/chT2/lisa_response.pdf,
        results/chT2/13_lisa_response.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, INK, theory_line
from lib_lisa import F_STAR, L_ARM, c

# ---------------------------------------------------------------- sky and polarisation grid
# Gauss-Legendre in cos(theta) and uniform grids in phi and psi integrate the trigonometric
# polynomials below exactly, so the low-frequency averages come out to machine precision.
nx, nphi, npsi = 32, 64, 16
x_gl, w_gl = np.polynomial.legendre.leggauss(nx)          # x = cos(theta), weights sum to 2
phi = 2 * np.pi * np.arange(nphi) / nphi
psi = np.pi * np.arange(npsi) / npsi                      # psi and psi + pi give the same wave
X, PHI, PSI = np.meshgrid(x_gl, phi, psi, indexing="ij")
W = np.broadcast_to(w_gl[:, None, None] / 2, X.shape) / (nphi * npsi)   # weights sum to 1
S = np.sqrt(1 - X**2)

n_src = np.stack([S * np.cos(PHI), S * np.sin(PHI), X], axis=-1)        # direction to the source
k_hat = -n_src                                                          # direction of travel
p0 = np.stack([X * np.cos(PHI), X * np.sin(PHI), -S], axis=-1)          # theta-hat
q0 = np.stack([-np.sin(PHI), np.cos(PHI), np.zeros_like(PHI)], axis=-1)  # phi-hat
cp, sp = np.cos(PSI)[..., None], np.sin(PSI)[..., None]
p = cp * p0 + sp * q0                                                   # polarisation axes
q = -sp * p0 + cp * q0


def proj(a, b, arm):
    """(arm . a)(arm . b) on the grid."""
    return (a @ arm) * (b @ arm)


def pattern(u, v):
    """Long-wavelength antenna patterns of a Michelson with unit arm vectors u, v."""
    Fp = 0.5 * ((proj(p, p, u) - proj(q, q, u)) - (proj(p, p, v) - proj(q, q, v)))
    Fx = 0.5 * ((2 * proj(p, q, u)) - (2 * proj(p, q, v)))
    return Fp, Fx


avg = lambda Z: float(np.sum(W * Z))

u = np.array([1.0, 0.0, 0.0])
v90 = np.array([0.0, 1.0, 0.0])
v60 = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0])
Fp90, Fx90 = pattern(u, v90)
Fp60, Fx60 = pattern(u, v60)

# inclination average of a circular binary and the angular integral of its flux
xi = x_gl
incl = 0.5 * np.sum(w_gl * ((1 + xi**2) ** 2 / 4 + xi**2))
flux_int = 2 * np.pi * np.sum(w_gl * ((1 + xi**2) ** 2 / 4 + xi**2))     # should be 16 pi / 5
print(f"<(1+x^2)^2/4 + x^2> = {incl:.6f} (4/5),  int dOmega = {flux_int / np.pi:.6f} pi (16/5 pi)")
print(f"90 deg: <F+^2>={avg(Fp90**2):.6f} <Fx^2>={avg(Fx90**2):.6f} <F+Fx>={avg(Fp90 * Fx90):.1e}")
print(f"60 deg: <F+^2>={avg(Fp60**2):.6f} <Fx^2>={avg(Fx60**2):.6f}")

# ---------------------------------------------------------------- figure 1: the sky map
setup(6.0, 3.0)
th_m = np.arccos(np.linspace(1, -1, 181))   # uniform in cos(theta), so the map has even rows
ph_m = np.linspace(0, 2 * np.pi, 361)
TH, PH = np.meshgrid(th_m, ph_m, indexing="ij")
Ffull = 0.25 * (1 + np.cos(TH) ** 2) ** 2 * np.cos(2 * PH) ** 2 + np.cos(TH) ** 2 * np.sin(2 * PH) ** 2
fig, ax = plt.subplots()
im = ax.pcolormesh(np.degrees(PH), np.cos(TH), Ffull, shading="auto", cmap="Blues", vmin=0, vmax=1, rasterized=True)
ax.set_xlabel(r"azimuth $\phi$ of the source [deg] (arms along $0^\circ$ and $90^\circ$)")
ax.set_ylabel(r"$\cos\theta$")
ax.set_xticks([0, 90, 180, 270, 360])
ax.grid(False)
cb = fig.colorbar(im, ax=ax)
cb.set_label(r"$F_+^2+F_\times^2$")
savefig(fig, "chT2", "antenna_pattern")

# ---------------------------------------------------------------- frequency-dependent response
# Units: one-way light time T = L/c = 1, so f/f* = 2 pi f T = 2 pi f.


def sinc(z):
    """sin(z)/z (numpy's sinc is sin(pi z)/(pi z))."""
    return np.sinc(z / np.pi)


def leg(fr, kdota, start, sign):
    """Integral of exp(2 pi i f [(1 - sign k.a) t' + const]) over one leg of length T = 1.

    Outgoing leg (sign=+1): t' in [-2, -1], const = -2 k.a ; returning leg (sign=-1): t' in [-1, 0].
    Closed form: int_A^{A+1} e^{i kappa t} dt = e^{i kappa (A + 1/2)} sinc(kappa/2).
    """
    alpha = 1 - sign * kdota
    kappa = 2 * np.pi * fr * alpha
    const = -2 * kdota if sign > 0 else 0.0
    return np.exp(1j * kappa * (start + 0.5) + 2j * np.pi * fr * const) * sinc(kappa / 2)


def arm_transfer(fr, arm):
    """Round-trip time perturbation of one arm divided by its low-frequency value."""
    kda = k_hat @ arm
    return (leg(fr, kda, -2.0, +1) + leg(fr, kda, -1.0, -1)) / 2.0


x_f = np.logspace(-2, 1.6, 120)                 # f / f*
R_X = np.empty_like(x_f)
hpu, hpv = proj(p, p, u) - proj(q, q, u), proj(p, p, v60) - proj(q, q, v60)   # e+ along each arm
for i, xf in enumerate(x_f):
    fr = xf / (2 * np.pi)
    Fp_f = 0.5 * (hpu * arm_transfer(fr, u) - hpv * arm_transfer(fr, v60))
    R_X[i] = avg(np.abs(Fp_f) ** 2)
R2 = 2 * R_X                                      # two low-frequency channels
fit = 0.3 / (1 + 0.6 * x_f**2)
normal = 0.3 * sinc(x_f) ** 2                     # wave arriving perpendicular to both arms

setup(6.0, 3.6)
fig, ax = plt.subplots()
ax.loglog(x_f, R2, color=SERIES[0], lw=2.0, label=r"$2\times$ one $60^\circ$ Michelson, sky average")
theory_line(ax, x_f, fit, label=r"fit $0.3/[1+0.6(f/f_*)^2]$")
ax.loglog(x_f, normal, color=SERIES[1], lw=1.0, label=r"perpendicular wave, $0.3\,\mathrm{sinc}^2$")
ax.axvline(1.0, color="0.5", lw=0.8, ls=":")
ax.text(1.08, 2e-4, r"$f_*$", color="0.35")
ax.set_ylim(1e-4, 0.6)
ax.set_xlabel(r"$f/f_*$")
ax.set_ylabel(r"response $R(f)$")
ax.legend(loc="lower left")
savefig(fig, "chT2", "lisa_response")

R_low = R2[0]
ratio_star = float(np.exp(np.interp(0.0, np.log(x_f), np.log(R2)))) / R_low
hi = x_f > 8
slope = float(np.polyfit(np.log(x_f[hi]), np.log(R2[hi]), 1)[0])
maxdev = float(np.max(np.abs(R2 / fit - 1)))
print(f"R(0)={R_low:.4f}  R(f*)/R(0)={ratio_star:.3f} (fit 0.625)  slope above 8 f* = {slope:.2f}  max|R/fit-1| = {maxdev:.2f}")

save_numbers("chT2", "13_lisa_response", {
    "TwInclAvg": incl,
    "TwFluxInt": flux_int / np.pi,
    "TwFpNinety": avg(Fp90**2),
    "TwFxNinety": avg(Fx90**2),
    "TwFpSixty": avg(Fp60**2),
    "TwRzero": R_low,
    "TwRstarRatio": ratio_star,
    "TwRslope": slope,
    "TwRfitDev": 100 * maxdev,
    "TwfZero": c / (2 * L_ARM) * 1e3,
})
