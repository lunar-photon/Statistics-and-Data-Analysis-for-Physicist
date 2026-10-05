"""How dense is dark matter where an EMRI orbits?

Question: start from an NFW halo of mass M200 = 1e8 Msun with a 1e4 Msun black hole at its
centre (the 1e4 + 1 Msun binary of Kim et al. 2023). How does adiabatic growth of the hole turn the
r^-1 cusp into a spike r^-7/3, how does an ultralight field change the profile inside the
self-gravity radius, and what density does the inspiralling companion actually feel?
Computes:
  * the NFW parameters from the concentration-mass relation (Correa et al. 2015 eq. 19, NFW 1997 eq. 2),
  * the spike radius r_sp from continuity at r_sp and  M(< 5 r_sp) = 2 M  (Eda et al. 2015 eq. 4-5),
  * inside r_sg = 1e5 r_h (virial estimate 4.5e4 r_h) the steady-infall profile rho ~ r^-3/2 (Hui et al. 2019, Hancock & Witek 2025),
  * the density on the EMRI orbit during the last 4 years, and the wave/particle thresholds.
Writes: figures/chT2/dm_profile.pdf, results/chT2/04_dm_profile.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from common import setup, savefig, save_numbers, SERIES, INK
from lib_lisa import G, c, MSUN, PC, YEAR, chirp_mass, f_isco, tau_of_f, f_of_tau

setup(6.0, 3.9)
M, m2, M200, h, z = 1e4, 1.0, 1e8, 0.674, 0.0
rh = 2 * G * M * MSUN / c**2 / PC                       # Schwarzschild radius [pc]
rho_crit = 2.775e11 * h**2 / 1e18                       # [Msun / pc^3]

# --- NFW from the concentration-mass relation
a0 = 1.62774 - 0.2458 * (1 + z) + 0.01716 * (1 + z) ** 2
a1 = 1.66079 + 0.00359 * (1 + z) - 1.6901 * (1 + z) ** 0.00417
a2 = -0.02049 + 0.0253 * (1 + z) ** (-0.1044)
lm = np.log10(M200)
conc = 10 ** (a0 + a1 * lm * (1 + a2 * lm**2))
r200 = (3 * M200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)
rs = r200 / conc
delta_c = 200 / 3 * conc**3 / (np.log(1 + conc) - conc / (1 + conc))
rho_s = rho_crit * delta_c
nfw = lambda r: rho_s / ((r / rs) * (1 + r / rs) ** 2)
# check: the mean density inside r200 is 200 rho_crit
M_nfw = lambda r: 4 * np.pi * rho_s * rs**3 * (np.log(1 + r / rs) - (r / rs) / (1 + r / rs))
assert abs(M_nfw(r200) / M200 - 1) < 1e-10

# --- spike: gamma_sp = (9 - 2 gamma_i)/(4 - gamma_i) with gamma_i = 1
gsp = 7 / 3
rmin = 3 * rh
def rho_spike_nfw(r, rsp):
    return np.where(r < rsp, nfw(rsp) * (rsp / r) ** gsp, nfw(r))
def mass_condition(rsp):
    m_in = 4*np.pi * nfw(rsp) * rsp**gsp * (rsp**(3-gsp) - rmin**(3-gsp)) / (3-gsp)
    return m_in + (M_nfw(5 * rsp) - M_nfw(rsp)) - 2 * M
rsp = brentq(mass_condition, 1e-4, 1e3)
rho_sp = nfw(rsp)

# --- inside the self-gravity radius: steady infall, rho ~ r^-3/2, matched at r_sg
v_disp = 1e6                                     # 1000 km/s
rsg_virial = 0.5 * (c / v_disp) ** 2 * rh        # virial theorem: v^2 = GM/r_sg, i.e. r_sg = (r_h/2)(c/v)^2
rsg = 1e5 * rh                                   # Hui et al. 2019: r_h/r_sg = (v/c)^2, about 1e5
rho_sg = rho_sp * (rsp / rsg) ** gsp
def rho_full(r):
    r = np.asarray(r, float)
    out = rho_spike_nfw(r, rsp)
    inner = r < rsg
    out[inner] = rho_sg * (rsg / r[inner]) ** 1.5
    out[r < rmin] = 0.0
    return out

# --- where the EMRI orbits during the last 4 years
Mc = chirp_mass(M, m2)
fi = f_isco(M + m2)
f4 = f_of_tau(tau_of_f(fi, Mc) + 4 * YEAR, Mc)
r_of_f = lambda f: (G * (M + m2) * MSUN / (np.pi**2 * f**2)) ** (1 / 3) / PC
r_orb4, r_orb_isco = r_of_f(f4), r_of_f(fi)
rho_orb = rho_full(np.array([r_orb4]))[0]
rho_local = 0.01                                  # solar neighbourhood, Msun/pc^3

# --- wave/particle thresholds for v = 1000 km/s (hbar c = 1.9733e-7 eV m)
hbarc = 1.97327e-7
rh_m = rh * PC
mu_particle = hbarc / ((v_disp / c) * rh_m)       # mu v r_h / hbar = 1
mu_compton = hbarc / rh_m                         # mu c r_h / hbar = 1

# --- figure
r = np.logspace(np.log10(rmin) - 0.3, np.log10(r200), 3000)
fig, ax = plt.subplots()
ax.loglog(r / rh, nfw(r), color="0.6", ls="--", lw=1.2, label="NFW without the black hole")
inside = r >= rmin                     # rho = 0 below r_min: leave it unplotted (log of zero)
ax.loglog(r[inside] / rh, rho_spike_nfw(r[inside], rsp), color=SERIES[1], ls=":", lw=1.4,
          label=r"particle spike $\rho\propto r^{-7/3}$ continued inward")
ax.loglog(r[inside] / rh, rho_full(r[inside]), color=SERIES[0], lw=2.0, label=r"with infall core $\rho\propto r^{-3/2}$ inside $r_{\rm sg}$")
ax.axvspan(r_orb_isco / rh, r_orb4 / rh, color=SERIES[2], alpha=0.25, lw=0)
ax.axhline(rho_local, color="0.5", lw=0.8)
ax.text(1e2, rho_local * 4, "solar neighbourhood", fontsize=8, color="0.35")
for x, lab in [(rmin, r"$r_{\rm min}$"), (rsg, r"$r_{\rm sg}$"), (rsp, r"$r_{\rm sp}$"), (rs, r"$r_s$")]:
    ax.axvline(x / rh, color="0.5", lw=0.6, ls=":")
    ax.text(x / rh * 1.3, 1e14, lab, fontsize=9, color="0.25")
ax.text(r_orb4 / rh * 1.5, 1e-5 * 3, "EMRI orbit,\nlast 4 yr", fontsize=8, color=SERIES[2])
ax.set_xlabel(r"$r/r_h$  ($r_h=2GM/c^2$)")
ax.set_ylabel(r"$\rho$ [$M_\odot\,{\rm pc}^{-3}$]")
ax.set_ylim(1e-6, 1e21)
ax.legend(loc="upper right", fontsize=8)
savefig(fig, "chT2", "dm_profile")

nums = {"conc": conc, "rtwohundred": r200 / 1e3, "rsNFW": rs / 1e3, "deltac": rf"{delta_c/1e4:.2f}\times10^{{4}}", "rhos": rho_s,
        "rsp": rsp, "rhosp": rho_sp, "rhpc": rh, "rsgOverRh": rf"{rsg_virial/rh/1e4:.1f}\times10^{{4}}", "rhosg": rho_sg,
        "rorbFour": r_orb4 / rh, "rorbIsco": r_orb_isco / rh, "rhoOrb": rho_orb,
        "rhoBoost": rho_orb / rho_local, "muParticle": mu_particle, "muCompton": mu_compton}
for k, v in nums.items():
    print(k, v)
save_numbers("chT2", "04_dm_profile", {"Tw" + k: v for k, v in nums.items()})   # "Tw" = chapter T2 prefix
