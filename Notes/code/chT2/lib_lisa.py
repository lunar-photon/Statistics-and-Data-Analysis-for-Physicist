"""Small library for chapter T2: Newtonian chirps, the stationary-phase waveform and the LISA noise.

Everything is in SI units unless a name says otherwise. Masses are passed in solar masses.

Contents
  constants                      G, c, MSUN, PC, MPC, YEAR, TSUN (= G MSUN / c^3 in seconds)
  chirp_mass(m1, m2)             (m1 m2)^{3/5} / (m1+m2)^{1/5}
  f_isco(M)                      GW frequency at the innermost stable circular orbit r = 6GM/c^2
  fdot_vac(f, Mc)                Newtonian chirp rate  df/dt = (96/5) pi^{8/3} (G Mc/c^3)^{5/3} f^{11/3}
  tau_of_f(f, Mc)                time left to coalescence at GW frequency f
  f_of_tau(tau, Mc)              inverse of tau_of_f
  cycles(f1, f2, Mc)             number of GW cycles between f1 and f2
  amp_spa(f, Mc, DL)             SPA amplitude A(f) of an optimally oriented source (Robson et al. eq. 20)
  psi_vac(f, Mc, tc, phic)       SPA phase  2 pi f tc - phic - pi/4 + (3/128)(pi G Mc f/c^3)^{-5/3}
  lisa_components(f)             the three pieces of the LISA sensitivity (Robson, Cornish & Liu 2019)
  Sn_lisa(f, confusion=True)     sky-averaged sensitivity S_n(f) [1/Hz], their eq. (13) + eq. (14)
  snr_sky_avg(f, A)              sky/inclination/polarisation averaged SNR, rho^2 = (16/5) int A^2/S_n df
"""
from __future__ import annotations

import numpy as np

G = 6.67430e-11
c = 2.99792458e8
MSUN = 1.98847e30
PC = 3.0856775814913673e16
MPC = 1e6 * PC
YEAR = 365.25 * 86400.0
TSUN = G * MSUN / c**3          # 4.925e-6 s: the Sun's mass expressed as a time


# ---------------------------------------------------------------- Newtonian chirp
def chirp_mass(m1, m2):
    """Chirp mass in solar masses."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def f_isco(M):
    """GW frequency (twice the orbital one) at r = 6 G M / c^2, M in solar masses."""
    return 1.0 / (6.0**1.5 * np.pi * M * TSUN)


def fdot_vac(f, Mc):
    """Newtonian chirp rate df/dt [Hz/s] from energy balance (quadrupole power)."""
    return 96.0 / 5.0 * np.pi ** (8 / 3) * (Mc * TSUN) ** (5 / 3) * f ** (11 / 3)


def tau_of_f(f, Mc):
    """Time to coalescence [s] when the GW frequency is f (integral of 1/fdot from f to infinity)."""
    return 5.0 / 256.0 * (Mc * TSUN) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


def f_of_tau(tau, Mc):
    """GW frequency [Hz] a time tau before coalescence."""
    return (1.0 / (8.0 * np.pi * Mc * TSUN)) * (5.0 * Mc * TSUN / tau) ** (3 / 8)


def cycles(f1, f2, Mc):
    """Number of GW cycles N = int f dt between f1 < f2."""
    k = 1.0 / (32.0 * np.pi ** (8 / 3)) * (Mc * TSUN) ** (-5 / 3)
    return k * (f1 ** (-5 / 3) - f2 ** (-5 / 3))


# ---------------------------------------------------------------- stationary-phase waveform
def amp_spa(f, Mc, DL):
    """SPA amplitude of h_+ for a face-on source [1/Hz]; DL in metres.

    A(f) = sqrt(5/24) pi^{-2/3} (G Mc/c^3)^{5/6} f^{-7/6} / (DL/c).
    """
    return np.sqrt(5.0 / 24.0) * np.pi ** (-2 / 3) * (Mc * TSUN) ** (5 / 6) * f ** (-7 / 6) / (DL / c)


def psi_vac(f, Mc, tc=0.0, phic=0.0):
    """Newtonian SPA phase Psi(f)."""
    return 2 * np.pi * f * tc - phic - np.pi / 4 + 3.0 / 128.0 * (np.pi * Mc * TSUN * f) ** (-5 / 3)


# ---------------------------------------------------------------- LISA (Robson, Cornish & Liu 2019)
L_ARM = 2.5e9                     # arm length [m]
F_STAR = c / (2 * np.pi * L_ARM)  # transfer frequency, 19.09 mHz
CONF_4YR = dict(A=9e-45, alpha=0.138, beta=-221.0, kappa=521.0, gamma=1680.0, fk=0.00113)


def P_oms(f):
    """Single-link optical metrology noise [m^2/Hz] (their eq. 10)."""
    return (1.5e-11) ** 2 * (1 + (2e-3 / f) ** 4)


def P_acc(f):
    """Single test-mass acceleration noise [m^2 s^-4 / Hz] (their eq. 11)."""
    return (3e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)


def S_conf(f, p=CONF_4YR):
    """Galactic confusion noise [1/Hz] (their eq. 14, 4-year fit)."""
    return (p["A"] * f ** (-7 / 3) * np.exp(-f ** p["alpha"] + p["beta"] * f * np.sin(p["kappa"] * f))
            * (1 + np.tanh(p["gamma"] * (p["fk"] - f))))


def lisa_components(f):
    """The three contributions to S_n(f) [1/Hz]: metrology, acceleration, confusion."""
    resp = 10.0 / (3.0 * L_ARM**2) * (1 + 0.6 * (f / F_STAR) ** 2)
    oms = resp * P_oms(f)
    acc = resp * 2 * (1 + np.cos(f / F_STAR) ** 2) * P_acc(f) / (2 * np.pi * f) ** 4
    return oms, acc, S_conf(f)


def Sn_lisa(f, confusion=True):
    """Sky-averaged LISA S_n(f) [1/Hz]: instrument (their eq. 13) + confusion (eq. 14)."""
    oms, acc, conf = lisa_components(f)
    return oms + acc + (conf if confusion else 0.0)


def snr_sky_avg(f, A, Sn=None):
    """Angle-averaged SNR: rho^2 = 4 int (4/5) A^2 / S_n df on the grid f (trapezoid rule)."""
    Sn = Sn_lisa(f) if Sn is None else Sn
    return np.sqrt(np.trapezoid(16.0 / 5.0 * A**2 / Sn, f))
