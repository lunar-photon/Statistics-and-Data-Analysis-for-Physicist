"""Library for part 10b: noise curves, stationary-phase waveforms and the gravitational-wave Fisher matrix.

SI units unless a name says otherwise; masses are passed in solar masses.

Noise curves (one-sided, sky-averaged sensitivities S_n(f) in 1/Hz)
  Sn_robson(f, tobs="4yr")    LISA, Robson, Cornish & Liu 2019, eqs. (13)+(14), confusion fit of their Table 1
  Sn_elisa(f)                 eLISA/NGO as used by Eda et al. 2015, eq. (32a)
  Sn_cf94(f)                  "advanced LIGO" analytic fit of Cutler & Flanagan 1994, eq. (4)

Waveform  h(f) = A f^{-7/6} exp(-i Psi(f))   (book convention: Fourier kernel exp(-2 pi i f t))
  psi_pn(f, Mc, eta, tc, phic)       restricted 1.5PN phase, Cutler & Flanagan eq. (42)
  dpsi_pn(f, Mc, eta)                its analytic derivatives with respect to ln Mc and ln eta
  u_of(f, Mc)                        the ppE frequency variable u = pi G Mc f / c^3

Inner products and Fisher matrices (integrals on a log-spaced grid: int g df = int g f dln f)
  integ(f, g)                        trapezoid in ln f
  snr2(f, amp2, Sn)                  rho^2 = 4 int |h|^2/S_n df
  fisher_from_dlogh(f, amp2, Sn, D)  Gamma_ij = 4 Re int |h|^2 D_i conj(D_j) / S_n df,  D_i = d ln h / d theta_i
  ppe_fisher(...)                    the six-parameter forecast (ln Mc, ln eta, tc, phic, ln A, beta) for one exponent b
"""
from __future__ import annotations

import numpy as np

G = 6.67430e-11
c = 2.99792458e8
MSUN = 1.98847e30
PC = 3.0856775814913673e16
MPC = 1e6 * PC
YEAR = 365.25 * 86400.0
TSUN = G * MSUN / c**3          # 4.925e-6 s


# ------------------------------------------------------------------ binaries
def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def sym_ratio(m1, m2):
    return m1 * m2 / (m1 + m2) ** 2


def f_isco(M):
    """GW frequency at r = 6GM/c^2 (M in Msun)."""
    return 1.0 / (6.0**1.5 * np.pi * M * TSUN)


def fdot_vac(f, Mc):
    """Newtonian chirp rate df/dt = (96/5) pi^{8/3} (G Mc/c^3)^{5/3} f^{11/3}."""
    return 96.0 / 5.0 * np.pi ** (8 / 3) * (Mc * TSUN) ** (5 / 3) * f ** (11 / 3)


def tau_of_f(f, Mc):
    """Newtonian time to coalescence."""
    return 5.0 / 256.0 * (Mc * TSUN) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


def f_of_tau(tau, Mc):
    return (256.0 / 5.0 * (Mc * TSUN) ** (5 / 3) * tau) ** (-3 / 8) / np.pi


def u_of(f, Mc):
    return np.pi * Mc * TSUN * f


def amp_avg(f, Mc, DL):
    """|h(f)| of the sky-, polarisation- and inclination-averaged SPA waveform (book T2a eq. hf, x sqrt(4/5))."""
    A = np.sqrt(5.0 / 24.0) * np.pi ** (-2 / 3) * (Mc * TSUN) ** (5 / 6) * c / DL
    return np.sqrt(4.0 / 5.0) * A * f ** (-7 / 6)


# ------------------------------------------------------------------ noise curves
L_LISA = 2.5e9
FSTAR = c / (2 * np.pi * L_LISA)
CONF = {  # Robson et al. Table 1: alpha, beta, kappa, gamma, f_k
    "6mo": (0.133, 243.0, 482.0, 917.0, 0.00258),
    "1yr": (0.171, 292.0, 1020.0, 1680.0, 0.00215),
    "2yr": (0.165, 299.0, 611.0, 1340.0, 0.00173),
    "4yr": (0.138, -221.0, 521.0, 1680.0, 0.00113),
}


def P_oms(f):
    return (1.5e-11) ** 2 * (1.0 + (2e-3 / f) ** 4)


def P_acc(f):
    return (3e-15) ** 2 * (1.0 + (4e-4 / f) ** 2) * (1.0 + (f / 8e-3) ** 4)


def Sn_instr(f):
    """Robson et al. eq. (13): (10/3L^2)[P_oms + 2(1+cos^2(f/f*)) P_acc/(2 pi f)^4][1 + 0.6 (f/f*)^2]."""
    return (10.0 / (3.0 * L_LISA**2) * (P_oms(f) + 2.0 * (1.0 + np.cos(f / FSTAR) ** 2) * P_acc(f) / (2 * np.pi * f) ** 4)
            * (1.0 + 0.6 * (f / FSTAR) ** 2))


def S_conf(f, tobs="4yr"):
    """Robson et al. eq. (14) with A = 9e-45."""
    a, b, k, g, fk = CONF[tobs]
    return 9e-45 * f ** (-7 / 3) * np.exp(-f**a + b * f * np.sin(k * f)) * (1.0 + np.tanh(g * (fk - f)))


def Sn_robson(f, tobs="4yr"):
    s = Sn_instr(f)
    return s if tobs is None else s + S_conf(f, tobs)


def Sn_elisa(f):
    """Eda et al. 2015 eq. (32a): eLISA/NGO, arm l = 1e9 m."""
    l = 1e9
    Sacc = 2.13e-29 * (1.0 + 1e-4 / f)
    Ssn, Somn = 6.28e-23, 5.25e-23
    return 20.0 / 3.0 * (4.0 * Sacc / (2 * np.pi * f) ** 4 + Ssn + Somn) / l**2 * (1.0 + (f / (0.41 * c / (2 * l))) ** 2)


def Sn_cf94(f):
    """Cutler & Flanagan 1994 eq. (4) (shape only; the scale S0 cancels once errors are normalised to an SNR)."""
    S0, f0 = 3e-48, 70.0
    return S0 * ((f0 / f) ** 4 + 2.0 * (1.0 + (f / f0) ** 2))


# ------------------------------------------------------------------ phase
def psi_pn(f, Mc, eta, tc=0.0, phic=0.0, order=1.5):
    """Restricted PN phase in the e^{-i Psi} convention:
    Psi = 2 pi f tc - phic - pi/4 + (3/128) u^{-5/3} [1 + (20/9)(743/336 + 11 eta/4) x - 16 pi x^{3/2}],
    u = pi G Mc f/c^3, x = (pi G M f/c^3)^{2/3} = u^{2/3} eta^{-2/5}."""
    u = u_of(f, Mc)
    x = u ** (2 / 3) * eta ** (-0.4)
    br = 1.0
    if order >= 1.0:
        br = br + 20.0 / 9.0 * (743.0 / 336.0 + 11.0 * eta / 4.0) * x
    if order >= 1.5:
        br = br - 16.0 * np.pi * x**1.5
    return 2 * np.pi * f * tc - phic - np.pi / 4 + 3.0 / 128.0 * u ** (-5 / 3) * br


def dpsi_pn(f, Mc, eta, order=1.5):
    """Analytic d Psi / d ln Mc and d Psi / d ln eta of psi_pn.
    u ~ Mc and x ~ Mc^{2/3} eta^{-2/5}, so each term u^{-5/3} x^k changes by (-5/3 + 2k/3) per unit ln Mc
    and by (-2k/5) per unit ln eta; the 1PN coefficient also contains eta explicitly."""
    u = u_of(f, Mc)
    x = u ** (2 / 3) * eta ** (-0.4)
    pre = 3.0 / 128.0 * u ** (-5 / 3)
    a1 = 20.0 / 9.0 * (743.0 / 336.0 + 11.0 * eta / 4.0)
    a15 = -16.0 * np.pi
    dM = -5.0 / 3.0 + 0 * f
    de = 0 * f
    if order >= 1.0:
        dM = dM + (-1.0) * a1 * x
        de = de + (-0.4) * a1 * x + 20.0 / 9.0 * 11.0 * eta / 4.0 * x
    if order >= 1.5:
        dM = dM + (-2.0 / 3.0) * a15 * x**1.5
        de = de + (-0.6) * a15 * x**1.5
    return pre * dM, pre * de


# ------------------------------------------------------------------ integrals
def integ(f, g):
    """int g(f) df on a log grid, as int g f dln f (trapezoid)."""
    lf = np.log(f)
    y = g * f
    return np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(lf), axis=-1)


def snr2(f, amp2, Sn):
    return 4.0 * integ(f, amp2 / Sn)


def fisher_from_dlogh(f, amp2, Sn, D):
    """D: array (P, Nf) of d ln h / d theta_i (complex). Returns the P x P Fisher matrix."""
    w = 4.0 * amp2 / Sn
    P = D.shape[0]
    F = np.empty((P, P))
    for i in range(P):
        for j in range(i, P):
            F[i, j] = F[j, i] = integ(f, w * np.real(D[i] * np.conj(D[j])))
    return F


def phase_derivs(f, Mc, eta, b=None, order=1.5):
    """d ln h/d theta for theta = (ln Mc, ln eta, tc, phic, ln A [, beta]) at beta = 0:
    d ln h = -i dPsi for phase parameters and 1 for ln A; beta enters Psi as beta u^b."""
    dM, de = dpsi_pn(f, Mc, eta, order)
    rows = [-1j * dM, -1j * de, -1j * 2 * np.pi * f, -1j * (-1.0) + 0 * f, 1.0 + 0j * f]
    if b is not None:
        rows.append(-1j * u_of(f, Mc) ** b)
    return np.array(rows)


NAMES6 = ["lnMc", "lneta", "tc", "phic", "lnA", "beta"]


def ppe_fisher(f, Mc, eta, DL, Sn, b, order=1.5):
    amp2 = amp_avg(f, Mc, DL) ** 2
    return fisher_from_dlogh(f, amp2, Sn, phase_derivs(f, Mc, eta, b, order))


def inv_scaled(F):
    """F^{-1} computed after rescaling to unit diagonal (the entries of a GW Fisher matrix span ~40 decades)."""
    d = np.sqrt(np.diag(F))
    w, V = np.linalg.eigh(F / np.outer(d, d))         # symmetric inverse through the eigenvectors
    return (V / w) @ V.T / np.outer(d, d)


def marg_err(F):
    return np.sqrt(np.diag(inv_scaled(F)))


def corr(F):
    C = inv_scaled(F)
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


# ------------------------------------------------------------------ environments: eps = P_env / P_GW
R6 = 1e-6 * PC


def r_of_f(f, M):
    """Kepler: orbital radius [m] at GW frequency f (orbital frequency f/2), total mass M [Msun]."""
    return (G * M * MSUN / (np.pi * f) ** 2) ** (1 / 3)


def eps_df(f, m1, m2, rho6, gam, xi=0.58, lnL=None):
    """Dynamical friction in a spike rho = rho6 (r6/r)^gam (rho6 in Msun/pc^3):
    eps = 5 pi c^5 xi lnL rho r^{11/2} / (8 G^{5/2} M^{3/2} m1^2)  (book T2a eq. eps)."""
    M = m1 + m2
    lnL = np.log(np.sqrt(m1 / m2)) if lnL is None else lnL
    r = r_of_f(f, M)
    rho = rho6 * MSUN / PC**3 * (R6 / r) ** gam
    return 5 * np.pi * c**5 * xi * lnL * rho * r**5.5 / (8 * G**2.5 * (M * MSUN) ** 1.5 * (m1 * MSUN) ** 2)


def eps_acc(f, m1, m2, rho6, gam, lnL=None):
    """Collisionless accretion: P_acc / P_DF = 4 v^2 / (c^2 lnL) with xi = 1 (book T2a)."""
    M = m1 + m2
    lnL = np.log(np.sqrt(m1 / m2)) if lnL is None else lnL
    v2 = G * M * MSUN / r_of_f(f, M)
    return eps_df(f, m1, m2, rho6, gam, xi=1.0, lnL=lnL) * 4 * v2 / (c**2 * lnL)


def eps_pull(f, m1, m2, rho6, gam):
    """Order of magnitude of the enclosed-mass correction: M_DM(<r)/M, M_DM = 4 pi rho6 r6^gam r^{3-gam}/(3-gam)."""
    M = m1 + m2
    r = r_of_f(f, M)
    mdm = 4 * np.pi * rho6 * MSUN / PC**3 * R6**gam * r ** (3 - gam) / (3 - gam)
    return mdm / (M * MSUN)


def eps_edd(f, m1, m2, fedd=1.0, tsal=4.5e7 * YEAR):
    """Central black hole accreting at fedd x Eddington (e-folding time tsal/fedd).  The companion's specific angular
    momentum sqrt(G M r) is conserved, so r ~ 1/M and f ~ M^2:  extra df/dt = 2 f Mdot/M,  eps = 2 f (fedd/tsal)/fdot_vac."""
    return 2 * f * (fedd / tsal) / fdot_vac(f, chirp_mass(m1, m2))


def eps_migI(f, m1, m2, fedd=0.1, alpha=0.1):
    """Type-I migration in a thin disc, Barausse et al. 2014 eq. (108) far from the inner edge:
    Ldot_migr/Ldot_GW = 1e-5 fedd^{2/5} (M/1e6)^{7/5} (alpha/0.1)^{-3/5} rtilde^{7/2},  rtilde = r c^2/(G M)."""
    M = m1 + m2
    rt = r_of_f(f, M) * c**2 / (G * M * MSUN)
    return 1e-5 * fedd**0.4 * (M / 1e6) ** 1.4 * (alpha / 0.1) ** (-0.6) * rt**3.5


def dpsi_powerlaw(f, Mc, eps, n):
    """Fourier-phase shift of a power-law eps = c f^{-n} << 1:  dPsi = -(2 pi f^2 / fdot_V) eps / ((n+8/3)(n+5/3))."""
    return -2 * np.pi * f**2 / fdot_vac(f, Mc) * eps / ((n + 8 / 3) * (n + 5 / 3))


def logfgrid(f1, f2, n=4000):
    return np.geomspace(f1, f2, n)


def tidy(nums):
    """common._fmt prints numbers between 1e4 and 1e5 as '2.0e+04'; store those as integers instead."""
    out = {}
    for k, v in nums.items():
        if not isinstance(v, str) and 1e4 <= abs(float(v)) < 1e5:
            v = int(round(float(v)))
        out[k] = v
    return out
