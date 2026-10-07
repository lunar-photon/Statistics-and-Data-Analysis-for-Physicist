"""lib_biref10.py -- forecasting and simulating a measurement of the rotation angle beta (part 10c).

  fiducial(lmax)                  lensed CAMB spectra TT, EE, BB, TE (muK^2), EB = TB = 0
  white(depth, fwhm, lmax)        beam-deconvolved white noise N_l / B_l^2 (muK^2)
  planck_like(lmax)               inverse-variance HFI 100+143+217 noise (N_T, N_P), from lib_fisher10
  dust(nu, lmax)                  Galactic dust EE, BB in CMB units at frequency nu (T1c model)
  info_beta(...)                  Fisher information on beta per multipole, closed form (EB, TB, joint)
  cov_eb(...)                     the 2N x 2N covariance of (E_1..E_N, B_1..B_N) at each l for N
                                  channels whose maps are rotated by alpha_i (+ beta for the CMB)
  fisher_numeric(...)             F_ij = sum_l (2l+1) fsky / 2 Tr[C^-1 C_,i C^-1 C_,j] by central
                                  differences in the angles
  rotate(E, B, angle)             (E, B) -> rotated by 2 angle, mode by mode

Angles are in radians inside the code; tables quote degrees.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from camb_fiducial import load_fiducial  # noqa: E402
import lib_fisher10 as LF  # noqa: E402

DEG = np.pi / 180.0
ARCMIN = DEG / 60.0

# Galactic dust: Planck 2018 XI power-law fit over 71% of the sky (the model of code/chT1/09):
# D_l^EE = 315.4 muK^2 at l = 80 and 353 GHz, D_l ~ l^-0.42, BB/EE = 0.53,
# modified black body beta_d = 1.53, T_d = 19.6 K.
A_EE353, ALPHA_L, BB_EE = 315.4, -0.42, 0.53
BETA_D, T_D, T_CMB, H_OVER_K = 1.53, 19.6, 2.7255, 0.0479924

# Planck 2018 I, Table 4: FWHM [arcmin], temperature and polarisation noise [muK deg]
PLANCK = {100: (9.66, 1.29, 1.96), 143: (7.22, 0.55, 1.17), 217: (4.90, 0.78, 1.75), 353: (4.92, 2.56, 7.31)}


def fiducial(lmax=3000):
    out = {k: load_fiducial(k)[1][: lmax + 1].astype(float) for k in ("TT", "EE", "BB", "TE")}
    return out


def white(depth, fwhm, lmax=3000):
    return LF.white_noise(depth, fwhm, lmax)


def planck_like(lmax=3000):
    return LF.noise_planck_like(lmax)


def sed(nu):
    """Dust amplitude in CMB temperature units, relative to 353 GHz."""
    def f(n):
        x = H_OVER_K * n / T_CMB
        rj = n ** (BETA_D + 1) / np.expm1(H_OVER_K * n / T_D)
        return rj * np.expm1(x) ** 2 / (x ** 2 * np.exp(x))
    return f(nu) / f(353.0)


def dust(nu, lmax=3000, amp=1.0):
    ell = np.arange(lmax + 1, dtype=float)
    ee = np.zeros(lmax + 1)
    l = ell[2:]
    ee[2:] = amp * A_EE353 * sed(nu) ** 2 * (l / 80.0) ** ALPHA_L * 2 * np.pi / (l * (l + 1))
    return ee, BB_EE * ee


def info_beta(cl, NT, NP, lmin, lmax, fsky=1.0, which="joint"):
    """Fisher information on beta from each multipole, at beta = 0 (closed form of the 3x3 trace).

    At each l the a_lm (T, E, B) have covariance [[TT~, TE, 0], [TE, EE~, 0], [0, 0, BB~]] (~ = + noise);
    beta enters only through EB = 2 beta (EE - BB) and TB = 2 beta TE to first order, so with
    v = (2 TE, 2 (EE - BB)) and A the (T, E) block,  F_l = (2l+1) fsky v^T A^-1 v / BB~.
    """
    l = np.arange(lmin, lmax + 1)
    tt, ee, bb, te = (cl[k][l] for k in ("TT", "EE", "BB", "TE"))
    T, E, B = tt + NT[l], ee + NP[l], bb + NP[l]
    b, e = 2 * te, 2 * (ee - bb)
    n = (2 * l + 1) * fsky
    if which == "EB":
        return l, n * e ** 2 / (E * B)
    if which == "TB":
        return l, n * b ** 2 / (T * B)
    with np.errstate(over="ignore", invalid="ignore"):
        det = T * E - te ** 2
    return l, n * (b ** 2 * E - 2 * b * e * te + e ** 2 * T) / det / B


def sigma_beta(cl, NT, NP, lmin, lmax, fsky=1.0, which="joint"):
    _, f = info_beta(cl, NT, NP, lmin, lmax, fsky, which)
    return 1.0 / np.sqrt(f.sum())


def rot(a):
    c, s = np.cos(2 * a), np.sin(2 * a)
    return np.array([[c, -s], [s, c]])


def cov_eb(l, cmb, fg, noise, alphas, beta):
    """Covariance of (E_1, B_1, E_2, B_2, ...) at the multipoles l for N channels.

    cmb = (EE, BB) arrays over l; fg = list of (EE, BB) per channel (fully correlated between
    channels: one dust pattern times an SED, so the cross-spectrum is sqrt(EE_i EE_j));
    noise = list of N_P arrays per channel (independent).  Channel i is rotated by alpha_i,
    the CMB additionally by beta.
    """
    N = len(alphas)
    L = len(l)
    C = np.zeros((L, 2 * N, 2 * N))
    Sc = np.zeros((L, 2, 2)); Sc[:, 0, 0], Sc[:, 1, 1] = cmb
    for i in range(N):
        for j in range(N):
            Ri, Rj = rot(alphas[i] + beta), rot(alphas[j] + beta)
            blk = np.einsum("ab,lbc,dc->lad", Ri, Sc, Rj)
            if fg is not None:
                Sd = np.zeros((L, 2, 2))
                Sd[:, 0, 0] = np.sqrt(fg[i][0] * fg[j][0])
                Sd[:, 1, 1] = np.sqrt(fg[i][1] * fg[j][1])
                Ri, Rj = rot(alphas[i]), rot(alphas[j])
                blk = blk + np.einsum("ab,lbc,dc->lad", Ri, Sd, Rj)
            if i == j:
                blk[:, 0, 0] += noise[i]
                blk[:, 1, 1] += noise[i]
            C[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2] = blk
    return C


def fisher_numeric(covfun, p0, l, fsky, h=1e-4):
    """F_ij = sum_l (2l+1) fsky / 2 Tr[C^-1 dC_i C^-1 dC_j], derivatives by central differences."""
    p0 = np.asarray(p0, float)
    C0 = covfun(p0)
    Ci = np.linalg.inv(C0)
    dC = []
    for k in range(len(p0)):
        e = np.zeros_like(p0); e[k] = h
        dC.append((covfun(p0 + e) - covfun(p0 - e)) / (2 * h))
    A = [np.einsum("lab,lbc->lac", Ci, d) for d in dC]
    n = (2 * l + 1) * fsky
    F = np.zeros((len(p0), len(p0)))
    for i in range(len(p0)):
        for j in range(i, len(p0)):
            F[i, j] = F[j, i] = 0.5 * np.sum(n * np.einsum("lab,lba->l", A[i], A[j]))
    return F


def rotate(E, B, angle):
    """(Q + iU) -> e^{2 i angle} (Q + iU), written for E and B coefficients."""
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    return c * E - s * B, s * E + c * B


def corr_alm(cl, lmax, rng):
    """Gaussian T, E, B a_lm (healpy layout) with spectra TT, EE, BB and cross TE, and EB = TB = 0.

    a_T = sqrt(TT) z1,  a_E = (TE / sqrt(TT)) z1 + sqrt(EE - TE^2 / TT) z2,  a_B = sqrt(BB) z3,
    with z unit complex normals (real for m = 0): then <a_T a_E*> = TE and <|a_E|^2> = EE.
    """
    import healpy as hp
    ell, m = hp.Alm.getlm(lmax)
    TT, EE, BB, TE = (np.asarray(cl[k], float)[: lmax + 1] for k in ("TT", "EE", "BB", "TE"))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(TT > 0, TE / np.sqrt(TT), 0.0)
        y = np.sqrt(np.clip(EE - x ** 2, 0, None))
    z = []
    for _ in range(3):
        re, im = rng.standard_normal(ell.size), rng.standard_normal(ell.size)
        z.append(np.where(m == 0, re, (re + 1j * im) / np.sqrt(2.0)).astype(complex))
    return np.sqrt(TT)[ell] * z[0], x[ell] * z[0] + y[ell] * z[1], np.sqrt(BB)[ell] * z[2]
