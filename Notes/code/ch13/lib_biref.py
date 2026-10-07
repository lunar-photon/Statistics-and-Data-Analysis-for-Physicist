"""lib_biref.py -- polarization pseudo-spectra and the rotation-angle estimator (part 13e).

Shared by 41_pol_data.py ... 45_bmodes.py:

    import lib_biref as lb
    pol = lb.load_pol()                      # degraded SMICA Q, U of hm1, hm2 (muK, RING) + masks
    E, B = lb.rotate(E, B, beta)             # rotate the polarization plane by beta (radians)
    s = lb.cross_spectra(h1, h2)             # binned EE, BB, EB, TE, TB of two halves
    b, sig = lb.estimate_beta(s, Cinv)       # linear estimator of the angle (radians)

Conventions: HEALPix RING, Galactic, Planck's (COSMO) polarization convention as in the files;
E and B a_lm from healpy's spin-2 transform; binned quantities are averages of
D_l = l(l+1) C_l / 2 pi over Delta l = 30 bins (50 <= l < 1490).
"""
from __future__ import annotations

import pathlib
import numpy as np
import healpy as hp

HERE = pathlib.Path(__file__).resolve().parent
NOTES = HERE.parents[1]
DATA = NOTES / "data" / "ch13"
NSIDE = 512
LS = 3 * NSIDE - 1                  # band limit of the degraded maps
EDGES = np.arange(50, 1491, 30)     # 48 bins, [50,80), ..., [1460,1490)
NB = len(EDGES) - 1
DEG = np.pi / 180.0


def load_pol(nside: int = NSIDE):
    z = np.load(DATA / f"smica_pol_n{nside}.npz")
    return {k: z[k].astype(np.float64) for k in z.files}


def binning_matrix(lmax: int = LS):
    """P (NB x lmax+1): the plain average of D_l over each bin."""
    P = np.zeros((NB, lmax + 1))
    for b in range(NB):
        ls = np.arange(EDGES[b], EDGES[b + 1])
        P[b, ls] = ls * (ls + 1) / (2 * np.pi) / ls.size
    return P


PBIN = binning_matrix()
LEFF = np.array([0.5 * (EDGES[b] + EDGES[b + 1] - 1) for b in range(NB)])


def binned(cl):
    return PBIN @ np.asarray(cl)[: LS + 1]


def rotate(E, B, beta):
    """(Q + iU) -> e^{2i beta} (Q + iU) at every pixel, written for the E and B coefficients."""
    c, s = np.cos(2 * beta), np.sin(2 * beta)
    return c * E - s * B, s * E + c * B


def xcl(a, b):
    """Symmetrised cross-spectrum 1/2 (a1 b2* + b1 a2*) given pairs a=(a1,a2), b=(b1,b2)."""
    return 0.5 * (hp.alm2cl(a[0], b[1]) + hp.alm2cl(b[0], a[1]))


def cross_spectra(h1, h2):
    """h = (T, E, B) pseudo-a_lm of one half.  Binned cross-spectra of the two halves."""
    T, E, B = zip(h1, h2)
    return {"EE": binned(hp.alm2cl(E[0], E[1])), "BB": binned(hp.alm2cl(B[0], B[1])),
            "EB": binned(xcl(E, B)), "TE": binned(xcl(T, E)), "TB": binned(xcl(T, B))}


def vectors(s, which="joint"):
    """Data vector e (the parity-odd spectra) and template D (what they become under rotation)."""
    if which == "EB":
        return s["EB"], s["EE"] - s["BB"]
    if which == "TB":
        return s["TB"], s["TE"]
    return np.concatenate([s["EB"], s["TB"]]), np.concatenate([s["EE"] - s["BB"], s["TE"]])


def estimate_beta(s, Cinv, which="joint"):
    """Least squares for e = 2 beta D + noise:  beta = D C^-1 e / (2 D C^-1 D),  sigma = 1/(2 sqrt(D C^-1 D))."""
    e, D = vectors(s, which)
    F = D @ Cinv @ D
    return (D @ Cinv @ e) / (2 * F), 1.0 / (2 * np.sqrt(F))


def correlated_alm(cls, lmax, rng):
    """Gaussian T, E, B a_lm with spectra TT, EE, BB and cross TE (EB = TB = 0).

    a_T = sqrt(TT) z1,  a_E = (TE/sqrt(TT)) z1 + sqrt(EE - TE^2/TT) z2,  a_B = sqrt(BB) z3,
    with z unit complex normals (real for m = 0), the layout of lib_cmbsim.synalm.
    """
    ell, m = hp.Alm.getlm(lmax)
    TT, EE, BB, TE = (np.asarray(cls[k], float)[: lmax + 1] for k in ("TT", "EE", "BB", "TE"))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(TT > 0, TE / np.sqrt(TT), 0.0)
        y = np.sqrt(np.clip(EE - x ** 2, 0, None))
    z = []
    for _ in range(3):
        re, im = rng.standard_normal(ell.size), rng.standard_normal(ell.size)
        z.append(np.where(m == 0, re, (re + 1j * im) / np.sqrt(2.0)).astype(complex))
    aT = np.sqrt(TT)[ell] * z[0]
    aE = x[ell] * z[0] + y[ell] * z[1]
    aB = np.sqrt(BB)[ell] * z[2]
    return aT, aE, aB
