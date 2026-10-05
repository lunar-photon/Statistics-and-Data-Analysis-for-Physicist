"""lib_harmonic.py -- spherical-harmonic coefficients a_lm written out by hand.

Question: how do we draw the a_lm of a statistically isotropic Gaussian sky, and
how do we turn them back into the full-sky estimate C_hat_l, without hiding the
counting of real degrees of freedom inside a library call?

Storage follows HEALPix/healpy: only m >= 0 is stored, because a real map has
a_{l,-m} = (-1)^m conj(a_lm).  The coefficients are ordered m-major:
    index(l, m) = m*(2*lmax + 1 - m)//2 + l ,   0 <= m <= l <= lmax .
Used by the scripts 21-26 of chapter 7.
"""
from __future__ import annotations

import numpy as np


def nalm(lmax: int) -> int:
    """Number of stored coefficients (m >= 0) up to lmax."""
    return (lmax + 1) * (lmax + 2) // 2


def index(l, m, lmax: int):
    """Position of a_lm (m >= 0) in the healpy-ordered array."""
    return m * (2 * lmax + 1 - m) // 2 + l


def getlm(lmax: int):
    """Arrays (ell, m) for every stored coefficient, in storage order."""
    ell = np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])
    m = np.concatenate([np.full(lmax + 1 - m, m) for m in range(lmax + 1)])
    return ell, m


def synalm(cl, lmax: int, rng: np.random.Generator):
    """One isotropic Gaussian realisation of the a_lm (m >= 0) with spectrum cl.

    m = 0 : a_l0 is real,        a_l0 ~ N(0, C_l)
    m > 0 : Re a_lm, Im a_lm are independent N(0, C_l / 2), so <|a_lm|^2> = C_l
    """
    cl = np.asarray(cl, dtype=float)
    ell, m = getlm(lmax)
    sig = np.sqrt(cl[ell])
    re = rng.standard_normal(ell.size)
    im = rng.standard_normal(ell.size)
    return np.where(m == 0, sig * re, sig * (re + 1j * im) / np.sqrt(2.0)).astype(complex)


def alm2cl(alm, lmax: int):
    """Full-sky estimator C_hat_l = (1/(2l+1)) sum_{m=-l}^{l} |a_lm|^2.

    The sum over negative m is folded in: |a_{l,-m}| = |a_lm|, so each m > 0 counts twice.
    """
    ell, m = getlm(lmax)
    w = np.where(m == 0, 1.0, 2.0)
    s = np.bincount(ell, weights=w * np.abs(alm) ** 2, minlength=lmax + 1)
    return s / (2 * np.arange(lmax + 1) + 1)


def chat_sims(cl, ells, nsim: int, rng: np.random.Generator):
    """C_hat_l for nsim independent full skies, multipole by multipole (memory-light).

    For each l we draw the 2l+1 real numbers that make up the a_lm of one sky:
    a_l0, and Re/Im of a_lm for m = 1..l.  Returns an array of shape (nsim, len(ells)).
    """
    out = np.empty((nsim, len(ells)))
    for j, l in enumerate(ells):
        z = rng.standard_normal((nsim, 2 * l + 1))
        a0 = np.sqrt(cl[l]) * z[:, 0]                       # real, variance C_l
        reim = np.sqrt(cl[l] / 2.0) * z[:, 1:]              # l real parts + l imaginary parts
        out[:, j] = (a0 ** 2 + 2.0 * np.sum(reim ** 2, axis=1)) / (2 * l + 1)
    return out


def ccoef(l, m):
    """c_lm = sqrt((l^2 - m^2)/((2l+1)(2l-1))):  cos(theta) Y_lm = c_{l+1,m} Y_{l+1,m} + c_lm Y_{l-1,m}."""
    l = np.asarray(l, dtype=float)
    m = np.asarray(m, dtype=float)
    num = np.clip(l ** 2 - m ** 2, 0, None)
    den = (2 * l + 1) * (2 * l - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, np.sqrt(num / np.where(den > 0, den, 1.0)), 0.0)


def modulate_dipole(alm, lmax: int, A: float):
    """a_lm of (1 + A cos theta) * map, computed exactly in harmonic space (result truncated at lmax-1).

    a'_LM = a_LM + A [ c_{L,M} a_{L-1,M} + c_{L+1,M} a_{L+1,M} ].
    """
    ell, m = getlm(lmax)
    out = alm.copy()
    down = ell - 1 >= m                     # a_{L-1,M} exists
    up = ell + 1 <= lmax                    # a_{L+1,M} exists
    i = np.arange(ell.size)
    out[down] += A * ccoef(ell[down], m[down]) * alm[i[down] - 1]
    out[up] += A * ccoef(ell[up] + 1, m[up]) * alm[i[up] + 1]
    return out
