"""lib_masks.py -- masks, mode coupling and the MASTER estimator (chapter 8, second half).

What a mask does to the power spectrum, and how to undo it:

    sky Theta(e)  --(multiply by the mask W(e))-->  W Theta  --(map2alm)-->  pseudo a_lm
    pseudo C_l = mean over m of |pseudo a_lm|^2,   <pseudo C_l> = sum_l' M_ll' T_l'^2 C_l' + N fsky w2
    MASTER:  bin, then solve the small linear system  K_bb' D_b' = P_bl (pseudo C_l - noise)

Contents
  * masks on HEALPix:  cap(), band(), holes(), with an optional cosine taper (apodisation)
  * mask_moments():    fsky and w_i  (fsky w_i = (1/4pi) int W^i)
  * three_j_000_sq():  (l1 l2 l3; 0 0 0)^2 from the closed form, in log space (no overflow)
  * coupling_matrix(): M_ll' = (2l'+1)/(4pi) sum_l3 (2l3+1) W_l3 (l l' l3; 0 0 0)^2
  * binning operators P_bl, Q_lb (Hivon et al. 2002, eqs 20-21) and the MASTER solve
Conventions: raw C_l in muK^2, angles in radians unless the name says _deg, RING ordering.
"""
from __future__ import annotations

import math

import numpy as np
import healpy as hp
from numba import njit, prange

DEG = np.pi / 180.0


# ------------------------------------------------------------------ masks
def taper(d, width):
    """Cosine taper of the distance d (rad) from the mask edge: 0 outside, 1 deep inside.

    W = 0 for d <= 0;  W = (1 - cos(pi d / width)) / 2  for 0 < d < width;  W = 1 beyond.
    width = 0 gives the binary (top-hat) mask.
    """
    d = np.asarray(d, dtype=float)
    if width <= 0:
        return (d > 0).astype(float)
    x = np.clip(d / width, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def _angles(nside):
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return theta, phi


def cap(nside, fsky, apo_deg=0.0):
    """Observed region = a cap around the north pole holding a fraction fsky of the sphere.

    A cap of half-opening theta_c has area 2 pi (1 - cos theta_c), so cos theta_c = 1 - 2 fsky.
    The taper (if any) eats into the cap from its edge inwards.
    """
    theta, _ = _angles(nside)
    theta_c = np.arccos(1.0 - 2.0 * fsky)
    return taper(theta_c - theta, apo_deg * DEG)


def band(nside, b_cut_deg, apo_deg=0.0):
    """Galactic-like cut: remove |latitude| < b_cut. The kept fraction is 1 - sin(b_cut)."""
    theta, _ = _angles(nside)
    lat = np.pi / 2 - theta
    return taper(np.abs(lat) - b_cut_deg * DEG, apo_deg * DEG)


def hole_centres(n, rng):
    """n directions drawn uniformly on the sphere (uniform in cos theta and phi)."""
    z = rng.uniform(-1.0, 1.0, n)
    ph = rng.uniform(0.0, 2 * np.pi, n)
    s = np.sqrt(1 - z ** 2)
    return np.stack([s * np.cos(ph), s * np.sin(ph), z], axis=1)


def holes(nside, centres, radius_deg, apo_deg=0.0):
    """Point-source mask: discs of the given radius cut out around each centre."""
    v = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T     # (npix, 3)
    cosmax = np.full(v.shape[0], -1.0)
    for c0 in range(0, len(centres), 50):                                  # nearest centre, in chunks
        cosmax = np.maximum(cosmax, (v @ centres[c0:c0 + 50].T).max(axis=1))
    dist = np.arccos(np.clip(cosmax, -1, 1))                              # angle to nearest centre
    return taper(dist - radius_deg * DEG, apo_deg * DEG)


def ellipse(nside, a_deg, b_deg, profile="tophat", centre=(1.0, 0.0, 0.0)):
    """Elliptical patch with the apodisations of Hivon et al. (2002), Table 1.

    The patch is centred on `centre` (default: on the equator at longitude 0); its semi-axis a
    runs east-west (along the scan of a ground-based or balloon telescope), b north-south.
    r is the angle from the centre, phi' the direction from the a-axis, and
    rho(phi') = a / sqrt(1 + (a^2/b^2 - 1) sin^2 phi') the radius of the ellipse in that direction.
    profile: "tophat" W = 1, "cosine" W = cos(pi r / 2 rho), "gauss" W = exp(-(3 r / rho)^2 / 2),
    all for r <= rho and 0 outside.
    """
    v = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    c = np.asarray(centre, float)
    r = np.arccos(np.clip(c @ v, -1, 1))
    # local east (a-axis) and north (b-axis) at the centre
    east = np.cross([0.0, 0.0, 1.0], c)
    east /= np.linalg.norm(east)
    north = np.cross(c, east)
    phi = np.arctan2(north @ v, east @ v)
    a, b = a_deg * DEG, b_deg * DEG
    rho = a / np.sqrt(1 + (a * a / (b * b) - 1) * np.sin(phi) ** 2)
    u = r / rho
    inside = (u <= 1.0) & (c @ v > 0)
    if profile == "tophat":
        w = np.ones_like(u)
    elif profile == "cosine":
        w = np.cos(0.5 * np.pi * u)
    elif profile == "gauss":
        w = np.exp(-0.5 * (3.0 * u) ** 2)
    else:
        raise ValueError(profile)
    return np.where(inside, w, 0.0)


def mask_moments(w, imax=4):
    """fsky and w_1..w_imax with  fsky w_i = (1/4pi) int W^i dOmega  (Hivon et al. eq. 9).

    fsky is the fraction of pixels where W is nonzero, so w_i = 1 for every i for a binary mask.
    """
    fsky = np.mean(w > 0)
    return fsky, {i: np.mean(w ** i) / fsky for i in range(1, imax + 1)}


def window_spectrum(w, lmax):
    """W_l = (1/(2l+1)) sum_m |w_lm|^2 of the mask, by the same pixel quadrature as the data."""
    return hp.anafast(w, lmax=lmax, iter=0)


# ------------------------------------------------------------------ Wigner 3j (0 0 0)
def log_factorials(nmax):
    """lf[n] = ln n!  for n = 0..nmax, from the log-gamma function."""
    from scipy.special import gammaln
    return gammaln(np.arange(nmax + 1) + 1.0)


@njit(cache=True)
def _w3j000_sq(l1, l2, l3, lf):
    """(l1 l2 l3; 0 0 0)^2 from the closed form (Hivon et al. eq. A25), evaluated in logs."""
    L = l1 + l2 + l3
    if L % 2 == 1 or l3 < abs(l1 - l2) or l3 > l1 + l2:
        return 0.0
    g = L // 2
    lnv = (lf[L - 2 * l1] + lf[L - 2 * l2] + lf[L - 2 * l3] - lf[L + 1]
           + 2.0 * (lf[g] - lf[g - l1] - lf[g - l2] - lf[g - l3]))
    return math.exp(lnv)


def three_j_000_sq(l1, l2, l3):
    lf = log_factorials(l1 + l2 + l3 + 2)
    return _w3j000_sq(int(l1), int(l2), int(l3), lf)


@njit(parallel=True, cache=True)
def _coupling(wl, lmax_row, lmax_col, lf):
    nW = wl.shape[0]
    M = np.zeros((lmax_row + 1, lmax_col + 1))
    for l1 in prange(lmax_row + 1):
        for l2 in range(lmax_col + 1):
            s = 0.0
            lo = abs(l1 - l2)
            hi = min(l1 + l2, nW - 1)
            l3 = lo
            if (l1 + l2 + l3) % 2 == 1:      # only even l1 + l2 + l3 survive
                l3 += 1
            while l3 <= hi:
                s += (2 * l3 + 1) * wl[l3] * _w3j000_sq(l1, l2, l3, lf)
                l3 += 2
            M[l1, l2] = (2 * l2 + 1) / (4 * np.pi) * s
    return M


def coupling_matrix(wl, lmax_row, lmax_col=None):
    """M_ll' (Hivon et al. eq. A31) for l <= lmax_row, l' <= lmax_col.

    wl must extend to at least lmax_row + lmax_col for the result to be exact.
    """
    lmax_col = lmax_row if lmax_col is None else lmax_col
    lf = log_factorials(2 * (len(wl) + lmax_row + lmax_col) + 4)
    return _coupling(np.asarray(wl, dtype=float), lmax_row, lmax_col, lf)


# ------------------------------------------------------------------ associated Legendre functions
def lambda_lm(lmax, m, theta):
    """lambda_lm(theta) for l = m..lmax (rows), with Y_lm = lambda_lm(theta) e^{i m phi}.

    Stable three-term recursion in l at fixed m (no factorials, no overflow up to l of several
    thousand):  lambda_mm = (-1)^m sqrt((2m+1)!!/(4 pi (2m)!!)) sin^m theta,
    lambda_{m+1,m} = sqrt(2m+3) cos theta lambda_mm,
    lambda_lm = a_lm (cos theta lambda_{l-1,m} - lambda_{l-2,m} / a_{l-1,m}),
    a_lm = sqrt((4 l^2 - 1)/(l^2 - m^2)).
    """
    x, s = np.cos(theta), np.sin(theta)
    out = np.zeros((lmax - m + 1, np.size(theta)))
    p = np.full(np.size(theta), 1.0 / np.sqrt(4 * np.pi))
    for k in range(1, m + 1):
        p = -np.sqrt((2 * k + 1) / (2.0 * k)) * s * p
    out[0] = p
    if lmax > m:
        out[1] = np.sqrt(2 * m + 3.0) * x * p
    a = lambda l: np.sqrt((4.0 * l * l - 1) / (l * l - m * m))
    for l in range(m + 2, lmax + 1):
        out[l - m] = a(l) * (x * out[l - m - 1] - out[l - m - 2] / a(l - 1))
    return out


# ------------------------------------------------------------------ binning and MASTER
def bin_edges(lmin, lmax, dl):
    """Edges [lmin, lmin+dl, ...] with the last bin closed at lmax+1."""
    e = np.arange(lmin, lmax + 2, dl)
    if e[-1] != lmax + 1:
        e = np.append(e, lmax + 1)
    return e


def binning_operators(edges, lmax):
    """P_bl averages l(l+1)C_l/2pi over a bin; Q_lb spreads a flat D_b back over the bin."""
    nb = len(edges) - 1
    P = np.zeros((nb, lmax + 1))
    Q = np.zeros((lmax + 1, nb))
    for b in range(nb):
        ls = np.arange(edges[b], edges[b + 1])
        P[b, ls] = ls * (ls + 1) / (2 * np.pi) / len(ls)
        Q[ls, b] = 2 * np.pi / (ls * (ls + 1))
    return P, Q


def master_matrix(M, transfer2, P, Q):
    """Binned coupling K_bb' = P_bl M_ll' F_l' B_l'^2 Q_l'b' (Hivon et al. eq. 25)."""
    return P @ (M * transfer2[None, :]) @ Q


def dl_from_cl(cl):
    ell = np.arange(cl.shape[-1])
    return ell * (ell + 1) * cl / (2 * np.pi)


def tex_sci(x, digits=2):
    """A number as LaTeX: 0.0123 -> 1.2\\times10^{-2}; numbers between 0.01 and 1e4 stay plain."""
    x = float(x)
    if x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e4):
        m, e = f"{x:.{digits - 1}e}".split("e")
        return rf"{m}\times10^{{{int(e)}}}"
    return f"{x:.{digits}g}"
