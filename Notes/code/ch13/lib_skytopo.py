"""lib_skytopo.py -- topology of a masked HEALPix map: Minkowski functionals, Betti numbers, H0 persistence.

Everything a sky map needs for the tests of part 13b, written so that the real map and every
simulation pass through exactly the same code:

* windows(nside, fwhm, lmax): Gaussian beam x pixel window of the working resolution.
* remove_monodipole(alm, obs, nside): fit a monopole and a dipole on the observed pixels and
  subtract them from the harmonic coefficients (as Planck does for the data).
* entry_levels(u, obs, mesh): the threshold at which each vertex, edge and face of the HEALPix
  pixel mesh joins the excursion set {u > nu} restricted to the observed pixels.  Then
  chi(nu) = #vertices - #edges - ... is a search in three sorted arrays, for any number of nu.
* h0_pairs(u, obs, nbr): (birth, death) pairs of the connected components of {u > nu} as nu is
  lowered (union-find with the elder rule, compiled with numba).  beta0(nu) counts the pairs
  with birth > nu >= death.
* sky_stats(alm, ...): v0, v1, v2 (Planck normalisation V_k / (sigma1/sigma0)^k), beta0 and
  beta1 = beta0 - chi of the hot set, and beta0 of the cold set, at the chosen thresholds.
"""
from __future__ import annotations

import pathlib

import numpy as np
import healpy as hp
from numba import njit

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch13"


# ------------------------------------------------------------------ windows
def pixwin(nside: int, lmax: int) -> np.ndarray:
    """HEALPix pixel window, cached in data/ch13 so that runs without network do not need healpy's files."""
    path = DATA / f"pixwin_{nside}.npy"
    if path.exists():
        w = np.load(path)
    else:
        w = hp.pixwin(nside, lmax=min(4 * nside, 3 * 2048))
        DATA.mkdir(parents=True, exist_ok=True)
        np.save(path, w)
    return w[:lmax + 1]


def windows(nside: int, fwhm_arcmin: float, lmax: int) -> np.ndarray:
    """b_l p_l of a working map: Gaussian beam of the given FWHM times the pixel window."""
    return hp.gauss_beam(np.radians(fwhm_arcmin / 60.0), lmax=lmax) * pixwin(nside, lmax)


# ------------------------------------------------------------------ monopole and dipole
def remove_monodipole(alm, obs, nside: int, lmax: int):
    """Subtract the monopole + dipole fitted on the observed pixels (least squares), in harmonic space."""
    t = hp.alm2map(alm, nside, lmax=lmax)
    tm = np.where(obs, t, hp.UNSEEN)
    mono, dip = hp.fit_dipole(tm)
    vec = np.array(hp.pix2vec(nside, np.arange(t.size)))
    fit = mono + dip @ vec                                # the fitted 4-parameter field on the whole sky
    a_fit = hp.map2alm(fit, lmax=1, iter=3)               # exactly band-limited at l <= 1
    out = alm.copy()
    for m in (0, 1):
        for l in range(m, 2):
            out[hp.Alm.getidx(lmax, l, m)] -= a_fit[hp.Alm.getidx(1, l, m)]
    return out


# ------------------------------------------------------------------ the pixel mesh
def mesh(nside: int):
    """(edge_pix (E,2), vert_pix (V,4) padded with npix): which pixels share each edge and each vertex.

    Same construction as code/ch12/lib_sphere.py (pixel corners from healpy, rounded to integer keys).
    """
    path = DATA / f"mesh_{nside}.npz"
    if path.exists():
        z = np.load(path)
        return z["edge_pix"], z["vert_pix"]
    npix = hp.nside2npix(nside)
    corners = np.transpose(hp.boundaries(nside, np.arange(npix), step=1), (0, 2, 1))   # (npix, 4, 3)
    key = np.round(corners.reshape(-1, 3) * 2 ** 30).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(npix, 4)
    nv = vid.max() + 1
    v_all, p_all = vid.ravel(), np.repeat(np.arange(npix), 4)
    order = np.argsort(v_all, kind="stable")
    v_s, p_s = v_all[order], p_all[order]
    start = np.searchsorted(v_s, np.arange(nv))
    slot = np.arange(v_s.size) - start[v_s]
    vert_pix = np.full((nv, 4), npix, dtype=np.int64)
    vert_pix[v_s, slot] = p_s
    a = np.sort(np.stack([vid, np.roll(vid, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
    _, eid = np.unique(a, axis=0, return_inverse=True)
    order = np.argsort(eid, kind="stable")
    edge_pix = np.repeat(np.arange(npix), 4)[order].reshape(-1, 2)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(path, edge_pix=edge_pix, vert_pix=vert_pix)
    return edge_pix, vert_pix


def entry_levels(u, obs, msh):
    """Sorted entry thresholds of vertices, edges and faces of the closed observed hot pixels.

    A closed pixel is in {u > nu} when its value exceeds nu.  A vertex or an edge belongs to the
    union of such pixels as soon as ONE of the observed pixels around it does, i.e. when nu is below
    the largest observed value around it.  Unobserved pixels count as -inf (never in the set).
    """
    edge_pix, vert_pix = msh
    w = np.append(np.where(obs, u, -np.inf), -np.inf)     # sentinel index npix -> -inf
    fv = np.sort(w[vert_pix].max(axis=1))
    fe = np.sort(w[edge_pix].max(axis=1))
    ff = np.sort(w[:-1])
    return fv, fe, ff


def chi_from_levels(levels, nus):
    """chi(nu) = V - E + F of the closed observed pixels with u > nu, for every nu at once."""
    fv, fe, ff = levels
    above = lambda f: f.size - np.searchsorted(f, nus, side="right")     # number of entries > nu
    return above(fv) - above(fe) + above(ff)


# ------------------------------------------------------------------ H0 persistence (union-find)
def neighbours(nside: int, edge_only: bool = False) -> np.ndarray:
    """(npix, k) neighbour table; 8 neighbours (edge or corner) or the 4 that share an edge."""
    nb = hp.get_all_neighbours(nside, np.arange(hp.nside2npix(nside)))   # (8, npix): SW W NW N NE E SE S
    rows = [0, 2, 4, 6] if edge_only else list(range(8))                 # SW, NW, NE, SE share an edge
    return np.ascontiguousarray(nb[rows].T.astype(np.int64))


@njit(cache=True)
def _find(parent, p):
    while parent[p] != p:
        parent[p] = parent[parent[p]]                     # path halving
        p = parent[p]
    return p


@njit(cache=True)
def _h0(vals, obs, nbr, order):
    n = vals.size
    parent = -np.ones(n, dtype=np.int64)                  # -1: pixel not yet in the set
    birth = np.zeros(n)
    bs = np.empty(n)
    ds = np.empty(n)
    k = 0
    for p in order:
        if not obs[p]:
            continue
        parent[p] = p
        birth[p] = vals[p]
        for j in range(nbr.shape[1]):
            q = nbr[p, j]
            if q < 0 or parent[q] < 0:
                continue
            rp = _find(parent, p)
            rq = _find(parent, q)
            if rp == rq:
                continue
            if birth[rp] < birth[rq]:                     # rp: the elder (born at the higher value)
                rp, rq = rq, rp
            if birth[rq] > vals[p]:                       # the younger dies here (skip zero-length pairs)
                bs[k] = birth[rq]
                ds[k] = vals[p]
                k += 1
            parent[rq] = rp
    for p in range(n):                                    # components that never merge: essential
        if obs[p] and parent[p] == p:
            bs[k] = birth[p]
            ds[k] = -np.inf
            k += 1
    return bs[:k], ds[:k]


def h0_pairs(u, obs, nbr):
    """(birth, death) of the components of {u > nu} on the observed pixels, as nu is lowered."""
    order = np.argsort(-u, kind="stable")
    return _h0(np.ascontiguousarray(u, dtype=np.float64), np.ascontiguousarray(obs), nbr, order)


def betti0_from_pairs(b, d, nus):
    """beta0(nu) = number of components alive at nu: born above nu, not yet merged (death <= nu)."""
    nus = np.asarray(nus)
    return ((b[None, :] > nus[:, None]) & (d[None, :] <= nus[:, None])).sum(axis=1)


# ------------------------------------------------------------------ all statistics of one map
NU12 = np.linspace(-3.0, 3.0, 12)                         # Planck's 12 thresholds


def sky_stats(alm, nside, lmax, obs, msh, nb8, nb4, nus=NU12, dnu=0.1, keep_pairs=False):
    """Every statistic of one map, computed on the observed pixels only.

    Returns a dict with v0, v1, v2 (normalised as V_k / (sigma1/sigma0)^k, per steradian),
    chi, b0 and b1 (= b0 - chi) of the hot set {u > nu}, b0c of the cold set {u <= nu},
    the measured ratio sigma1/sigma0 (rad^-1) and, optionally, the H0 pairs of the hot set.
    """
    t, dth, dph = hp.alm2map_der1(alm, nside, lmax=lmax)
    to = t[obs]
    mu, s0 = to.mean(), to.std()
    u = (t - mu) / s0                                      # threshold in units of the map's own rms
    g2 = (dth ** 2 + dph ** 2)[obs]
    r = np.sqrt(g2.mean()) / s0                            # sigma1 / sigma0 measured on this map
    uo, go = u[obs], np.sqrt(g2) / s0
    area = obs.sum() * hp.nside2pixarea(nside)             # observed solid angle (sr)
    v0 = np.array([(uo > nu).mean() for nu in nus])
    length = np.array([(go * (np.abs(uo - nu) < dnu / 2)).mean() / dnu for nu in nus])   # per sr
    chi = chi_from_levels(entry_levels(u, obs, msh), nus)
    b, d = h0_pairs(u, obs, nb8)
    b0 = betti0_from_pairs(b, d, nus)
    bc, dc = h0_pairs(-u, obs, nb4)                        # cold spots: components of {u <= nu}
    b0c = betti0_from_pairs(bc, dc, -np.asarray(nus))
    out = {"v0": v0, "v1": 0.25 * length / r, "v2": chi / area / r ** 2,
           "chi": chi, "b0": b0, "b1": b0 - chi, "b0c": b0c, "ratio": r, "s0": s0, "area": area}
    if keep_pairs:
        out["pairs"] = (b, d)
    return out
