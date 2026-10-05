"""lib_sphere.py -- excursion-set topology and Minkowski functionals on the HEALPix sphere.

* chi of the union of closed HEALPix pixels above a threshold: V - E + F of the pixel mesh.
  The mesh (which pixels share each vertex and each edge) is built once per nside from the
  pixel corners returned by healpy and cached in data/ch12/.
* beta0 of the hot set (pixels touching at an edge or a corner) and of the cold set
  (pixels sharing an edge) from healpy's neighbour lists; on the sphere Alexander duality
  gives beta1(hot) = beta0(cold) - 1, so chi = beta0(hot) - beta0(cold) + 1.
* Minkowski functionals per steradian: v0 area fraction, v1 = (1/4) x boundary length per
  steradian (coarea formula with exact first derivatives from the a_lm), and
  v2 = (chi - 2 v0) / (4 pi), the boundary curvature with the sphere's own curvature removed.
"""
from __future__ import annotations

import pathlib

import numpy as np
import healpy as hp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch12"


def mesh(nside: int):
    """(edge_pix (E,2), vert_pix (V,4) padded with npix) for the HEALPix pixel mesh."""
    path = DATA / f"healpix_mesh_{nside}.npz"
    if path.exists():
        z = np.load(path)
        return z["edge_pix"], z["vert_pix"]
    npix = hp.nside2npix(nside)
    corners = np.transpose(hp.boundaries(nside, np.arange(npix), step=1), (0, 2, 1))   # (npix, 4, 3)
    key = np.round(corners.reshape(-1, 3) * 2 ** 30).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(npix, 4)
    nv = vid.max() + 1
    # vertex -> incident pixels (3 or 4), padded with the sentinel npix
    # (a vertex can be the k-th corner of two different pixels near the base-pixel corners,
    #  so the slots are filled by sorting (vertex, pixel) pairs rather than by fancy indexing)
    v_all = vid.ravel()
    p_all = np.repeat(np.arange(npix), 4)
    order = np.argsort(v_all, kind="stable")
    v_s, p_s = v_all[order], p_all[order]
    start = np.searchsorted(v_s, np.arange(nv))
    slot = np.arange(v_s.size) - start[v_s]                # 0, 1, 2 (, 3) within each vertex
    assert slot.max() < 4
    vert_pix = np.full((nv, 4), npix, dtype=np.int64)
    vert_pix[v_s, slot] = p_s
    # edges: consecutive corners of each pixel; each edge is shared by exactly two pixels
    a = np.sort(np.stack([vid, np.roll(vid, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
    _, eid = np.unique(a, axis=0, return_inverse=True)
    pix = np.repeat(np.arange(npix), 4)
    order = np.argsort(eid, kind="stable")
    edge_pix = pix[order].reshape(-1, 2)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(path, edge_pix=edge_pix, vert_pix=vert_pix)
    return edge_pix, vert_pix


def chi_sphere(mask, msh) -> int:
    """V - E + F of the union of closed pixels in `mask` (bool array of length npix)."""
    edge_pix, vert_pix = msh
    m = np.append(np.asarray(mask, dtype=bool), False)     # sentinel pixel is never in the set
    V = m[vert_pix].any(axis=1).sum()
    E = m[edge_pix].any(axis=1).sum()
    F = m[:-1].sum()
    return int(V) - int(E) + int(F)


def betti_sphere(mask, nside: int):
    """(beta0 hot, beta0 cold) with the dual neighbour rules (8 for hot, 4 edge-sharing for cold)."""
    npix = hp.nside2npix(nside)
    nb = hp.get_all_neighbours(nside, np.arange(npix))     # (8, npix): SW, W, NW, N, NE, E, SE, S
    m = np.asarray(mask, dtype=bool)

    def count(sel, rows):
        src = np.repeat(np.arange(npix)[None, :], len(rows), 0).ravel()
        dst = nb[rows].ravel()
        ok = (dst >= 0) & sel[src] & sel[np.where(dst >= 0, dst, 0)]
        g = coo_matrix((np.ones(ok.sum()), (src[ok], dst[ok])), shape=(npix, npix))
        n, lab = connected_components(g, directed=False)
        return len(np.unique(lab[sel]))                    # components that contain selected pixels

    hot = count(m, list(range(8))) if m.any() else 0
    cold = count(~m, [0, 2, 4, 6]) if (~m).any() else 0    # SW, NW, NE, SE share an edge
    return hot, cold


def mfs_sphere(alm, nside: int, nus, msh, lmax: int | None = None, dnu: float = 0.1):
    """v0, v1, v2 (per steradian) and chi of {u > nu}, u = map / its rms, from a_lm."""
    lmax = hp.Alm.getlmax(len(alm)) if lmax is None else lmax
    u, dth, dph = hp.alm2map_der1(alm, nside, lmax=lmax)
    sig = u.std()
    u, g = (u - u.mean()) / sig, np.hypot(dth, dph) / sig
    npix = u.size
    v0 = np.array([(u > nu).mean() for nu in nus])
    v1 = np.array([0.25 * (g * (np.abs(u - nu) < dnu / 2)).mean() / dnu for nu in nus])
    chi = np.array([chi_sphere(u > nu, msh) for nu in nus])
    v2 = (chi - 2 * v0) / (4 * np.pi)
    return v0, v1, v2, chi


def gauss_mfs_sphere(nu, sigma2, tau):
    """Schmalzing & Gorski (1998) eq. (14) in sigma units: tau = <u_;i u_;i>/2, lam = tau/sigma2."""
    from scipy.special import erfc
    nu = np.asarray(nu, dtype=float)
    lam = tau / sigma2
    e = np.exp(-nu ** 2 / 2)
    return 0.5 * erfc(nu / np.sqrt(2)), np.sqrt(lam) / 8 * e, lam * nu * e / (2 * np.pi) ** 1.5


def sigma_tau(cl):
    """Variance and tau of an isotropic map with spectrum C_l (l from 0): the CORRECT eq. (16)."""
    ell = np.arange(len(cl))
    s2 = np.sum((2 * ell + 1) * cl) / (4 * np.pi)
    tau = np.sum((2 * ell + 1) * cl * ell * (ell + 1) / 2) / (4 * np.pi)
    return s2, tau
