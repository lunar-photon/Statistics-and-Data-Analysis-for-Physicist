"""lib_topo.py -- excursion-set topology and Minkowski functionals, written from scratch.

Shared by the scripts of chapter 11.  Import with

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from lib_topo import betti_plane, euler_cubical, mfs_flat, gkf_plane

Conventions
-----------
* A pixel map phi[i, j] on an N x N grid of unit pixels.  The excursion set at threshold nu
  is the union of the CLOSED unit squares of the pixels with phi > nu.  Two such squares
  that touch only at a corner are therefore connected: the set is 8-connected, and its
  complement (open squares) is 4-connected.  This is the dual pair that makes the
  counting consistent (Jordan curve theorem on the grid).
* Thresholds nu are in units of the standard deviation sigma of the field.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EIGHT = np.ones((3, 3), dtype=int)                       # 8-neighbour stencil
FOUR = ndimage.generate_binary_structure(2, 1)           # 4-neighbour stencil (a plus sign)


# ------------------------------------------------------------------ Betti numbers in the plane
def n_components(mask, conn: int = 8, periodic: bool = False) -> int:
    """Number of connected components of a binary image (scipy.ndimage.label).

    periodic: the grid is a torus, so pieces cut by an edge of the square are glued back
    together.  Labels that touch across the seam are merged with a small union-find.
    """
    lab, n = ndimage.label(mask, structure=EIGHT if conn == 8 else FOUR)
    if not periodic or n == 0:
        return int(n)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = []
    for a, b in ((lab[0, :], lab[-1, :]), (lab[:, 0], lab[:, -1])):
        pairs.append((a, b))                     # straight across the seam
        if conn == 8:                            # and diagonally across it
            pairs.append((a, np.roll(b, 1)))
            pairs.append((a, np.roll(b, -1)))
    for a, b in pairs:
        ok = (a > 0) & (b > 0)
        for x, y in zip(a[ok], b[ok]):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
    return len({find(i) for i in range(1, n + 1)})


def betti_plane(mask, conn: int = 8):
    """(beta0, beta1) of a pixel set in the plane.

    beta0: components of the set with `conn`-connectivity.
    beta1: holes = components of the complement that cannot reach the outside, counted
           with the DUAL connectivity (4 if conn == 8, else 8).  Padding the image with a
           frame of background makes everything outside the grid one single component, so
           beta1 = (components of the padded complement) - 1.
    """
    mask = np.asarray(mask, dtype=bool)
    b0 = n_components(mask, conn)
    comp = ~np.pad(mask, 1, constant_values=False)
    b1 = n_components(comp, 4 if conn == 8 else 8) - 1
    return b0, b1


# ------------------------------------------------------------------ Euler characteristic by cell counting
def euler_cubical(mask, conn: int = 8, periodic: bool = False) -> int:
    """chi = V - E + F of the cubical complex of a binary image.

    conn = 8: union of closed pixel squares (pixels are faces).  A lattice vertex or edge
              belongs to the set if ANY pixel touching it does.
    conn = 4: the dual construction (pixels are vertices).  An edge joins two 4-neighbour
              pixels that are both in the set, a face fills a 2x2 block that is entirely in.
    periodic: the grid is a torus (opposite sides glued), as for an FFT-generated field.
    """
    m = np.asarray(mask, dtype=bool)
    if conn == 8:
        if periodic:
            r0 = np.roll(m, 1, 0)
            r1 = np.roll(m, 1, 1)
            V = (m | r0 | r1 | np.roll(r0, 1, 1)).sum()
            E = (m | r0).sum() + (m | r1).sum()
        else:
            P = np.pad(m, 1)
            V = (P[:-1, :-1] | P[1:, :-1] | P[:-1, 1:] | P[1:, 1:]).sum()
            E = (P[:-1, 1:-1] | P[1:, 1:-1]).sum() + (P[1:-1, :-1] | P[1:-1, 1:]).sum()
        F = m.sum()
    else:
        if periodic:
            r0 = np.roll(m, -1, 0)
            r1 = np.roll(m, -1, 1)
            E = (m & r0).sum() + (m & r1).sum()
            F = (m & r0 & r1 & np.roll(r0, -1, 1)).sum()
        else:
            E = (m[:-1, :] & m[1:, :]).sum() + (m[:, :-1] & m[:, 1:]).sum()
            F = (m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]).sum()
        V = m.sum()
    return int(V) - int(E) + int(F)


def euler_cubical_3d(mask, periodic: bool = True) -> int:
    """chi = V - E + F - C of the union of closed unit cubes (voxels > threshold)."""
    m = np.asarray(mask, dtype=bool)
    if not periodic:
        m = np.pad(m, 1)

    def sh(a, ax):                       # neighbour across axis ax (periodic roll)
        return np.roll(a, 1, ax)

    # faces: the two cubes on either side of a face normal to axis a
    F = sum((m | sh(m, a)).sum() for a in range(3))
    # edges along axis a: the four cubes around it (shifts in the two other axes)
    E = 0
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        mb = m | sh(m, b)
        E += (mb | sh(mb, c)).sum()
    # vertices: the eight cubes around a lattice point
    v = m | sh(m, 0)
    v = v | sh(v, 1)
    v = v | sh(v, 2)
    V = v.sum()
    C = m.sum()
    return int(V) - int(E) + int(F) - int(C)


# ------------------------------------------------------------------ Minkowski functionals of a flat periodic map
def spectral_gradient(phi):
    """Exact derivatives of a periodic band-limited map (unit pixels) via the FFT."""
    N0, N1 = phi.shape
    k0 = 2 * np.pi * np.fft.fftfreq(N0)[:, None]
    k1 = 2 * np.pi * np.fft.rfftfreq(N1)[None, :]
    f = np.fft.rfft2(phi)
    gx = np.fft.irfft2(1j * k0 * f, s=phi.shape)
    gy = np.fft.irfft2(1j * k1 * f, s=phi.shape)
    return gx, gy


def mfs_flat(phi, nus, dnu: float = 0.1, periodic: bool = True):
    """Area fraction, boundary length per unit area and chi per unit area of {phi/sigma > nu}.

    Area: fraction of pixels above threshold.
    Length: the coarea formula L = int d^2x delta(phi - nu) |grad phi|, with the delta
            function replaced by a top hat of width dnu (in sigma units).
    chi: V - E + F of the closed-pixel complex (exact topology of the pixel set).
    Returns three arrays over nus (per unit pixel area, lengths in pixels).
    """
    u = (phi - phi.mean()) / phi.std()
    gx, gy = spectral_gradient(u)
    g = np.hypot(gx, gy)
    A = u.size
    area = np.array([(u > nu).mean() for nu in nus])
    length = np.array([(g * (np.abs(u - nu) < dnu / 2)).sum() / dnu / A for nu in nus])
    chi = np.array([euler_cubical(u > nu, 8, periodic) / A for nu in nus])
    return area, length, chi


# ------------------------------------------------------------------ analytic Gaussian kinematic formula
def tail(nu):
    """Psi(nu) = P(Z > nu) for a standard normal Z."""
    from scipy.special import erfc
    return 0.5 * erfc(np.asarray(nu) / np.sqrt(2))


def gkf_plane(nu, lam):
    """Expected (area fraction, length per area, chi per area) of a 2-D Gaussian field.

    lam = <|grad phi|^2> / (2 sigma^2)  (variance of ONE derivative over sigma^2).
    """
    nu = np.asarray(nu, dtype=float)
    e = np.exp(-nu ** 2 / 2)
    return tail(nu), 0.5 * np.sqrt(lam) * e, lam * nu * e / (2 * np.pi) ** 1.5


def gkf_square(nu, lam, L):
    """Expected chi of the excursion set in an L x L square WITH its edges and corners.

    area term + (perimeter / 2) x (1-D crossing density) + (number of corners / 4) x Psi(nu).
    """
    nu = np.asarray(nu, dtype=float)
    e = np.exp(-nu ** 2 / 2)
    return L * L * lam * nu * e / (2 * np.pi) ** 1.5 + 2 * L * np.sqrt(lam) * e / (2 * np.pi) + tail(nu)


def lam_from_pk(P, N: int, L: float):
    """lambda of a field made by lib_fields.gaussian_field(P, N, L, 2): exact lattice sums."""
    k0 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky = np.meshgrid(k0, k0, indexing="ij")
    k = np.hypot(kx, ky)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(k > 0, P(np.where(k > 0, k, 1.0)), 0.0)
    s0 = p.sum()
    s1 = (kx ** 2 * p).sum()          # one derivative component
    return s1 / s0
