"""lib_persist.py -- persistent homology written from scratch (numpy/scipy only).

Used by the b0*.py scripts of chapter 11.  Import with

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from lib_persist import islands, lakes, betti_curve, kruskal_h0, rips_bruteforce

Conventions
-----------
* A field f[i, j] on a grid of pixels.  The superlevel set at level nu is the union of the
  closed unit squares of the pixels with f >= nu.  Squares touching at a corner are connected
  (8-neighbours); the complement is then 4-connected.  This matches gudhi's CubicalComplex
  built from top-dimensional cells, so the two codes must give identical diagrams.
* A persistence pair (b, d) of the superlevel filtration has b >= d: the feature appears when
  the level drops to b and disappears when it drops below d.  The island that never dies has
  d = -inf.
"""
from __future__ import annotations

import itertools

import numpy as np

NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGH4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]


# ------------------------------------------------------------------ H0 of superlevel sets
def islands(f, conn: int = 8):
    """Birth-death pairs of the islands of {f >= nu} as the level nu is lowered (union-find)."""
    ny, nx = f.shape
    h = f.ravel()
    order = np.argsort(-h, kind="stable")       # pixels from the highest to the lowest
    parent = np.full(h.size, -1)                # -1 means: still under water
    peak = np.empty(h.size)                     # for the root of each island: its peak height
    steps = NEIGH8 if conn == 8 else NEIGH4

    def root(p):                                # follow parents to the island's root
        while parent[p] != p:
            parent[p] = parent[parent[p]]       # path halving keeps the trees shallow
            p = parent[p]
        return p

    pairs = []
    for p in order:                             # lower the water level pixel by pixel
        parent[p], peak[p] = p, h[p]            # p emerges as a one-pixel island
        i, j = divmod(p, nx)
        for di, dj in steps:
            a, b = i + di, j + dj
            if not (0 <= a < ny and 0 <= b < nx) or parent[a * nx + b] < 0:
                continue                        # off the grid, or neighbour still under water
            r1, r2 = root(p), root(a * nx + b)
            if r1 == r2:
                continue                        # already the same island
            if peak[r1] < peak[r2]:
                r1, r2 = r2, r1                 # r1: the elder island (higher peak)
            if peak[r2] > h[p]:                 # the younger one dies at this level
                pairs.append((peak[r2], h[p]))
            parent[r2] = r1                     # ... and becomes part of the elder
    pairs.append((h[order[0]], -np.inf))        # the highest island never dies
    return np.array(pairs)


def lakes(f):
    """Birth-death pairs of the holes (lakes) of {f >= nu}, by Alexander duality in the plane.

    A hole of the land is a piece of the water {f < nu} that does not reach the open sea.  We
    surround the map by a frame of sea (+inf in -f), compute the islands of -f with
    4-neighbours, and translate: a water body born at -f = -m (the lake bottom, height m) and
    merging into another at -f = -s (the pass, height s) is a lake of the land that appears
    when the level drops to s and fills up when it drops below m.
    """
    g = np.pad(-np.asarray(f, float), 1, constant_values=np.inf)
    p = islands(g, conn=4)
    p = p[np.isfinite(p[:, 0]) & np.isfinite(p[:, 1])]  # drop the open sea itself
    return np.column_stack([-p[:, 1], -p[:, 0]])        # (birth, death) = (pass, bottom)


# ------------------------------------------------------------------ summaries of a diagram
def betti_curve(pairs, nus):
    """beta(nu) = number of pairs alive at level nu, i.e. with d < nu <= b."""
    b, d = pairs[:, 0][:, None], pairs[:, 1][:, None]
    return ((d < nus[None, :]) & (nus[None, :] <= b)).sum(axis=0)


def to_sublevel(pairs):
    """(b, d) of the superlevel filtration of f -> (birth, death) of the sublevel filtration of -f,
    the convention of gudhi, ripser and persim (birth <= death)."""
    return np.column_stack([-pairs[:, 0], -pairs[:, 1]])


def finite(pairs):
    return pairs[np.all(np.isfinite(pairs), axis=1)]


# ------------------------------------------------------------------ point clouds
def kruskal_h0(X):
    """H0 of the Vietoris-Rips filtration = the edges of the minimum spanning tree (Kruskal).

    All points are born at r = 0; a component dies at the length of the edge that merges it
    into another.  Returns the sorted death values (n - 1 of them; one component never dies).
    """
    n = len(X)
    i, j = np.triu_indices(n, k=1)
    length = np.linalg.norm(X[i] - X[j], axis=1)
    parent = list(range(n))

    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    deaths = []
    for e in np.argsort(length, kind="stable"):  # shortest edges first
        r1, r2 = root(i[e]), root(j[e])
        if r1 != r2:                             # the edge joins two components: one dies
            parent[r1] = r2
            deaths.append(length[e])
            if len(deaths) == n - 1:
                break
    return np.array(deaths)


def rips_bruteforce(X, maxdim: int = 1):
    """Persistence of the Vietoris-Rips filtration by reducing the boundary matrix over Z_2.

    Simplices up to dimension maxdim + 1.  Filtration value of a simplex = its longest edge.
    Returns {dim: array of (birth, death)}, deaths of essential classes = inf.  Small clouds only.
    """
    n = len(X)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    simplices = []
    for k in range(1, maxdim + 3):
        for s in itertools.combinations(range(n), k):
            val = 0.0 if k == 1 else max(D[a, b] for a, b in itertools.combinations(s, 2))
            simplices.append((val, k - 1, s))
    simplices.sort(key=lambda t: (t[0], t[1]))          # by value, faces before cofaces
    index = {s: c for c, (_, _, s) in enumerate(simplices)}
    low_of = {}                                         # pivot row -> column that owns it
    out = {k: [] for k in range(maxdim + 1)}
    paired = set()
    for c, (val, dim, s) in enumerate(simplices):
        col = {index[f] for f in itertools.combinations(s, dim)} if dim > 0 else set()
        while col and max(col) in low_of:               # add earlier columns until the pivot is new
            col ^= low_of[max(col)][1]
        if col:
            r = max(col)                                # simplex r (dim-1) is killed by c
            low_of[r] = (c, col)
            paired.update((r, c))
            b = simplices[r][0]
            if dim - 1 <= maxdim and val > b:
                out[dim - 1].append((b, val))
    for c, (val, dim, s) in enumerate(simplices):       # unpaired simplices: essential classes
        if c not in paired and dim <= maxdim:
            # a simplex whose column reduced to zero creates a class; if never killed it is essential
            out[dim].append((val, np.inf))
    return {k: np.array(v).reshape(-1, 2) for k, v in out.items()}


# ------------------------------------------------------------------ polarization
def flat_qu(clee, N: int, side_deg: float, rng, lmax: float | None = None, nsim: int | None = None):
    """E-mode-only Q, U on a periodic flat patch (B = 0) from C_ell^EE (muK^2), pixel units.

    Flat sky: a Fourier mode ell = (lx, ly) of E gives Q = E cos 2phi_l, U = E sin 2phi_l.
    Returns Q, U and the ratio <ell^2> = sigma_1^2 / sigma_0^2 of the realised grid modes.
    """
    L = np.deg2rad(side_deg)
    D = L / N
    lx = 2 * np.pi * np.fft.fftfreq(N, d=D)
    ly = 2 * np.pi * np.fft.rfftfreq(N, d=D)
    LX, LY = np.meshgrid(lx, ly, indexing="ij")
    ell = np.hypot(LX, LY)
    cl = np.interp(ell, np.arange(len(clee)), clee, right=0.0)
    cl[ell < 2] = 0.0
    if lmax is not None:
        cl[ell > lmax] = 0.0
    amp = np.sqrt(cl / D ** 2)
    phi = np.arctan2(LY, LX)
    shape = ([nsim] if nsim else []) + [N, N]
    w = rng.standard_normal(shape)
    Ek = np.fft.rfft2(w) * amp
    Q = np.fft.irfft2(Ek * np.cos(2 * phi), s=(N, N))
    U = np.fft.irfft2(Ek * np.sin(2 * phi), s=(N, N))
    wgt = np.full(cl.shape, 2.0)
    wgt[:, 0] = 1.0
    if N % 2 == 0:
        wgt[:, -1] = 1.0
    ell2 = (wgt * ell ** 2 * cl).sum() / (wgt * cl).sum()
    return Q, U, ell2


def winding(Q, U, periodic: bool = True):
    """Index of every plaquette: sum of the changes of arg(Q+iU) around it, each wrapped into
    (-pi, pi], divided by 2pi gives n; the polarization angle turns by n*pi, index n/2.

    Returns an array n/2 of shape (N, N) (periodic) or (N-1, N-1) (open patch)."""
    a = np.angle(Q + 1j * U)

    def d(x, y):
        return np.angle(np.exp(1j * (y - x)))

    if periodic:
        a10 = np.roll(a, -1, 0)
        a11 = np.roll(a10, -1, 1)
        a01 = np.roll(a, -1, 1)
    else:
        a, a10, a11, a01 = a[:-1, :-1], a[1:, :-1], a[1:, 1:], a[:-1, 1:]
    w = d(a, a10) + d(a10, a11) + d(a11, a01) + d(a01, a)
    return np.rint(w / (2 * np.pi)) / 2
