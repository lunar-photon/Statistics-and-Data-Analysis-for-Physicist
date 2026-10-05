"""lib_wl.py -- weak-lensing maps on a flat periodic patch, written from scratch.

Shared by the w0*.py scripts of chapter 11.  Import with

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import lib_wl as wl

Conventions (the flat-box conventions of code/ch07/lib_fields.py, in two dimensions):

    kappa_l = sum_x kappa(x) e^{-i l.x} A_pix        (A_pix = pixel solid angle, sr)
    <|FFT_l|^2> = N_pix P(l) / A_pix,   so a field with spectrum P is made by multiplying the
    FFT of unit white noise by sqrt(P(l) / A_pix) and transforming back.

The patch is N x N pixels of PIX_ARCMIN arcmin, periodic (a torus), as an FFT simulation is.
Shear and convergence are related mode by mode by the phase e^{2 i phi_l}, phi_l the angle of
the wavevector l (Kaiser and Squires 1993):  gamma_l = e^{2 i phi_l} kappa_l.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.integrate import trapezoid

# ------------------------------------------------------------------ the patch and the survey
N = 256                                   # pixels per side
PIX_ARCMIN = 1.5                          # pixel side, arcmin
ARCMIN = np.pi / (180 * 60)               # one arcmin in radians
PIX = PIX_ARCMIN * ARCMIN                 # pixel side, rad
A_PIX = PIX**2                            # pixel solid angle, sr
SIDE_DEG = N * PIX_ARCMIN / 60            # 6.4 deg
A_PATCH_DEG2 = SIDE_DEG**2                # 40.96 deg^2

SIGMA_E = 0.27                            # rms intrinsic ellipticity per component (KiDS-like)
NGAL = 6.2                                # galaxies per arcmin^2 (KiDS-1000 effective density)
AREA_DEG2 = 1000.0                        # survey area
THETA_G = 3.0                             # Gaussian smoothing scale (standard deviation), arcmin

# the source sample of part T1b: p(z) ~ z^2 exp[-(z/0.5)^1.5], median redshift 0.71
ZGRID = np.linspace(0.0, 3.5, 701)
Z0 = 0.5
OM_FID, S8_FID = 0.3153, 0.8111           # Planck 2018 values used throughout the book
STEP = 0.05                               # +/- 5 per cent steps for the derivatives
C_KMS = 299792.458


def source_nz(z, z0=Z0):
    nz = z**2 * np.exp(-(z / z0) ** 1.5)
    return nz / trapezoid(nz, z)


# ------------------------------------------------------------------ cosmology: C_ell and the shift
def camb_setup(om, s8):
    """CAMB parameters for flat LCDM at (Omega_m, sigma_8), the rest at the Planck values.
    sigma_8 is set by rescaling A_s, since linear sigma_8^2 is proportional to A_s."""
    import camb
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from camb_fiducial import FIDUCIAL
    par = dict(FIDUCIAL)
    h = par["H0"] / 100
    p0 = camb.set_params(**par)
    par["omch2"] = om * h**2 - par["ombh2"] - p0.omnuh2
    plin = camb.set_params(**par, WantTransfer=True)
    plin.set_matter_power(redshifts=[0.0], kmax=2.0)
    s8_ref = camb.get_results(plin).get_sigma8_0()
    par["As"] = FIDUCIAL["As"] * (s8 / s8_ref) ** 2
    return par


def limber_and_shift(om, s8, ells, kmax=100.0):
    """Limber C_ell^{kappa kappa}(ells) for the source sample, and the empty-beam convergence.

    C_ell = (9/4)(H0/c)^4 Om^2 int dchi [g/a]^2 P_delta((l+1/2)/chi, chi)     (part T1b)
    kappa_empty = (3/2)(H0/c)^2 Om int dchi g chi / a                         (delta = -1)
    """
    import camb
    par = camb_setup(om, s8)
    p = camb.set_params(**par, NonLinear=camb.model.NonLinear_both, halofit_version="takahashi")
    res = camb.get_background(p)
    pk = camb.get_matter_power_interpolator(p, nonlinear=True, hubble_units=False, k_hunit=False,
                                            kmax=kmax, zmax=ZGRID[-1])
    z = ZGRID[1:]
    nz = source_nz(ZGRID)[1:]
    chi = res.comoving_radial_distance(z)
    g = np.array([trapezoid(nz[i:] * (1.0 - chi[i] / chi[i:]), z[i:]) for i in range(len(z))])
    dchi_dz = C_KMS / res.hubble_parameter(z)
    a = 1.0 / (1.0 + z)
    H0c = par["H0"] / C_KMS
    pref = 2.25 * H0c**4 * om**2
    cl = np.zeros(len(ells))
    for i, l in enumerate(ells):
        k = (l + 0.5) / chi
        ok = k < kmax
        cl[i] = pref * trapezoid(((g / a) ** 2 * pk.P(z, k, grid=False) * dchi_dz)[ok], z[ok])
    kappa_empty = 1.5 * H0c**2 * om * trapezoid(g * chi / a * dchi_dz, z)
    return cl, kappa_empty


# ------------------------------------------------------------------ the Fourier grid
def lgrid(n=N, pix=PIX):
    """(l1, l2, |l|) on the full FFT grid, in inverse radians."""
    f = 2 * np.pi * np.fft.fftfreq(n, d=pix)
    l1, l2 = np.meshgrid(f, f, indexing="ij")
    return l1, l2, np.hypot(l1, l2)


L1, L2, LMAG = lgrid()
with np.errstate(invalid="ignore", divide="ignore"):
    E2IPHI = np.where(LMAG > 0, (L1 + 1j * L2) ** 2 / LMAG**2, 0.0)   # e^{2 i phi_l}


def cl_on_grid(ells, cl):
    """C_ell interpolated (log-log) onto |l| of the grid; zero at l = 0."""
    out = np.zeros_like(LMAG)
    ok = LMAG > 0
    out[ok] = np.exp(np.interp(np.log(LMAG[ok]), np.log(ells), np.log(cl)))
    return out


# ------------------------------------------------------------------ maps
def gaussian_map(P, white):
    """Gaussian field with grid spectrum P from a unit white-noise map (same white -> same phases)."""
    return np.fft.ifft2(np.fft.fft2(white) * np.sqrt(P / A_PIX)).real


def lognormal_spectrum(P, lam):
    """Spectrum P_G of the Gaussian field G such that kappa = lam (e^{G - s^2/2} - 1) has spectrum P.

    xi_kappa(x) = ifft2(P)/A_pix on the grid;  xi_G = ln(1 + xi_kappa / lam^2);  P_G = A_pix fft2(xi_G).
    Negative P_G (a few modes at the highest l) are set to zero.
    """
    xi = np.fft.ifft2(P).real / A_PIX
    xig = np.log1p(xi / lam**2)
    PG = np.fft.fft2(xig).real * A_PIX
    PG[0, 0] = 0.0
    return np.clip(PG, 0.0, None), xig[0, 0]


def lognormal_map(PG, lam, white):
    """Shifted lognormal convergence with zero mean, built from the Gaussian spectrum PG."""
    G = gaussian_map(PG, white)
    s2 = G.var()
    k = lam * (np.exp(G - s2 / 2) - 1.0)
    return k - k.mean()


def shear_from_kappa(kappa):
    """gamma_1 + i gamma_2 from kappa: multiply each Fourier mode by e^{2 i phi_l}."""
    g = np.fft.ifft2(E2IPHI * np.fft.fft2(kappa))
    return g.real, g.imag


def kaiser_squires(g1, g2):
    """kappa_E + i kappa_B: multiply each mode of gamma_1 + i gamma_2 by e^{-2 i phi_l}."""
    k = np.fft.ifft2(np.conj(E2IPHI) * np.fft.fft2(g1 + 1j * g2))
    return k.real, k.imag


def smooth(m, theta_arcmin):
    """Gaussian smoothing with standard deviation theta (arcmin): multiply by exp(-l^2 theta^2/2)."""
    w = np.exp(-0.5 * (LMAG * theta_arcmin * ARCMIN) ** 2)
    return np.fft.ifft2(np.fft.fft2(m) * w).real


def sigma_pix(ngal=NGAL, sigma_e=SIGMA_E):
    """Shape-noise rms per shear component in one pixel: sigma_e / sqrt(n A_pix)."""
    return sigma_e / np.sqrt(ngal * PIX_ARCMIN**2)


def sigma_noise_smoothed(theta=THETA_G, ngal=NGAL, sigma_e=SIGMA_E):
    """rms of the smoothed kappa_E noise on the grid (exact lattice sum) and the continuum formula
    sigma_e^2 / (4 pi theta^2 n)."""
    w2 = np.exp(-(LMAG * theta * ARCMIN) ** 2)
    lattice = sigma_pix(ngal, sigma_e) ** 2 * w2.mean()
    continuum = sigma_e**2 / (4 * np.pi * theta**2 * ngal)
    return np.sqrt(lattice), np.sqrt(continuum)


# ------------------------------------------------------------------ a survey mask on the patch
def make_mask(rng, nholes=40, rmin=3.0, rmax=9.0, strip_deg=0.4):
    """1 = observed, 0 = masked: circular holes around bright stars (radius rmin..rmax arcmin)
    and a strip of width strip_deg along one edge (where the survey footprint ends)."""
    x = np.arange(N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    m = np.ones((N, N), bool)
    for _ in range(nholes):
        cx, cy = rng.integers(0, N, 2)
        r = rng.uniform(rmin, rmax) / PIX_ARCMIN
        dx = (X - cx + N / 2) % N - N / 2
        dy = (Y - cy + N / 2) % N - N / 2
        m &= dx**2 + dy**2 > r**2
    strip = int(round(strip_deg * 60 / PIX_ARCMIN))
    m[:, :strip] = False
    return m


# ------------------------------------------------------------------ two-point estimator
CL_EDGES = np.logspace(np.log10(112.0), np.log10(5000.0), 9)    # 8 band powers
def band_power(m, edges):
    """Binned power P_hat(l) = A_pix |FFT|^2 / N_pix averaged over the modes with edges[b] <= |l| < edges[b+1]."""
    p = np.abs(np.fft.fft2(m)) ** 2 * A_PIX / m.size
    idx = np.digitize(LMAG.ravel(), edges) - 1
    nb = len(edges) - 1
    ok = (idx >= 0) & (idx < nb)
    s = np.bincount(idx[ok], weights=p.ravel()[ok], minlength=nb)
    c = np.bincount(idx[ok], minlength=nb)
    return s / c


def modes_per_bin(edges):
    """Number of grid modes in each bin (both l and -l counted)."""
    idx = np.digitize(LMAG.ravel(), edges) - 1
    nb = len(edges) - 1
    ok = (idx >= 0) & (idx < nb)
    return np.bincount(idx[ok], minlength=nb)


def ell_mean(edges):
    idx = np.digitize(LMAG.ravel(), edges) - 1
    nb = len(edges) - 1
    ok = (idx >= 0) & (idx < nb)
    return np.bincount(idx[ok], weights=LMAG.ravel()[ok], minlength=nb) / np.bincount(idx[ok], minlength=nb)


# ------------------------------------------------------------------ beyond two points
def peaks(u, edges):
    """Histogram of the heights of local maxima (higher than all 8 neighbours, periodic)."""
    mx = ndimage.maximum_filter(u, footprint=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="wrap")
    h = u[u > mx]
    return np.histogram(h, bins=edges)[0], h


def minkowski(u, nus, dnu=0.1):
    """V0 (area fraction), V1 (boundary length per area, 1/arcmin), V2 (chi per area, 1/arcmin^2)
    of {u > nu}; u is in units of the noise rms, so the thresholds are FIXED, not rescaled per map."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from lib_topo import spectral_gradient, euler_cubical
    gx, gy = spectral_gradient(u)
    g = np.hypot(gx, gy) / PIX_ARCMIN                       # per arcmin
    area = u.size * PIX_ARCMIN**2
    v0 = np.array([(u > nu).mean() for nu in nus])
    v1 = np.array([(g * (np.abs(u - nu) < dnu / 2)).sum() * PIX_ARCMIN**2 / dnu / area for nu in nus])
    v2 = np.array([euler_cubical(u > nu, 8, periodic=True) / area for nu in nus])
    return v0, v1, v2


def _uf_pairs_py(h, ny, nx, order, steps):
    """Union-find on a periodic grid (the loop of lib_persist.islands, with wrap-around).
    Pixels enter from the highest h down; returns the (birth, death) pairs of the islands that
    die (the elder rule: when two islands meet, the one with the lower peak dies)."""
    parent = np.full(h.size, -1, dtype=np.int64)
    peak = np.empty(h.size)
    pairs = np.empty((h.size, 2))
    npair = 0
    for p in order:
        parent[p], peak[p] = p, h[p]
        i, j = p // nx, p % nx
        for k in range(steps.shape[0]):
            q = ((i + steps[k, 0]) % ny) * nx + (j + steps[k, 1]) % nx    # periodic neighbour
            if parent[q] < 0:
                continue                                    # still under water
            r1 = p
            while parent[r1] != r1:
                parent[r1] = parent[parent[r1]]
                r1 = parent[r1]
            r2 = q
            while parent[r2] != r2:
                parent[r2] = parent[parent[r2]]
                r2 = parent[r2]
            if r1 == r2:
                continue
            if peak[r1] < peak[r2]:
                r1, r2 = r2, r1                             # r1: the elder island
            if peak[r2] > h[p]:                             # zero-length pairs are not kept
                pairs[npair, 0], pairs[npair, 1] = peak[r2], h[p]
                npair += 1
            parent[r2] = r1
    return pairs[:npair]


try:                                                        # the same loop, compiled (100x faster)
    from numba import njit
    _uf_pairs = njit(cache=False)(_uf_pairs_py)
except ImportError:                                         # pragma: no cover
    _uf_pairs = _uf_pairs_py

_STEPS8 = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int64)
_STEPS4 = np.array([(-1, 0), (0, -1), (0, 1), (1, 0)], dtype=np.int64)


def islands_periodic(u, conn=8):
    """Finite H0 pairs (b, d), b > d, of the superlevel sets {u >= nu} on the torus."""
    h = np.ascontiguousarray(u, dtype=float).ravel()
    order = np.argsort(-h, kind="stable").astype(np.int64)
    return _uf_pairs(h, u.shape[0], u.shape[1], order, _STEPS8 if conn == 8 else _STEPS4)


def persistence_pairs(u):
    """Superlevel persistence of a periodic map: (H0 pairs, H1 pairs) as (b, d), b >= d.

    H0: islands of {u >= nu} (pixels are closed squares, so corner-touching pixels connect:
    8 neighbours), by union-find; the one island that never dies keeps d = -inf.
    H1: the lakes, by duality.  A lake of the land is a body of water {u < nu}, and the water
    (4-connected) is the superlevel set of -u.  A water body born at its bottom m and merging at
    the pass s is a land loop that closes when the level drops to s and fills when it drops
    below m: (b, d) = (s, m).  On the torus the duality pairs the finite features only; the two
    loops that wrap around the torus never close up and are not counted (as with gudhi's
    PeriodicCubicalComplex after dropping its infinite H1 intervals)."""
    h0 = islands_periodic(u, 8)
    h0 = np.vstack([h0, [[float(np.max(u)), -np.inf]]])
    w = islands_periodic(-u, 4)
    h1 = np.column_stack([-w[:, 1], -w[:, 0]])
    return h0, h1


def betti_from_pairs(pairs, nus):
    """beta(nu) = number of features alive at nu: d < nu <= b."""
    b, d = pairs[:, 0][:, None], pairs[:, 1][:, None]
    return ((d < nus[None, :]) & (nus[None, :] <= b)).sum(axis=0)


def persistent_betti(pairs, grid):
    """beta(nu_hi, nu_lo) = number of features alive over the whole range [nu_lo, nu_hi]:
    born at or above nu_hi and dying below nu_lo (pairs nu_hi >= nu_lo from the grid)."""
    b, d = pairs[:, 0], pairs[:, 1]
    out = []
    for i, hi in enumerate(grid):
        for lo in grid[: i + 1]:
            out.append(int(((b >= hi) & (d < lo)).sum()))
    return np.array(out)
