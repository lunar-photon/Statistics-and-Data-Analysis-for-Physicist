"""lib_t3.py -- small flat-sky toolkit for the CMB morphology chapter (T3).

Everything here works on a square periodic patch of the sky, N x N pixels of side dx
radians (patch side L = N dx). The flat-sky approximation replaces the multipole ell by
the length of the 2-D wave vector, |k| = ell.

  gaussian_flat(cl, N, dx, rng)        Gaussian map with angular power spectrum C_ell
  polarization_flat(clee, clbb, ...)   Q and U maps from E- and B-mode spectra
  moments_flat(cl, N, dx)              sigma_0^2 = <u^2>, sigma_1^2 = <|grad u|^2> on the grid
  euler_torus(mask)                    Euler characteristic (V - E + F) of a pixel set on the torus
  minkowski(u, nus, dx)                area fraction, boundary length per area, chi per area
  mf_gauss(nus, s0, s1)                the Gaussian predictions for the three functionals
  singularities(Q, U)                  zeros of Q + iU and their index (+1/2 or -1/2)
  camb_unlensed_phi()                  unlensed TT, EE and C_ell^{phi phi} (cached in data/chT3)
"""
from __future__ import annotations

import pathlib

import numpy as np
from scipy.special import erfc

NOTES = pathlib.Path(__file__).resolve().parents[2]
ARCMIN = np.pi / 180 / 60


# ------------------------------------------------------------------ grids and spectra
def ell_grid(N: int, dx: float):
    """|ell|, ell_x, ell_y on the FFT grid of an N x N patch with pixel side dx (radians)."""
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    lx, ly = np.meshgrid(k, k, indexing="ij")
    return np.hypot(lx, ly), lx, ly


def cl_on_grid(cl, ell):
    """Interpolate C_ell (array indexed by integer ell) onto the grid; zero beyond the table."""
    cl = np.asarray(cl, float)
    return np.interp(ell, np.arange(cl.size), cl, right=0.0)


def beam(ell, fwhm_arcmin: float):
    """Gaussian beam window b_ell = exp(-ell(ell+1) s^2 / 2), s = FWHM / sqrt(8 ln 2)."""
    s = fwhm_arcmin * ARCMIN / np.sqrt(8 * np.log(2))
    return np.exp(-0.5 * ell * (ell + 1) * s ** 2)


# ------------------------------------------------------------------ simulation
def gaussian_flat(cl, N: int, dx: float, rng, nsim: int | None = None, fourier=False):
    """Gaussian map(s) whose Fourier modes have variance C_ell / pixel area (flat sky).

    Unit white noise has <|FFT_k|^2> = N^2; multiplying by sqrt(C(|k|)/dx^2) gives a map with
    variance sum_k C(k) / (N dx)^2 = int d^2 ell/(2 pi)^2 C_ell, the flat-sky <u^2>.
    """
    ell, _, _ = ell_grid(N, dx)
    amp = np.sqrt(cl_on_grid(cl, ell)) / dx
    amp[0, 0] = 0.0                                    # no monopole on the patch
    shape = ((nsim,) if nsim else ()) + (N, N)
    wk = np.fft.fft2(rng.standard_normal(shape), axes=(-2, -1)) * amp
    if fourier:
        return wk
    return np.fft.ifft2(wk, axes=(-2, -1)).real


def polarization_flat(clee, clbb, N: int, dx: float, rng):
    """Q, U maps: Q_k = E_k cos 2phi - B_k sin 2phi,  U_k = E_k sin 2phi + B_k cos 2phi."""
    ell, lx, ly = ell_grid(N, dx)
    phi = np.arctan2(ly, lx)
    Ek = gaussian_flat(clee, N, dx, rng, fourier=True)
    Bk = gaussian_flat(clbb, N, dx, rng, fourier=True) if clbb is not None else 0.0
    c, s = np.cos(2 * phi), np.sin(2 * phi)
    Q = np.fft.ifft2(Ek * c - Bk * s).real
    U = np.fft.ifft2(Ek * s + Bk * c).real
    return Q, U


def smooth(u, fwhm_arcmin: float, dx: float):
    ell, _, _ = ell_grid(u.shape[-1], dx)
    return np.fft.ifft2(np.fft.fft2(u, axes=(-2, -1)) * beam(ell, fwhm_arcmin), axes=(-2, -1)).real


def gradient(u, dx: float):
    """Spectral gradient (exact for a band-limited periodic map)."""
    ell, lx, ly = ell_grid(u.shape[-1], dx)
    uk = np.fft.fft2(u, axes=(-2, -1))
    return (np.fft.ifft2(1j * lx * uk, axes=(-2, -1)).real,
            np.fft.ifft2(1j * ly * uk, axes=(-2, -1)).real)


def moments_flat(cl, N: int, dx: float):
    """Expected sigma_0^2 = <u^2> and sigma_1^2 = <|grad u|^2> of gaussian_flat(cl, N, dx)."""
    ell, _, _ = ell_grid(N, dx)
    c = cl_on_grid(cl, ell)
    c[0, 0] = 0.0
    A = (N * dx) ** 2
    return c.sum() / A, (c * ell ** 2).sum() / A


def moments_sphere(cl):
    """Full-sky sigma_0^2 = sum (2l+1) C_l / 4 pi and sigma_1^2 = sum (2l+1) l(l+1) C_l / 4 pi."""
    l = np.arange(len(cl))
    w = (2 * l + 1) / (4 * np.pi) * np.asarray(cl, float)
    return w.sum(), (w * l * (l + 1)).sum()


# ------------------------------------------------------------------ morphology
def euler_torus(mask):
    """Euler characteristic of the union of closed pixel squares in `mask`, on the periodic grid.

    F = pixels in the set; an edge belongs to the complex if either pixel sharing it is in the set;
    a vertex belongs if any of the four pixels around it is in the set.  chi = V - E + F.
    (Closed squares touching at a corner are connected: 8-connectivity for the set.)
    """
    m = np.asarray(mask, bool)
    F = m.sum(axis=(-2, -1))
    r = lambda a, s, ax: np.roll(a, s, axis=ax)
    Eh = (m | r(m, 1, -2)).sum(axis=(-2, -1))      # edges between vertical neighbours
    Ev = (m | r(m, 1, -1)).sum(axis=(-2, -1))      # edges between horizontal neighbours
    V = (m | r(m, 1, -2) | r(m, 1, -1) | r(r(m, 1, -2), 1, -1)).sum(axis=(-2, -1))
    return V - (Eh + Ev) + F


def minkowski(u, nus, dx: float, width: float | None = None):
    """Minkowski functionals per unit area of the excursion sets {u > nu} of a map in sigma units.

    v0 : area fraction (pixel count);
    v1 : (1/4) x boundary length per area, from <|grad u| delta(u - nu)> with a binned delta
         (the estimator of Schmalzing & Gorski 1998, eq. 23-24);
    v2 : Euler characteristic per area, by counting V - E + F on the torus.
    """
    nus = np.asarray(nus, float)
    gx, gy = gradient(u, dx)
    g = np.hypot(gx, gy)
    width = width if width is not None else (nus[1] - nus[0])
    A = u.size * dx ** 2
    v0 = np.array([(u > n).mean() for n in nus])
    v1 = np.array([0.25 * np.mean(g * (np.abs(u - n) < width / 2)) / width for n in nus])
    v2 = np.array([euler_torus(u > n) for n in nus]) / A
    return v0, v1, v2


def mf_gauss(nus, s0: float, s1: float):
    """Gaussian predictions (Tomita 1986; Schmalzing & Gorski 1998 eq. 14), per unit area."""
    nus = np.asarray(nus, float)
    e = np.exp(-nus ** 2 / 2)
    v0 = 0.5 * erfc(nus / np.sqrt(2))
    v1 = (1 / 8) * (s1 / (np.sqrt(2) * s0)) * e
    v2 = (1 / (2 * np.pi) ** 1.5) * (s1 ** 2 / (2 * s0 ** 2)) * nus * e
    return v0, v1, v2


# ------------------------------------------------------------------ polarization singularities
def singularities(Q, U):
    """Zeros of P = Q + iU on the periodic grid, one per plaquette, with their index.

    Around each plaquette (i,j)->(i+1,j)->(i+1,j+1)->(i,j+1)->(i,j) add the changes of
    arg(Q + iU), each wrapped into (-pi, pi]. The total is 2 pi n with n an integer; the
    polarization angle alpha = arg/2 turns by n pi, so the index is n/2.
    Returns (i, j, index) arrays; positions are plaquette centres (i + 1/2, j + 1/2).
    """
    a = np.angle(Q + 1j * U)
    def d(x, y):
        return np.angle(np.exp(1j * (y - x)))
    a10 = np.roll(a, -1, 0)
    a11 = np.roll(a10, -1, 1)
    a01 = np.roll(a, -1, 1)
    w = d(a, a10) + d(a10, a11) + d(a11, a01) + d(a01, a)
    n = np.rint(w / (2 * np.pi)).astype(int)
    i, j = np.nonzero(n)
    return i, j, n[i, j] / 2


# ------------------------------------------------------------------ CAMB (one call, cached)
def camb_unlensed_phi(lmax: int = 5000):
    """Unlensed TT, EE (muK^2) and the lensing-potential spectrum C_ell^{phi phi}, fiducial model."""
    path = NOTES / "data" / "chT3" / "camb_unlensed_phi.npz"
    if path.exists():
        z = np.load(path)
        return z["ell"], z["TT"], z["EE"], z["PP"]
    import sys
    sys.path.insert(0, str(NOTES / "code"))
    import camb
    from camb_fiducial import FIDUCIAL
    p = camb.set_params(**FIDUCIAL, lmax=lmax + 300, lens_potential_accuracy=1)
    r = camb.get_results(p)
    un = r.get_unlensed_scalar_cls(CMB_unit="muK", raw_cl=True)[: lmax + 1]
    pp = r.get_lens_potential_cls(raw_cl=True)[: lmax + 1, 0]     # C_ell^{phi phi}
    ell = np.arange(lmax + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ell=ell, TT=un[:, 0], EE=un[:, 1], PP=pp)
    return ell, un[:, 0], un[:, 1], pp
