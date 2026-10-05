"""lib_fields.py -- Gaussian random fields on a flat periodic box and on the sphere.

Shared by chapters 7, 8, 10 and 11.  Import with

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
    from lib_fields import gaussian_field, measure_pk, measure_xi, gaussian_field_sphere

Conventions (flat box, d dimensions, side L, N cells per side, cell size D = L/N, V = L^d):

    delta_k = int d^dx delta(x) e^{-i k.x}   ~=  D^d * FFT(delta)_k
    delta(x) = (1/V) sum_k delta_k e^{i k.x}  =   IFFT(delta_k) / D^d
    <delta_k delta_k'^*> = V P(k) Kronecker(k, k')       (box version of (2 pi)^d delta_D P)

so that <|FFT_k|^2> = N^d P(k) / D^d.  A field with spectrum P is made by
multiplying the FFT of unit white noise by sqrt(P(k)/D^d) and transforming back.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ k grids
def kgrid(N: int, L: float, d: int, real: bool = True):
    """|k| on the FFT grid (rfft layout if real=True), and the list of component arrays."""
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kl = 2 * np.pi * np.fft.rfftfreq(N, d=L / N) if real else k1
    axes = [k1] * (d - 1) + [kl]
    comps = np.meshgrid(*axes, indexing="ij")
    kmag = np.sqrt(sum(c ** 2 for c in comps))
    return kmag, comps


def rfft_weights(N: int, d: int):
    """How many full-grid modes each rfft cell stands for (1 for the self-conjugate planes, else 2)."""
    shape = [N] * (d - 1) + [N // 2 + 1]
    w = np.full(shape, 2.0)
    w[..., 0] = 1.0
    if N % 2 == 0:
        w[..., -1] = 1.0
    return w


# ------------------------------------------------------------------ simulation
def gaussian_field(P, N: int, L: float, d: int = 2, rng=None, nsim: int | None = None):
    """Gaussian random field(s) with power spectrum P(k) on a periodic box.

    P    : callable P(kmag) (vectorised), in units of length^d
    nsim : if given, return an array of nsim independent fields (axis 0)
    The k = 0 mode is set to zero, so each box has exactly zero mean.
    """
    rng = np.random.default_rng() if rng is None else rng
    D = L / N
    kmag, _ = kgrid(N, L, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.sqrt(np.where(kmag > 0, P(np.where(kmag > 0, kmag, 1.0)), 0.0) / D ** d)
    shape = ([nsim] if nsim else []) + [N] * d
    axes = tuple(range(-d, 0))
    w = rng.standard_normal(shape)                      # unit white noise, one number per cell
    wk = np.fft.rfftn(w, axes=axes)                     # its FFT: <|wk|^2> = N^d, Hermitian
    return np.fft.irfftn(wk * amp, s=[N] * d, axes=axes)  # colour it, transform back (real)


def gaussian_field_sphere(cl, nside: int, rng=None, lmax: int | None = None):
    """Statistically isotropic Gaussian map on the sphere (HEALPix), from C_ell, with our own rng.

    a_l0 ~ N(0, C_l) real;  a_lm (m>0): Re, Im ~ N(0, C_l/2);  a_l,-m = (-1)^m a_lm^* (implicit in healpy).
    """
    import healpy as hp
    rng = np.random.default_rng() if rng is None else rng
    cl = np.asarray(cl, dtype=float)
    lmax = min(len(cl) - 1, 3 * nside - 1) if lmax is None else lmax
    ell, m = hp.Alm.getlm(lmax)
    sig = np.sqrt(cl[ell])
    re = rng.standard_normal(ell.size)
    im = rng.standard_normal(ell.size)
    alm = np.where(m == 0, sig * re, sig * (re + 1j * im) / np.sqrt(2)).astype(complex)
    return hp.alm2map(alm, nside, lmax=lmax), alm


# ------------------------------------------------------------------ estimators
def _shells(N, L, d, nbins, kmin, kmax, log):
    kmag, _ = kgrid(N, L, d)
    kmin = 2 * np.pi / L if kmin is None else kmin      # fundamental mode k_f = 2 pi / L
    kmax = np.pi * N / L if kmax is None else kmax      # Nyquist k_Ny = pi N / L
    edges = np.geomspace(kmin, kmax, nbins + 1) if log else np.linspace(kmin, kmax, nbins + 1)
    idx = (np.digitize(kmag.ravel(), edges) - 1)
    ok = (idx >= 0) & (idx < nbins)
    return kmag, rfft_weights(N, d).ravel(), idx, ok


def measure_pk(field, L: float, nbins: int = 40, kmin=None, kmax=None, log=False, P=None):
    """Shell-averaged power spectrum estimate  P_hat(k) = |D^d FFT_k|^2 / V  averaged over the modes in a shell.

    Returns (k_mean, P_hat, N_modes), N_modes counting both k and -k (full grid).
    If a callable P is given, also returns P averaged over the same modes (the exact expectation of P_hat).
    """
    field = np.asarray(field)
    d = field.ndim
    N = field.shape[0]
    D = L / N
    p = np.abs(np.fft.rfftn(field) * D ** d) ** 2 / L ** d
    kmag, w, idx, ok = _shells(N, L, d, nbins, kmin, kmax, log)
    def avg(q):
        return np.bincount(idx[ok], weights=(w * q.ravel())[ok], minlength=nbins)
    nm = np.bincount(idx[ok], weights=w[ok], minlength=nbins)
    g = nm > 0
    out = [avg(kmag)[g] / nm[g], avg(p)[g] / nm[g], nm[g]]
    if P is not None:
        km = np.where(kmag > 0, kmag, 1.0)
        out.append(avg(P(km))[g] / nm[g])
    return tuple(out)


def radial_average(grid, L: float, nbins: int = 40, rmax=None):
    """Average a function of the separation vector (FFT layout, origin at index 0) over shells of |r|."""
    grid = np.asarray(grid)
    d = grid.ndim
    N = grid.shape[0]
    x1 = np.fft.fftfreq(N, d=1.0 / N) * (L / N)                    # signed separations
    comps = np.meshgrid(*([x1] * d), indexing="ij")
    r = np.sqrt(sum(c ** 2 for c in comps))
    rmax = L / 2 if rmax is None else rmax
    edges = np.linspace(0, rmax, nbins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    n = np.bincount(idx[ok], minlength=nbins)
    s = np.bincount(idx[ok], weights=grid.ravel()[ok], minlength=nbins)
    rs = np.bincount(idx[ok], weights=r.ravel()[ok], minlength=nbins)
    good = n > 0
    return rs[good] / n[good], s[good] / n[good]


def measure_xi(field, L: float, nbins: int = 40, rmax=None):
    """Correlation function estimate xi_hat(r) = (1/N^d) sum_x delta(x) delta(x+r), periodic, shell-averaged."""
    field = np.asarray(field)
    d = field.ndim
    N = field.shape[0]
    fk = np.fft.rfftn(field)
    xi_grid = np.fft.irfftn(np.abs(fk) ** 2, s=[N] * d, axes=tuple(range(d))) / N ** d   # Wiener-Khinchin on the grid
    return radial_average(xi_grid, L, nbins, rmax)


# ------------------------------------------------------------------ operations on fields
def smooth(field, L: float, W):
    """Convolve with a window whose Fourier transform is W(kmag):  delta_R(k) = W(k) delta(k)."""
    field = np.asarray(field)
    d = field.ndim
    N = field.shape[0]
    kmag, _ = kgrid(N, L, d)
    return np.fft.irfftn(np.fft.rfftn(field) * W(kmag), s=[N] * d, axes=tuple(range(d)))


def randomise_phases(field, rng=None):
    """Same |FFT| as field, phases replaced by those of fresh white noise (Hermitian symmetry automatic)."""
    rng = np.random.default_rng() if rng is None else rng
    field = np.asarray(field)
    d = field.ndim
    N = field.shape[0]
    fk = np.fft.rfftn(field)
    wk = np.fft.rfftn(rng.standard_normal(field.shape))
    u = wk / np.abs(wk)                      # unit complex numbers, uniform phases
    out = np.abs(fk) * u
    out.flat[0] = fk.flat[0]                 # keep the mean
    return np.fft.irfftn(out, s=[N] * d, axes=tuple(range(d)))


def swap_phases(amp_field, phase_field):
    """Field with the Fourier amplitudes of amp_field and the Fourier phases of phase_field."""
    a = np.fft.rfftn(amp_field)
    b = np.fft.rfftn(phase_field)
    out = np.abs(a) * np.exp(1j * np.angle(b))
    N = np.asarray(amp_field).shape[0]
    return np.fft.irfftn(out, s=[N] * np.asarray(amp_field).ndim, axes=tuple(range(np.asarray(amp_field).ndim)))


# ------------------------------------------------------------------ analytic helpers
def xi_from_pk(P, r, d: int = 2, kmax: float = 50.0, nk: int = 20000):
    """Isotropic xi(r) from P(k): d=2 uses J0(kr) k dk/2pi, d=3 uses sin(kr)/kr k^2 dk/2pi^2."""
    from scipy.special import j0
    k = np.linspace(1e-6, kmax, nk)
    pk = P(k)
    kr = np.outer(np.atleast_1d(r), k)
    if d == 2:
        integ = k * pk * j0(kr) / (2 * np.pi)
    elif d == 3:
        integ = k ** 2 * pk * np.sinc(kr / np.pi) / (2 * np.pi ** 2)
    elif d == 1:
        integ = pk * np.cos(kr) / np.pi
    else:
        raise ValueError(d)
    return np.trapezoid(integ, k, axis=1)


def xi_box(P, N: int, L: float, d: int = 2):
    """The exact expectation of measure_xi for a field made by gaussian_field: (1/V) sum_k P(k) e^{ikr}."""
    D = L / N
    kmag, _ = kgrid(N, L, d, real=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk = np.where(kmag > 0, P(np.where(kmag > 0, kmag, 1.0)), 0.0)
    return np.real(np.fft.ifftn(pk)) / D ** d       # grid of xi at every separation vector
