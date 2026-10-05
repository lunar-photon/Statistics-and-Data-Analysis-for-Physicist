"""lib_cmbsim.py -- a small simulator of an observed CMB temperature sky.

What an instrument does to the sky, step by step, and how the power spectrum is read back:

    sky a_lm  --(beam B_l, pixel window p_l)-->  map  --(+ pixel noise)-->  observed map d
    d  --(map2alm, average |a|^2 over m)-->  C_obs_l  --(- N_l, / (B_l p_l)^2)-->  C_hat_l

Shared by chapter 8 (both parts), chapter 9 (CMB likelihood) and chapter 11 (topology of
noisy, beamed maps).  Import with

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch08"))
    import lib_cmbsim as cs
    exp = cs.Experiment()                      # the toy experiment of chapter 8
    sims = cs.simulate(exp, cl, nsim, rng)     # dict of spectra, one row per simulated sky

Conventions: raw C_l in muK^2 (not D_l), angles in radians unless the name ends in _arcmin,
noise depth Delta_T in muK*arcmin, white-noise power N_l = sigma_pix^2 * Omega_pix in muK^2.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import healpy as hp

ARCMIN = np.pi / 180.0 / 60.0          # one arcminute in radians
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"


# ------------------------------------------------------------------ the instrument
def gaussian_beam(fwhm_arcmin: float, lmax: int) -> np.ndarray:
    """B_l = exp(-l(l+1) sigma_b^2 / 2) with sigma_b = FWHM / sqrt(8 ln 2)  (small-beam formula)."""
    sb = fwhm_arcmin * ARCMIN / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(lmax + 1)
    return np.exp(-0.5 * ell * (ell + 1) * sb ** 2)


def gaussian_beam_exact(fwhm_arcmin: float, lmax: int, nth: int = 20001) -> np.ndarray:
    """B_l = 2 pi int b(theta) P_l(cos theta) sin theta d theta for a normalised Gaussian profile.

    The profile b(theta) ~ exp(-theta^2 / 2 sigma_b^2) is normalised on the sphere (B_0 = 1);
    Legendre polynomials come from the three-term recursion, so nothing is approximated
    except the quadrature (the integrand dies within ~8 sigma_b of the pole).
    """
    sb = fwhm_arcmin * ARCMIN / np.sqrt(8.0 * np.log(2.0))
    th = np.linspace(0.0, min(np.pi, 12 * sb), nth)
    w = np.exp(-0.5 * (th / sb) ** 2) * np.sin(th)
    x = np.cos(th)
    out = np.empty(lmax + 1)
    p0, p1 = np.ones_like(x), x.copy()
    norm = np.trapezoid(w * p0, th)
    out[0] = 1.0
    if lmax >= 1:
        out[1] = np.trapezoid(w * p1, th) / norm
    for l in range(1, lmax):
        p0, p1 = p1, ((2 * l + 1) * x * p1 - l * p0) / (l + 1)   # Bonnet recursion
        out[l + 1] = np.trapezoid(w * p1, th) / norm
    return out


def pixel_window(nside: int, lmax: int) -> np.ndarray:
    """HEALPix pixel window p_l: the harmonic response of averaging the sky over one pixel."""
    return np.asarray(hp.pixwin(nside, lmax=lmax))


@dataclass
class Experiment:
    """A full-sky temperature experiment: resolution, depth and the analysis band limit."""
    nside: int = 256
    fwhm_arcmin: float = 30.0
    depth_uK_arcmin: float = 200.0      # Delta_T: white-noise level per square arcminute
    lmax: int = 512                     # analysis band limit, 2 nside (trusted range, ch. 7)
    pixwin: bool = True

    @property
    def lmax_sim(self) -> int:          # synthesis band limit, 3 nside - 1
        return 3 * self.nside - 1

    @property
    def npix(self) -> int:
        return hp.nside2npix(self.nside)

    @property
    def omega_pix(self) -> float:       # sr
        return 4.0 * np.pi / self.npix

    @property
    def sigma_pix(self) -> float:       # muK per pixel: Delta_T / pixel side
        return self.depth_uK_arcmin * ARCMIN / np.sqrt(self.omega_pix)

    def nl(self, lmax: int | None = None) -> np.ndarray:
        """White-noise power N_l = sigma_pix^2 Omega_pix = (Delta_T in muK rad)^2, flat in l."""
        lmax = self.lmax if lmax is None else lmax
        return np.full(lmax + 1, self.sigma_pix ** 2 * self.omega_pix)

    def bl(self, lmax: int | None = None) -> np.ndarray:
        return gaussian_beam(self.fwhm_arcmin, self.lmax_sim if lmax is None else lmax)

    def pl(self, lmax: int | None = None) -> np.ndarray:
        lmax = self.lmax_sim if lmax is None else lmax
        return pixel_window(self.nside, lmax) if self.pixwin else np.ones(lmax + 1)

    def transfer(self, lmax: int | None = None) -> np.ndarray:
        """T_l = B_l p_l: what the instrument multiplies each a_lm of the sky by."""
        return self.bl(lmax) * self.pl(lmax)


# ------------------------------------------------------------------ analytic statistics
def expected_cl_obs(cl, exp: Experiment, lmax=None):
    """<C_obs_l> = T_l^2 C_l + N_l  (full sky, white noise)."""
    lmax = exp.lmax if lmax is None else lmax
    return exp.transfer(lmax) ** 2 * cl[: lmax + 1] + exp.nl(lmax)


def noise_over_transfer(exp: Experiment, lmax=None):
    """N_l / T_l^2: the noise as it looks after the beam has been divided out."""
    lmax = exp.lmax if lmax is None else lmax
    return exp.nl(lmax) / exp.transfer(lmax) ** 2


def var_fullsky(cl, exp: Experiment, lmax=None):
    """Var(C_hat_l) = 2 (C_l + N_l/T_l^2)^2 / (2l+1)  for the debiased, deconvolved estimator."""
    lmax = exp.lmax if lmax is None else lmax
    ell = np.arange(lmax + 1)
    return 2.0 * (cl[: lmax + 1] + noise_over_transfer(exp, lmax)) ** 2 / (2 * ell + 1)


def var_knox(cl, exp: Experiment, fsky: float = 1.0, dl: int = 1, lmax=None):
    """Knox's rule of thumb: Var = 2 (C_l + N_l/T_l^2)^2 / ((2l+1) dl fsky)  (an approximation)."""
    return var_fullsky(cl, exp, lmax) / (fsky * dl)


def snr_per_ell(cl, exp: Experiment, lmax=None):
    """C_l / sigma(C_hat_l) = sqrt((2l+1)/2) C_l / (C_l + N_l/T_l^2)."""
    lmax = exp.lmax if lmax is None else lmax
    ell = np.arange(lmax + 1)
    s = np.sqrt((2 * ell + 1) / 2.0) * cl[: lmax + 1] / (cl[: lmax + 1] + noise_over_transfer(exp, lmax))
    s[:2] = 0.0
    return s


def snr_cumulative(cl, exp: Experiment, lmax=None):
    """Cumulative SNR(<= l): sqrt of the running sum of the squared per-l SNR (l >= 2)."""
    return np.sqrt(np.cumsum(snr_per_ell(cl, exp, lmax) ** 2))


def ell_equality(cl, exp: Experiment, lmin: int = 30, lmax=None) -> int:
    """First multipole above lmin where the beamed signal T_l^2 C_l drops below the noise N_l."""
    lmax = exp.lmax_sim if lmax is None else lmax
    s = exp.transfer(lmax) ** 2 * cl[: lmax + 1]
    ell = np.arange(lmax + 1)
    idx = np.where((s < exp.nl(lmax)) & (ell >= lmin))[0]
    return int(idx[0]) if idx.size else -1


# ------------------------------------------------------------------ simulation
def synalm(cl, lmax: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian a_lm (healpy m >= 0 layout) with <|a_lm|^2> = C_l, from our own generator.

    a_l0 ~ N(0, C_l) is real; for m > 0, Re and Im ~ N(0, C_l/2) independently.
    """
    ell, m = hp.Alm.getlm(lmax)
    sig = np.sqrt(np.asarray(cl, dtype=float)[ell])
    re = rng.standard_normal(ell.size)
    im = rng.standard_normal(ell.size)
    return np.where(m == 0, sig * re, sig * (re + 1j * im) / np.sqrt(2.0)).astype(complex)


def observe_signal(alm_sky, exp: Experiment) -> np.ndarray:
    """Pixel map of the sky as the instrument sees it: a_lm -> T_l a_lm -> map."""
    alm = hp.almxfl(alm_sky, exp.transfer(exp.lmax_sim))
    return hp.alm2map(alm, exp.nside, lmax=exp.lmax_sim)


def variance_pattern(nside: int, contrast: float = 4.0) -> np.ndarray:
    """Relative noise variance v_p (mean 1) of a scan that piles up hits near the poles.

    hits(theta) ~ 1 + contrast cos^2(theta); the noise variance of a pixel is ~ 1/hits.
    """
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    v = 1.0 / (1.0 + contrast * np.cos(theta) ** 2)
    return v / v.mean()


def noise_map(exp: Experiment, rng: np.random.Generator, varmap=None, scale: float = 1.0):
    """Independent Gaussian pixel noise with variance (scale sigma_pix)^2 v_p (v_p = 1 if varmap is None)."""
    sd = scale * exp.sigma_pix * (1.0 if varmap is None else np.sqrt(varmap))
    return sd * rng.standard_normal(exp.npix)


def map2alm(m, lmax: int, iter: int = 3):
    return hp.map2alm(m, lmax=lmax, iter=iter)


def cross_cl(alm1, alm2=None):
    """C_l = (1/(2l+1)) sum_m Re(a1_lm a2_lm^*)  (the auto-spectrum when alm2 is None)."""
    return hp.alm2cl(alm1) if alm2 is None else hp.alm2cl(alm1, alm2)


def debias(cl_obs, exp: Experiment, nl=None, lmax=None):
    """Debiased, deconvolved estimator  C_hat_l = (C_obs_l - N_l) / T_l^2  (nl=0 for a cross-spectrum)."""
    lmax = exp.lmax if lmax is None else lmax
    nl = exp.nl(lmax) if nl is None else nl
    return (cl_obs[..., : lmax + 1] - nl) / exp.transfer(lmax) ** 2


def simulate(exp: Experiment, cl, nsim: int, rng: np.random.Generator, varmap=None, mask=None,
             progress: int = 0, iter_signal: int = 3, iter_noise: int = 1):
    """Monte Carlo of the experiment. Each simulated sky is observed four ways.

    Returns a dict of arrays of shape (nsim, lmax+1), float32:
      sky  -- C_hat of the sky a_lm themselves (cosmic variance only)
      sig  -- spectrum of the beamed, pixelised signal map (no noise)
      auto -- signal + white noise, full map           (noise = (n1 + n2)/2)
      cross-- cross-spectrum of two half maps s + n1 and s + n2 (each half twice as noisy)
      noise-- white noise alone, full map
      inh  -- signal + inhomogeneous noise with the same mean variance (if varmap is given)
      ninh -- inhomogeneous noise alone
    An optional mask multiplies every map before the transform (pseudo-C_l, chapter 8b).
    iter_* are healpy's quadrature iterations: 3 for the band-limited signal (error ~1e-5 below
    l = 400), 1 for pixel-white noise (its power is reproduced to the Monte Carlo precision).
    """
    L = exp.lmax
    keys = ["sky", "sig", "auto", "cross", "noise"] + (["inh", "ninh"] if varmap is not None else [])
    out = {k: np.empty((nsim, L + 1), dtype=np.float32) for k in keys}
    w = 1.0 if mask is None else mask
    for i in range(nsim):
        a = synalm(cl[: exp.lmax_sim + 1], exp.lmax_sim, rng)
        s = observe_signal(a, exp)
        n1 = noise_map(exp, rng, scale=np.sqrt(2.0))          # half-mission noise, variance 2 sigma^2
        n2 = noise_map(exp, rng, scale=np.sqrt(2.0))
        a_s = map2alm(w * s, L, iter_signal)
        a_1 = map2alm(w * n1, L, iter_noise)
        a_2 = map2alm(w * n2, L, iter_noise)
        a_n = 0.5 * (a_1 + a_2)                               # map2alm is linear
        out["sky"][i] = cross_cl(hp.resize_alm(a, exp.lmax_sim, exp.lmax_sim, L, L))
        out["sig"][i] = cross_cl(a_s)
        out["auto"][i] = cross_cl(a_s + a_n)
        out["cross"][i] = cross_cl(a_s + a_1, a_s + a_2)
        out["noise"][i] = cross_cl(a_n)
        if varmap is not None:
            a_i = map2alm(w * noise_map(exp, rng, varmap=varmap), L, iter_noise)
            out["inh"][i] = cross_cl(a_s + a_i)
            out["ninh"][i] = cross_cl(a_i)
        if progress and (i + 1) % progress == 0:
            print(f"  {i + 1}/{nsim} skies", flush=True)
    return out


def cached(name: str, make, force: bool = False):
    """Load data/ch08/<name>.npz if it exists, else run make() -> dict, save it and return it."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.npz"
    if path.exists() and not force:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = make()
    np.savez(path, **d)
    print(f"[cache] wrote {path}")
    return d


def bin_spectrum(x, edges):
    """Average the last axis of x over multipole bins [edges[i], edges[i+1])."""
    return np.stack([x[..., a:b].mean(axis=-1) for a, b in zip(edges[:-1], edges[1:])], axis=-1)
