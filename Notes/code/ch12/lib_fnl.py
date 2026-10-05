"""lib_fnl.py -- two random fields with the same power spectrum, and their topological summaries.

Used by the c0*.py scripts of chapter 11.  Import with

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from lib_fnl import Model, summaries

The model
---------
* g : a Gaussian field on an N x N periodic grid with power P_g(k) ~ k^-2 (equal power per
      octave, the flat-sky analogue of a scale-invariant spectrum), normalised to unit variance
      per pixel.
* phi = g + eps (g^2 - 1): local-type non-Gaussianity, eps = f_NL sigma_g.
* u = W * R_eps * phi : R_eps(k) = sqrt(S_g / S_phi) removes the extra power that g^2 adds,
      W(k) = exp(-k^2 s^2 / 2) is a Gaussian smoothing of s pixels (beam + transfer function).
  The ensemble spectrum of u is W^2 S_g for EVERY eps: the power spectrum cannot see eps.

The summaries (all thresholds in units of the map's own standard deviation)
--------------------------------------------------------------------------
pk     : binned periodogram                              (two-point information)
skew   : <u^3>/sigma^3                                    (one-point information)
chi    : Euler characteristic V - E + F                   (lib_topo, from scratch)
b0, b1 : Betti numbers of {u >= nu}                       (lib_topo, from scratch, scipy labels)
area   : area fraction above each threshold (the one-point distribution)
chiA,b0A,b1A: chi and Betti numbers at thresholds of fixed AREA fraction (one-point information removed)
pd0,pd1: histograms of the persistence diagrams in (birth, persistence) bins (gudhi)
pd0A,pd1A: the same for the gaussianised map (area-fraction labelling)
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib_topo import betti_plane, euler_cubical  # noqa: E402

NU = np.linspace(-3.0, 3.0, 13)                     # thresholds for chi and Betti curves
# area-fraction thresholds: the Gaussian thresholds NU translated into area fractions
from scipy.special import erfc  # noqa: E402
AREA = 0.5 * erfc(NU / np.sqrt(2))
# persistence-diagram bins: birth level (sigma units) x persistence (sigma units)
PD_BIRTH = np.array([-np.inf, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, np.inf])
PD_PERS = np.array([0.0, 0.1, 0.3, 0.6, 1.0, np.inf])
NPK = 12


class Model:
    """Spectra and filters on an N x N grid; draw(eps, rng) returns one map u."""

    def __init__(self, N: int = 256, smooth: float = 3.0, slope: float = -2.0):
        self.N, self.s = N, smooth
        k1 = np.fft.fftfreq(N) * N                       # wavenumbers in units of the fundamental
        kx, ky = np.meshgrid(k1, k1, indexing="ij")
        k = np.hypot(kx, ky)
        Sg = np.where(k > 0, np.where(k > 0, k, 1.0) ** slope, 0.0)
        Sg /= Sg.mean()                                  # variance per pixel = mean of S over modes = 1
        self.k, self.Sg = k, Sg
        xi = np.fft.ifft2(Sg).real                       # correlation function, xi(0) = mean(S) = 1
        self.xi = xi
        self.Sg2 = np.fft.fft2(2.0 * xi ** 2).real       # mode power of g^2 - 1: Wick, 2 xi^2
        kr = 2 * np.pi * k / N                           # radians per pixel
        self.W = np.exp(-0.5 * (kr * smooth) ** 2)
        # pk bins (log spaced in k, from the fundamental to where W has fallen to 1e-2)
        kmax = np.sqrt(2 * np.log(100.0)) / (2 * np.pi * smooth / N)
        self.kedges = np.geomspace(1.0, kmax, NPK + 1)
        self.kbin = np.digitize(k, self.kedges) - 1

    def renorm(self, eps: float):
        Sphi = self.Sg + eps ** 2 * self.Sg2
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(Sphi > 0, np.sqrt(self.Sg / Sphi), 0.0)

    def gauss(self, rng):
        w = rng.standard_normal((self.N, self.N))
        return np.fft.ifft2(np.fft.fft2(w) * np.sqrt(self.Sg)).real

    def observe(self, phi, eps: float):
        return np.fft.ifft2(np.fft.fft2(phi) * self.renorm(eps) * self.W).real

    def draw(self, eps: float, rng=None, g=None):
        g = self.gauss(rng) if g is None else g
        phi = g + eps * (g ** 2 - 1.0)
        return self.observe(phi, eps)

    def pk(self, u):
        P = np.abs(np.fft.fft2(u)) ** 2 / u.size
        ok = (self.kbin >= 0) & (self.kbin < NPK)
        return np.bincount(self.kbin[ok], P[ok], NPK) / np.bincount(self.kbin[ok], None, NPK)


def standardise(u):
    return (u - u.mean()) / u.std()


def diagrams(x):
    """Superlevel persistence of x with gudhi (pixels = closed squares, 8-connected islands).

    Returns (pd0, pd1) as arrays of (birth, death) with birth >= death (superlevel convention);
    the essential island gets death = min(x).
    """
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=-x)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    out = []
    for dim in (0, 1):
        p = cc.persistence_intervals_in_dimension(dim)
        p = -np.asarray(p).reshape(-1, 2)                # back to superlevel levels
        p[~np.isfinite(p[:, 1]), 1] = x.min()
        out.append(p)
    return out


def pd_hist(p):
    pers = p[:, 0] - p[:, 1]
    H, _, _ = np.histogram2d(p[:, 0], pers, bins=[PD_BIRTH, PD_PERS])
    return H.ravel()


def summaries(u, model: Model | None = None, persistence: bool = True):
    x = standardise(u)
    out = {}
    if model is not None:
        out["pk"] = model.pk(u)
    out["skew"] = np.array([np.mean(x ** 3)])
    out["area"] = np.array([(x >= nu).mean() for nu in NU])
    out["chi"] = np.array([euler_cubical(x >= nu) for nu in NU], float)
    b = np.array([betti_plane(x >= nu) for nu in NU], float)
    out["b0"], out["b1"] = b[:, 0], b[:, 1]
    xg = gaussianise(x)                                  # same order of pixels, Gaussian histogram
    out["chiA"] = np.array([euler_cubical(xg >= nu) for nu in NU], float)
    bA = np.array([betti_plane(xg >= nu) for nu in NU], float)
    out["b0A"], out["b1A"] = bA[:, 0], bA[:, 1]
    if persistence:
        p0, p1 = diagrams(x)
        out["pd0"], out["pd1"] = pd_hist(p0), pd_hist(p1)
        p0, p1 = diagrams(xg)
        out["pd0A"], out["pd1A"] = pd_hist(p0), pd_hist(p1)
    return out


def gaussianise(x):
    """Replace each value by the Gaussian quantile of its rank: a monotone relabelling that makes
    the histogram exactly Gaussian, so a threshold nu now means 'the top Psi(nu) of the area'."""
    from scipy.special import ndtri
    r = np.empty(x.size)
    r[np.argsort(x, axis=None, kind="stable")] = np.arange(x.size)
    return ndtri((r + 0.5) / x.size).reshape(x.shape)


KEYS = ["pk", "skew", "area", "chi", "b0", "b1", "chiA", "b0A", "b1A", "pd0", "pd1", "pd0A", "pd1A"]


def stack(list_of_dicts, keys=KEYS):
    return {k: np.array([d[k] for d in list_of_dicts]) for k in keys if k in list_of_dicts[0]}
