"""lib_lss.py -- small toolkit for chapter 14 (galaxy clustering as data).

Conventions follow code/ch07/lib_fields.py (periodic box of side L, N cells per side,
cell size D = L/N, V = L^d):
    delta_k = D^d FFT(delta),   <delta_k delta_k'^*> = V P(k) Kronecker(k, k').
Lengths in Mpc/h, wavenumbers in h/Mpc, power in (Mpc/h)^3 (or (Mpc/h)^2 in 2-D).

Contents
  cosmology inputs : linear P(k) at z = 0 (cached by chT1), growth D(z) and f(z) for flat LCDM,
                     the Eisenstein-Hu (1998) no-wiggle spectrum, the damped BAO template
  fields           : Gaussian field from a gridded spectrum, lognormal field with a target
                     two-point function, Poisson sampling of a density field (a Cox process)
  estimators       : xi(r) on a periodic grid by FFT, P(k) multipoles, xi(r) from P(k)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "ch07"))
from common import DATA  # noqa: E402
from camb_fiducial import FIDUCIAL  # noqa: E402
from lib_fields import kgrid, rfft_weights  # noqa: E402

# ------------------------------------------------------------------ cosmology inputs
H = FIDUCIAL["H0"] / 100.0
OMB = FIDUCIAL["ombh2"] / H ** 2
OMNU = FIDUCIAL["mnu"] / 93.14 / H ** 2
OMM = (FIDUCIAL["ombh2"] + FIDUCIAL["omch2"]) / H ** 2 + OMNU
NS = FIDUCIAL["ns"]


def linear_pk_z0():
    """k [h/Mpc] and the CAMB linear matter P(k) [(Mpc/h)^3] at z = 0, cached by chapter T1."""
    z = np.load(DATA / "chT1" / "pk.npz")
    return z["k"], z["pk_lin"][0], float(z["sigma8"]), float(z["rdrag"])


def E_of_z(z, om=OMM):
    """H(z)/H0 for flat LCDM (radiation neglected, adequate at z < 5)."""
    return np.sqrt(om * (1 + z) ** 3 + 1 - om)


def growth(z, om=OMM):
    """Linear growth D(z) normalised to D(0) = 1, and the growth rate f = dlnD/dlna.

    For matter + Lambda, D(a) is proportional to E(a) int_0^a da' / (a' E(a'))^3 (see the
    cosmology notes); f follows by differentiating that expression with respect to ln a.
    """
    def Dun(a):
        aa = np.linspace(1e-6, a, 4000)
        e = E_of_z(1 / aa - 1, om)
        return E_of_z(1 / a - 1, om) * np.trapezoid(1 / (aa * e) ** 3, aa)
    z = np.atleast_1d(np.asarray(z, float))
    D = np.array([Dun(1 / (1 + zz)) for zz in z]) / Dun(1.0)
    a = 1 / (1 + z)
    e = E_of_z(z, om)
    om_a = om * (1 + z) ** 3 / e ** 2
    # f = dlnE/dlna + a^{-2} E^{-3} / int(...);  dlnE/dlna = -3/2 om_a
    integ = np.array([Dun(aa) / E_of_z(1 / aa - 1, om) for aa in a])
    f = -1.5 * om_a + 1.0 / (a ** 2 * e ** 3 * integ)
    return D, f


def nowiggle_pk(k, k_tab, pk_tab):
    """Eisenstein & Hu (1998) no-wiggle spectrum (their eqs. 26, 28-31), matched to P_lin.

    The EH shape k^ns T0(q_eff)^2 has no acoustic wiggles; a smooth low-order polynomial in ln k
    fitted to ln(P_lin / P_EH) fixes its normalisation and small broad-band differences without
    letting the wiggles back in.
    """
    def eh(kk):
        omh2, obh2 = OMM * H ** 2, OMB * H ** 2
        fb = OMB / OMM
        s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2 ** 0.75)          # Mpc, eq. 26
        aG = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb ** 2  # eq. 31
        gam = OMM * H * (aG + (1 - aG) / (1 + (0.43 * kk * H * s) ** 4))             # eq. 30, k in Mpc^-1
        q = kk * (2.7255 / 2.7) ** 2 / gam                                              # eq. 28
        L0 = np.log(2 * np.e + 1.8 * q)
        C0 = 14.2 + 731 / (1 + 62.5 * q)
        return kk ** NS * (L0 / (L0 + C0 * q ** 2)) ** 2                                # eq. 29
    sel = (k_tab > 1e-3) & (k_tab < 1.0)
    x = np.log(k_tab[sel])
    c = np.polyfit(x, np.log(pk_tab[sel] / eh(k_tab[sel])), 5)
    xk = np.clip(np.log(k), x.min(), x.max())
    return eh(k) * np.exp(np.polyval(c, xk))


def bao_template(k, sigma_nl=8.0, wiggles=True):
    """Template P_m(k) = [P_lin - P_nw] exp(-k^2 Sigma^2/2) + P_nw (Anderson et al. 2012, eq. 26)."""
    k_tab, pk_tab, _, _ = linear_pk_z0()
    plin = np.interp(np.log(k), np.log(k_tab), pk_tab)
    pnw = nowiggle_pk(k, k_tab, pk_tab)
    if not wiggles:
        return pnw
    return (plin - pnw) * np.exp(-0.5 * (k * sigma_nl) ** 2) + pnw


def xi_from_pk(k, pk, r, damp=1.0):
    """xi(r) = int dk/(2 pi^2) k^2 P(k) sin(kr)/(kr) exp(-k^2 a^2), integrated in ln k (a = damp)."""
    lk = np.log(k)
    w = k ** 3 * pk * np.exp(-(k * damp) ** 2) / (2 * np.pi ** 2)
    kr = np.outer(np.atleast_1d(r), k)
    return np.trapezoid(w * np.sinc(kr / np.pi), lk, axis=1)


# ------------------------------------------------------------------ fields
def rgrid(N, L, d):
    """|r| of every separation vector on the periodic grid (full FFT layout, origin at index 0)."""
    n = np.fft.fftfreq(N, d=1.0 / N)            # 0, 1, ..., -1 in cell units
    comps = np.meshgrid(*([n * (L / N)] * d), indexing="ij")
    return np.sqrt(sum(c ** 2 for c in comps))


def gaussian_from_pgrid(pg_r, N, L, d, rng, nsim=None):
    """Gaussian field(s) whose spectrum on the rfft grid is the array pg_r (zero mode ignored)."""
    D = L / N
    amp = np.sqrt(np.clip(pg_r, 0, None) / D ** d)
    amp.flat[0] = 0.0
    shape = ([nsim] if nsim else []) + [N] * d
    axes = tuple(range(-d, 0))
    w = rng.standard_normal(shape)
    return np.fft.irfftn(np.fft.rfftn(w, axes=axes) * amp, s=[N] * d, axes=axes)


class Lognormal:
    """Lognormal field 1 + delta = exp(G - sigma_G^2/2) with a prescribed grid two-point function.

    xi_target is the two-point function the lognormal field should have on the grid. Because
    <(1+d1)(1+d2)> = exp(xi_G(r)) for jointly Gaussian G (the mean of exp of a Gaussian), the
    Gaussian field needs xi_G = ln(1 + xi_target) [Xavier et al. 2016, eq. 7 with shift 1].
    Its spectrum P_G is the FFT of xi_G; rare negative values are clipped to zero.
    """

    def __init__(self, xi_target_grid, N, L, d):
        self.N, self.L, self.d = N, L, d
        D = L / N
        xg = np.log1p(xi_target_grid)
        pg = np.fft.rfftn(xg).real * D ** d         # P_G on the rfft grid
        self.clipped = float(np.mean(pg < 0))
        self.pg = np.clip(pg, 0, None)
        self.pg.flat[0] = 0.0
        w = rfft_weights(N, d)
        self.sigma2 = float(np.sum(w * self.pg) / L ** d)   # variance of G on the grid

    def draw(self, rng, nsim=None, return_gauss=False):
        g = gaussian_from_pgrid(self.pg, self.N, self.L, self.d, rng, nsim)
        out = np.expm1(g - 0.5 * self.sigma2)
        return (out, g) if return_gauss else out


def xi_box_from_pk(Pfun, N, L, d):
    """Exact grid two-point function (1/V) sum_k P(k) e^{ik.r} of a field with spectrum P on the grid."""
    D = L / N
    kmag, _ = kgrid(N, L, d, real=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk = np.where(kmag > 0, Pfun(np.where(kmag > 0, kmag, 1.0)), 0.0)
    return np.real(np.fft.ifftn(pk)) / D ** d


def poisson_sample(density, nbar, L, rng, window=None):
    """Counts per cell of a Cox process: N_cell ~ Poisson(nbar * dV * W * (1 + delta)).

    density : 1 + delta on the grid (must be >= 0); window: optional W(x) in [0, 1] (or n(x)/nbar).
    """
    d = density.ndim
    dV = (L / density.shape[0]) ** d
    lam = nbar * dV * density * (1.0 if window is None else window)
    return rng.poisson(np.clip(lam, 0, None))


def points_from_counts(counts, L, rng):
    """Place counts[cell] points uniformly inside each cell; returns an (Npts, d) array."""
    d = counts.ndim
    D = L / counts.shape[0]
    idx = np.nonzero(counts)
    reps = counts[idx]
    base = np.repeat(np.stack(idx, axis=1), reps, axis=0).astype(float)
    return (base + rng.random(base.shape)) * D


# ------------------------------------------------------------------ estimators
class RadialBins:
    """Average a grid function of the separation vector over shells [edges[i], edges[i+1])."""

    def __init__(self, N, L, d, edges):
        r = rgrid(N, L, d).ravel()
        self.idx = np.digitize(r, edges) - 1
        self.ok = (self.idx >= 0) & (self.idx < len(edges) - 1) & (r > 0)
        self.nb = len(edges) - 1
        self.cnt = np.bincount(self.idx[self.ok], minlength=self.nb)
        self.rmean = np.bincount(self.idx[self.ok], weights=r[self.ok], minlength=self.nb) / self.cnt

    def __call__(self, grid):
        g = np.asarray(grid).ravel()
        return np.bincount(self.idx[self.ok], weights=g[self.ok], minlength=self.nb) / self.cnt


def xi_grid_fft(delta):
    """Periodic estimate (1/N^d) sum_x delta(x) delta(x+r) for every separation vector r."""
    fk = np.fft.rfftn(delta)
    return np.fft.irfftn(np.abs(fk) ** 2, s=delta.shape) / delta.size


class KBins:
    """Shell averages on the rfft grid, with mu = k_z/k for multipoles (line of sight = last axis)."""

    def __init__(self, N, L, d, edges):
        kmag, comps = kgrid(N, L, d)
        self.w = rfft_weights(N, d).ravel()
        k = kmag.ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mu = np.where(k > 0, comps[-1].ravel() / np.where(k > 0, k, 1), 0.0)
        self.idx = np.digitize(k, edges) - 1
        self.ok = (self.idx >= 0) & (self.idx < len(edges) - 1) & (k > 0)
        self.nb = len(edges) - 1
        self.nm = np.bincount(self.idx[self.ok], weights=self.w[self.ok], minlength=self.nb)
        self.kmean = np.bincount(self.idx[self.ok], weights=(self.w * k)[self.ok], minlength=self.nb) / self.nm

    def avg(self, q):
        q = np.asarray(q).ravel()
        return np.bincount(self.idx[self.ok], weights=(self.w * q)[self.ok], minlength=self.nb) / self.nm

    def multipoles(self, p):
        """P_0, P_2, P_4 = (2l+1) <P(k,mu) L_l(mu)> over each shell."""
        mu2 = self.mu ** 2
        L2 = 0.5 * (3 * mu2 - 1)
        L4 = (35 * mu2 ** 2 - 30 * mu2 + 3) / 8
        p = np.asarray(p).ravel()
        return self.avg(p), 5 * self.avg(p * L2), 9 * self.avg(p * L4)


def power_raw(field, L):
    """|delta_k|^2 / V on the rfft grid, with delta_k = D^d FFT(field)."""
    d = field.ndim
    D = L / field.shape[0]
    return np.abs(np.fft.rfftn(field) * D ** d) ** 2 / L ** d
