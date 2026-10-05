"""lib_web.py -- haloes and voids: the small toolkit behind code/ch11/08-12.

Units: masses in M_sun/h, lengths in Mpc/h, wavenumbers in h/Mpc, P(k) in (Mpc/h)^3, z = 0.

Contents
  sigma(M)          : rms linear density in a top hat holding mass M (from the cached CAMB P_lin)
  mass functions    : Press-Schechter and Sheth-Tormen multiplicities, their peak-background-split
                      biases, dn/dlnM
  random walks      : excursion-set walks with independent Gaussian steps (sharp-k filter)
  NFW               : concentration-mass relation, the normalised Fourier transform u(k|M)
  Zel'dovich box    : particles x = q + Psi(q) at z = 0 (cached in data/ch11), CIC density
  void finder       : spherical underdensity finder on a gridded density field
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.special import gamma, sici

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ch07"))
from common import DATA  # noqa: E402
from lib_lss import OMM, linear_pk_z0  # noqa: E402
from lib_fields import kgrid  # noqa: E402

RHO_CRIT = 2.775e11                 # critical density today in (M_sun/h) / (Mpc/h)^3
RHO_M = OMM * RHO_CRIT              # mean matter density
DELTA_C = 1.686                     # linear collapse threshold (spherical collapse, EdS value)
DELTA_V = -2.81                     # linear shell-crossing threshold of a spherical void
DELTA_HALO = 330.0                  # halo = sphere of mean density 330 rho_m (virial, LCDM z = 0)

K_TAB, PK_TAB, SIGMA8, _ = linear_pk_z0()


# ------------------------------------------------------------------ sigma(M)
def tophat_w(x):
    """Fourier transform of a unit-volume sphere, W(x) = 3 (sin x - x cos x) / x^3."""
    x = np.asarray(x, float)
    out = np.ones_like(x)
    big = x > 1e-3
    xb = x[big]
    out[big] = 3 * (np.sin(xb) - xb * np.cos(xb)) / xb ** 3
    return out


def radius_of_mass(M):
    """Lagrangian radius R with (4 pi/3) R^3 rho_m = M."""
    return (3 * np.asarray(M, float) / (4 * np.pi * RHO_M)) ** (1 / 3)


def sigma_R(R, k=K_TAB, pk=PK_TAB):
    """sigma^2(R) = int dlnk  k^3 P(k)/(2 pi^2) W(kR)^2 (top-hat window), vectorised over R."""
    R = np.atleast_1d(np.asarray(R, float))
    lk = np.log(k)
    w = k ** 3 * pk / (2 * np.pi ** 2)
    out = np.array([np.trapezoid(w * tophat_w(k * r) ** 2, lk) for r in R])
    return np.sqrt(out)


def sigma_M(M):
    return sigma_R(radius_of_mass(M))


# ------------------------------------------------------------------ mass functions
ST_A, ST_a, ST_p = None, 0.707, 0.3
ST_A = 1.0 / (1.0 + 2 ** (-ST_p) * gamma(0.5 - ST_p) / np.sqrt(np.pi))   # normalisation, = 0.3222


def F_ps(nu):
    """Press-Schechter: fraction of mass per unit nu, nu = delta_c / sigma (includes the factor 2)."""
    return np.sqrt(2 / np.pi) * np.exp(-0.5 * nu ** 2)


def F_st(nu, A=ST_A, a=ST_a, p=ST_p):
    """Sheth-Tormen fraction of mass per unit nu (nu = delta_c/sigma)."""
    an2 = a * nu ** 2
    return A * np.sqrt(2 * a / np.pi) * (1 + an2 ** (-p)) * np.exp(-0.5 * an2)


def bias_ps(nu, dc=DELTA_C):
    """Eulerian peak-background-split bias of the PS mass function, 1 + (nu^2 - 1)/delta_c."""
    return 1 + (nu ** 2 - 1) / dc


def bias_st(nu, dc=DELTA_C, a=ST_a, p=ST_p):
    """Eulerian PBS bias of the ST mass function, 1 - (1/dc) dln(nu F)/dln nu."""
    an2 = a * nu ** 2
    return 1 + (an2 - 1) / dc + 2 * p / (dc * (1 + an2 ** p))


def dndlnM(M, which="st"):
    """Haloes per unit volume per unit ln M: (rho_m/M) nu F(nu) |dln sigma/dln M|."""
    M = np.asarray(M, float)
    s = sigma_M(M)
    nu = DELTA_C / s
    dls = np.gradient(np.log(s), np.log(M))
    F = F_st(nu) if which == "st" else F_ps(nu)
    return RHO_M / M * nu * F * np.abs(dls), nu, s


# ------------------------------------------------------------------ random walks
def first_crossings(S_grid, nwalk, rng, barrier=DELTA_C, lower=None, chunk=2000):
    """Excursion-set walks delta(S) with independent N(0, dS) steps on the grid S_grid.

    Returns, per walk, the S of the first up-crossing of `barrier` (np.inf if none) and, if
    `lower` is given, the S of the first down-crossing of `lower` that happens before any
    up-crossing (np.inf otherwise). Between two grid points the walk is a Brownian bridge, which
    crosses a level b that both end points lie below with probability
    exp(-2 (b - x0)(b - x1)/dS); drawing that event too makes the result exact for continuous
    Brownian motion instead of only for the discrete walk.
    """
    dS = np.diff(np.concatenate([[0.0], S_grid]))
    up = np.full(nwalk, np.inf)
    down = np.full(nwalk, np.inf)
    done = 0
    while done < nwalk:
        n = min(chunk, nwalk - done)
        walk = np.cumsum(rng.standard_normal((n, len(S_grid))) * np.sqrt(dS), axis=1)
        prev = np.concatenate([np.zeros((n, 1)), walk[:, :-1]], axis=1)
        u = rng.random((n, len(S_grid)))
        bridge = np.exp(-2 * np.clip(barrier - prev, 0, None) * np.clip(barrier - walk, 0, None) / dS)
        hit = (walk >= barrier) | (u < bridge)
        first = np.where(hit.any(1), hit.argmax(1), -1)
        up[done:done + n] = np.where(first >= 0, S_grid[np.maximum(first, 0)], np.inf)
        if lower is not None:
            bridge_v = np.exp(-2 * np.clip(prev - lower, 0, None) * np.clip(walk - lower, 0, None) / dS)
            hitv = (walk <= lower) | (rng.random((n, len(S_grid))) < bridge_v)
            fv = np.where(hitv.any(1), hitv.argmax(1), -1)
            ok = (fv >= 0) & ((first < 0) | (fv < first))
            down[done:done + n] = np.where(ok, S_grid[np.maximum(fv, 0)], np.inf)
        done += n
    return up, down


# ------------------------------------------------------------------ NFW haloes
def m_star():
    """Mass at which sigma(M) = delta_c (nu = 1)."""
    lm = np.linspace(10, 15, 400)
    s = sigma_M(10 ** lm)
    return 10 ** np.interp(0.0, np.log(DELTA_C / s), lm)


def concentration(M, mstar):
    """Mean NFW concentration c(M) = 9 (M/M*)^-0.13 (Cooray & Sheth 2002, eq. 78, z = 0)."""
    return 9.0 * (np.asarray(M, float) / mstar) ** (-0.13)


def r_halo(M, delta=DELTA_HALO):
    """Halo radius: the sphere of mean density DELTA_HALO * rho_m (the virial radius) that holds M."""
    return (3 * np.asarray(M, float) / (4 * np.pi * delta * RHO_M)) ** (1 / 3)


def u_nfw(k, M, c, delta=DELTA_HALO):
    """Normalised Fourier transform of an NFW profile truncated at r_halo (CS02 eq. 81).

    k may be 1-D and M, c 1-D; returns shape (len(M), len(k)).
    """
    k = np.atleast_1d(k)[None, :]
    M = np.atleast_1d(M)[:, None]
    c = np.atleast_1d(c)[:, None]
    rs = r_halo(M, delta) / c
    x = k * rs
    si1, ci1 = sici((1 + c) * x)
    si0, ci0 = sici(x)
    mnorm = np.log(1 + c) - c / (1 + c)
    return (np.sin(x) * (si1 - si0) - np.sin(c * x) / ((1 + c) * x) + np.cos(x) * (ci1 - ci0)) / mnorm


# ------------------------------------------------------------------ Zel'dovich box
BOX_L, BOX_N = 600.0, 192          # the Zel'dovich universe shared by scripts 08, 11, 12


def zeldovich_box(L, N, rng, D=1.0):
    """Zel'dovich particles at growth factor D: x = q + D Psi(q), Psi_k = i k delta_k / k^2.

    Returns positions (N^3, 3) wrapped into [0, L), displacements Psi (N^3, 3) and the linear
    density delta_lin on the grid.
    """
    cell = L / N
    kmag, comps = kgrid(N, L, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk = np.where(kmag > 0, np.interp(kmag, K_TAB, PK_TAB), 0.0)
        inv_k2 = np.where(kmag > 0, 1 / kmag ** 2, 0.0)
    dk = np.fft.rfftn(rng.standard_normal((N, N, N))) * np.sqrt(pk / cell ** 3) * D
    psi = np.stack([np.fft.irfftn(1j * c_ * inv_k2 * dk, s=(N, N, N), axes=(0, 1, 2)).ravel() for c_ in comps], axis=1)
    q = (np.indices((N, N, N)).reshape(3, -1).T + 0.5) * cell
    x = (q + psi) % L
    dlin = np.fft.irfftn(dk, s=(N, N, N), axes=(0, 1, 2))
    return x, psi, dlin


def cic(pos, N, L):
    """Cloud-in-cell density 1 + delta of particles (periodic box), on an N^3 grid."""
    cell = L / N
    g = pos / cell - 0.5
    i0 = np.floor(g).astype(np.int64)
    w1 = g - i0
    rho = np.zeros(N ** 3)
    for dx in (0, 1):
        wx = w1[:, 0] if dx else 1 - w1[:, 0]
        for dy in (0, 1):
            wy = w1[:, 1] if dy else 1 - w1[:, 1]
            for dz in (0, 1):
                wz = w1[:, 2] if dz else 1 - w1[:, 2]
                idx = (((i0[:, 0] + dx) % N) * N + (i0[:, 1] + dy) % N) * N + (i0[:, 2] + dz) % N
                rho += np.bincount(idx, weights=wx * wy * wz, minlength=N ** 3)
    rho = rho.reshape(N, N, N)
    return rho / rho.mean()


def sphere_average(field, R, L):
    """Mean of `field` inside a sphere of radius R about every grid cell (periodic FFT convolution)."""
    N = field.shape[0]
    cell = L / N
    n = np.fft.fftfreq(N, d=1.0 / N) * cell
    r2 = n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2
    ker = (r2 <= R ** 2).astype(float)
    ker /= ker.sum()
    return np.fft.irfftn(np.fft.rfftn(field) * np.fft.rfftn(ker), s=field.shape)


def find_voids(rho, L, radii, smooth=1.0, seed_max=-0.5, threshold=-0.8, rmin=0.0):
    """Spherical underdensity void finder.

    rho      : 1 + delta on an N^3 periodic grid
    radii    : increasing trial radii [Mpc/h]
    Seeds are local minima of rho smoothed by a Gaussian of `smooth` cells with delta < seed_max.
    About each seed the mean enclosed contrast Delta(<R) is read off sphere-averaged copies of
    rho; the void radius is where Delta(<R) first rises through `threshold` (linear interpolation).
    Overlapping spheres are removed, largest first. Returns centres (n, 3) and radii (n,).
    """
    from scipy import ndimage
    N = rho.shape[0]
    cell = L / N
    rs = ndimage.gaussian_filter(rho, smooth, mode="wrap")
    mins = (rs == ndimage.minimum_filter(rs, size=3, mode="wrap")) & (rs - 1 < seed_max)
    idx = np.argwhere(mins)
    Dl = np.empty((len(idx), len(radii)))
    for j, R in enumerate(radii):
        Dl[:, j] = sphere_average(rho, R, L)[tuple(idx.T)] - 1
    rv = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        above = np.nonzero(Dl[i] >= threshold)[0]
        if len(above) == 0 or above[0] == 0:
            continue
        j = above[0]
        d0, d1 = Dl[i, j - 1], Dl[i, j]
        rv[i] = radii[j - 1] + (threshold - d0) / (d1 - d0) * (radii[j] - radii[j - 1])
    ok = np.isfinite(rv) & (rv >= rmin)
    cen = (idx[ok] + 0.5) * cell
    rv = rv[ok]
    order = np.argsort(rv)[::-1]
    keep_c, keep_r = [], []
    for i in order:
        c = cen[i]
        if keep_c:
            d = np.array(keep_c) - c
            d -= L * np.round(d / L)
            if np.any(np.sqrt((d ** 2).sum(1)) < np.array(keep_r) + rv[i]):
                continue
        keep_c.append(c)
        keep_r.append(rv[i])
    return np.array(keep_c), np.array(keep_r)


def hamaus_profile(r, dc, rs, alpha, beta):
    """Hamaus, Sutter & Wandelt (2014) eq. (2), r in units of the void radius."""
    return dc * (1 - (r / rs) ** alpha) / (1 + r ** beta)


def watershed_voids(rho, L, smooth=1.0, min_delta=-0.8, pad=24):
    """Watershed voids: the basins of the density landscape, one per local minimum.

    The density (smoothed by a Gaussian of `smooth` cells) is treated as a landscape and flooded
    from every local minimum (scikit-image watershed); each basin ends on the density ridges that
    separate it from its neighbours. The periodic box is padded by `pad` cells on every side so
    that basins near a face are complete, and only basins lying wholly inside the original box are
    kept, together with the requirement that the basin floor has delta < min_delta.
    Returns the volume-weighted centres (n, 3), effective radii R = (3V/4pi)^(1/3), and floor
    densities.
    """
    from scipy import ndimage
    from skimage.segmentation import watershed
    N = rho.shape[0]
    cell = L / N
    rs = ndimage.gaussian_filter(rho, smooth, mode="wrap")
    rp = np.pad(rs, pad, mode="wrap")
    markers, _ = ndimage.label(rp == ndimage.minimum_filter(rp, size=3, mode="wrap"))
    lab = watershed(rp, markers)
    core = lab[pad:-pad, pad:-pad, pad:-pad]
    tot = np.bincount(lab.ravel())
    inside = np.bincount(core.ravel(), minlength=len(tot))
    floor = np.asarray(ndimage.minimum(rp, lab, index=np.arange(len(tot)))) - 1
    ids = np.nonzero((inside == tot) & (tot > 0) & (floor < min_delta))[0]
    R = (3 * tot[ids] * cell ** 3 / (4 * np.pi)) ** (1 / 3)
    cen = (np.array(ndimage.center_of_mass(np.ones(core.shape), core, ids)) + 0.5) * cell
    return cen, R, floor[ids]
