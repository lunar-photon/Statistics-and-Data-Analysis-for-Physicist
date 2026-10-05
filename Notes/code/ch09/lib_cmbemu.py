"""lib_cmbemu.py -- a fast model C_l(theta) for MCMC: Taylor expansion of ln C_l around the fiducial.

  camb_tt(**change)          lensed TT C_l (muK^2), l = 0..LMAX, from CAMB, fiducial with changes
  build(path)                derivatives d lnC/d theta_i and second derivatives, cached in data/ch09/emu.npz
  Emulator(order)            emu.cl(theta) for theta = (ln 10^10 A_s, n_s, H0, ombh2, omch2, tau),
                             or for a subset of names with the others fixed at the fiducial
  planck_like_noise(l)       N_l / B_l^2 of a single 143 GHz-like channel (7.22', 33 muK arcmin)

Parameter conventions follow the Planck papers: ln(10^10 A_s) with A_s at k = 0.05 / Mpc.  The
fiducial is the Planck 2018 best fit used everywhere in the notes (code/camb_fiducial.py).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from camb_fiducial import FIDUCIAL  # noqa: E402

NOTES = pathlib.Path(__file__).resolve().parents[2]
CACHE = NOTES / "data" / "ch09" / "emu.npz"
LMAX = 2500
NAMES = ["lnAs", "ns", "H0", "ombh2", "omch2", "tau"]
LABELS = [r"$\ln(10^{10}A_s)$", r"$n_s$", r"$H_0$", r"$\omega_b$", r"$\omega_c$", r"$\tau$"]
FID = np.array([np.log(1e10 * FIDUCIAL["As"]), FIDUCIAL["ns"], FIDUCIAL["H0"], FIDUCIAL["ombh2"],
                FIDUCIAL["omch2"], FIDUCIAL["tau"]])
STEPS = np.array([0.02, 0.01, 0.5, 0.0003, 0.003, 0.01])     # central-difference steps (as in ch. 8)
ARCMIN = np.pi / 180 / 60


def camb_tt(theta=None, lmax=LMAX):
    """Lensed TT C_l in muK^2 for the full parameter vector theta (default: fiducial)."""
    import camb
    th = FID if theta is None else np.asarray(theta, float)
    p = dict(FIDUCIAL)
    p.update(As=np.exp(th[0]) * 1e-10, ns=th[1], H0=th[2], ombh2=th[3], omch2=th[4], tau=th[5])
    pars = camb.set_params(**p, lmax=lmax + 500, lens_potential_accuracy=1)
    res = camb.get_results(pars)
    return res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["total"][: lmax + 1, 0]


def build(path=CACHE, force=False):
    """First and second derivatives of ln C_l (l >= 2) by central differences: 1 + 2d + 2d(d-1) calls."""
    if path.exists() and not force:
        return dict(np.load(path))
    d = len(NAMES)
    c0 = camb_tt()
    l = np.arange(LMAX + 1)
    good = l >= 2
    lnc0 = np.zeros(LMAX + 1); lnc0[good] = np.log(c0[good])

    def lnc(th):
        out = np.zeros(LMAX + 1); out[good] = np.log(camb_tt(th)[good]); return out

    E = np.eye(d) * STEPS
    plus = [lnc(FID + E[i]) for i in range(d)]
    minus = [lnc(FID - E[i]) for i in range(d)]
    D = np.array([(plus[i] - minus[i]) / (2 * STEPS[i]) for i in range(d)])
    H = np.zeros((d, d, LMAX + 1))
    for i in range(d):
        H[i, i] = (plus[i] - 2 * lnc0 + minus[i]) / STEPS[i] ** 2
        for j in range(i):
            pp, pm = lnc(FID + E[i] + E[j]), lnc(FID + E[i] - E[j])
            mp, mm = lnc(FID - E[i] + E[j]), lnc(FID - E[i] - E[j])
            H[i, j] = H[j, i] = (pp - pm - mp + mm) / (4 * STEPS[i] * STEPS[j])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, c0=c0, lnc0=lnc0, D=D, H=H, fid=FID, steps=STEPS)
    return dict(np.load(path))


class Emulator:
    """ln C_l(theta) = ln C_l(fid) + D . dtheta [+ 1/2 dtheta . H . dtheta]   (order 1 or 2).

    names: which parameters are free (others fixed at the fiducial); theta is given in that order.
    lmax: truncate the returned spectrum."""

    def __init__(self, order=2, names=NAMES, lmax=LMAX, linear_in_C=False):
        z = build()
        idx = [NAMES.index(n) for n in names]
        self.idx = idx
        self.fid = FID[idx]
        self.c0 = z["c0"][: lmax + 1]
        self.lnc0 = z["lnc0"][: lmax + 1]
        self.D = z["D"][idx][:, : lmax + 1]
        self.H = z["H"][np.ix_(idx, idx)][:, :, : lmax + 1]
        self.order = order
        self.linear_in_C = linear_in_C
        self.lmax = lmax
        self.l = np.arange(lmax + 1)

    def cl(self, theta):
        dt = np.asarray(theta, float) - self.fid
        if self.linear_in_C:                       # Taylor series of C itself, first order
            return self.c0 * (1.0 + dt @ self.D)
        x = dt @ self.D
        if self.order == 2:
            x = x + 0.5 * np.einsum("i,ijl,j->l", dt, self.H, dt)
        out = np.exp(self.lnc0 + x)
        out[:2] = 0.0
        return out


def gaussian_beam(fwhm_arcmin, lmax):
    s = fwhm_arcmin * ARCMIN / np.sqrt(8 * np.log(2))
    l = np.arange(lmax + 1)
    return np.exp(-0.5 * l * (l + 1) * s ** 2)


def planck_like_noise(lmax=LMAX, fwhm=7.22, depth=33.0):
    """Effective noise N_l / B_l^2 (muK^2) of one 143-GHz-like channel: white noise of `depth`
    muK arcmin, Gaussian beam of `fwhm` arcmin (Planck 2018 I, Table 4: 7.22', 0.55 muK deg)."""
    nl = (depth * ARCMIN) ** 2
    return nl / gaussian_beam(fwhm, lmax) ** 2
