"""lib_fisher10.py -- CMB Fisher-matrix machinery for chapter 10 (part 10a).

  NAMES, LABELS, FID, STEPS      the six LCDM parameters in the Planck basis
                                 (omega_b, omega_c, 100 theta_MC, tau, ln 10^10 A_s, n_s)
  camb_spectra(theta)            lensed TT, EE, TE (muK^2, l = 0..LMAX) from CAMB, plus H0
  derivatives(mult, stencil)     dC/dtheta_i by central differences (2- or 4-point), cached in data/ch10/
  noise_*()                      beam-deconvolved noise spectra N_l^T, N_l^P for the experiments used
  fisher_cmb(...)                F_ij = sum_l (2l+1) f_sky / 2 Tr[C^-1 C,i C^-1 C,j]  (T, E or T+E)
  fisher_spectra(...)            the same, written with the covariance of the estimated spectra
  marg(F), corr(F)               marginalised errors and correlation matrix

Everything follows the notes' fiducial (code/camb_fiducial.py, Planck 2018 best fit); the angle
100 theta_MC is the value CAMB returns for that fiducial, so the two parametrisations describe the
same model.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from camb_fiducial import FIDUCIAL  # noqa: E402

NOTES = pathlib.Path(__file__).resolve().parents[2]
DATA = NOTES / "data" / "ch10"
LMAX = 3000
NAMES = ["ombh2", "omch2", "theta", "tau", "logA", "ns"]
LABELS = [r"$\omega_b$", r"$\omega_c$", r"$100\theta_{\rm MC}$", r"$\tau$", r"$\ln(10^{10}A_s)$", r"$n_s$"]
TEXNAMES = [r"\omega_b", r"\omega_c", r"100\theta_{\rm MC}", r"\tau", r"\ln(10^{10}A_s)", r"n_s"]
# step sizes for the central differences: roughly 1/3 to 1 of Planck's 1-sigma error
STEPS = np.array([0.0001, 0.001, 0.0002, 0.005, 0.01, 0.004])
ARCMIN = np.pi / 180 / 60


def _theta_fid():
    import camb
    p = camb.set_params(**FIDUCIAL)
    return 100 * camb.get_background(p).cosmomc_theta()


def fiducial_vector():
    path = DATA / "fid_theta.npy"
    if path.exists():
        return np.load(path)
    th = np.array([FIDUCIAL["ombh2"], FIDUCIAL["omch2"], _theta_fid(), FIDUCIAL["tau"],
                   np.log(1e10 * FIDUCIAL["As"]), FIDUCIAL["ns"]])
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(path, th)
    return th


def camb_params(th):
    import camb
    p = dict(mnu=FIDUCIAL["mnu"], ombh2=th[0], omch2=th[1], cosmomc_theta=th[2] / 100, tau=th[3],
             As=np.exp(th[4]) * 1e-10, ns=th[5])
    return camb.set_params(**p, lmax=LMAX + 500, lens_potential_accuracy=1)


def camb_spectra(th, lmax=LMAX):
    """Lensed TT, EE, TE in muK^2 for l = 0..lmax, and H0, for the Planck-basis vector th."""
    import camb
    pars = camb_params(th)
    res = camb.get_results(pars)
    tot = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["total"][: lmax + 1]
    return dict(TT=tot[:, 0].copy(), EE=tot[:, 1].copy(), TE=tot[:, 3].copy(), H0=pars.H0,
                omnuh2=pars.omnuh2)


def h0_of(th):
    """H0 for a Planck-basis vector: only the background is needed (fast)."""
    return camb_params(th).H0


def derivatives(mult=1.0, stencil=2, force=False):
    """dC_l/dtheta_i for TT, EE, TE with steps mult*STEPS. stencil=2: (f(+h)-f(-h))/2h,
    stencil=4: (-f(+2h)+8f(+h)-8f(-h)+f(-2h))/12h.  Cached in data/ch10/derivs_<mult>_<stencil>.npz."""
    tag = f"{mult:g}".replace(".", "p")
    path = DATA / f"derivs_{tag}_{stencil}.npz"
    if path.exists() and not force:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    fid = fiducial_vector()
    out = {s: np.zeros((6, LMAX + 1)) for s in ("TT", "EE", "TE")}
    for i in range(6):
        h = mult * STEPS[i]
        e = np.zeros(6); e[i] = h
        p1, m1 = camb_spectra(fid + e), camb_spectra(fid - e)
        if stencil == 4:
            p2, m2 = camb_spectra(fid + 2 * e), camb_spectra(fid - 2 * e)
        for s in out:
            if stencil == 2:
                out[s][i] = (p1[s] - m1[s]) / (2 * h)
            else:
                out[s][i] = (-p2[s] + 8 * p1[s] - 8 * m1[s] + m2[s]) / (12 * h)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def fiducial_spectra():
    path = DATA / "fid_spectra.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    c = camb_spectra(fiducial_vector())
    np.savez(path, TT=c["TT"], EE=c["EE"], TE=c["TE"], H0=c["H0"], omnuh2=c["omnuh2"])
    return fiducial_spectra()


# ---------------------------------------------------------------- noise models
def beam2(fwhm_arcmin, lmax=LMAX):
    s = fwhm_arcmin * ARCMIN / np.sqrt(8 * np.log(2))
    l = np.arange(lmax + 1)
    return np.exp(-l * (l + 1) * s ** 2)


def white_noise(depth_uk_arcmin, fwhm_arcmin, lmax=LMAX):
    """Beam-deconvolved white noise N_l / B_l^2 in muK^2 for a depth in muK arcmin."""
    return (depth_uk_arcmin * ARCMIN) ** 2 / beam2(fwhm_arcmin, lmax)


# Planck 2018 I, Table 4: FWHM [arcmin], temperature and polarisation noise [muK deg]
PLANCK_HFI = {100: (9.66, 1.29, 1.96), 143: (7.22, 0.55, 1.17), 217: (4.90, 0.78, 1.75)}


def noise_planck_like(lmax=LMAX, channels=(100, 143, 217)):
    """Inverse-variance combination of the HFI CMB channels: 1/N = sum_nu 1/N_nu."""
    invT = np.zeros(lmax + 1); invP = np.zeros(lmax + 1)
    for nu in channels:
        fwhm, dT, dP = PLANCK_HFI[nu]
        invT += 1 / white_noise(60 * dT, fwhm, lmax)
        invP += 1 / white_noise(60 * dP, fwhm, lmax)
    return 1 / invT, 1 / invP


def noise_s4_like(lmax=LMAX):
    """A futuristic ground-based survey: 1 muK arcmin in T (sqrt 2 in P), 1.4' beam."""
    return white_noise(1.0, 1.4, lmax), white_noise(np.sqrt(2.0), 1.4, lmax)


def noise_143(lmax=LMAX):
    """The single 143-GHz-like channel of chapter 9 (7.22', 33 muK arcmin), temperature only."""
    return white_noise(33.0, 7.22, lmax), white_noise(33.0 * np.sqrt(2), 7.22, lmax)


# ---------------------------------------------------------------- Fisher matrices
def fisher_cmb(dC, C, NT, NP, fsky, lmin, lmax, which="TTTEEE"):
    """F_ij = sum_l (2l+1) fsky/2 Tr[Cl^-1 dCl_i Cl^-1 dCl_j], with Cl the 2x2 (T,E) covariance of
    the a_lm (signal + noise) when which='TTTEEE', or the 1x1 TT / EE case."""
    l = np.arange(lmin, lmax + 1)
    nu = (2 * l + 1) * fsky
    if which in ("TT", "EE"):
        n = NT if which == "TT" else NP
        c = C[which][l] + n[l]
        d = dC[which][:, l]
        return (d * (nu / 2 / c ** 2)) @ d.T
    if which == "TE":           # the TE spectrum alone: Var(C_hat^TE) = [(C^TE)^2 + C^TT C^EE] / nu
        var = (C["TE"][l] ** 2 + (C["TT"][l] + NT[l]) * (C["EE"][l] + NP[l])) / nu
        d = dC["TE"][:, l]
        return (d / var) @ d.T
    a = C["TT"][l] + NT[l]; b = C["TE"][l]; e = C["EE"][l] + NP[l]
    det = a * e - b * b
    # inverse of [[a, b], [b, e]] is [[e, -b], [-b, a]] / det
    ia, ib, ie = e / det, -b / det, a / det
    F = np.zeros((6, 6))
    for i in range(6):
        Ma = ia * dC["TT"][i, l] + ib * dC["TE"][i, l]      # (Cinv dC_i), elements 11, 12, 21, 22
        Mb = ia * dC["TE"][i, l] + ib * dC["EE"][i, l]
        Mc = ib * dC["TT"][i, l] + ie * dC["TE"][i, l]
        Md = ib * dC["TE"][i, l] + ie * dC["EE"][i, l]
        for j in range(i + 1):
            Na = ia * dC["TT"][j, l] + ib * dC["TE"][j, l]
            Nb = ia * dC["TE"][j, l] + ib * dC["EE"][j, l]
            Nc = ib * dC["TT"][j, l] + ie * dC["TE"][j, l]
            Nd = ib * dC["TE"][j, l] + ie * dC["EE"][j, l]
            tr = Ma * Na + Mb * Nc + Mc * Nb + Md * Nd
            F[i, j] = F[j, i] = np.sum(nu / 2 * tr)
    return F


def fisher_spectra(dC, C, NT, NP, fsky, lmin, lmax):
    """The same T+E Fisher matrix written as sum_l dC^T Cov^-1 dC, with Cov the 3x3 covariance of the
    estimated (TT, EE, TE) spectra (Verde 2010, eq. 27 with noise added)."""
    F = np.zeros((6, 6))
    for l in range(lmin, lmax + 1):
        a = C["TT"][l] + NT[l]; b = C["TE"][l]; e = C["EE"][l] + NP[l]
        nu = (2 * l + 1) * fsky
        cov = np.array([[2 * a * a, 2 * b * b, 2 * a * b],
                        [2 * b * b, 2 * e * e, 2 * e * b],
                        [2 * a * b, 2 * e * b, b * b + a * e]]) / nu
        d = np.array([dC["TT"][:, l], dC["EE"][:, l], dC["TE"][:, l]])
        F += d.T @ np.linalg.solve(cov, d)
    return F


def tau_prior(sigma=0.0086):
    P = np.zeros((6, 6)); P[3, 3] = 1 / sigma ** 2
    return P


def marg(F):
    return np.sqrt(np.diag(np.linalg.inv(F)))


def corr(F):
    C = np.linalg.inv(F)
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)
