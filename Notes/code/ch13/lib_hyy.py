"""lib_hyy.py -- the pieces of the diphoton likelihood shared by the Higgs-search scripts.

Signal: a Crystal Ball line shape (Gaussian core, power-law tail towards low mass) whose bin
integrals are computed exactly from its cumulative distribution. Background: smooth families
(exponentials of polynomials, Bernstein polynomials, a power law) normalised over the window.
Likelihood: binned Poisson in 0.25 GeV bins with Gaussian-constrained nuisance parameters.
The Bernstein background is linear in its coefficients, so a vectorised Newton fitter
(lin_fit, scan_q0) handles thousands of toy spectra at once; the full likelihood with its
nuisance parameters (class Search) is minimised with MINUIT (iminuit).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

MLO, MHI, BW = 105.0, 160.0, 0.25
EDGES = np.arange(MLO, MHI + 1e-9, BW)
CENT = 0.5 * (EDGES[1:] + EDGES[:-1])
U = (CENT - 132.5) / 27.5                     # window mapped onto [-1, 1]
SQ2 = np.sqrt(2.0)


# ------------------------------------------------------------------ signal line shape
def cb_cdf(x, mbar, sig, alpha, n):
    """Cumulative distribution of the normalised Crystal Ball (tail below mbar - alpha*sig)."""
    t = (np.asarray(x, float) - mbar) / sig
    A = (n / alpha) ** n * np.exp(-alpha ** 2 / 2)
    B = n / alpha - alpha
    tail = A / (n - 1) * (n / alpha) ** (1 - n)          # area of the tail, t < -alpha
    core = np.sqrt(np.pi / 2) * (1 + erf(alpha / SQ2))    # area of the Gaussian part, t > -alpha
    lo = A / (n - 1) * (B - np.minimum(t, -alpha)) ** (1 - n)
    hi = np.sqrt(np.pi / 2) * (erf(np.maximum(t, -alpha) / SQ2) + erf(alpha / SQ2))
    return np.where(t < -alpha, lo, tail + hi) / (tail + core)


def cb_pdf(x, mbar, sig, alpha, n):
    t = (np.asarray(x, float) - mbar) / sig
    A = (n / alpha) ** n * np.exp(-alpha ** 2 / 2)
    B = n / alpha - alpha
    tail = A / (n - 1) * (n / alpha) ** (1 - n)
    core = np.sqrt(np.pi / 2) * (1 + erf(alpha / SQ2))
    f = np.where(t > -alpha, np.exp(-0.5 * t ** 2), A * np.abs(B - t) ** (-n))
    return f / (sig * (tail + core))


def gauss_cdf(x, m, s):
    return 0.5 * (1 + erf((np.asarray(x, float) - m) / (SQ2 * s)))


class SignalShape:
    """Crystal Ball fitted to the m_H = 125 GeV simulation, carried to other masses by scaling."""

    def __init__(self, dm, sig, alpha, n, s_tot):
        self.dm, self.sig, self.alpha, self.n, self.s_tot = dm, sig, alpha, n, s_tot

    def params(self, mH, ksig=1.0, kscale=1.0):
        return (mH + self.dm * mH / 125.0) * kscale, self.sig * mH / 125.0 * ksig, self.alpha, self.n

    def bins(self, mH, ksig=1.0, kscale=1.0):
        """Expected signal events per bin for mu = 1 (total yield s_tot before the window cut)."""
        c = cb_cdf(EDGES, *self.params(mH, ksig, kscale))
        return self.s_tot * np.diff(c)


def fit_cb(m, w):
    """Weighted unbinned maximum likelihood of a Crystal Ball truncated to the window."""
    def nll(p):
        mbar, sig, alpha, n = p
        if sig <= 0.3 or alpha <= 0.2 or n <= 1.05:
            return 1e30
        norm = cb_cdf(MHI, *p) - cb_cdf(MLO, *p)
        return -np.sum(w * np.log(cb_pdf(m, *p) / norm + 1e-300))
    r = minimize(nll, [124.5, 1.6, 1.5, 5.0], method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-6, maxiter=20000))
    return r.x


def fit_gauss(m, w):
    def nll(p):
        mu, s = p
        norm = gauss_cdf(MHI, mu, s) - gauss_cdf(MLO, mu, s)
        return -np.sum(w * (-0.5 * ((m - mu) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi) * norm)))
    return minimize(nll, [125.0, 2.0], method="Nelder-Mead").x


# ------------------------------------------------------------------ background families
def _bern(u, c):
    """Bernstein polynomial of degree k = len(c) on [0, 1]; the last coefficient is fixed to 1
    (the overall normalisation is carried by B)."""
    from math import comb
    x = 0.5 * (u + 1)
    k = len(c)
    coef = list(c) + [1.0]
    return sum(coef[j] * comb(k, j) * x ** j * (1 - x) ** (k - j) for j in range(k + 1))


BKG = {   # name: (number of shape parameters, log of the unnormalised density, start values)
    "Exp1": (1, lambda u, p: p[0] * u, [-1.5]),
    "Exp2": (2, lambda u, p: p[0] * u + p[1] * u ** 2, [-1.5, 0.2]),
    "Exp3": (3, lambda u, p: p[0] * u + p[1] * u ** 2 + p[2] * u ** 3, [-1.5, 0.2, 0.0]),
    "Pow": (1, lambda u, p: -p[0] * np.log((27.5 * u + 132.5) / 132.5), [4.0]),
    "Bern3": (3, lambda u, p: np.log(np.maximum(_bern(u, p), 1e-12)), [3.0, 1.8, 1.0]),
    "Bern4": (4, lambda u, p: np.log(np.maximum(_bern(u, p), 1e-12)), [3.5, 2.5, 1.6, 1.0]),
    "Exp4": (4, lambda u, p: p[0] * u + p[1] * u ** 2 + p[2] * u ** 3 + p[3] * u ** 4,
             [-0.8, 0.05, -0.05, 0.0]),
    "Bern5": (5, lambda u, p: np.log(np.maximum(_bern(u, p), 1e-12)), [5.6, 4.0, 2.8, 2.0, 1.5]),
}


def bkg_shape(name, p, u=U):
    g = np.exp(BKG[name][1](u, p) - BKG[name][1](np.zeros(1), p))
    return g / g.sum()


# ------------------------------------------------------------------ Bernstein background, linear in its coefficients
def bern_basis(deg, u=U):
    """Bins x (deg+1) matrix: the Bernstein polynomials of degree deg on the window, x = (u+1)/2."""
    from math import comb
    x = 0.5 * (u + 1)
    return np.stack([comb(deg, j) * x ** j * (1 - x) ** (deg - j) for j in range(deg + 1)], 1)


def lin_fit(n, A, offset=None, c0=None, nit=30):
    """Maximise sum(n ln nu - nu) for nu = offset + A c, for many spectra at once (Newton's method).

    n: (T, nb) counts; A: (nb, k) or (T, nb, k); offset: (T, nb) or (nb,) fixed part of the mean.
    The log-likelihood is concave in c, so Newton steps (halved if a mean turns negative) converge.
    Returns c (T, k) and lnL (T,) without the constant -ln n!.
    """
    n = np.atleast_2d(np.asarray(n, float))
    T, nb = n.shape
    A = np.broadcast_to(A, (T,) + A.shape[-2:]) if A.ndim == 2 else A
    off = np.zeros((T, nb)) if offset is None else np.broadcast_to(offset, (T, nb))
    if c0 is None:   # start: least squares of the counts on the columns
        c = np.linalg.lstsq(A[0], np.maximum(n - off, 1.0).mean(0), rcond=None)[0]
        c = np.tile(c, (T, 1))
    else:
        c = np.array(c0, float)
    nu = off + np.einsum("tbk,tk->tb", A, c)
    for _ in range(nit):
        grad = np.einsum("tbk,tb->tk", A, n / nu - 1)
        H = np.einsum("tbk,tb,tbl->tkl", A, n / nu ** 2, A) + 1e-9 * np.eye(A.shape[-1])
        step = np.linalg.solve(H, grad[..., None])[..., 0]
        lam = np.ones(T)
        for _ in range(20):                                  # halve steps that make a mean negative
            trial = off + np.einsum("tbk,tk->tb", A, c + lam[:, None] * step)
            bad = np.any(trial <= 0, 1)
            if not bad.any():
                break
            lam[bad] *= 0.5
        c = c + lam[:, None] * step
        nu = off + np.einsum("tbk,tk->tb", A, c)
    return c, np.sum(n * np.log(nu) - nu, 1)


def scan_q0(n, templates, B, cb=None, nit=30):
    """Discovery statistic q0(m) for many spectra: background = B c, signal = mu * template(m).

    n: (T, nb); templates: (M, nb) unit-yield signal shapes; B: (nb, k) background basis.
    Returns q0 (T, M), mu_hat (T, M), and the background-only coefficients (T, k).
    """
    n = np.atleast_2d(np.asarray(n, float))
    T = n.shape[0]
    cb, l0 = lin_fit(n, B, nit=nit) if cb is None else cb
    q0 = np.zeros((T, len(templates)))
    muh = np.zeros_like(q0)
    for j, t in enumerate(templates):
        A = np.concatenate([t[:, None], B], 1)
        c, l1 = lin_fit(n, A, c0=np.concatenate([np.zeros((T, 1)), cb], 1), nit=nit)
        muh[:, j] = c[:, 0]
        q0[:, j] = np.where(c[:, 0] > 0, np.maximum(2 * (l1 - l0), 0.0), 0.0)
    return q0, muh, cb


# ------------------------------------------------------------------ the full likelihood with nuisance parameters
class Search:
    """nu_i = mu (1+dy)^th_y s_i(mH; th_sig, th_m) + th_ss S_spur f_i(mH) + sum_j c_j B_j(m_i).

    Parameters: [mu, c_0..c_deg, th_y, th_sig, th_m, th_ss]; every th has a unit-Gaussian
    constraint centred on its auxiliary measurement (0 for the data, drawn afresh for toys).
    th_y: signal yield (luminosity, efficiencies), relative error dy, entering as (1+dy)^th_y;
    th_sig: mass resolution, sigma -> sigma (1+ds)^th_sig; th_m: photon energy scale, peak
    position -> position (1 + dm th_m); th_ss: spurious signal of size S_spur.
    """

    def __init__(self, sig, deg=5, dy=0.10, ds=0.14, dm=0.0086, s_spur=0.0, n_bins=None):
        self.sig, self.deg = sig, deg
        self.B = bern_basis(deg)
        self.k = deg + 1
        self.dy, self.ds, self.dm, self.s_spur = dy, ds, dm, s_spur
        self.npar = 1 + self.k + 4
        self.names = ["mu"] + [f"c{j}" for j in range(self.k)] + ["th_y", "th_sig", "th_m", "th_ss"]

    def nu(self, p, mH):
        mu, c = p[0], np.asarray(p[1:1 + self.k])
        ty, ts, tm, tss = p[1 + self.k:]
        s = self.sig.bins(mH, ksig=(1 + self.ds) ** ts, kscale=1 + self.dm * tm)
        f = self.sig.bins(mH) / self.sig.s_tot
        return mu * (1 + self.dy) ** ty * s + tss * self.s_spur * f + self.B @ c

    def nll(self, p, n, mH, glob=None):
        """-2 ln L without constants; glob = the auxiliary measurements (default all 0)."""
        nu = self.nu(p, mH)
        g = np.zeros(4) if glob is None else glob
        th = np.asarray(p[1 + self.k:])
        if np.any(nu <= 0):
            return 1e12 + 1e6 * np.sum(np.minimum(nu, 0) ** 2)
        return 2 * np.sum(nu - n * np.log(nu)) + np.sum((th - g) ** 2)

    def fit(self, n, mH, mu=None, fix_theta=False, glob=None, p0=None):
        """Minimise -2 ln L with MINUIT. mu=None: free, else fixed; fix_theta: nuisances held at glob."""
        from iminuit import Minuit
        n = np.asarray(n, float)
        if p0 is None:
            cb, _ = lin_fit(n[None], self.B)
            p0 = np.r_[0.0, cb[0], np.zeros(4)]
        p0 = np.array(p0, float)
        g = np.zeros(4) if glob is None else np.asarray(glob, float)
        if mu is not None:
            p0[0] = mu
        if fix_theta:
            p0[1 + self.k:] = g
        m = Minuit(lambda q: self.nll(q, n, mH, g), p0, name=self.names)
        m.errordef = Minuit.LEAST_SQUARES          # we minimise -2 ln L: one unit = one sigma
        m.strategy = 1
        m.fixed["mu"] = mu is not None
        for nm in ("th_y", "th_sig", "th_m", "th_ss"):
            m.fixed[nm] = fix_theta
            m.limits[nm] = (-6.0, 6.0)                 # six standard deviations of each calibration
        if self.s_spur == 0.0:
            m.values["th_ss"] = 0.0
            m.fixed["th_ss"] = True
        m.migrad()
        if not m.valid:
            m.simplex()
            m.migrad()
        return np.array(m.values), m.fval

    def q0(self, n, mH, fix_theta=False, glob=None):
        """q0 = -2 ln lambda(0) if mu_hat > 0, else 0 (Cowan et al. 2011, eq. 12)."""
        p1, f1 = self.fit(n, mH, fix_theta=fix_theta, glob=glob)
        p0, f0 = self.fit(n, mH, mu=0.0, fix_theta=fix_theta, glob=glob, p0=p1)
        return (max(f0 - f1, 0.0) if p1[0] > 0 else 0.0), p1, p0

    def qtilde(self, n, mH, mu, free=None, fix_theta=False, glob=None):
        """Upper-limit statistic q~_mu (Cowan et al. 2011, eq. 16)."""
        p1, f1 = self.fit(n, mH, fix_theta=fix_theta, glob=glob) if free is None else free
        if p1[0] > mu:
            return 0.0
        pm, fm = self.fit(n, mH, mu=mu, fix_theta=fix_theta, glob=glob, p0=p1)
        if p1[0] < 0:
            pz, fz = self.fit(n, mH, mu=0.0, fix_theta=fix_theta, glob=glob, p0=p1)
            return max(fm - fz, 0.0)
        return max(fm - f1, 0.0)
