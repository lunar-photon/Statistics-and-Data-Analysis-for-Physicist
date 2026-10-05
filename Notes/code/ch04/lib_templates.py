"""lib_templates.py -- a binned template search with nuisance parameters (used by 19 and 20).

Model: a diphoton-like mass spectrum, 60 bins of 1 GeV from 100 to 160 GeV.
  expected count in bin i:  nu_i = mu * kappa * S0 * f_i(m + delta)  +  beta * B0 * g_i
  f_i(m): fraction of a Gaussian peak (width SIG_M) at mass m falling in bin i (signal template)
  g_i   : fraction of a falling exponential (decay length SLOPE) in bin i (background template)
Parameter of interest: the signal strength mu (0 = background only, 1 = nominal signal).
Nuisance parameters theta = (beta, kappa, delta):
  beta  background normalisation, constrained only by the spectrum itself (the sidebands);
  kappa signal yield factor (luminosity x efficiency), auxiliary measurement a_kappa = 1 +- 0.10;
  delta photon energy scale shift in GeV, auxiliary calibration a_delta = 0 +- 0.5 GeV.
The auxiliary measurements enter as Gaussian likelihood factors (constraint terms).
"""
import numpy as np
from scipy import stats, optimize

EDGES = np.arange(100.0, 161.0, 1.0)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])
B0, SLOPE = 3000.0, 30.0          # expected background events, decay length (GeV)
S0, SIG_M = 60.0, 1.5             # nominal signal events, mass resolution (GeV)
SIG_KAPPA, SIG_DELTA = 0.10, 0.5  # widths of the two calibration constraints
A_KAPPA, A_DELTA = 1.0, 0.0       # the calibration results (auxiliary data)

_cdf_b = lambda x: 1 - np.exp(-(x - EDGES[0]) / SLOPE)
G = np.diff(_cdf_b(EDGES)) / _cdf_b(EDGES[-1])        # background template, sums to 1


def sig_template(m):
    """Fraction of a unit Gaussian peak at mass m in each bin."""
    return np.diff(stats.norm.cdf(EDGES, m, SIG_M))


def expected(mu, theta, m=125.0):
    beta, kappa, delta = theta
    return mu * kappa * S0 * sig_template(m + delta) + beta * B0 * G


def nll(mu, theta, n, m=125.0, aux=(A_KAPPA, A_DELTA)):
    """-ln L(mu, theta) up to a constant: Poisson bins times Gaussian constraint terms."""
    nu = expected(mu, theta, m)
    if np.any(nu <= 0):
        return 1e12
    beta, kappa, delta = theta
    pois = np.sum(nu - n * np.log(nu))
    cons = 0.5 * ((kappa - aux[0]) / SIG_KAPPA) ** 2 + 0.5 * ((delta - aux[1]) / SIG_DELTA) ** 2
    return pois + cons


def fit(n, m=125.0, mu=None, aux=(A_KAPPA, A_DELTA)):
    """Maximise L. mu=None: over (mu, theta) (unconditional); else over theta at fixed mu.
    Returns (min nll, mu_hat or mu, theta_hat)."""
    th0 = np.array([max(n.sum() / B0, 0.5), aux[0], aux[1]])
    bnd = [(0.3, 3.0), (0.2, 2.5), (-3.0, 3.0)]
    if mu is None:
        f = lambda p: nll(p[0], p[1:], n, m, aux)
        best = None
        for mu0 in (0.0, 1.0, -0.5):
            r = optimize.minimize(f, np.r_[mu0, th0], method="L-BFGS-B",
                                  bounds=[(-5.0, 10.0)] + bnd)
            if best is None or r.fun < best.fun:
                best = r
        return best.fun, best.x[0], best.x[1:]
    f = lambda t: nll(mu, t, n, m, aux)
    r = optimize.minimize(f, th0, method="L-BFGS-B", bounds=bnd)
    return r.fun, mu, r.x


def asimov(mu_prime=0.0, m=125.0):
    """Asimov data set: every count equal to its expectation, nuisances and aux at nominal."""
    return expected(mu_prime, (1.0, A_KAPPA, A_DELTA), m)


def q_tilde(mu, n, m=125.0, glob=None):
    """The upper-limit statistic q~_mu (Cowan et al. 2011, eq. 16). glob = fit(n, m) if known."""
    nll_hat, mu_hat, _ = glob if glob is not None else fit(n, m)
    if mu_hat > mu:
        return 0.0, mu_hat
    nll_mu = fit(n, m, mu)[0]
    ref = nll_hat if mu_hat >= 0 else fit(n, m, 0.0)[0]
    return max(2 * (nll_mu - ref), 0.0), mu_hat


def q0(n, m=125.0):
    """The discovery statistic q0 (eq. 12): -2 ln lambda(0) if mu_hat >= 0, else 0."""
    nll_hat, mu_hat, _ = fit(n, m)
    if mu_hat < 0:
        return 0.0, mu_hat
    return max(2 * (fit(n, m, 0.0)[0] - nll_hat), 0.0), mu_hat


def pseudo_data(rng, mu_true=1.0, m_true=125.0):
    """One pseudo-experiment: Poisson counts around the nominal model with signal mu_true."""
    return rng.poisson(expected(mu_true, (1.0, 1.0, 0.0), m_true)).astype(float)
