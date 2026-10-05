"""20_cls_limits.py -- discovery and exclusion with nuisance parameters: q0, q~_mu, CLs, Brazil band.

Question: do the asymptotic formulae of Cowan, Cranmer, Gross and Vitells (2011) for q0 and q~_mu
hold for a small counting experiment whose background is measured in a control region (on/off)?
How different are the CL_{s+b} and CL_s exclusions when the experiment has almost no
sensitivity?  What are the observed and expected CLs upper limits, with their +-1 and +-2 sigma
bands, from toys and from the Asimov data set?  And what do the Higgs-style plots (limit
versus mass with the "Brazil band", local p0 versus mass) look like for the template search
of 19_hep_templates.py?

Computes:
  A. on/off counting, s = 10, b = 10, tau = 1: toys of q0 under mu' = 0 and 1, of q~_1 under
     mu' = 1 and 0, against the asymptotic densities; median q0 vs its Asimov value.
  B. toy-based CLs: q~_mu toy tables on a grid of mu; the observed limit for (n, m) = (14, 9);
     the distribution of limits for background-only experiments vs the Asimov band
     mu_up(N) = sigma [N + Phi^-1(1 - alpha Phi(N))].
  C. a nearly blind experiment (s = 1): how often CL_{s+b} and CL_s exclude mu = 1 when there is
     no signal.
  D. the template search: observed CLs limit and expected band versus hypothesised mass, and
     the local p0 versus mass with its Asimov expectation for mu = 1.
Writes: figures/ch04/hep_qdists.pdf, figures/ch04/hep_brazil.pdf, results/ch04/20_cls_limits.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import lib_templates as T

rng = rng_for("ch04", "20_cls_limits")
setup()
Phi, Phinv = stats.norm.cdf, stats.norm.ppf
ALPHA = 0.05


# ---------- A. the on/off counting experiment, vectorised ----------
class OnOff:
    """n ~ Pois(mu s + b) in the signal region, m ~ Pois(tau b) in the control region."""

    def __init__(self, s, b, tau):
        self.s, self.b, self.tau = s, b, tau

    def bhh(self, mu, n, m):          # profiled background, Cowan et al. (93)
        s, t = self.s, self.tau
        a = n + m - (1 + t) * mu * s
        return np.maximum(a / (2 * (1 + t)) + np.sqrt(a**2 + 4 * (1 + t) * m * mu * s) / (2 * (1 + t)), 1e-12)

    def lnL(self, mu, b, n, m):
        nu = np.maximum(mu * self.s + b, 1e-300)
        return n * np.log(nu) - nu + m * np.log(self.tau * b) - self.tau * b

    def mle(self, n, m):
        return (n - m / self.tau) / self.s, m / self.tau

    def lnL_hat(self, n, m):          # at the MLE: mu s + b = n, tau b = m (0 ln 0 = 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(n > 0, n * np.log(np.maximum(n, 1e-300)), 0) - n + \
                   np.where(m > 0, m * np.log(np.maximum(m, 1e-300)), 0) - m

    def q0(self, n, m):
        mu_h, _ = self.mle(n, m)
        q = 2 * (self.lnL_hat(n, m) - self.lnL(0.0, self.bhh(0.0, n, m), n, m))
        return np.where(mu_h > 0, np.maximum(q, 0), 0.0)

    def qt(self, mu, n, m):           # q~_mu, Cowan et al. (16)
        mu_h, _ = self.mle(n, m)
        l_mu = self.lnL(mu, self.bhh(mu, n, m), n, m)
        l_ref = np.where(mu_h >= 0, self.lnL_hat(n, m), self.lnL(0.0, self.bhh(0.0, n, m), n, m))
        return np.where(mu_h > mu, 0.0, np.maximum(2 * (l_ref - l_mu), 0.0))

    def toys(self, mu, N):
        return (rng.poisson(mu * self.s + self.b, N).astype(float),
                rng.poisson(self.tau * self.b, N).astype(float))

    def asimov(self, mu):
        return np.array(mu * self.s + self.b, float), np.array(self.tau * self.b, float)

    def sigmaA(self, mu):             # sigma_A^2 = mu^2 / q~_{mu,A} with mu' = 0, Cowan (31)
        nA, mA = self.asimov(0.0)
        return mu / np.sqrt(self.qt(mu, nA, mA))


def F_qt(q, mu, mu_p, sig):
    """Asymptotic CDF of q~_mu when the true strength is mu' (Cowan et al. 2011, eq. 65)."""
    q = np.asarray(q, float)
    a = (mu / sig) ** 2
    return np.where(q <= a, Phi(np.sqrt(q) - (mu - mu_p) / sig),
                    Phi((q - (mu**2 - 2 * mu * mu_p) / sig**2) / (2 * mu / sig)))


def cls_asym(q, mu, sig):
    p_sb = 1 - F_qt(q, mu, mu, sig)
    one_minus_pb = 1 - F_qt(q, mu, 0.0, sig)
    return p_sb / one_minus_pb, p_sb


E = OnOff(10.0, 10.0, 1.0)
NT = 400_000
n0, m0 = E.toys(0.0, NT)
n1, m1 = E.toys(1.0, NT)
q0_b, q0_sb = E.q0(n0, m0), E.q0(n1, m1)
q0A = float(E.q0(*E.asimov(1.0)))
sig0 = 1.0 / np.sqrt(q0A)                     # sigma of mu_hat from Asimov, Cowan (32)
qt_sb, qt_b = E.qt(1.0, n1, m1), E.qt(1.0, n0, m0)
sig1 = float(E.sigmaA(1.0))
qtA = float(E.qt(1.0, *E.asimov(0.0)))

# tail checks: P(q0 >= 9 | 0) toys vs Phi(-3); median q0 under mu' = 1 vs Asimov
p_q0_9 = np.mean(q0_b >= 9)
med_q0 = np.median(q0_sb)


# ---------- B. toy-based CLs limits on a grid of mu ----------
mu_grid = np.round(np.arange(0.1, 4.01, 0.05), 3)
NG = 40_000
tab_sb, tab_b = [], []
nb, mb = E.toys(0.0, NG)                       # background-only toys (shared across mu)
for mu in mu_grid:
    ns, ms = E.toys(mu, NG)
    tab_sb.append(np.sort(E.qt(mu, ns, ms)))
    tab_b.append(np.sort(E.qt(mu, nb, mb)))


def tail(sorted_q, x):                          # P(q >= x) from a sorted toy table
    return 1.0 - np.searchsorted(sorted_q, x, side="left") / len(sorted_q)


def limit_toys(n, m):
    cls = np.array([tail(tab_sb[k], E.qt(mu, n, m)) / max(tail(tab_b[k], E.qt(mu, n, m)), 1e-12)
                    for k, mu in enumerate(mu_grid)])
    k = np.argmax(cls < ALPHA)
    if k == 0:
        return mu_grid[0]
    return np.interp(ALPHA, [cls[k], cls[k - 1]], [mu_grid[k], mu_grid[k - 1]])


def limit_asym(n, m):
    def g(mu):
        return cls_asym(E.qt(mu, n, m), mu, float(E.sigmaA(mu)))[0] - ALPHA
    return optimize.brentq(g, 0.02, 10.0)


def band_asym(N):
    """Expected CLs limit for mu_hat = N sigma, sigma = sigma_A(mu) solved self-consistently."""
    k = N + Phinv(1 - ALPHA * Phi(N))
    return optimize.brentq(lambda mu: mu - float(E.sigmaA(mu)) * k, 0.02, 10.0)


n_obs, m_obs = 14.0, 9.0
lim_obs_toy, lim_obs_asym = limit_toys(n_obs, m_obs), limit_asym(n_obs, m_obs)
cls_obs_toy_1 = tail(tab_sb[np.argmin(abs(mu_grid - 1))], E.qt(1.0, n_obs, m_obs)) / \
    tail(tab_b[np.argmin(abs(mu_grid - 1))], E.qt(1.0, n_obs, m_obs))
NB = 4000
nbb, mbb = E.toys(0.0, NB)
lims = np.array([limit_toys(a, b) for a, b in zip(nbb, mbb)])
qs = [Phi(-2), Phi(-1), 0.5, Phi(1), Phi(2)]
band_toy = np.quantile(lims, qs)
band_as = np.array([band_asym(N) for N in (-2, -1, 0, 1, 2)])
coef = {N: N + Phinv(1 - ALPHA * Phi(N)) for N in (-2, -1, 0, 1, 2)}

# ---------- C. a nearly blind experiment ----------
W = OnOff(1.0, 10.0, 1.0)
nw1, mw1 = W.toys(1.0, NT)
nw0, mw0 = W.toys(0.0, NT)
qw_sb = np.sort(W.qt(1.0, nw1, mw1))
qw_b_sorted = np.sort(W.qt(1.0, nw0, mw0))
qw_obs = W.qt(1.0, nw0[:20000], mw0[:20000])     # background-only "observations"
p_sb_w = tail(qw_sb, qw_obs)
cl_b_w = tail(qw_b_sorted, qw_obs)
excl_sb = np.mean(p_sb_w < ALPHA)
excl_cls = np.mean(p_sb_w / np.maximum(cl_b_w, 1e-12) < ALPHA)
sigw = float(W.sigmaA(1.0))
excl_sb_asym = Phi(1.0 / sigw - Phinv(1 - ALPHA))

# ---------- figure 1: toys against the asymptotic densities ----------
fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8))
bins = np.linspace(0, 30, 61)
w = bins[1] - bins[0]
x = np.linspace(0.01, 30, 600)
a = ax[0]
for q, c, lab in ((q0_b, SERIES[0], r"toys, $\mu'=0$"), (q0_sb, SERIES[1], r"toys, $\mu'=1$")):
    h, _ = np.histogram(q[q > 0], bins)
    a.step(bins[:-1], h / (len(q) * w), where="post", color=c, label=lab)
f0 = 0.5 / np.sqrt(2 * np.pi * x) * np.exp(-x / 2)
f1 = 0.5 / np.sqrt(2 * np.pi * x) * np.exp(-0.5 * (np.sqrt(x) - 1 / sig0) ** 2)
theory_line(a, x, f0, label="asymptotic")
a.plot(x, f1, color="k", ls="--", lw=1.4, zorder=5)
a.axvline(q0A, color="0.4", ls=":", lw=1)
a.set_yscale("log"); a.set_ylim(1e-7, 1); a.set_xlim(0, 30)
a.set_xlabel(r"$q_0$"); a.set_ylabel("density (continuous part)")
a.legend(fontsize=8)
a = ax[1]
bins2 = np.linspace(0, 12, 49); w2 = bins2[1] - bins2[0]
x2 = np.linspace(0.01, 12, 600)
for q, c, lab in ((qt_sb, SERIES[0], r"toys, $\mu'=1$"), (qt_b, SERIES[1], r"toys, $\mu'=0$")):
    h, _ = np.histogram(q[q > 0], bins2)
    a.step(bins2[:-1], h / (len(q) * w2), where="post", color=c, label=lab)
for mup in (1.0, 0.0):
    F = F_qt(x2, 1.0, mup, sig1)
    dens = np.gradient(F, x2)
    a.plot(x2, dens, color="k", ls="--", lw=1.4, zorder=5, label="asymptotic" if mup == 1 else None)
a.axvline(qtA, color="0.4", ls=":", lw=1)
a.set_yscale("log"); a.set_ylim(1e-4, 2); a.set_xlim(0, 12)
a.set_xlabel(r"$\tilde q_\mu$ at $\mu=1$"); a.legend(fontsize=8)
savefig(fig, "ch04", "hep_qdists")

# ---------- D. the template search: Brazil band and local p0 versus mass ----------
n = T.pseudo_data(rng_for("ch04", "lib_templates"))
masses = np.arange(110.0, 150.01, 1.0)


def sigA_T(mu, m, globA):
    nA = T.asimov(0.0, m)
    q, _ = T.q_tilde(mu, nA, m, glob=globA)
    return mu / np.sqrt(max(q, 1e-12))


obs_lim, exp_band, p0_obs, p0_exp = [], [], [], []
for m in masses:
    glob, globA = T.fit(n, m), T.fit(T.asimov(0.0, m), m)
    g = lambda mu: cls_asym(T.q_tilde(mu, n, m, glob)[0], mu, sigA_T(mu, m, globA))[0] - ALPHA
    obs_lim.append(optimize.brentq(g, 0.05, 6.0, xtol=1e-3))
    row = []
    for N in (-2, -1, 0, 1, 2):
        k = N + Phinv(1 - ALPHA * Phi(N))
        row.append(optimize.brentq(lambda mu: mu - sigA_T(mu, m, globA) * k, 0.05, 6.0, xtol=1e-3))
    exp_band.append(row)
    q0o, _ = T.q0(n, m)
    p0_obs.append(1 - Phi(np.sqrt(q0o)))
    q0e, _ = T.q0(T.asimov(1.0, m), m)
    p0_exp.append(1 - Phi(np.sqrt(q0e)))
obs_lim, exp_band = np.array(obs_lim), np.array(exp_band)
p0_obs, p0_exp = np.array(p0_obs), np.array(p0_exp)

fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8))
a = ax[0]
a.fill_between(masses, exp_band[:, 0], exp_band[:, 4], color="#f2d53c", lw=0, label=r"expected $\pm2\sigma$")
a.fill_between(masses, exp_band[:, 1], exp_band[:, 3], color="#3cb043", lw=0, label=r"expected $\pm1\sigma$")
a.plot(masses, exp_band[:, 2], color="k", ls="--", lw=1.2, label="expected (median)")
a.plot(masses, obs_lim, color="k", lw=1.6, marker="o", ms=2.5, label="observed")
a.axhline(1.0, color=SERIES[7], lw=1.0)
a.set_xlabel(r"hypothesised mass $m_H$ [GeV]"); a.set_ylabel(r"95% CL$_s$ limit on $\mu$")
a.set_xlim(110, 150); a.set_ylim(0, 3.2); a.legend(fontsize=8, loc="upper left")
a = ax[1]
a.semilogy(masses, p0_obs, color="k", lw=1.6, label="observed local $p_0$")
a.semilogy(masses, p0_exp, color="k", ls="--", lw=1.2, label=r"expected for $\mu=1$")
for z in (1, 2, 3, 4):
    a.axhline(1 - Phi(z), color=SERIES[7], lw=0.7, ls=":")
    a.text(150.3, 1 - Phi(z), rf"{z}$\sigma$", va="center", fontsize=8, color=SERIES[7])
a.set_xlabel(r"hypothesised mass $m_H$ [GeV]"); a.set_ylabel(r"local $p_0$")
a.set_xlim(110, 150); a.set_ylim(1e-5, 1); a.legend(fontsize=8, loc="lower left")
savefig(fig, "ch04", "hep_brazil")

i125 = np.argmin(abs(masses - 125))
f = lambda x, d=2: f"{x:.{d}f}"
save_numbers("ch04", "20_cls_limits", {
    "FourBHlToys": f"{NT:,}".replace(",", r"\,"), "FourBHlGridToys": f"{NG:,}".replace(",", r"\,"),
    "FourBHlBandToys": f"{NB}",
    "FourBHlQzeroA": f(q0A), "FourBHlZA": f(np.sqrt(q0A)), "FourBHlMedQzero": f(med_q0),
    "FourBHlZmed": f(np.sqrt(med_q0)),
    "FourBHlZAform": f(np.sqrt(2 * (20 * np.log(2) - 10))),
    "FourBHlSigZero": f(sig0, 3), "FourBHlSigOne": f(sig1, 3), "FourBHlQtA": f(qtA),
    "FourBHlFracZero": f(np.mean(q0_b == 0), 3),
    "FourBHlPnine": float(p_q0_9), "FourBHlPnineTh": r"1.35\times10^{-3}",
    "FourBHlNobs": f"{n_obs:.0f}", "FourBHlMobs": f"{m_obs:.0f}",
    "FourBHlLimToy": f(lim_obs_toy), "FourBHlLimAsym": f(lim_obs_asym),
    "FourBHlBandToyMm": f(band_toy[0]), "FourBHlBandToyM": f(band_toy[1]), "FourBHlBandToyMed": f(band_toy[2]),
    "FourBHlBandToyP": f(band_toy[3]), "FourBHlBandToyPp": f(band_toy[4]),
    "FourBHlBandAsMm": f(band_as[0]), "FourBHlBandAsM": f(band_as[1]), "FourBHlBandAsMed": f(band_as[2]),
    "FourBHlBandAsP": f(band_as[3]), "FourBHlBandAsPp": f(band_as[4]),
    "FourBHlCoefMm": f(coef[-2]), "FourBHlCoefM": f(coef[-1]), "FourBHlCoefMed": f(coef[0]),
    "FourBHlCoefP": f(coef[1]), "FourBHlCoefPp": f(coef[2]),
    "FourBHlBlindSig": f(sigw), "FourBHlExclSB": f(excl_sb, 3), "FourBHlExclSBAsym": f(excl_sb_asym, 3),
    "FourBHlExclCLs": f(excl_cls, 4),
    "FourBHlObsLimOneTwoFive": f(obs_lim[i125]), "FourBHlExpLimOneTwoFive": f(exp_band[i125, 2]),
    "FourBHlExpLimMin": f(exp_band[:, 2].min()), "FourBHlExpLimMax": f(exp_band[:, 2].max()),
    "FourBHlZlocOneTwoFive": f(-Phinv(p0_obs[i125])), "FourBHlZexpOneTwoFive": f(-Phinv(p0_exp[i125])),
    "FourBHlMassMinP": f"{masses[np.argmin(p0_obs)]:.0f}", "FourBHlZlocMax": f(-Phinv(p0_obs.min())),
    "FourBHlNexcl": f"{np.sum(obs_lim < 1)}",
})
print("q0A", q0A, "med q0", med_q0, "P(q0>=9)", p_q0_9, "sig0", sig0, "sig1", sig1)
print("limits obs toy/asym", lim_obs_toy, lim_obs_asym, "band toy", band_toy, "band asym", band_as)
print("blind: excl sb", excl_sb, "asym", excl_sb_asym, "cls", excl_cls, "sigw", sigw)
print("template: lim125", obs_lim[i125], "exp", exp_band[i125], "Zloc125", -Phinv(p0_obs[i125]))
