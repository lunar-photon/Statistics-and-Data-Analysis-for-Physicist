"""19_hep_templates.py -- a search as a likelihood: templates and nuisance parameters.

Question: a mass spectrum contains a possible peak at 125 GeV on a falling background. The
background normalisation, the signal yield (luminosity x efficiency) and the energy scale are
known only from calibrations. How do we turn that into one likelihood, remove the nuisance
parameters, and how much does each way of removing them (fix, profile, marginalise) change
what we learn about the signal strength mu?  And what does the same search look like if we
keep only one signal window and its sidebands (a counting experiment with a control region)?

Computes: one pseudo-data spectrum (signal mu = 1 at 125 GeV); the global fit; the profile,
conditional (nuisances fixed at their best values) and marginal (Bayesian, flat prior on the
background normalisation, the calibrations as Gaussian priors) curves of mu; the
stat-only and total errors; the signal-window + sidebands counting version (on/off), with
mu_hat, the profiled background, q0 and Z, against the known-background and full-template
results; Asimov expected significances.

Writes: figures/ch04/hep_templates.pdf, results/ch04/19_hep_templates.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import lib_templates as T

setup()
n = T.pseudo_data(rng_for("ch04", "lib_templates"))     # the "observed" spectrum
M = 125.0

# ---------- global fit and the three one-dimensional curves ----------
nll_hat, mu_hat, th_hat = T.fit(n, M)
mus = np.linspace(-0.6, 2.6, 81)
prof = np.array([2 * (T.fit(n, M, mu)[0] - nll_hat) for mu in mus])
cond = np.array([2 * (T.nll(mu, th_hat, n, M) - nll_hat) for mu in mus])

# marginal: integrate exp(-nll) over a grid of the three nuisance parameters
bg = np.linspace(th_hat[0] - 0.1, th_hat[0] + 0.1, 41)
kg = np.linspace(0.55, 1.45, 37)
dg = np.linspace(-2.0, 2.0, 33)
Bb, Kk, Dd = np.meshgrid(bg, kg, dg, indexing="ij")
ftab = np.array([T.sig_template(M + d) for d in dg])                 # (33, bins)
cons = 0.5 * ((Kk - T.A_KAPPA) / T.SIG_KAPPA) ** 2 + 0.5 * ((Dd - T.A_DELTA) / T.SIG_DELTA) ** 2


def log_marg(mu):
    nu = mu * Kk[..., None] * T.S0 * ftab[None, None, :, :] + Bb[..., None] * T.B0 * T.G
    ll = -(np.sum(nu - n * np.log(nu), axis=-1) + cons)
    mx = ll.max()
    return mx + np.log(np.exp(ll - mx).sum())


lm = np.array([log_marg(mu) for mu in mus])
marg = -2 * (lm - lm.max())


def interval(curve, level=1.0):
    """Points where the curve crosses `level` on either side of its minimum (linear interp)."""
    i0 = np.argmin(curve)
    fine = np.linspace(mus[0], mus[-1], 4001)
    c = np.interp(fine, mus, curve)
    j0 = np.argmin(c)
    lo = fine[:j0][c[:j0] > level][-1]
    hi = fine[j0:][c[j0:] > level][0]
    return fine[j0], lo, hi


p_c, p_lo, p_hi = interval(prof)
c_c, c_lo, c_hi = interval(cond)
m_c, m_lo, m_hi = interval(marg)
sig_tot = 0.5 * (p_hi - p_lo)
sig_stat = 0.5 * (c_hi - c_lo)
sig_syst = np.sqrt(max(sig_tot**2 - sig_stat**2, 0.0))

# discovery at the fixed mass 125 GeV, and its Asimov expectation
q0_obs, _ = T.q0(n, M)
nA = T.asimov(1.0, M)
q0_A, _ = T.q0(nA, M)

# ---------- the counting version: signal window + sidebands ----------
win = (T.CENTRES > 122) & (T.CENTRES < 128)
side = ((T.CENTRES > 110) & (T.CENTRES < 120)) | ((T.CENTRES > 130) & (T.CENTRES < 140))
tau = T.G[side].sum() / T.G[win].sum()          # background in sidebands / in window (known shape)
s_win = T.S0 * T.sig_template(M)[win].sum()     # nominal signal in the window
b_win = T.B0 * T.G[win].sum()                   # true background in the window
n_on, n_off = n[win].sum(), n[side].sum()


def bhh(mu, n_on, n_off):
    """Profiled background (Cowan et al. 2011, eq. 93): positive root of the quadratic."""
    a = n_on + n_off - (1 + tau) * mu * s_win
    return a / (2 * (1 + tau)) + np.sqrt(a**2 + 4 * (1 + tau) * n_off * mu * s_win) / (2 * (1 + tau))


def lnL(mu, b, n_on, n_off):
    return n_on * np.log(mu * s_win + b) - (mu * s_win + b) + n_off * np.log(tau * b) - tau * b


def q0_onoff(n_on, n_off):
    mu_h = (n_on - n_off / tau) / s_win
    if mu_h <= 0:
        return 0.0, mu_h
    return 2 * (lnL(mu_h, n_off / tau, n_on, n_off) - lnL(0, bhh(0, n_on, n_off), n_on, n_off)), mu_h


q0_oo, mu_oo = q0_onoff(n_on, n_off)
b_hat = n_off / tau
b_hh0 = bhh(0, n_on, n_off)
sig_oo = np.sqrt(n_on + n_off / tau**2) / s_win            # Gaussian-limit error on mu_hat
# known background: the 4a formula
q0_kb = 2 * (n_on * np.log(n_on / b_win) - (n_on - b_win)) if n_on > b_win else 0.0
# Asimov expectations for the three analyses (mu' = 1)
q0A_oo, _ = q0_onoff(s_win + b_win, tau * b_win)
ZA_kb = np.sqrt(2 * ((s_win + b_win) * np.log(1 + s_win / b_win) - s_win))

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
a = ax[0]
fine = np.linspace(100, 160, 601)
a.errorbar(T.CENTRES, n, yerr=np.sqrt(n), fmt="o", color="k", ms=2.5, lw=0.8, label="pseudo-data")
_, _, th0 = T.fit(n, M, 0.0)
a.step(T.EDGES, np.r_[T.expected(0.0, th0, M), np.nan], where="post", color=SERIES[0],
       ls="--", lw=1.2, label=r"background-only fit ($\mu=0$)")
a.step(T.EDGES, np.r_[T.expected(mu_hat, th_hat, M), np.nan], where="post", color=SERIES[1],
       lw=1.4, label=rf"best fit, $\hat\mu={mu_hat:.2f}$")
a.axvspan(122, 128, color=SERIES[2], alpha=0.12, lw=0)
for lo_, hi_ in ((110, 120), (130, 140)):
    a.axvspan(lo_, hi_, color=SERIES[3], alpha=0.10, lw=0)
a.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]"); a.set_ylabel("events / GeV")
a.set_xlim(100, 160); a.legend(loc="upper right", fontsize=8)
b = ax[1]
b.plot(mus, prof, color=SERIES[0], label="profile")
b.plot(mus, cond, color=SERIES[2], ls="-.", label="nuisances fixed")
b.plot(mus, marg, color=SERIES[1], ls=":", lw=2.0, label="marginal (Bayes)")
b.axhline(1.0, color="0.5", lw=0.8)
b.set_ylim(0, 6); b.set_xlim(mus[0], mus[-1])
b.set_xlabel(r"signal strength $\mu$"); b.set_ylabel(r"$-2\ln(\mathcal{L}/\mathcal{L}_{\max})$")
b.legend(loc="upper center", fontsize=8)
savefig(fig, "ch04", "hep_templates")

f = lambda x, d=3: f"{x:.{d}f}"
save_numbers("ch04", "19_hep_templates", {
    "FourBTpBtot": f"{T.B0:.0f}", "FourBTpSzero": f"{T.S0:.0f}", "FourBTpSigM": f"{T.SIG_M}",
    "FourBTpSlope": f"{T.SLOPE:.0f}",
    "FourBTpSigKappa": f"{T.SIG_KAPPA:.2f}", "FourBTpSigDelta": f"{T.SIG_DELTA}",
    "FourBTpMuHat": f(mu_hat, 2), "FourBTpBetaHat": f(th_hat[0], 3),
    "FourBTpKappaHat": f(th_hat[1], 2), "FourBTpDeltaHat": f(th_hat[2], 2),
    "FourBTpProfLo": f(p_c - p_lo, 2), "FourBTpProfHi": f(p_hi - p_c, 2),
    "FourBTpCondLo": f(c_c - c_lo, 2), "FourBTpCondHi": f(c_hi - c_c, 2),
    "FourBTpMargMode": f(m_c, 2), "FourBTpMargLo": f(m_c - m_lo, 2), "FourBTpMargHi": f(m_hi - m_c, 2),
    "FourBTpSigTot": f(sig_tot, 2), "FourBTpSigStat": f(sig_stat, 2), "FourBTpSigSyst": f(sig_syst, 2),
    "FourBTpMaxDiffPM": f(np.max(np.abs(prof - marg)[prof < 4]), 2),
    "FourBTpQzero": f(q0_obs, 2), "FourBTpZzero": f(np.sqrt(q0_obs), 2),
    "FourBTpQzeroA": f(q0_A, 2), "FourBTpZA": f(np.sqrt(q0_A), 2),
    "FourBTpTau": f(tau, 3), "FourBTpSwin": f(s_win, 1), "FourBTpBwin": f(b_win, 1),
    "FourBTpNon": f"{n_on:.0f}", "FourBTpNoff": f"{n_off:.0f}",
    "FourBTpMuOO": f(mu_oo, 2), "FourBTpBhatOO": f(b_hat, 1), "FourBTpBhhOO": f(b_hh0, 1),
    "FourBTpSigOO": f(sig_oo, 2), "FourBTpQzeroOO": f(q0_oo, 2), "FourBTpZOO": f(np.sqrt(q0_oo), 2),
    "FourBTpZKB": f(np.sqrt(q0_kb), 2), "FourBTpZAOO": f(np.sqrt(q0A_oo), 2), "FourBTpZAKB": f(ZA_kb, 2),
    "FourBTpSigOOKB": f(np.sqrt(n_on) / s_win, 2),
})
print(f"mu_hat={mu_hat:.3f} theta={th_hat} prof [{p_lo:.3f},{p_hi:.3f}] cond [{c_lo:.3f},{c_hi:.3f}]"
      f" marg {m_c:.3f} [{m_lo:.3f},{m_hi:.3f}] q0={q0_obs:.2f} q0A={q0_A:.2f}")
print(f"on/off: n_on={n_on} n_off={n_off} tau={tau:.3f} s={s_win:.1f} b={b_win:.1f} mu={mu_oo:.3f}"
      f" Z={np.sqrt(q0_oo):.2f} Zkb={np.sqrt(q0_kb):.2f} ZA_oo={np.sqrt(q0A_oo):.2f} ZA_kb={ZA_kb:.2f}")
