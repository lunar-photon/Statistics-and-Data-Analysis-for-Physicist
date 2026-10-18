"""22_em_mixture.py -- the EM algorithm for a peak on a flat background.

Question: a mass spectrum holds events from a Gaussian peak and from a flat background,
and no event carries a label saying which.  If we knew the labels, the signal fraction
would be a simple count.  EM guesses the labels softly (E-step), counts (M-step) and
repeats.  Does the likelihood really never go down, where does the iteration settle,
and what sets its speed?

  A. Fraction only (peak position and width known): f_{k+1} = mean_i tau_i(f_k).
     The fixed point against the root of the score; log L non-decreasing; the
     contraction factor from the data, (1/n) sum phi_i u_i / m_i^2, against the ratio
     of successive errors and against 1 - I_obs / I_complete.
  B. Fraction, position and width unknown: weighted mean and weighted variance as the
     M-step; the result against a general-purpose minimiser.
  C. Speed against overlap: data sets with peaks of different widths; measured
     contraction factor against the population value 1 - f(1-f) int (phi-u)^2/m dx.

Writes: figures/chT5/em_fit.pdf, figures/chT5/em_speed.pdf, results/chT5/22_em_mixture.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq, minimize
from scipy.integrate import quad

rng = rng_for("chT5", "22_em_mixture")
out = {}
setup()

a, b = 100.0, 160.0                      # mass window (GeV)
u = 1.0 / (b - a)                        # flat background density
n, f0, mu0, sig0 = 1000, 0.10, 125.0, 2.0


def phi_trunc(x, mu, sig):               # Gaussian peak normalised inside the window
    return stats.norm.pdf(x, mu, sig) / (stats.norm.cdf(b, mu, sig) - stats.norm.cdf(a, mu, sig))


def draw(n, f, mu, sig):
    is_sig = rng.random(n) < f
    xs = stats.truncnorm.rvs((a - mu) / sig, (b - mu) / sig, loc=mu, scale=sig,
                             size=n, random_state=rng)
    return np.where(is_sig, xs, a + (b - a) * rng.random(n)), is_sig


def em_fraction(phi, f=0.5, iters=2000, tol=1e-15):
    path, ll = [f], [np.sum(np.log(f * phi + (1 - f) * u))]
    for _ in range(iters):
        tau = f * phi / (f * phi + (1 - f) * u)          # E-step: soft labels
        f_new = tau.mean()                                # M-step: count them
        path.append(f_new)
        ll.append(np.sum(np.log(f_new * phi + (1 - f_new) * u)))
        if abs(f_new - f) < tol:
            break
        f = f_new
    return np.array(path), np.array(ll)


# ---------------------------------------------------------------- A. fraction only
x, lab = draw(n, f0, mu0, sig0)
phi = phi_trunc(x, mu0, sig0)
path, ll = em_fraction(phi)
assert np.all(np.diff(ll) >= -1e-9), "log-likelihood decreased"
f_hat = brentq(lambda f: np.sum((phi - u) / (f * phi + (1 - f) * u)), 1e-6, 1 - 1e-6)
assert abs(path[-1] - f_hat) < 1e-10
m = f_hat * phi + (1 - f_hat) * u
rate = np.mean(phi * u / m**2)
I_obs = np.sum((phi - u) ** 2 / m**2)
I_com = n / (f_hat * (1 - f_hat))
assert abs(rate - (1 - I_obs / I_com)) < 1e-10, "rate identity"
e = np.abs(path - f_hat)
j = int(np.argmax(e < 1e-7))
rate_meas = e[j] / e[j - 1]
assert abs(rate_meas - rate) < 1e-3 * rate
out.update(TFiveBEmN=n, TFiveBEmF=f0, TFiveBEmMu=mu0, TFiveBEmSig=sig0, TFiveBEmA=a, TFiveBEmB=b,
           TFiveBEmNsig=int(lab.sum()), TFiveBEmFhat=f_hat, TFiveBEmFerr=1 / np.sqrt(I_obs),
           TFiveBEmFerrCom=1 / np.sqrt(I_com),
           TFiveBEmRate=rate, TFiveBEmRateMeas=rate_meas, TFiveBEmItersA=int(np.argmax(e < 1e-10)),
           TFiveBEmPathOne=path[1], TFiveBEmPathTwo=path[2], TFiveBEmPathThree=path[3])
print(f"A: f_hat={f_hat:.5f} (true labels: {lab.mean():.4f}), rate {rate:.4f} (measured {rate_meas:.4f}), "
      f"iterations to 1e-10: {np.argmax(e < 1e-10)}; path {np.round(path[:5], 4)}")

# ---------------------------------------------------------------- B. fraction, position, width
th = np.array([0.5, 120.0, 5.0])         # deliberately poor start
llB = []
for k in range(5000):
    f, mu, sg = th
    p = stats.norm.pdf(x, mu, sg)        # peak is far from the window edges: no truncation needed
    mk = f * p + (1 - f) * u
    llB.append(np.sum(np.log(mk)))
    tau = f * p / mk
    f_new = tau.mean()
    mu_new = np.sum(tau * x) / np.sum(tau)
    sg_new = np.sqrt(np.sum(tau * (x - mu_new) ** 2) / np.sum(tau))
    th_new = np.array([f_new, mu_new, sg_new])
    if np.max(np.abs(th_new - th)) < 1e-12:
        th = th_new
        break
    th = th_new
llB = np.array(llB)
assert np.all(np.diff(llB) >= -1e-9), "log-likelihood decreased (B)"
nll = lambda q: -np.sum(np.log(q[0] * stats.norm.pdf(x, q[1], q[2]) + (1 - q[0]) * u))
res = minimize(nll, np.array([0.12, 124.0, 2.5]), method="Nelder-Mead",
               options=dict(xatol=1e-9, fatol=1e-11, maxiter=20000))
assert np.max(np.abs(res.x - th) / np.array([1e-2, 1e-1, 1e-1])) < 1e-3, (res.x, th)
out.update(TFiveBEmBf=th[0], TFiveBEmBmu=th[1], TFiveBEmBsig=th[2], TFiveBEmItersB=len(llB),
           TFiveBEmBstartLL=llB[0] - llB[-1])
print(f"B: f={th[0]:.4f} mu={th[1]:.3f} sigma={th[2]:.3f} after {len(llB)} steps; Nelder-Mead {np.round(res.x, 4)}")

# ---------------------------------------------------------------- C. speed against overlap
widths = np.array([0.5, 1, 2, 4, 8, 12])
pop, meas = [], []
for s in widths:
    ph = lambda t: phi_trunc(t, mu0, s)
    integrand = lambda t: (ph(t) - u) ** 2 / (f0 * ph(t) + (1 - f0) * u)
    I1 = quad(integrand, a, b, points=[mu0], limit=400)[0]
    pop.append(1 - f0 * (1 - f0) * I1)
    rr = []
    for rep in range(20):                # 20 data sets per width: median measured rate
        xx, _ = draw(n, f0, mu0, s)
        pp = phi_trunc(xx, mu0, s)
        pth, _ = em_fraction(pp, iters=20000, tol=1e-14)
        ee = np.abs(pth - pth[-1])
        jj = int(np.argmax(ee < 1e-6))
        rr.append(ee[jj] / ee[jj - 1])
    meas.append(np.median(rr))
pop, meas = np.array(pop), np.array(meas)
for s, p_, m_ in zip(widths, pop, meas):
    print(f"C: width {s:5.1f}: population rate {p_:.3f}, measured (median of 20) {m_:.3f}")
out.update(TFiveBEmRateHalf=pop[0], TFiveBEmRateTwo=pop[2], TFiveBEmRateEight=pop[4],
           TFiveBEmRateTwelve=pop[5], TFiveBEmMeasTwelve=meas[5], TFiveBEmMeasHalf=meas[0],
           TFiveBEmStepsTwelve=np.log(1e-6) / np.log(pop[5]), TFiveBEmStepsHalf=np.log(1e-6) / np.log(pop[0]))
assert np.all(np.diff(pop) > 0), "rate should grow with overlap"

save_numbers("chT5", "22_em_mixture", out)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))
bins = np.arange(a, b + 1, 1.0)
ax[0].hist(x, bins, histtype="step", color="k", lw=1, label="%d events" % n)
xx = np.linspace(a, b, 600)
f, mu, sg = th
ax[0].plot(xx, n * (f * stats.norm.pdf(xx, mu, sg) + (1 - f) * u), color=SERIES[0], label="EM fit, total")
ax[0].plot(xx, n * (1 - f) * u * np.ones_like(xx), "--", color=SERIES[1], label="background part")
ax[0].set_xlabel("mass $x$ (GeV)")
ax[0].set_ylabel("events per GeV")
ax[0].legend(fontsize=8)
ax[0].set_title(r"$\hat f=%.3f$, $\hat\mu=%.2f$, $\hat\sigma=%.2f$" % (f, mu, sg))
ax[1].semilogy(np.arange(len(ll)), np.maximum(ll[-1] - ll, 1e-14), "o-", ms=3, color=SERIES[0], label="fraction only")
ax[1].semilogy(np.arange(len(llB)), np.maximum(llB[-1] - llB, 1e-14), "-", color=SERIES[2], label="fraction, position, width")
ax[1].set_xlim(0, len(llB) + 5)
ax[1].set_ylim(1e-13, 1e3)
ax[1].set_xlabel("EM step $k$")
ax[1].set_ylabel(r"$\ln\mathcal{L}(\hat\theta)-\ln\mathcal{L}(\theta_k)$")
ax[1].legend(fontsize=8)
fig.tight_layout()
savefig(fig, "chT5", "em_fit")

fig, ax = plt.subplots(figsize=(5.2, 3.5))
ss = np.linspace(0.4, 13, 60)
popc = []
for s in ss:
    ph = lambda t: phi_trunc(t, mu0, s)
    popc.append(1 - f0 * (1 - f0) * quad(lambda t: (ph(t) - u) ** 2 / (f0 * ph(t) + (1 - f0) * u),
                                          a, b, points=[mu0], limit=400)[0])
ax.plot(ss, popc, "--", color="k", lw=1.4, label=r"$1-f(1-f)\int(\varphi-u)^2/m\,\mathrm{d}x$")
ax.plot(widths, meas, "o", color=SERIES[0], label="measured on simulated spectra")
ax.set_xlabel(r"peak width $\sigma$ (GeV), window %d GeV wide" % (b - a))
ax.set_ylabel("EM contraction factor")
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
savefig(fig, "chT5", "em_speed")
