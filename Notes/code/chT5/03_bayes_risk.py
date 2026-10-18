"""03_bayes_risk.py -- how a Bayes estimator behaves when we repeat the experiment.

Question (a): an average xbar ~ N(theta, s^2) is combined with a Gaussian prior N(mu0, tau^2).
The posterior mean pulls xbar toward mu0. For which true theta is its mean squared error below the
s^2 of xbar itself? (Theory: when |theta - mu0| < sqrt(2 tau^2 + s^2).)
Question (b): a success probability p from y successes in n trials. Risk of y/n, of the uniform
-prior posterior mean (y+1)/(n+2), and of the constant-risk (minimax) choice alpha = beta = sqrt(n)/2.
Question (c): a smooth signal s with power spectrum P_s plus white noise of power P_n. The Wiener
filter multiplies each Fourier mode by P_s/(P_s+P_n). How much does it lower the error, and by how
much does it shrink the signal's amplitude (its bias)?

Computes: analytic risk curves with Monte Carlo points; exact binomial risks; a Wiener-filtered
          realisation, Monte Carlo total errors against theory; a check that the matrix forms
          (S^-1 + N^-1)^-1 N^-1 d and S (S + N)^-1 d agree.
Writes:   figures/chT5/bayes_risk.pdf, figures/chT5/wiener.pdf, results/chT5/03_bayes_risk.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("chT5", "03_bayes_risk")
setup()

# ---------- (a) Gaussian: risk of the posterior mean ----------
s, mu0 = 1.0, 0.0
delta = np.linspace(-4, 4, 401)                     # theta - mu0 in units of s
def risk_bayes(dl, tau):
    w = tau**2 / (tau**2 + s**2)
    return w**2 * s**2 + (1 - w)**2 * dl**2
Mg = 40_000
dpts = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
tau_mc = 1.0
w_mc = tau_mc**2 / (tau_mc**2 + s**2)
mc = []
for dl in dpts:
    xb = dl + s * rng.standard_normal(Mg)
    est = w_mc * xb + (1 - w_mc) * mu0
    mc.append(np.mean((est - dl)**2))
mc = np.array(mc)
assert np.all(np.abs(mc - risk_bayes(dpts, tau_mc)) < 0.05 * risk_bayes(dpts, tau_mc) + 0.02)

# ---------- (b) binomial: risk of three estimators of p ----------
nb = 20
p = np.linspace(0.001, 0.999, 500)
def risk_beta(p, a, b):
    """MSE of (y+a)/(n+a+b): [n p(1-p) + (a - (a+b) p)^2] / (n+a+b)^2."""
    return (nb * p * (1 - p) + (a - (a + b) * p)**2) / (nb + a + b)**2
r_mle = p * (1 - p) / nb
r_unif = risk_beta(p, 1, 1)
am = np.sqrt(nb) / 2
r_minimax = risk_beta(p, am, am)
assert np.allclose(r_minimax, 1 / (4 * (1 + np.sqrt(nb))**2))
# exact check of the closed form by summing over y
y = np.arange(nb + 1)
pp = 0.3
exact = np.sum(stats.binom.pmf(y, nb, pp) * ((y + am) / (nb + 2 * am) - pp)**2)
assert abs(exact - risk_beta(pp, am, am)) < 1e-12

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1))
axs[0].axhline(s**2, color=SERIES[0], label=r"$\bar x$ (ML)")
for tau, c in [(0.5, SERIES[1]), (1.0, SERIES[2]), (2.0, SERIES[3])]:
    axs[0].plot(delta, risk_bayes(delta, tau), color=c, label=rf"posterior mean, $\tau={tau:g}$")
axs[0].plot(dpts, mc, "o", color=SERIES[2], ms=4, label=r"Monte Carlo, $\tau=1$")
axs[0].set_ylim(0, 3); axs[0].set_xlabel(r"$(\theta-\mu_0)/s$"); axs[0].set_ylabel(r"risk $R(\theta)/s^2$")
axs[0].legend(fontsize=7, loc="upper center")
axs[1].plot(p, r_mle, color=SERIES[0], label=r"$y/n$ (ML)")
axs[1].plot(p, r_unif, color=SERIES[1], label=r"$(y+1)/(n+2)$")
axs[1].plot(p, r_minimax, color=SERIES[2], label=r"$\alpha=\beta=\sqrt{n}/2$ (minimax)")
axs[1].set_xlabel(r"true $p$"); axs[1].set_ylabel("risk (mean squared error)")
axs[1].set_title(rf"$n={nb}$ trials"); axs[1].legend(fontsize=7, loc="lower center")
fig.tight_layout()
savefig(fig, "chT5", "bayes_risk")

# ---------- (c) Wiener filter in one dimension ----------
Npix, ell0, Pn = 512, 12.0, 1.0
k = np.fft.fftfreq(Npix, d=1.0 / Npix)            # integer wavenumbers 0, 1, ..., -1
Ps = 20.0 / (1.0 + (np.abs(k) / ell0)**4)         # smooth signal: power concentrated at |k| < 12
wk = Ps / (Ps + Pn)                               # Wiener weight of each mode


def draw(power, size):
    """Stationary periodic Gaussian field with power spectrum `power` (per-pixel normalisation:
    E|FFT(x)_k|^2 = Npix * power_k, so Var x = mean(power))."""
    white = rng.standard_normal((size, Npix))
    return np.fft.ifft(np.fft.fft(white, axis=-1) * np.sqrt(power), axis=-1).real


Mw = 4000
sig = draw(Ps, Mw)
dat = sig + np.sqrt(Pn) * rng.standard_normal((Mw, Npix))
rec = np.fft.ifft(np.fft.fft(dat, axis=-1) * wk, axis=-1).real
mse_raw = np.mean((dat - sig)**2)
mse_wf = np.mean((rec - sig)**2)
mse_wf_th = np.mean(Ps * Pn / (Ps + Pn))
var_sig, var_rec = np.mean(sig**2), np.mean(rec**2)
var_rec_th = np.mean(Ps**2 / (Ps + Pn))
assert abs(mse_wf / mse_wf_th - 1) < 0.02 and abs(mse_raw - Pn) < 0.02

# matrix forms agree (small periodic grid, so the matrices are cheap)
n8 = 32
kk = np.fft.fftfreq(n8, d=1.0 / n8)
P8 = 5.0 / (1.0 + (np.abs(kk) / 3.0)**4)
F = np.fft.fft(np.eye(n8)) / np.sqrt(n8)              # unitary DFT matrix
S = (F.conj().T @ np.diag(P8) @ F).real               # stationary signal covariance
Nm = 0.7 * np.eye(n8)
dv = rng.standard_normal(n8)
form1 = np.linalg.solve(np.linalg.inv(S) + np.linalg.inv(Nm), np.linalg.solve(Nm, dv))
form2 = S @ np.linalg.solve(S + Nm, dv)
form3 = np.fft.ifft(np.fft.fft(dv) * P8 / (P8 + 0.7)).real
assert np.allclose(form1, form2) and np.allclose(form2, form3)

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
xs = np.arange(Npix)
axs[0].plot(xs, dat[0], color=SERIES[0], lw=0.6, alpha=0.6, label="data $d=s+n$")
axs[0].plot(xs, sig[0], color="k", lw=1.4, label="true signal $s$")
axs[0].plot(xs, rec[0], color=SERIES[1], lw=1.4, label="Wiener filter $\\hat s$")
axs[0].set_xlim(0, 256); axs[0].set_xlabel("pixel"); axs[0].legend(fontsize=7, loc="upper right", ncol=3)
kp = np.arange(1, Npix // 2)
axs[1].loglog(kp, Ps[kp], color="k", label=r"$P_s$")
axs[1].axhline(Pn, color=SERIES[0], ls=":", label=r"$P_n$")
axs[1].loglog(kp, wk[kp], color=SERIES[1], label=r"weight $P_s/(P_s+P_n)$")
axs[1].set_xlabel(r"wavenumber $k$"); axs[1].set_ylim(1e-4, 50); axs[1].legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "chT5", "wiener")

kcross = float(ell0 * (20.0 / Pn - 1) ** 0.25)   # P_s = P_n there
save_numbers("chT5", "03_bayes_risk", {
    "TFiveABrM": Mg, "TFiveABrCrossTauOne": f"{np.sqrt(2 * tau_mc**2 + s**2):.3f}",
    "TFiveABinN": nb, "TFiveABinMinimaxRisk": f"{1 / (4 * (1 + np.sqrt(nb))**2):.5f}",
    "TFiveABinMLEMax": f"{0.25 / nb:.5f}",
    "TFiveABinAlpha": f"{am:.3f}",
    "TFiveAWfNpix": Npix, "TFiveAWfM": Mw, "TFiveAWfKcross": f"{kcross:.1f}",
    "TFiveAWfMseRaw": f"{mse_raw:.4f}", "TFiveAWfMse": f"{mse_wf:.4f}", "TFiveAWfMseTh": f"{mse_wf_th:.4f}",
    "TFiveAWfVarSig": f"{var_sig:.3f}", "TFiveAWfVarRec": f"{var_rec:.3f}", "TFiveAWfVarRecTh": f"{var_rec_th:.3f}",
    "TFiveAWfVarSigTh": f"{np.mean(Ps):.3f}",
})
