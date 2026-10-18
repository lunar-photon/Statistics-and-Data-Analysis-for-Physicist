"""06_quadratic.py -- when the parameter lives in the covariance: quadratic estimators.

Question (a): one pixel x ~ N(0, p + sigma^2). Is x^2 biased by the noise variance sigma^2, and is
x^2 - sigma^2 unbiased with variance 2 (p + sigma^2)^2?
Question (b): two pixels, the first noiseless, the second with noise variance 3. Does the minimum-variance
quadratic estimator (weights proportional to 1/(p + sigma_i^2)^2) beat the plain average of the
two debiased squares, with the variances 1/F and 2 tr(E C E C) derived in the text?
Question (c): a periodic one-dimensional field of 64 pixels with a gap of 16 unobserved pixels,
signal power in four bands of wavenumber, white noise of variance 0.2 on pixels 0-23 and 4 elsewhere
(deep and shallow coverage). Compare the quadratic maximum-likelihood
estimator p_hat = F^{-1}(q - b) with a debiased pseudo-spectrum estimator (band-averaged power of
the gapped data, corrected by its own coupling matrix): both unbiased, but which scatters less, and
does the QML covariance equal F^{-1}?

Computes: Monte Carlo means and variances for (a)-(c) against the analytic formulas.
Writes:   figures/chT5/quadratic.pdf, results/chT5/06_quadratic.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("chT5", "06_quadratic")
setup()
M = 200_000

# ---------- (a) one pixel ----------
p1, s2 = 1.0, 0.5
x = np.sqrt(p1 + s2) * rng.standard_normal(M)
naive, deb = x**2, x**2 - s2
a_mean_naive, a_mean, a_var = naive.mean(), deb.mean(), deb.var()
a_var_th = 2 * (p1 + s2)**2

# ---------- (b) two pixels, one noisy ----------
p2, sig = 1.0, np.array([0.0, 3.0])          # noise variances of the two pixels
lam = p2 + sig                                # pixel variances 1 and 4
X = np.sqrt(lam) * rng.standard_normal((M, 2))
E_avg = np.diag([0.5, 0.5])
E_qml = np.diag(1 / lam**2) / np.sum(1 / lam**2)       # tr(E C_p) = 1 with C_p = identity
def quad(E, X, Nd):
    return np.einsum("mi,ij,mj->m", X, E, X) - np.trace(E @ Nd)
Nd, Cd = np.diag(sig), np.diag(lam)
qa, qq = quad(E_avg, X, Nd), quad(E_qml, X, Nd)
var_avg_th = 2 * np.trace(E_avg @ Cd @ E_avg @ Cd)
F2 = 0.5 * np.sum(1 / lam**2)
var_qml_th = 1 / F2
assert abs(np.trace(E_qml @ Cd) - np.trace(E_qml @ Nd) - p2) < 1e-12

# ---------- (c) masked periodic field, four bands ----------
N = 64
obs = np.ones(N, bool); obs[40:56] = False              # a gap of 16 pixels
k = np.abs(np.fft.fftfreq(N, d=1.0 / N)).astype(int)    # |k| = 0..32
edges = [0, 4, 8, 16, 33]                               # band a holds |k| in [edges[a], edges[a+1])
p_true = np.array([8.0, 4.0, 1.5, 0.5])
noise_pix = np.where(np.arange(N) < 24, 0.2, 4.0)   # deep (0.2) and shallow (4.0) coverage
i = np.arange(N)
lag = i[:, None] - i[None, :]
Ca_full = []
for a in range(4):
    ks = np.arange(N)[(k >= edges[a]) & (k < edges[a + 1])]
    kk = np.fft.fftfreq(N, d=1.0 / N)[ks]
    Ca_full.append(np.cos(2 * np.pi * kk[None, None, :] * lag[:, :, None] / N).sum(2) / N)
Ca = [c[np.ix_(obs, obs)] for c in Ca_full]             # covariance templates of observed pixels
n_obs = obs.sum()
Nm = np.diag(noise_pix[obs])
C = sum(p * c for p, c in zip(p_true, Ca)) + Nm
Ci = np.linalg.inv(C)
F = np.array([[0.5 * np.trace(Ci @ Ca[a] @ Ci @ Ca[b]) for b in range(4)] for a in range(4)])
E_q = [0.5 * Ci @ Ca[a] @ Ci for a in range(4)]
b_q = np.array([np.trace(E @ Nm) for E in E_q])
# pseudo-spectrum: band-averaged |FFT|^2 of the zero-filled data = x^T E x with E a band projector
E_p = []
for a in range(4):
    sel = (k >= edges[a]) & (k < edges[a + 1])
    Fm = np.fft.fft(np.eye(N), axis=0)[sel]               # rows: Fourier modes of the band
    proj = np.real(Fm.conj().T @ Fm) / (N * sel.sum())    # average |d_k|^2 / N over the band
    E_p.append(proj[np.ix_(obs, obs)])
W_q = np.array([[np.trace(E_q[a] @ Ca[b]) for b in range(4)] for a in range(4)])    # = F
W_p = np.array([[np.trace(E_p[a] @ Ca[b]) for b in range(4)] for a in range(4)])
b_p = np.array([np.trace(E @ Nm) for E in E_p])
assert np.allclose(W_q, F)
cov_q_th = np.linalg.inv(F)
Wpi = np.linalg.inv(W_p)
cov_p_raw = np.array([[2 * np.trace(E_p[a] @ C @ E_p[b] @ C) for b in range(4)] for a in range(4)])
cov_p_th = Wpi @ cov_p_raw @ Wpi.T

L = np.linalg.cholesky(C)
Mc = 20_000
xs = rng.standard_normal((Mc, n_obs)) @ L.T
q_q = np.stack([np.einsum("mi,ij,mj->m", xs, E, xs) for E in E_q], 1) - b_q
q_p = np.stack([np.einsum("mi,ij,mj->m", xs, E, xs) for E in E_p], 1) - b_p
ph_q = q_q @ np.linalg.inv(F).T
ph_p = q_p @ Wpi.T
sd_q, sd_p = ph_q.std(0), ph_p.std(0)
assert np.allclose(ph_q.mean(0), p_true, atol=4 * sd_q / np.sqrt(Mc) + 1e-9)
assert np.allclose(sd_q, np.sqrt(np.diag(cov_q_th)), rtol=0.03)
assert np.allclose(sd_p, np.sqrt(np.diag(cov_p_th)), rtol=0.03)

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1))
bins = np.linspace(-1, 12, 131)
axs[0].hist(naive, bins=bins, density=True, histtype="step", color=SERIES[1], lw=1.4, label=r"$x^2$")
axs[0].hist(deb, bins=bins, density=True, histtype="step", color=SERIES[0], lw=1.4, label=r"$x^2-\sigma^2$")
axs[0].axvline(p1, color="k", ls="--", lw=1.2, label=r"true $p$")
axs[0].axvline(a_mean_naive, color=SERIES[1], ls=":", lw=1.2)
axs[0].set_yscale("log"); axs[0].set_xlabel("estimate of $p$ from one pixel"); axs[0].set_ylabel("density")
axs[0].legend(fontsize=7, loc="upper right")
xb = np.arange(4)
axs[1].bar(xb - 0.18, sd_p / p_true, width=0.34, color=SERIES[1], label="pseudo-spectrum (debiased)")
axs[1].bar(xb + 0.18, sd_q / p_true, width=0.34, color=SERIES[0], label="quadratic ML")
axs[1].plot(xb + 0.18, np.sqrt(np.diag(cov_q_th)) / p_true, "k_", ms=14, mew=2, label=r"$\sqrt{(F^{-1})_{aa}}$")
axs[1].set_xticks(xb, [f"$|k|\\in[{edges[a]},{edges[a+1]-1}]$" for a in range(4)], fontsize=7)
axs[1].set_ylabel("fractional error of band power"); axs[1].legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "chT5", "quadratic")

print("a", a_mean_naive, a_mean, a_var, a_var_th)
print("b", qa.mean(), qa.var(), var_avg_th, qq.mean(), qq.var(), var_qml_th)
print("c sd_q", sd_q, "sd_p", sd_p, "th", np.sqrt(np.diag(cov_q_th)), np.sqrt(np.diag(cov_p_th)))
save_numbers("chT5", "06_quadratic", {
    "TFiveAQM": M, "TFiveAQMc": Mc,
    "TFiveAQaNaive": f"{a_mean_naive:.3f}", "TFiveAQaMean": f"{a_mean:.3f}",
    "TFiveAQaVar": f"{a_var:.3f}", "TFiveAQaVarTh": f"{a_var_th:.3f}",
    "TFiveAQbAvgMean": f"{qa.mean():.3f}", "TFiveAQbAvgVar": f"{qa.var():.2f}", "TFiveAQbAvgVarTh": f"{var_avg_th:.2f}",
    "TFiveAQbQmlMean": f"{qq.mean():.3f}", "TFiveAQbQmlVar": f"{qq.var():.3f}", "TFiveAQbQmlVarTh": f"{var_qml_th:.3f}",
    "TFiveAQcN": N, "TFiveAQcNobs": int(n_obs), "TFiveAQcNoiseDeep": 0.2, "TFiveAQcNoiseShallow": 4.0,
    **{f"TFiveAQcSdQ{w}": f"{sd_q[a]:.3f}" for a, w in enumerate(["One", "Two", "Three", "Four"])},
    **{f"TFiveAQcSdP{w}": f"{sd_p[a]:.3f}" for a, w in enumerate(["One", "Two", "Three", "Four"])},
    **{f"TFiveAQcSdTh{w}": f"{np.sqrt(cov_q_th[a, a]):.3f}" for a, w in enumerate(["One", "Two", "Three", "Four"])},
    **{f"TFiveAQcVarRatio{w}": f"{(sd_p[a] / sd_q[a])**2:.2f}" for a, w in enumerate(["One", "Two", "Three", "Four"])},
    "TFiveAQcCorrQ": f"{cov_q_th[2, 3] / np.sqrt(cov_q_th[2, 2] * cov_q_th[3, 3]):.2f}",
})
