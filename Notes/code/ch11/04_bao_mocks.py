"""04_bao_mocks.py -- the acoustic peak in mock surveys the size of the first detection.

Question: in a survey with the volume and galaxy density of the SDSS luminous-red-galaxy sample
(Eisenstein et al. 2005: 0.72 (Gpc/h)^3, 46,748 galaxies), how strongly is the acoustic peak in
xi(r) detected, i.e. how much worse does a template without the peak fit (Delta chi^2), and how
well is the dilation alpha measured?
Computes: lognormal galaxy fields in a periodic box of side 900 Mpc/h (96^3 cells), with spectrum
b^2 D(z)^2 (1 + 2 beta/3 + beta^2/5) P_t(k) (the angle-averaged redshift-space amplitude), where P_t is the damped acoustic template (Sigma_nl = 8 Mpc/h, before
reconstruction); Poisson galaxies; xi(r) in 16 bins of 10 Mpc/h from 20 to 180 Mpc/h by FFT pair
counting on the grid. NCOV mocks give the covariance matrix; NFIT further mocks are each fitted
with xi_model(r) = B^2 xi_t(alpha r) + a0 + a1/r + a2/r^2 for the template with the peak and for
the template without it (Eisenstein-Hu no-wiggle), on a grid of alpha; and again with the\namplitude B^2 as the only free coefficient, as Eisenstein et al. did.
Writes: figures/ch11/bao_mock_fit.pdf, figures/ch11/bao_mock_detect.pdf,
results/ch11/04_bao_mocks.tex, data/ch11/bao_mocks.npz
"""
import pathlib
import sys
import os
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA
from lib_lss import (linear_pk_z0, nowiggle_pk, growth, xi_box_from_pk, Lognormal,
                     poisson_sample, RadialBins, xi_grid_fft)

L, N = 900.0, 96                         # box side [Mpc/h], cells per side
Z_EFF, B, SIG_NL = 0.35, 2.0, 8.0        # effective redshift, galaxy bias, damping [Mpc/h]
NBAR = 46748 / 0.72e9                    # Eisenstein's galaxy count over his survey volume
NCOV, NFIT = int(os.environ.get("NCOV", 1500)), int(os.environ.get("NFIT", 500))
EDGES = np.arange(20.0, 181.0, 10.0)
ALPHAS = np.round(np.arange(0.80, 1.2001, 0.004), 4)
rng = rng_for("ch11", "04_bao_mocks")

# ---- the two templates: with the acoustic feature (damped) and without it
k_tab, pk_tab, sigma8, rdrag = linear_pk_z0()
D, f = growth(Z_EFF)
beta = f[0] / B
kaiser0 = 1 + 2 * beta / 3 + beta ** 2 / 5     # monopole of (1 + beta mu^2)^2, derived in the RSD section
amp = (B * D[0]) ** 2 * kaiser0                 # spherically averaged redshift-space amplitude
kk = np.geomspace(1e-4, 20, 6000)
p_nw = nowiggle_pk(kk, k_tab, pk_tab)
p_lin = np.interp(np.log(kk), np.log(k_tab), pk_tab)
p_w = (p_lin - p_nw) * np.exp(-0.5 * (kk * SIG_NL) ** 2) + p_nw
interp = lambda tab: (lambda k: amp * np.exp(np.interp(np.log(k), np.log(kk), np.log(tab))))
P_w, P_nw = interp(p_w), interp(p_nw)

rb = RadialBins(N, L, 3, EDGES)
r = rb.rmean


def template_bins(P, alpha):
    """Bin-averaged grid two-point function of the spectrum P(k/alpha)/alpha^3, i.e. xi(alpha r)."""
    return rb(xi_box_from_pk(lambda k: P(k / alpha) / alpha ** 3, N, L, 3))


t0 = time.time()
T_w = np.array([template_bins(P_w, a) for a in ALPHAS])     # (n_alpha, n_bins)
T_nw = np.array([template_bins(P_nw, a) for a in ALPHAS])
print(f"templates: {time.time()-t0:.0f} s")

# ---- mocks: lognormal galaxy field with the template spectrum, Poisson galaxies, xi by FFT
ln = Lognormal(xi_box_from_pk(P_w, N, L, 3), N, L, 3)
xis, ngal = [], []
for s in range(NCOV + NFIT):
    delta = ln.draw(rng)
    c = poisson_sample(1 + delta, NBAR, L, rng)
    dn = c / c.mean() - 1                      # density contrast of the galaxies, mean from the box
    xis.append(rb(xi_grid_fft(dn)))
    ngal.append(c.sum())
xis = np.array(xis)
print(f"mocks: {time.time()-t0:.0f} s")
xc, xf = xis[:NCOV], xis[NCOV:]
C = np.cov(xc, rowvar=False)
nb = len(r)
hartlap = (NCOV - nb - 2) / (NCOV - 1)
Ci = hartlap * np.linalg.inv(C)


def design(t, poly=True):
    """Columns of the linear part of the model: B^2 times the template, then a0, a1/r, a2/r^2."""
    return np.column_stack([t, np.ones(nb), 1 / r, 1 / r ** 2]) if poly else t[:, None]


def fit(d, T, poly=True):
    """chi^2(alpha) with the linear coefficients solved by generalised least squares at each alpha."""
    chi2 = np.empty(len(ALPHAS))
    for i, t in enumerate(T):
        X = design(t, poly)
        A = X.T @ Ci @ X
        beta = np.linalg.solve(A, X.T @ Ci @ d)
        res = d - X @ beta
        chi2[i] = res @ Ci @ res
    i = int(np.argmin(chi2))
    # parabola through the three points around the minimum, for a sub-grid alpha
    if 0 < i < len(ALPHAS) - 1:
        y0, y1, y2 = chi2[i - 1], chi2[i], chi2[i + 1]
        h = ALPHAS[1] - ALPHAS[0]
        den = y0 - 2 * y1 + y2
        da = 0.5 * h * (y0 - y2) / den if den > 0 else 0.0
        a_hat, c_min = ALPHAS[i] + da, y1 - 0.25 * (y0 - y2) * da / h
    else:
        a_hat, c_min = ALPHAS[i], chi2[i]
    p = np.exp(-0.5 * (chi2 - chi2.min()))
    p /= p.sum()
    sig = np.sqrt(np.sum(p * ALPHAS ** 2) - np.sum(p * ALPHAS) ** 2)     # Anderson et al. eq. (29)
    return a_hat, c_min, sig, chi2


res_w = [fit(d, T_w) for d in xf]
res_nw = [fit(d, T_nw) for d in xf]
# Eisenstein et al. (2005) style: amplitude only, no broad-band polynomial
eis_w = [fit(d, T_w, poly=False) for d in xf]
eis_nw = [fit(d, T_nw, poly=False) for d in xf]
dchi2_e = np.array([q[1] for q in eis_nw]) - np.array([q[1] for q in eis_w])
signif_e = np.sqrt(np.clip(dchi2_e, 0, None))
a_e = np.array([q[0] for q in eis_w])
a_w = np.array([q[0] for q in res_w]); c_w = np.array([q[1] for q in res_w]); s_w = np.array([q[2] for q in res_w])
a_nw = np.array([q[0] for q in res_nw]); c_nw = np.array([q[1] for q in res_nw]); s_nw = np.array([q[2] for q in res_nw])
dchi2 = c_nw - c_w
signif = np.sqrt(np.clip(dchi2, 0, None))
dof = nb - 5
np.savez(DATA / "ch11" / "bao_mocks.npz", a_w=a_w, a_nw=a_nw, c_w=c_w, c_nw=c_nw, s_w=s_w, s_nw=s_nw,
         dchi2=dchi2, xis=xis, r=r, C=C, L=L, nbar=NBAR, b=B, z=Z_EFF, signl=SIG_NL)

# ---- figure 1: one mock survey (the one with the median Delta chi^2) and its chi^2(alpha) curves
j = int(np.argsort(dchi2)[len(dchi2) // 2])
d = xf[j]
err = np.sqrt(np.diag(C))
setup(6.6, 3.0)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
rr = np.linspace(r[0], r[-1], 300)
for T, chi, col, lab in [(T_w, res_w[j][3], SERIES[0], "with the peak"), (T_nw, res_nw[j][3], SERIES[1], "without the peak")]:
    i = int(np.argmin(chi))
    X = design(T[i])
    coef = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ d)
    axs[0].plot(r, r ** 2 * (X @ coef), color=col, lw=1.4, label=lab)
    axs[1].plot(ALPHAS, chi, color=col, lw=1.4, label=lab)
axs[0].errorbar(r, r ** 2 * d, r ** 2 * err, fmt="o", ms=3, color="k", lw=0.8, label="one mock survey")
axs[0].plot(r, r ** 2 * xc.mean(0), color="0.5", ls=":", lw=1.2, label="mean of the mocks")
axs[0].set_xlabel(r"$r\ [h^{-1}\mathrm{Mpc}]$")
axs[0].set_ylabel(r"$r^2\xi(r)\ [h^{-2}\mathrm{Mpc}^2]$")
axs[0].legend(fontsize=7, loc="lower left")
axs[1].set_xlabel(r"dilation $\alpha$")
axs[1].set_ylabel(r"$\chi^2(\alpha)$")
axs[1].set_ylim(res_w[j][3].min() - 2, res_w[j][3].min() + 22)
axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch11", "bao_mock_fit")

# ---- figure 2: the distribution of the detection significance and of alpha-hat
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
axs[0].hist(signif, bins=np.arange(0, 7.01, 0.4), color=SERIES[0], alpha=0.8, label="with polynomial")
axs[0].hist(signif_e, bins=np.arange(0, 7.01, 0.4), histtype="step", color=SERIES[2], lw=1.4, label="amplitude only")
axs[0].axvline(np.sqrt(11.7), color=SERIES[1], lw=1.6, label="Eisenstein et al. 2005")
axs[0].axvline(np.median(signif), color="k", ls="--", lw=1.2, label="median of the mocks")
axs[0].set_xlabel(r"$\sqrt{\Delta\chi^2}$ (no peak $-$ peak)")
axs[0].set_ylabel("mock surveys")
axs[0].legend(fontsize=7)
bins = np.arange(0.8, 1.2001, 0.01)
axs[1].hist(a_w, bins=bins, color=SERIES[0], alpha=0.8, label="template with the peak")
axs[1].hist(a_nw, bins=bins, histtype="step", color=SERIES[1], lw=1.4, label="template without")
axs[1].set_xlabel(r"best-fit $\hat\alpha$ (truth $\alpha=1$)")
axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch11", "bao_mock_detect")

frac = float(np.mean(dchi2 >= 11.7))
save_numbers("ch11", "04_bao_mocks", {
    "BaoL": f"{L:g}", "BaoN": N, "BaoNcov": NCOV, "BaoNfit": NFIT, "BaoZ": f"{Z_EFF:g}", "BaoB": f"{B:g}",
    "BaoNbar": f"{NBAR*1e5:.2f}", "BaoNgal": f"{np.mean(ngal):.0f}", "BaoVol": f"{L**3/1e9:.2f}",
    "BaoD": f"{D[0]:.3f}", "BaoSigNl": f"{SIG_NL:g}", "BaoHartlap": f"{hartlap:.3f}", "BaoNbins": nb, "BaoDof": dof,
    "BaoMedSig": f"{np.median(signif):.1f}", "BaoFracEis": f"{100*frac:.0f}",
    "BaoSigLo": f"{np.percentile(signif, 16):.1f}", "BaoSigHi": f"{np.percentile(signif, 84):.1f}",
    "BaoMeanAlpha": f"{a_w.mean():.4f}", "BaoErrMeanAlpha": f"{a_w.std(ddof=1)/np.sqrt(NFIT):.4f}",
    "BaoSdAlpha": f"{100*a_w.std(ddof=1):.1f}", "BaoMedSigAlpha": f"{100*np.median(s_w):.1f}",
    "BaoSdAlphaNw": f"{100*a_nw.std(ddof=1):.1f}", "BaoMedSigAlphaNw": f"{100*np.median(s_nw):.1f}",
    "BaoMeanChiW": f"{c_w.mean():.1f}", "BaoMeanChiNw": f"{c_nw.mean():.1f}",
    "BaoExDchi": f"{dchi2[j]:.1f}", "BaoExAlpha": f"{a_w[j]:.3f}", "BaoExChiW": f"{c_w[j]:.1f}",
    "BaoMedSigE": f"{np.median(signif_e):.1f}", "BaoFracEisE": f"{100*np.mean(dchi2_e >= 11.7):.0f}",
    "BaoSdAlphaE": f"{100*a_e.std(ddof=1):.1f}", "BaoKaiserZero": f"{kaiser0:.3f}", "BaoBeta": f"{beta:.3f}",
    "BaoFz": f"{f[0]:.3f}", "BaoSigEightG": f"{B*D[0]*sigma8*np.sqrt(kaiser0):.2f}",
    "BaoClip": f"{100*ln.clipped:.2f}", "BaoGauVar": f"{ln.sigma2:.2f}",
})
print(f"total {time.time()-t0:.0f} s; median sqrt(dchi2) {np.median(signif):.2f}; sd alpha {a_w.std():.4f}")
