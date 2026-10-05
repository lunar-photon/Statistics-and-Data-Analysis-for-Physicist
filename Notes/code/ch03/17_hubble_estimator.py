"""17_hubble_estimator.py -- which function of supernova data estimates the Hubble constant?

Question: given redshifts z_i and apparent magnitudes m_i of type Ia supernovae, which estimator of
H0 is unbiased and has the smallest scatter?  Three candidates are compared on mock surveys drawn
from a known H0: (a) convert each magnitude to a distance and fit v = H0 d through the origin,
(b) average the per-object ratios v_i/d_i, (c) the maximum-likelihood estimator, an
inverse-variance weighted mean in the variable a = 5 log10 H0 where the noise is Gaussian.
Also: the degeneracy between H0 and the absolute magnitude M, and what an external calibration
of M does to the error bar; the flat-LambdaCDM luminosity distance (cross-checked with CAMB),
the profile over Omega_m at high redshift, and the bias of the Hubble-law estimator vs z_max.

Computes: 20000 low-z mock surveys (100 SNe, 0.01 < z < 0.1) with the three estimators, their
          bias, scatter and the coverage of the ML error bar (with M known and with M calibrated);
          likelihood contours in the (M, H0) plane for one survey; 1000 mock surveys of 300 SNe up
          to z = 1 fitted for (H0, Omega_m); the bias of the Hubble-law and q0 estimators vs z_max.
Writes:   figures/ch03/hubble_mock.pdf, hubble_estimators.pdf, hubble_degeneracy.pdf,
          hubble_highz.pdf; results/ch03/17_hubble_estimator.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import cumulative_trapezoid

rng = rng_for("ch03", "17_hubble_estimator")
setup()
out = {}

C = 299792.458                       # speed of light, km/s
H0, M_TRUE = 70.0, -19.25            # truth of the mock universe: km/s/Mpc, mag
SIG_INT, SIG_V = 0.15, 300.0         # magnitude scatter (intrinsic + measurement), peculiar velocity km/s
SIG_M = 0.03                         # uncertainty of the external calibration of M (mag)
KAPPA = np.log(10) / 5               # d ln(d) / d(mag): a distance modulus error delta gives d -> d e^{kappa delta}
N, NMOCK = 100, 20000

# ---------- the model: where the noise lives ----------
def sigma_mag(z):
    """Total magnitude scatter: intrinsic plus the peculiar velocity propagated into magnitudes."""
    return np.sqrt(SIG_INT**2 + (5 / np.log(10) * SIG_V / (C * z))**2)

def mock_lowz(shape):
    """A universe where the Hubble law is exact: cz = H0 d + v_pec, m = M + 5 log10(d/10pc) + eps."""
    z = rng.uniform(0.01, 0.1, shape)
    d = (C * z - rng.normal(0, SIG_V, shape)) / H0                 # true distance, Mpc
    m = M_TRUE + 25 + 5 * np.log10(d) + rng.normal(0, SIG_INT, shape)
    return z, m

# ---------- the three candidate estimators ----------
def naive(z, m, M=M_TRUE):
    d = 10 ** ((m - M - 25) / 5)                                  # distance "measured" from each magnitude
    v = C * z
    return (v * d).sum(-1) / (d * d).sum(-1)                      # least squares of v = H0 d through the origin

def ratio(z, m, M=M_TRUE):
    d = 10 ** ((m - M - 25) / 5)
    return (C * z / d).mean(-1)

def ml(z, m, M=M_TRUE):
    w = 1 / sigma_mag(z)**2
    y = M + 25 + 5 * np.log10(C * z) - m                          # each SN's own estimate of a = 5 log10 H0
    a = (w * y).sum(-1) / w.sum(-1)                               # inverse-variance weighted mean
    sa = 1 / np.sqrt(w.sum(-1))                                   # 1/sqrt(Fisher information)
    return 10 ** (a / 5), a, sa

z, m = mock_lowz((NMOCK, N))
Hn, Hr = naive(z, m), ratio(z, m)
Hml, a, sa = ml(z, m)
aTrue = 5 * np.log10(H0)
for name, H in (("Naive", Hn), ("Ratio", Hr), ("ML", Hml)):
    out[f"H{name}Mean"], out[f"H{name}Bias"], out[f"H{name}Sd"] = H.mean(), H.mean() - H0, H.std()
    out[f"H{name}Rmse"] = np.sqrt(np.mean((H - H0)**2))
# large-N limits: log-normal distance moments, plus the peculiar-velocity term x = v_pec/cz
cz2 = (C * z[:2000])**2
out["HNaiveBiasLead"] = H0 * (np.exp(-1.5 * (KAPPA * SIG_INT)**2) - 1)
out["HNaiveBiasTh"] = np.mean(H0 * np.exp(-1.5 * (KAPPA * SIG_INT)**2) * cz2.sum(-1) / (cz2 + SIG_V**2).sum(-1)) - H0
out["HRatioBiasLead"] = H0 * (np.exp(0.5 * (KAPPA * SIG_INT)**2) - 1)
out["HRatioBiasTh"] = np.mean(H0 * np.exp(0.5 * (KAPPA * SIG_INT)**2) * (1 + SIG_V**2 / cz2).mean(-1)) - H0
out["HaBias"] = a.mean() - aTrue
out["HaSd"], out["HaSdTh"] = a.std(), sa.mean()
out["HMLSdTh"] = KAPPA * H0 * sa.mean()
out["HMLBiasTh"] = H0 * (np.exp(0.5 * (KAPPA * sa.mean())**2) - 1)       # convexity of 10^{a/5}
out["HMLBiasCorr"] = (Hml * np.exp(-0.5 * (KAPPA * sa)**2)).mean() - H0
out["HMLMedianBias"] = np.median(Hml) - H0
out["HCovOne"] = np.mean(np.abs(a - aTrue) < sa)
out["HCovTwo"] = np.mean(np.abs(a - aTrue) < 1.96 * sa)
# residual bias of a from linearising -5 log10(1 - v/cz): E ~ (5/ln10) <sigma_v^2/(cz)^2>/2, weighted
w0 = 1 / sigma_mag(z[:2000])**2
out["HaBiasLin"] = np.mean((w0 * 5 / np.log(10) * 0.5 * (SIG_V / (C * z[:2000]))**2).sum(-1) / w0.sum(-1))
out["HmagLowZ"] = 5 / np.log(10) * SIG_V / (C * 0.01)
out["HmagHighZ"] = 5 / np.log(10) * SIG_V / (C * 0.1)
# repeat for N = 400: the naive and ratio biases do not shrink (inconsistent estimators)
z4, m4 = mock_lowz((5000, 4 * N))
out["HNaiveBiasFour"] = naive(z4, m4).mean() - H0
out["HRatioBiasFour"] = ratio(z4, m4).mean() - H0
out["HMLBiasFour"] = ml(z4, m4)[0].mean() - H0
out["HMLSdFour"] = ml(z4, m4)[0].std()

# ---------- calibrated M: M_cal ~ N(M_true, SIG_M) per survey ----------
Mcal = M_TRUE + rng.normal(0, SIG_M, NMOCK)
Hc, ac, _ = ml(z, m, Mcal[:, None])
sac = np.sqrt(sa**2 + SIG_M**2)
out["HCalSd"], out["HCalSdTh"] = Hc.std(), KAPPA * H0 * sac.mean()
out["HCalCovOne"] = np.mean(np.abs(ac - aTrue) < sac)
out["HCalCovNaive"] = np.mean(np.abs(ac - aTrue) < sa)                    # forgetting sigma_M
out["HsigM"] = SIG_M

# ---------- one survey: the (M, H0) likelihood ----------
z1, m1 = z[0], m[0]
w1 = 1 / sigma_mag(z1)**2
Mg = np.linspace(-19.55, -18.95, 301)
Hg = np.linspace(58, 84, 301)
MM, HH = np.meshgrid(Mg, Hg)
pred = MM[..., None] + 25 + 5 * np.log10(C * z1) - 5 * np.log10(HH[..., None])
chi2 = ((m1 - pred)**2 * w1).sum(-1)
chi2_prior = chi2 + ((MM - M_TRUE) / SIG_M)**2
out["HOneML"] = ml(z1, m1)[0]
out["HOneSd"] = KAPPA * out["HOneML"] * ml(z1, m1)[2]
# Fisher matrix in (M, a): singular without the calibration
W = w1.sum()
F = np.array([[W, -W], [-W, W]])
out["HFishDet"] = np.linalg.det(F)
Fp = F + np.array([[1 / SIG_M**2, 0], [0, 0]])
out["HsaCal"] = np.sqrt(np.linalg.inv(Fp)[1, 1])
out["HsaStat"] = 1 / np.sqrt(W)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
dchi = chi2 - chi2.min()
ax[0].contour(MM, HH, dchi, levels=[2.30, 6.18], colors=[SERIES[0]], linestyles=["-", "--"])
ax[0].axvspan(M_TRUE - SIG_M, M_TRUE + SIG_M, color=SERIES[1], alpha=.25, lw=0)
ax[0].contour(MM, HH, chi2_prior - chi2_prior.min(), levels=[2.30, 6.18], colors=[SERIES[2]],
              linestyles=["-", "--"])
ax[0].plot([M_TRUE], [H0], "k+", ms=8)
ax[0].set_xlabel("absolute magnitude $M$"); ax[0].set_ylabel(r"$H_0$ [km/s/Mpc]")
ax[0].text(-19.53, 81, "SNe alone", color=SERIES[0], fontsize=8)
ax[0].text(-19.20, 60, "calibration\nof $M$", color=SERIES[1], fontsize=8)
ax[0].text(-19.30, 74.5, "SNe + calibration", color=SERIES[2], fontsize=8, ha="right")
Hline = np.linspace(58, 84, 400)
for Mv, c in zip((-19.35, -19.25, -19.15), SERIES[3:6]):
    pr = Mv + 25 + 5 * np.log10(C * z1)[None, :] - 5 * np.log10(Hline)[:, None]
    ch = ((m1 - pr)**2 * w1).sum(-1)
    ax[1].plot(Hline, ch - chi2.min(), color=c, label=f"$M={Mv}$")
ax[1].set_ylim(0, 25); ax[1].set_xlabel(r"$H_0$ [km/s/Mpc]"); ax[1].set_ylabel(r"$\chi^2-\chi^2_{\min}$")
ax[1].legend(fontsize=7)
fig.tight_layout(); savefig(fig, "ch03", "hubble_degeneracy")

# ---------- figure: one mock survey ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
d1 = 10 ** ((m1 - M_TRUE - 25) / 5)
ax[0].plot(d1, C * z1, ".", color=SERIES[0], ms=3, label="one survey")
dd = np.linspace(0, d1.max() * 1.05, 10)
theory_line(ax[0], dd, H0 * dd, label="$v=H_0d$ (truth)")
ax[0].plot(dd, naive(z1, m1) * dd, color=SERIES[1], lw=1, label="least-squares line")
ax[0].set_xlabel(r"distance from magnitude $d_i$ [Mpc]"); ax[0].set_ylabel(r"$cz_i$ [km/s]")
ax[0].legend(fontsize=7)
res = m1 - (M_TRUE + 25 + 5 * np.log10(C * z1 / H0))
zz = np.linspace(0.01, 0.1, 200)
ax[1].plot(z1, res, ".", color=SERIES[0], ms=3)
ax[1].fill_between(zz, -sigma_mag(zz), sigma_mag(zz), color=SERIES[0], alpha=.15, lw=0)
theory_line(ax[1], zz, 0 * zz, label=r"model, $\pm\sigma_i$ band")
ax[1].set_xlabel("redshift $z$"); ax[1].set_ylabel(r"$m_i-$ model [mag]")
ax[1].legend(fontsize=7)
fig.tight_layout(); savefig(fig, "ch03", "hubble_mock")

# ---------- figure: sampling distributions ----------
fig, ax = plt.subplots(figsize=(6.0, 2.9))
bins = np.linspace(64, 76, 90)
for H, lab, c in ((Hn, "naive least squares", SERIES[1]), (Hr, "mean of ratios", SERIES[2]),
                  (Hml, "maximum likelihood", SERIES[0])):
    ax.hist(H, bins=bins, density=True, histtype="step", color=c, lw=1.4, label=lab)
    ax.axvline(H.mean(), color=c, lw=0.8, ls=":")
ax.axvline(H0, color="k", ls="--", lw=1.0, label="truth")
ax.set_xlabel(r"$\hat H_0$ [km/s/Mpc]"); ax.set_ylabel("density"); ax.legend(fontsize=7)
fig.tight_layout(); savefig(fig, "ch03", "hubble_estimators")

# ---------- beyond low z: flat LambdaCDM luminosity distance ----------
ZG = np.linspace(0, 1.0, 4001)
def dl_bar(zv, Om):
    """H0 D_L / c for flat LambdaCDM (radiation neglected): (1+z) int_0^z dz'/E(z')."""
    E = np.sqrt(Om * (1 + ZG)**3 + 1 - Om)
    chi = cumulative_trapezoid(1 / E, ZG, initial=0)
    return (1 + zv) * np.interp(zv, ZG, chi)

# cross-check with CAMB at the fiducial Planck parameters
import camb
fid = np.load(pathlib.Path(__file__).resolve().parents[2] / "data" / "camb_fiducial.npz")
pars = camb.CAMBparams()
pars.set_cosmology(H0=float(fid["param_H0"]), ombh2=float(fid["param_ombh2"]),
                   omch2=float(fid["param_omch2"]), mnu=float(fid["param_mnu"]))
bg = camb.get_background(pars)
zc = np.linspace(0.01, 1.0, 100)
dl_camb = bg.luminosity_distance(zc)
dl_mine = C / float(fid["param_H0"]) * dl_bar(zc, pars.omegam)
out["HCambDiff"] = np.max(np.abs(dl_mine / dl_camb - 1))
out["HCambOm"] = pars.omegam

OM = 0.3
NH, NMH = 300, 1000
zh = rng.uniform(0.01, 1.0, (NMH, NH))
vp = rng.normal(0, SIG_V, (NMH, NH))
# peculiar velocity enters as a Doppler shift of the observed redshift; its effect on D_L is the low-z term
mh = (M_TRUE + 25 + 5 * np.log10(C / H0 * dl_bar(zh, OM)) - 5 / np.log(10) * vp / (C * zh)
      + rng.normal(0, SIG_INT, (NMH, NH)))

def profile_chi2(Om, zv, mv, M=M_TRUE):
    """chi^2 minimised analytically over a = 5 log10 H0 at fixed Omega_m; returns (chi2_p, a_hat)."""
    w = 1 / sigma_mag(zv)**2
    y = M + 25 + 5 * np.log10(C * dl_bar(zv, Om)) - mv
    ah = (w * y).sum() / w.sum()
    return (w * (y - ah)**2).sum(), ah

Hfit, Omfit = np.empty(NMH), np.empty(NMH)
for k in range(NMH):
    r = optimize.minimize_scalar(lambda o: profile_chi2(o, zh[k], mh[k])[0], bounds=(0.02, 0.9),
                                 method="bounded", options={"xatol": 1e-5})
    Omfit[k] = r.x
    Hfit[k] = 10 ** (profile_chi2(r.x, zh[k], mh[k])[1] / 5)
out["HhzMean"], out["HhzSd"] = Hfit.mean(), Hfit.std()
out["OmhzMean"], out["OmhzSd"] = Omfit.mean(), Omfit.std()
# Fisher matrix in (a, Omega_m) for one survey's redshifts, averaged over surveys
def fisher_hz(zv, Om=OM, eps=1e-4):
    w = 1 / sigma_mag(zv)**2
    g = 5 * (np.log10(dl_bar(zv, Om + eps)) - np.log10(dl_bar(zv, Om - eps))) / (2 * eps)
    F = np.array([[w.sum(), -(w * g).sum()], [-(w * g).sum(), (w * g * g).sum()]])
    return np.linalg.inv(F), 1 / np.sqrt(w.sum())
Vs = [fisher_hz(zh[k]) for k in range(200)]
out["HhzSdTh"] = KAPPA * H0 * np.mean([np.sqrt(V[0][0, 0]) for V in Vs])
out["HhzSdFixed"] = KAPPA * H0 * np.mean([V[1] for V in Vs])
out["OmhzSdTh"] = np.mean([np.sqrt(V[0][1, 1]) for V in Vs])
out["HhzRho"] = np.mean([V[0][0, 1] / np.sqrt(V[0][0, 0] * V[0][1, 1]) for V in Vs])
# cross-check on survey 0: direct 2-D minimisation of chi^2(H0, Omega_m) agrees with the profile
def chi2_full(p, zv, mv):
    Hv, Om = p
    return ((mv - (M_TRUE + 25 + 5 * np.log10(C / Hv * dl_bar(zv, Om))))**2 / sigma_mag(zv)**2).sum()
r2 = optimize.minimize(chi2_full, x0=[65, 0.4], args=(zh[0], mh[0]), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8})
out["HhzTwoD"], out["HhzProf"] = r2.x[0], Hfit[0]
out["OmhzTwoD"], out["OmhzProf"] = r2.x[1], Omfit[0]
# one survey's profile over Omega_m (for the figure)
Omg = np.linspace(0.18, 0.42, 181)
chp = np.array([profile_chi2(o, zh[0], mh[0])[0] for o in Omg])

# the Hubble-law and q0 estimators applied to a LambdaCDM universe: bias vs z_max (noiseless limit)
q0 = 1.5 * OM - 1
zmaxs = np.linspace(0.012, 1.0, 248)
bias_h, bias_q, sig_h = [], [], []
for zm in zmaxs:
    zz = np.linspace(0.01, zm, 2000)
    w = 1 / sigma_mag(zz)**2
    mtrue = M_TRUE + 25 + 5 * np.log10(C / H0 * dl_bar(zz, OM))
    ah = (w * (M_TRUE + 25 + 5 * np.log10(C * zz) - mtrue)).sum() / w.sum()
    aq = (w * (M_TRUE + 25 + 5 * np.log10(C * zz) + 5 / np.log(10) * (1 - q0) * zz / 2 - mtrue)).sum() / w.sum()
    bias_h.append(10 ** (ah / 5) - H0); bias_q.append(10 ** (aq / 5) - H0)
    sig_h.append(KAPPA * H0 / np.sqrt(w.mean() * N))           # ML scatter for N = 100 SNe up to z_max
bias_h, bias_q, sig_h = map(np.array, (bias_h, bias_q, sig_h))
out["HqZero"] = q0
out["HlawBiasTen"] = np.interp(0.1, zmaxs, bias_h)
out["HlawBiasFive"] = np.interp(0.05, zmaxs, bias_h)
out["HqBiasTen"] = np.interp(0.1, zmaxs, bias_q)
out["HqBiasFiveH"] = np.interp(0.5, zmaxs, bias_q)
out["HlawSigTen"] = np.interp(0.1, zmaxs, sig_h)
# where the Hubble-law bias first exceeds the statistical error of 100 SNe
out["HlawZcross"] = zmaxs[np.argmax(np.abs(bias_h) > sig_h)]
out["HqZcross"] = zmaxs[np.argmax(np.abs(bias_q) > sig_h)]
# Monte Carlo check of the bias curve at z_max = 0.1 and 0.3
for zm, tag in ((0.1, "Ten"), (0.3, "Thirty")):
    zz = rng.uniform(0.01, zm, (2000, N))
    mm = (M_TRUE + 25 + 5 * np.log10(C / H0 * dl_bar(zz, OM))
          - 5 / np.log(10) * rng.normal(0, SIG_V, zz.shape) / (C * zz) + rng.normal(0, SIG_INT, zz.shape))
    out[f"HlawBiasMC{tag}"] = ml(zz, mm)[0].mean() - H0
out["HlawBiasThirty"] = np.interp(0.3, zmaxs, bias_h)

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
zs = np.linspace(0.01, 1.0, 300)
res = mh[0] - (M_TRUE + 25 + 5 * np.log10(C * zh[0] / H0))
ax[0].plot(zh[0], res, ".", color=SERIES[0], ms=2, alpha=.7)
ax[0].plot(zs, 5 * np.log10(dl_bar(zs, OM) / zs), color=SERIES[1], label=r"$\Lambda$CDM, $\Omega_{m0}=0.3$")
ax[0].plot(zs, 5 / np.log(10) * (1 - q0) * zs / 2, color=SERIES[2], lw=1, label=r"$q_0$ term")
theory_line(ax[0], zs, 0 * zs, label="Hubble law")
ax[0].set_xlabel("$z$"); ax[0].set_ylabel(r"$m-m_{\rm Hubble\ law}$ [mag]"); ax[0].legend(fontsize=6)
ax[1].plot(Omg, chp - chp.min(), color=SERIES[0])
ax[1].axhline(1, color="k", lw=.6, ls=":")
ax[1].axvline(OM, color="k", lw=.8, ls="--")
ax[1].set_ylim(0, 8); ax[1].set_xlabel(r"$\Omega_{m0}$"); ax[1].set_ylabel(r"$\chi^2_p-\chi^2_{p,\min}$")
ax[2].fill_between(zmaxs, -sig_h, sig_h, color=SERIES[0], alpha=.15, lw=0, label=r"$\pm\sigma$, 100 SNe")
ax[2].plot(zmaxs, bias_h, color=SERIES[1], label="Hubble law")
ax[2].plot(zmaxs, bias_q, color=SERIES[2], label=r"with $q_0$ term")
ax[2].plot([0.1, 0.3], [out["HlawBiasMCTen"], out["HlawBiasMCThirty"]], "o", color=SERIES[1], ms=4,
           mfc="none", label="Monte Carlo")
ax[2].axhline(0, color="k", lw=.6)
ax[2].set_xlabel(r"$z_{\max}$"); ax[2].set_ylabel(r"bias of $\hat H_0$ [km/s/Mpc]"); ax[2].legend(fontsize=6)
ax[2].set_ylim(-12, 3); ax[2].set_xlim(0, 0.7)
fig.tight_layout(); savefig(fig, "ch03", "hubble_highz")

out.update({"HTrue": H0, "HMtrue": M_TRUE, "HsigInt": SIG_INT, "HsigV": SIG_V, "HN": N, "HNmock": NMOCK,
            "HNhz": NH, "HNmockhz": NMH, "HKappaSigSq": (KAPPA * SIG_INT)**2})
save_numbers("ch03", "17_hubble_estimator", {"ThreeB" + k: v for k, v in out.items()})
print({k: round(float(v), 5) for k, v in out.items()})
