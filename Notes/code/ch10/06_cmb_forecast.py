"""06_cmb_forecast.py -- a Fisher forecast of the six LCDM parameters from the CMB, against Planck 2018.

Question: before Planck flew, how well could one have predicted its errors on (omega_b, omega_c,
100 theta_MC, tau, ln 10^10 A_s, n_s)?  We build the Fisher matrix of a Planck-like experiment from
CAMB derivatives and white-noise spectra, and compare with Planck 2018 VI (Table 2: TT+lowE and
TT,TE,EE+lowE) and with the covariance of the official Planck chains (TT+lowE).  Then: a futuristic
experiment, the effect of l_max and f_sky, and the classic degeneracies A_s e^{-2 tau} and Omega_m h^3.

Experiments (all with a Gaussian prior sigma_tau = 0.0086 standing in for Planck's lowE):
  PTT  Planck-like TT, 2 <= l <= 2500, f_sky = 0.57, HFI 100+143+217 noise (inverse-variance)
  PTE  Planck-like TT (2..29) + T and E (30..2000) + TT (2001..2500)
  S4   a futuristic ground survey: 1 muK arcmin (T), sqrt 2 muK arcmin (E), 1.4' beam, T and E at
       30 <= l <= 3000, f_sky = 0.4
Derivatives: the 4-point rule at the nominal steps (code/ch10/04_derivatives.py, cached).
Writes: figures/ch10/cmb_triangle.pdf, figures/ch10/cmb_corr.pdf, figures/ch10/cmb_lmax.pdf,
        data/ch10/fisher_cmb.npz, results/ch10/06_cmb_forecast.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, DIV
import lib_fisher10 as L

setup()
NOTES = pathlib.Path(__file__).resolve().parents[2]
dC = L.derivatives(1.0, 4)
C = L.fiducial_spectra()
fid = L.fiducial_vector()
NT, NP = L.noise_planck_like()
NT4, NP4 = L.noise_s4_like()
PRIOR = L.tau_prior(0.0086)
FS = 0.57
nums = {}


def F_ptt(fsky=FS, lmax=2500):
    return L.fisher_cmb(dC, C, NT, NP, fsky, 2, lmax, "TT")


def F_pte(fsky=FS, lmax=2500, lmaxP=2000):
    F = L.fisher_cmb(dC, C, NT, NP, fsky, 2, 29, "TT") + L.fisher_cmb(dC, C, NT, NP, fsky, 30, min(lmax, lmaxP))
    if lmax > lmaxP:
        F += L.fisher_cmb(dC, C, NT, NP, fsky, lmaxP + 1, lmax, "TT")
    return F


def F_s4(fsky=0.4, lmax=3000):
    return L.fisher_cmb(dC, C, NT4, NP4, fsky, 30, lmax)


FPTT, FPTE, FS4 = F_ptt() + PRIOR, F_pte() + PRIOR, F_s4() + PRIOR
FPTT_noprior = F_ptt()

# check: the T+E Fisher matrix written with the covariance of the three estimated spectra (Verde eq. 27)
Fchk = L.fisher_spectra(dC, C, NT, NP, FS, 30, 2000)
Fdir = L.fisher_cmb(dC, C, NT, NP, FS, 30, 2000)
nums["TenFCmChk"] = f"{np.max(np.abs(Fchk / Fdir - 1)):.0e}".replace("e-", r"\times10^{-") + "}"

# ---------------------------------------------------------------- Planck 2018 VI, Table 2
PUB_TT = np.array([0.00022, 0.0021, 0.00047, 0.0080, 0.016, 0.0057])
PUB_TE = np.array([0.00015, 0.0014, 0.00031, 0.0076, 0.016, 0.0044])   # tau +0.0070 -0.0081 -> mean width
covfile = NOTES / "data" / "ch09" / "planck_covmats" / "base_plikHM_TT_lowl_lowE.covmat"
names = open(covfile).readline()[1:].split()
idx = [names.index(n) for n in ["omegabh2", "omegach2", "theta", "tau", "logA", "ns"]]
CPL = np.loadtxt(covfile)[np.ix_(idx, idx)]

sPTT, sPTE, sS4 = L.marg(FPTT), L.marg(FPTE), L.marg(FS4)
cond = 1 / np.sqrt(np.diag(FPTT))
short = ["Ob", "Oc", "Th", "Tau", "As", "Ns"]


def fmt(v):
    if v < 1e-4:
        return f"{v * 1e5:.1f}" + r"\times10^{-5}"
    return f"{v:.2g}" if v < 0.001 else (f"{v:.4f}" if v < 0.01 else f"{v:.3f}")


for k, s in enumerate(short):
    nums[f"TenFCmPtt{s}"] = fmt(sPTT[k])
    nums[f"TenFCmPte{s}"] = fmt(sPTE[k])
    nums[f"TenFCmSf{s}"] = fmt(sS4[k])
    nums[f"TenFCmCond{s}"] = fmt(cond[k])
    nums[f"TenFCmRatTT{s}"] = f"{sPTT[k] / PUB_TT[k]:.2f}"
    nums[f"TenFCmRatTE{s}"] = f"{sPTE[k] / PUB_TE[k]:.2f}"
    nums[f"TenFCmGain{s}"] = f"{sPTE[k] / sS4[k]:.1f}"
    nums[f"TenFCmMC{s}"] = f"{sPTT[k] / cond[k]:.1f}"
nums["TenFCmFsky"] = FS

# correlations: ours (PTT) against the Planck TT+lowE chains
Cours = np.linalg.inv(FPTT)
Rours = Cours / np.outer(sPTT, sPTT)
spl = np.sqrt(np.diag(CPL))
Rpl = CPL / np.outer(spl, spl)
nums["TenFCmRhoOcNsOurs"] = f"{Rours[1, 5]:+.2f}"; nums["TenFCmRhoOcNsPl"] = f"{Rpl[1, 5]:+.2f}"
nums["TenFCmRhoTauAsOurs"] = f"{Rours[3, 4]:+.2f}"; nums["TenFCmRhoTauAsPl"] = f"{Rpl[3, 4]:+.2f}"
nums["TenFCmRhoObNsOurs"] = f"{Rours[0, 5]:+.2f}"; nums["TenFCmRhoObNsPl"] = f"{Rpl[0, 5]:+.2f}"
nums["TenFCmRhoObOcOurs"] = f"{Rours[0, 1]:+.2f}"; nums["TenFCmRhoObOcPl"] = f"{Rpl[0, 1]:+.2f}"
nums["TenFCmRhoMaxDiff"] = f"{np.max(np.abs(Rours - Rpl)):.2f}"

# ---------------------------------------------------------------- derived parameters by the delta method
h0 = L.h0_of(fid)
grad_h0 = np.zeros(6)
for i in (0, 1, 2):
    e = np.zeros(6); e[i] = L.STEPS[i]
    grad_h0[i] = (L.h0_of(fid + e) - L.h0_of(fid - e)) / (2 * L.STEPS[i])
omnu = float(C["omnuh2"])
h = h0 / 100
omm = (fid[0] + fid[1] + omnu) / h**2
# gradients of ln h, ln omega_m (omega_m = omega_b + omega_c + omega_nu), ln(A_s e^{-2 tau})
g_lnh = grad_h0 / h0
g_lnwm = np.array([1, 1, 0, 0, 0, 0]) / (fid[0] + fid[1] + omnu)
g_lnAe = np.array([0, 0, 0, -2, 1, 0])        # d ln(A_s e^{-2 tau}) / d(..., tau, ln A_s, ...)


def sig(g, Cv):
    return np.sqrt(g @ Cv @ g)


AE = 1e9 * np.exp(fid[4]) * 1e-10 * np.exp(-2 * fid[3])
for tag, F in [("Ptt", FPTT), ("Pte", FPTE), ("Sf", FS4)]:
    Cv = np.linalg.inv(F)
    nums[f"TenFCm{tag}Hz"] = f"{h0 * sig(g_lnh, Cv):.2f}"
    nums[f"TenFCm{tag}Om"] = f"{omm * sig(g_lnwm - 2 * g_lnh, Cv):.4f}"
    nums[f"TenFCm{tag}OmhThree"] = f"{omm * h**3 * sig(g_lnwm + g_lnh, Cv):.5f}"
    nums[f"TenFCm{tag}OmhThreeRel"] = f"{100 * sig(g_lnwm + g_lnh, Cv):.2f}"
    nums[f"TenFCm{tag}AsE"] = f"{AE * sig(g_lnAe, Cv):.4f}"
    nums[f"TenFCm{tag}AsERel"] = f"{100 * sig(g_lnAe, Cv):.2f}"
    # the best exponent p in Omega_m h^p: ln(Omega_m h^p) = ln omega_m + (p - 2) ln h
    cov_wh = g_lnwm @ Cv @ g_lnh
    var_h = g_lnh @ Cv @ g_lnh
    nums[f"TenFCm{tag}Pbest"] = f"{2 - cov_wh / var_h:.2f}"
nums["TenFCmHzFid"] = f"{h0:.2f}"; nums["TenFCmOmFid"] = f"{omm:.4f}"
nums["TenFCmOmhThreeFid"] = f"{omm * h**3:.5f}"; nums["TenFCmAsEFid"] = f"{AE:.3f}"
nums["TenFCmGradHzTh"] = f"{grad_h0[2]:.0f}"; nums["TenFCmGradHzOc"] = f"{grad_h0[1]:.0f}"
nums["TenFCmGradHzOb"] = f"{grad_h0[0]:.0f}"
# the A_s - tau degeneracy direction from temperature alone (no tau prior): the long axis of the
# marginalised (tau, ln A_s) ellipse
Cn = np.linalg.inv(FPTT_noprior)
sub = Cn[np.ix_([3, 4], [3, 4])]
w, v = np.linalg.eigh(sub)
major = v[:, np.argmax(w)]
nums["TenFCmSlopeAsTau"] = f"{major[1] / major[0]:.2f}"
nums["TenFCmTauNoPrior"] = f"{np.sqrt(Cn[3, 3]):.3f}"
nums["TenFCmAsNoPrior"] = f"{np.sqrt(Cn[4, 4]):.3f}"
nums["TenFCmAsENoPriorRel"] = f"{100 * sig(g_lnAe, Cn):.2f}"
nums["TenFCmAxisRatio"] = f"{np.sqrt(w.max() / w.min()):.0f}"

# ---------------------------------------------------------------- a nuisance parameter: the calibration
# Planck compares the data with C_l / y_P^2, where the map calibration y_P has a prior of 0.25 per cent.
# Add ln y_P as a seventh parameter: dC_l/d ln y = -2 C_l (signal only), prior 1/0.0025^2, then
# marginalise over it by inverting the 7x7 matrix.
def with_calibration(lmin, lmax, which, sig_y=0.0025):
    dC7 = {s: np.vstack([dC[s], -2 * C[s][None, :]]) for s in ("TT", "EE", "TE")}
    l = np.arange(lmin, lmax + 1)
    nu = (2 * l + 1) * FS
    c = C["TT"][l] + NT[l]
    d = dC7["TT"][:, l]
    F7 = (d * (nu / 2 / c**2)) @ d.T
    F7[:6, :6] += PRIOR
    F7[6, 6] += 1 / sig_y**2
    return F7


F7 = with_calibration(2, 2500, "TT")
C7 = np.linalg.inv(F7)
g7 = np.append(g_lnAe, 0.0)
nums["TenFCmCalAsERel"] = f"{100 * np.sqrt(g7 @ C7 @ g7):.2f}"
nums["TenFCmCalAsE"] = f"{AE * np.sqrt(g7 @ C7 @ g7):.4f}"
nums["TenFCmCalAs"] = f"{np.sqrt(C7[4, 4]):.4f}"
nums["TenFCmCalNs"] = f"{np.sqrt(C7[5, 5]):.4f}"
nums["TenFCmCalTau"] = f"{np.sqrt(C7[3, 3]):.4f}"
nums["TenFCmCalY"] = f"{np.sqrt(C7[6, 6]):.4f}"
nums["TenFCmPttAsSharp"] = f"{np.sqrt(np.linalg.inv(FPTT)[4, 4]):.4f}"

# ---------------------------------------------------------------- l_max and f_sky
lmaxes = np.array([500, 750, 1000, 1250, 1500, 2000, 2500, 3000])
sig_lmax_pte = np.array([L.marg(F_pte(lmax=lm, lmaxP=min(lm, 2000)) + PRIOR) for lm in lmaxes])
sig_lmax_s4 = np.array([L.marg(F_s4(lmax=lm) + PRIOR) for lm in lmaxes])
for k, s in enumerate(short):
    nums[f"TenFCmLmThousand{s}"] = f"{sig_lmax_pte[2, k] / sig_lmax_pte[6, k]:.1f}"
fsk = [0.3, 0.57, 0.8]
for fv, w_ in zip(fsk, ["Low", "Mid", "High"]):
    s_ = L.marg(F_pte(fsky=fv) + PRIOR)
    nums[f"TenFCmFsky{w_}Ns"] = f"{s_[5]:.4f}"
    nums[f"TenFCmFsky{w_}Tau"] = f"{s_[3]:.4f}"
    nums[f"TenFCmFsky{w_}Val"] = f"{fv:g}"

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
for ax, S, ttl in [(axs[0], sig_lmax_pte, "Planck-like T+E"), (axs[1], sig_lmax_s4, "S4-like T+E")]:
    for k in range(6):
        ax.plot(lmaxes, S[:, k] / S[-2, k], "o-", ms=3, color=SERIES[k], label=L.LABELS[k])
    ax.set_xlabel(r"$\ell_{\max}$")
    ax.set_title(ttl, fontsize=9)
    ax.set_yscale("log")
    ax.axhline(1, color="k", lw=0.6)
axs[0].set_ylabel(r"$\sigma_i(\ell_{\max})\,/\,\sigma_i(2500)$")
axs[1].legend(fontsize=7, ncol=2)
savefig(fig, "ch10", "cmb_lmax")

# ---------------------------------------------------------------- triangle of 1-sigma (2.30) ellipses
t = np.linspace(0, 2 * np.pi, 300)
fig, axs = plt.subplots(5, 5, figsize=(7.4, 7.4))
sets = [(np.linalg.inv(FPTT), SERIES[0], "-", "PTT (ours, Fisher)"),
        (CPL, "k", "--", "Planck 2018 TT+lowE chains"),
        (np.linalg.inv(FPTE), SERIES[1], "-", "PTE (ours, Fisher)"),
        (np.linalg.inv(FS4), SERIES[2], "-", "S4-like (ours, Fisher)")]
for r in range(1, 6):
    for c in range(5):
        ax = axs[r - 1, c]
        if c >= r:
            ax.axis("off"); continue
        for Cv, col, ls, lab in sets:
            sub = Cv[np.ix_([c, r], [c, r])]
            Lc = np.linalg.cholesky(sub)
            e = np.sqrt(2.30) * Lc @ np.vstack([np.cos(t), np.sin(t)])
            ax.plot(fid[c] + e[0], fid[r] + e[1], color=col, ls=ls, lw=1.1, label=lab)
        ax.tick_params(labelsize=5.5)
        ax.locator_params(nbins=3)
        if c == 0:
            ax.set_ylabel(L.LABELS[r], fontsize=8)
        else:
            ax.set_yticklabels([])
        if r == 5:
            ax.set_xlabel(L.LABELS[c], fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.grid(False)
h_, l_ = axs[4, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper right", bbox_to_anchor=(0.97, 0.95), fontsize=8)
fig.subplots_adjust(wspace=0.08, hspace=0.08)
savefig(fig, "ch10", "cmb_triangle")

# ---------------------------------------------------------------- correlation matrices side by side
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6))
for ax, R, ttl in [(axs[0], Rours, "Fisher, Planck-like TT + $\\tau$ prior"),
                   (axs[1], Rpl, "Planck 2018 TT+lowE chains")]:
    im = ax.imshow(R, cmap=DIV, vmin=-1, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{R[i, j]:+.2f}" if i != j else "", ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(L.LABELS, fontsize=6.5, rotation=45)
    ax.set_yticklabels(L.LABELS, fontsize=6.5)
    ax.set_title(ttl, fontsize=8.5)
    ax.grid(False)
savefig(fig, "ch10", "cmb_corr")

np.savez(NOTES / "data" / "ch10" / "fisher_cmb.npz", FPTT=FPTT, FPTE=FPTE, FS4=FS4, FPTT_noprior=FPTT_noprior,
         fid=fid, grad_h0=grad_h0, CPL=CPL)
save_numbers("ch10", "06_cmb_forecast", nums)
for k in nums:
    print(k, nums[k])
