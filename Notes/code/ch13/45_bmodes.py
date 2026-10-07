"""45_bmodes.py -- the B modes of the SMICA half-missions against the lensing prediction.

Question: gravitational lensing turns part of the E polarization into B, with a spectrum the
standard model predicts (CAMB, lensing on).  Do the SMICA hm1 x hm2 BB cross-spectra show it, and
how well can they at SMICA's noise level?
Estimator, built and validated on the simulated skies of 43_pol_sims.py:
  * L_b = mean noise-free pseudo-BB of the E part of the sky (E-to-B leakage through the window);
  * F_b = mean noise-free pseudo-BB of the lensing B part / binned fiducial BB (window x beam x pixels);
  * BB_hat_b = (pseudo cross-BB_b - L_b) / F_b, for the data and for every simulated sky;
  * errors and correlations from the skies; the null from the skies with no lensing B;
  * a lensing amplitude A (BB = A x fiducial lensing BB) by least squares, its distribution with
    and without lensing in the skies.
Writes: figures/ch13/bmodes.pdf, results/ch13/45_bmodes.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
from camb_fiducial import load_fiducial
import lib_biref as lb

setup()
files = sorted(lb.DATA.glob("polsims_*.npz"))
S = {}
for f in files:
    z = np.load(f)
    for k in z.files:
        S.setdefault(k, []).append(z[k])
S = {k: np.concatenate(v) for k, v in S.items()}
nsim = S["b0_BB"].shape[0]
N_COV = int(round(2 * nsim / 3))
cov_idx, test_idx = np.arange(N_COV), np.arange(N_COV, nsim)
data = np.load(lb.DATA / "pol_data_spectra.npz")
bbf = load_fiducial("BB")[1][: lb.LS + 1]
eef = load_fiducial("EE")[1][: lb.LS + 1]
t = lb.binned(bbf)                                   # lensing BB, binned D_b
Lk = S["sigE_BB"].mean(0)                            # leakage
F = S["sigB_BB"].mean(0) / t                         # transfer of a B-mode spectrum
nums = {}


def bb_hat(pseudo):
    return (pseudo - Lk) / F


Bd = bb_hat(data["x_BB"])
Bs = bb_hat(S["b0_BB"])
Bn = bb_hat(S["nolens_BB"])
C = np.cov(Bs[cov_idx], rowvar=False)
n, p = N_COV, lb.NB
h = (n - p - 2) / (n - 1)
Ci = h * np.linalg.inv(C)
sd = np.sqrt(np.diag(C))

# lensing amplitude
def amp(B):
    return (t @ Ci @ B) / (t @ Ci @ t)


sigA = 1 / np.sqrt(t @ Ci @ t)
Ad = amp(Bd)
As = np.array([amp(B) for B in Bs[test_idx]])
An = np.array([amp(B) for B in Bn[test_idx]])
nums.update({"ThirteenEAlens": f"{Ad:.2f}", "ThirteenESigAlens": f"{sigA:.2f}",
             "ThirteenEAlensSimMean": f"{As.mean():.2f}", "ThirteenEAlensSimSd": f"{As.std(ddof=1):.2f}",
             "ThirteenEAnullMean": f"{An.mean():.2f}", "ThirteenEAnullSd": f"{An.std(ddof=1):.2f}",
             "ThirteenEAnullExceed": int(np.sum(An >= Ad)), "ThirteenEAlensBelow": int(np.sum(As <= Ad)),
             "ThirteenENtestB": len(test_idx), "ThirteenEAlensSignif": f"{Ad / sigA:.1f}"})
# chi^2 of the data against lensing and against zero, with the simulated distributions
chi = lambda B, m: (B - m) @ Ci @ (B - m)
cd_l, cd_0 = chi(Bd, t), chi(Bd, 0 * t)
cs_l = np.array([chi(B, t) for B in Bs[test_idx]])
nums.update({"ThirteenEChiBBLens": f"{cd_l:.1f}", "ThirteenEChiBBZero": f"{cd_0:.1f}",
             "ThirteenEPteBBLens": f"{np.mean(cs_l >= cd_l):.2f}", "ThirteenEBBdof": p})
# the stricter window: its own leakage and transfer, same skies
Lc = S["cutE_BB"].mean(0)
Fc = S["cutB_BB"].mean(0) / t
Bdc = (data["cut_BB"] - Lc) / Fc
Bsc = (S["cut_BB"] - Lc) / Fc
Cc = np.cov(Bsc[cov_idx], rowvar=False)
Cic = h * np.linalg.inv(Cc)
ampc = lambda B: (t @ Cic @ B) / (t @ Cic @ t)
Adc = ampc(Bdc)
dA = np.array([amp(Bs[j]) - ampc(Bsc[j]) for j in test_idx])
Asc = np.array([ampc(Bsc[j]) for j in test_idx])
nums["ThirteenEAlensCutSimSd"] = f"{Asc.std(ddof=1):.2f}"
nums["ThirteenEAcutAboveOne"] = f"{(Adc - 1) / Asc.std(ddof=1):.1f}"
nums.update({"ThirteenEAlensCut": f"{Adc:.2f}", "ThirteenESigAlensCut": f"{1 / np.sqrt(t @ Cic @ t):.2f}",
             "ThirteenEAshiftSd": f"{dA.std(ddof=1):.2f}", "ThirteenEAshiftZ": f"{(Ad - Adc) / dA.std(ddof=1):.1f}"})
# the size of things at l ~ 1000 (bin containing 1000)
kb = int(np.searchsorted(lb.EDGES, 1000) - 1)
nums.update({"ThirteenELensDb": f"{t[kb]:.3f}", "ThirteenESigBBbin": f"{sd[kb]:.2f}",
             "ThirteenELbin": int(lb.EDGES[kb]),
             "ThirteenELeakOverLens": f"{(Lk / S['sigB_BB'].mean(0))[kb]:.2f}",
             "ThirteenENoiseOverLens": f"{(S['noise_BB'].mean(0) / S['sigB_BB'].mean(0))[kb]:.0f}"})
bin_ok = ~np.isnan(S["binE_BB"][:, 0])
Lbin = S["binE_BB"][bin_ok].mean(0)
EEbin = S["binE_EE"][bin_ok].mean(0)
EEap = S["sigE_EE"][bin_ok].mean(0)
LkA = S["sigE_BB"][bin_ok].mean(0)
nums["ThirteenELeakFracBinary"] = f"{100 * np.median(Lbin / EEbin):.1f}"
nums["ThirteenELeakFracApod"] = f"{100 * np.median(LkA / EEap):.2f}"
nums["ThirteenELeakGain"] = f"{np.median(Lbin / EEbin) / np.median(LkA / EEap):.0f}"
nums["ThirteenELeakLowBinary"] = f"{100 * (Lbin / EEbin)[0]:.1f}"
nums["ThirteenELeakLowApod"] = f"{100 * (LkA / EEap)[0]:.2f}"
nums["ThirteenEBbFirstData"] = f"{Bd[0]:.2f}"
nums["ThirteenEBbFirstSig"] = f"{sd[0]:.2f}"
# data null: the half-difference BB against the simulated half-differences
nd, ns = data["null_BB"], S["null_BB"]
z = (nd - ns.mean(0)) / ns.std(0, ddof=1)
nums["ThirteenENullBBzMax"] = f"{np.max(np.abs(z)):.1f}"
nums["ThirteenENullBBratio"] = f"{np.median(nd / ns.mean(0)):.3f}"
ze = (data["null_EE"] - S["null_EE"].mean(0)) / S["null_EE"].std(0, ddof=1)
nums["ThirteenENullEEratio"] = f"{np.median(data['null_EE'] / S['null_EE'].mean(0)):.3f}"
nums["ThirteenENullEEzMed"] = f"{np.median(ze):.1f}"
nums["ThirteenENullBBzMed"] = f"{np.median(z):.1f}"
# an excess shared by EE and BB? (data cross-spectra minus the mean of the skies, pseudo D_b)
hiw = (lb.LEFF > 800) & (lb.LEFF < 1400)
for w, pre in (("Main", "x"), ("Cut", "cut")):
    simpre = "b0" if pre == "x" else "cut"
    for k in ("EE", "BB"):
        ex = data[f"{pre}_{k}"] - S[f"{simpre}_{k}"].mean(0)
        err = S[f"{simpre}_{k}"].std(0, ddof=1) / np.sqrt(hiw.sum())
        nums[f"ThirteenEExcess{k}{w}"] = f"{ex[hiw].mean():.2f}"
        nums[f"ThirteenEExcessErr{k}{w}"] = f"{err[hiw].mean():.2f}"
exB = data["x_BB"] - S["b0_BB"].mean(0)
lo_, hi_ = (lb.LEFF > 550) & (lb.LEFF < 750), (lb.LEFF > 1100) & (lb.LEFF < 1400)
nums["ThirteenEExcessSlope"] = f"{np.log(exB[hi_].mean() / exB[lo_].mean()) / np.log(lb.LEFF[hi_].mean() / lb.LEFF[lo_].mean()):.1f}"
r = data["null_BB"] / S["null_BB"].mean(0)
nums["ThirteenENullLowDev"] = f"{100 * np.max(np.abs(r[lb.LEFF < 1100] - 1)):.0f}"
nums["ThirteenENullHighDev"] = f"{100 * (1 - r[-1]):.0f}"
above = S["noise_BB"].mean(0) > S["sigE_EE"].mean(0)
nums["ThirteenENoiseCrossEE"] = int(lb.EDGES[np.argmax(above)]) if above.any() else ">1490"
print(nums)
save_numbers("ch13", "45_bmodes", nums)

# ---------------------------------------------------------------- figure 1: what is in a pseudo-BB spectrum
x = lb.LEFF
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.semilogy(x, S["noise_BB"].mean(0), color=SERIES[3], label="noise of one half-mission map")
ax.semilogy(x, S["sigE_EE"].mean(0), color="k", lw=0.8, label="E of the sky (EE)")
ax.semilogy(x, Lbin, color=SERIES[1], ls=":", label="E leaked into B, binary mask")
ax.semilogy(x, Lk, color=SERIES[1], label="E leaked into B, apodised window")
ax.semilogy(x, S["sigB_BB"].mean(0), color=SERIES[2], label="lensing B")
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"pseudo-$D_b$ [$\mu$K$^2$]")
ax.set_ylim(5e-4, 1e5)                               # headroom for the legend
ax.legend(fontsize=7, ncol=2, loc="upper center")    # above the curves, clear of the E power
savefig(fig, "ch13", "leakage")

# ---------------------------------------------------------------- figure 2: B modes against lensing
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
lo, hi = np.percentile(Bn[test_idx], [16, 84], axis=0)
ax.fill_between(x, lo, hi, color="0.85", label="skies without lensing B (68%)")
ax.errorbar(x, Bd, yerr=sd, fmt="o", ms=3, color="k", lw=0.8, label="SMICA hm1$\\times$hm2")
ax.plot(x, t, color=SERIES[2], lw=1.6, label="lensing prediction")
ax.axhline(0, color="0.5", lw=0.6)
ax.set_ylim(-3, 3)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$\widehat D^{BB}_b$ [$\mu$K$^2$]")
ax.set_title("(a) B modes, leakage removed")
ax.legend(fontsize=7, loc="lower right")
ax = axs[1]
bins = np.linspace(min(An.min(), As.min(), Ad) - 0.2, max(An.max(), As.max(), Ad) + 0.2, 30)
ax.hist(An, bins=bins, color="0.6", alpha=0.6, label="no lensing B")
ax.hist(As, bins=bins, color=SERIES[2], alpha=0.5, label="with lensing B")
ax.axvline(Ad, color="k", lw=1.4, label="SMICA")
ax.set_xlabel(r"lensing amplitude $\hat A$")
ax.set_ylabel("skies")
ax.set_title("(b) amplitude over the test skies")
ax.legend(fontsize=7)
savefig(fig, "ch13", "bmodes")
