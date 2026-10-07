"""44_angle.py -- the rotation angle of the SMICA polarization, and how far to trust the estimator.

Question: a rotation of the polarization plane by an angle beta turns E into B and makes
  EB = 1/2 sin(4 beta) (EE - BB),   TB = sin(2 beta) TE,
so for a small angle EB ~ 2 beta (EE - BB) and TB ~ 2 beta TE.  Which angle do the SMICA
cross-spectra prefer, with what error, and does the estimator find a known angle on simulated
skies (bias, scatter, coverage over many skies, never one)?
Estimator (least squares with the simulated covariance of the parity-odd spectra):
  e = (EB_b, TB_b),  D = (EE_b - BB_b, TE_b),  beta = D C^-1 e / (2 D C^-1 D),
  sigma = 1 / (2 sqrt(D C^-1 D)),  C^-1 with the Hartlap factor.
The covariance comes from the first N_COV simulated skies (beta = 0); the tests use the other
skies, so that no sky is both a ruler and a test object.
Writes: figures/ch13/biref_spectra.pdf, figures/ch13/biref_sims.pdf, data/ch13/biref_results.npz,
        results/ch13/44_angle.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, SERIES
import lib_biref as lb

setup()
DEG = lb.DEG
files = sorted(lb.DATA.glob("polsims_*.npz"))
S = {}
for f in files:
    z = np.load(f)
    for k in z.files:
        S.setdefault(k, []).append(z[k])
S = {k: np.concatenate(v) for k, v in S.items()}
nsim = S["b0_EB"].shape[0]
N_COV = int(round(2 * nsim / 3))
cov_idx, test_idx = np.arange(N_COV), np.arange(N_COV, nsim)
BETAS = np.array([0.0, 0.3, 1.0])
z = np.load(lb.DATA / "pol_data_spectra.npz")
data = {k: z[f"x_{k}"] for k in ("EE", "BB", "EB", "TE", "TB")}
nums = {"ThirteenENsimAll": nsim, "ThirteenENcov": N_COV, "ThirteenENtest": nsim - N_COV, "ThirteenENbins": lb.NB}


def sim_spec(prefix, i):
    return {k: S[f"{prefix}_{k}"][i] for k in ("EE", "BB", "EB", "TE", "TB")}


def inv_cov(which, prefix="b0", idx=cov_idx):
    E = np.array([lb.vectors(sim_spec(prefix, i), which)[0] for i in idx])
    C = np.cov(E, rowvar=False)
    n, p = len(idx), C.shape[0]
    h = (n - p - 2) / (n - 1)
    return h * np.linalg.inv(C), C, h


results = {}
for which in ("EB", "TB", "joint"):
    Ci, C, h = inv_cov(which)
    b, s = lb.estimate_beta(data, Ci, which)
    results[which] = (b, s)
    tag = {"EB": "EB", "TB": "TB", "joint": "Joint"}[which]
    nums[f"ThirteenEBeta{tag}"] = f"{b / DEG:.2f}"
    nums[f"ThirteenESig{tag}"] = f"{s / DEG:.3f}"
    nums[f"ThirteenEHartlap{tag}"] = f"{h:.2f}"
    # tests on the held-out skies
    for i, bt in enumerate(BETAS):
        est = np.array([lb.estimate_beta(sim_spec(f"b{i}", j), Ci, which) for j in test_idx])
        d = est[:, 0] - bt * DEG
        bias, sd, mean_sig = d.mean(), d.std(ddof=1), est[:, 1].mean()
        cov68 = np.mean(np.abs(d) < est[:, 1])
        cov95 = np.mean(np.abs(d) < 1.96 * est[:, 1])
        bt_tag = ["Zero", "Small", "One"][i]
        if which == "joint" or i == 0:
            nums[f"ThirteenEBias{tag}{bt_tag}"] = f"{round(bias / DEG, 3) + 0.0:.3f}"   # + 0.0: no "-0.000"
            nums[f"ThirteenEBiasErr{tag}{bt_tag}"] = f"{sd / np.sqrt(len(d)) / DEG:.3f}"
            nums[f"ThirteenESd{tag}{bt_tag}"] = f"{sd / DEG:.3f}"
            nums[f"ThirteenEMeanSig{tag}{bt_tag}"] = f"{mean_sig / DEG:.3f}"
            nums[f"ThirteenECovSix{tag}{bt_tag}"] = f"{100 * cov68:.0f}"
            nums[f"ThirteenECovNine{tag}{bt_tag}"] = f"{100 * cov95:.0f}"
        results[(which, i)] = est
n_test = len(test_idx)
nums["ThirteenECovErrSix"] = f"{100 * np.sqrt(0.68 * 0.32 / n_test):.0f}"
nums["ThirteenECovErrNine"] = f"{100 * np.sqrt(0.95 * 0.05 / n_test):.0f}"
sd_adopt = np.std(results[("joint", 0)][:, 0], ddof=1)      # scatter over the test skies
nums["ThirteenESigAdopt"] = f"{sd_adopt / DEG:.2f}"
nums["ThirteenESigRatio"] = f"{sd_adopt / results[('joint', 0)][:, 1].mean():.2f}"
nums["ThirteenESignifAdopt"] = f"{results['joint'][0] / sd_adopt:.1f}"
nums["ThirteenEDSFactor"] = f"{np.sqrt(1 + 2 * lb.NB / N_COV):.2f}"
nums["ThirteenEDSFactorEB"] = f"{np.sqrt(1 + lb.NB / N_COV):.2f}"
nums["ThirteenESigRatioEB"] = f"{np.std(results[('EB', 0)][:, 0], ddof=1) / results[('EB', 0)][:, 1].mean():.2f}"
nums["ThirteenESigTotal"] = f"{np.hypot(sd_adopt / DEG, 0.28):.2f}"
nums["ThirteenESignifJoint"] = f"{results['joint'][0] / results['joint'][1]:.1f}"
nums["ThirteenERatioEBTB"] = f"{results['TB'][1] / results['EB'][1]:.1f}"

# lensing B: variance, not bias (sky with and without lensing B, beta = 0, same skies)
Ci, C, h = inv_cov("joint")
lens = np.array([lb.estimate_beta(sim_spec("b0", j), Ci)[0] for j in test_idx])
nol = np.array([lb.estimate_beta(sim_spec("nolens", j), Ci)[0] for j in test_idx])
nums["ThirteenELensMean"] = f"{lens.mean() / DEG:.3f}"
nums["ThirteenENoLensMean"] = f"{nol.mean() / DEG:.3f}"
nums["ThirteenELensSd"] = f"{lens.std(ddof=1) / DEG:.4f}"
nums["ThirteenENoLensSd"] = f"{nol.std(ddof=1) / DEG:.4f}"
nums["ThirteenELensDiffSd"] = f"{(lens - nol).std(ddof=1) / DEG:.4f}"

# goodness of fit of the data: chi^2 of e - 2 beta D against the distribution over the test skies
def chi2(s, b):
    e, D = lb.vectors(s, "joint")
    r = e - 2 * b * D
    return r @ Ci @ r


chi_d = chi2(data, results["joint"][0])
chi_s = np.array([chi2(sim_spec("b0", j), results[("joint", 0)][k, 0]) for k, j in enumerate(test_idx)])
nums["ThirteenEChiData"] = f"{chi_d:.1f}"
nums["ThirteenEChiDof"] = 2 * lb.NB - 1
nums["ThirteenEChiPte"] = f"{np.mean(chi_s >= chi_d):.2f}"
nums["ThirteenEChiSimMean"] = f"{chi_s.mean():.1f}"
# no-rotation test: chi^2 at beta = 0 minus the best fit, against the skies (beta = 0)
dchi_d = chi2(data, 0.0) - chi_d
dchi_s = np.array([chi2(sim_spec("b0", j), 0.0) - chi_s[k] for k, j in enumerate(test_idx)])
nums["ThirteenEDchiData"] = f"{dchi_d:.1f}"
nums["ThirteenEDchiPte"] = f"{max(np.mean(dchi_s >= dchi_d), 0.0):.3f}"
nums["ThirteenEDchiNexceed"] = int(np.sum(dchi_s >= dchi_d))
nums["ThirteenEPteChiOne"] = f"{stats.chi2.sf(dchi_d, 1):.4f}"

# split in multipole: low and high halves of the bins (EB + TB)
half = lb.NB // 2
for name, sl in (("Low", slice(0, half)), ("High", slice(half, lb.NB))):
    sub = {k: v[sl] for k, v in data.items()}
    E = np.array([lb.vectors({k: S[f"b0_{k}"][i][sl] for k in data}, "joint")[0] for i in cov_idx])
    Cs = np.cov(E, rowvar=False)
    hs = (len(cov_idx) - Cs.shape[0] - 2) / (len(cov_idx) - 1)
    b, s = lb.estimate_beta(sub, hs * np.linalg.inv(Cs))
    nums[f"ThirteenEBeta{name}"] = f"{b / DEG:.2f}"
    nums[f"ThirteenESig{name}"] = f"{s / DEG:.2f}"
nums["ThirteenELsplit"] = int(lb.EDGES[half])

# the stricter window (Galactic check): same skies, so the shift main - cut is compared with the
# scatter of that shift over the test skies, not with either error bar
Ci_c, _, _ = inv_cov("joint", prefix="cut")
data_cut = {k: z[f"cut_{k}"] for k in data}
bc, sc = lb.estimate_beta(data_cut, Ci_c)
nums["ThirteenEBetaCut"] = f"{bc / DEG:.2f}"
nums["ThirteenESigCut"] = f"{sc / DEG:.2f}"
shift = np.array([lb.estimate_beta(sim_spec("b0", j), Ci)[0] - lb.estimate_beta(sim_spec("cut", j), Ci_c)[0]
                  for j in test_idx])
nums["ThirteenEShiftData"] = f"{(results['joint'][0] - bc) / DEG:.2f}"
nums["ThirteenEShiftSd"] = f"{shift.std(ddof=1) / DEG:.2f}"
nums["ThirteenEShiftZ"] = f"{(results['joint'][0] - bc) / shift.std(ddof=1):.1f}"

# template from the simulations' mean instead of the data (a choice, not a correction)
mean_spec = {k: S[f"b0_{k}"][cov_idx].mean(0) for k in data}
e, _ = lb.vectors(data, "joint")
_, Dm = lb.vectors(mean_spec, "joint")
F = Dm @ Ci @ Dm
nums["ThirteenEBetaFidTemplate"] = f"{(Dm @ Ci @ e) / (2 * F) / DEG:.2f}"
nums["ThirteenEEERatio"] = f"{np.sum(data['EE'][:20]) / np.sum(mean_spec['EE'][:20]):.3f}"

# the smallest example: one multipole, l = 1000, what a single ratio EB/2EE would give
from camb_fiducial import load_fiducial
pol = lb.load_pol()
nm = np.load(lb.DATA / "pol_noise_model.npz")
l0 = 1000
TP2 = (pol["beam_P"] * pol["pixwin_P"])[l0] ** 2
ee = load_fiducial("EE")[1][l0] * TP2
bb = load_fiducial("BB")[1][l0] * TP2
nh = 2 * np.mean(nm["ndEE"][l0 - 15:l0 + 16])           # noise of one half (beamed units)
W = np.load(lb.DATA / f"pol_windows_n{lb.NSIDE}.npz")["main"].astype(float)
fe = np.mean(W ** 2) ** 2 / np.mean(W ** 4)      # effective sky fraction (mode counting)
var1 = (ee + nh) * (bb + nh) / ((2 * l0 + 1) * fe)
fac = l0 * (l0 + 1) / (2 * np.pi)
nums["ThirteenEDEEthou"] = f"{fac * load_fiducial('EE')[1][l0]:.1f}"
nums["ThirteenEDEBthou"] = f"{fac * load_fiducial('EE')[1][l0] * 0.5 * np.sin(4 * 0.3 * DEG):.2f}"
nums["ThirteenENoiseOverEEthou"] = f"{nh / ee:.0f}"
nums["ThirteenESigNaiveThou"] = f"{np.sqrt(var1) / (2 * ee) / DEG:.0f}"
nums["ThirteenEModesThou"] = f"{(2 * l0 + 1) * fe:.0f}"

np.savez(lb.DATA / "biref_results.npz", beta=np.array([results[w][0] for w in ("EB", "TB", "joint")]),
         sig=np.array([results[w][1] for w in ("EB", "TB", "joint")]))
print(nums)
save_numbers("ch13", "44_angle", nums)

# ---------------------------------------------------------------- figure 1: the spectra and the fit
b = results["joint"][0]
sd = np.sqrt(np.diag(inv_cov("joint")[1]))
nb = lb.NB
fig, axs = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
x = lb.LEFF
for ax, k, Dk, sl, lab in ((axs[0], "EB", data["EE"] - data["BB"], slice(0, nb), "EB"),
                           (axs[1], "TB", data["TE"], slice(nb, 2 * nb), "TB")):
    ax.errorbar(x, data[k], yerr=sd[sl], fmt="o", ms=3, color="k", lw=0.8, label=f"SMICA $D^{{{lab}}}_b$ (hm1$\\times$hm2)")
    ax.plot(x, 2 * b * Dk, color=SERIES[1], lw=1.4,
            label=(r"$2\hat\beta\,(D^{EE}_b-D^{BB}_b)$" if k == "EB" else r"$2\hat\beta\,D^{TE}_b$"))
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel(rf"$D^{{{lab}}}_b$ [$\mu$K$^2$]")
    ax.legend(fontsize=8, loc="upper right")
axs[1].set_xlabel(r"multipole $\ell$")
savefig(fig, "ch13", "biref_spectra")

# ---------------------------------------------------------------- figure 2: the estimator on simulated skies
fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.8), sharey=True)
for i, bt in enumerate(BETAS):
    est = results[("joint", i)][:, 0] / DEG
    ax = axs[i]
    ax.hist(est, bins=25, density=True, color=SERIES[0], alpha=0.6)
    xx = np.linspace(est.min() - 0.05, est.max() + 0.05, 200)
    sg = results[("joint", i)][:, 1].mean() / DEG
    ax.plot(xx, stats.norm.pdf(xx, bt, sg), "k--", lw=1.2)
    ax.axvline(bt, color=SERIES[1], lw=1.2)
    ax.set_title(rf"injected $\beta={bt:g}^\circ$")
    ax.set_xlabel(r"$\hat\beta$ [deg]")
axs[0].set_ylabel("density")
savefig(fig, "ch13", "biref_sims")

# ---------------------------------------------------------------- figure 3: two checks against the skies
fig, axs = plt.subplots(1, 2, figsize=(8.0, 2.9))
ax = axs[0]
ax.hist(dchi_s, bins=30, color=SERIES[0], alpha=0.6, density=True, label=r"test skies, $\beta=0$")
xx = np.linspace(0, max(dchi_s.max(), dchi_d) * 1.05, 200)
ax.plot(xx, stats.chi2.pdf(xx, 1), "k--", lw=1.0, label=r"$\chi^2_1$")
ax.axvline(dchi_d, color=SERIES[1], lw=1.6, label="SMICA")
ax.set_xlabel(r"$\Delta\chi^2=\chi^2(0)-\chi^2(\hat\beta)$")
ax.set_ylim(0, 1.2)
ax.set_title(r"(a) is the angle zero?")
ax.legend(fontsize=7)
ax = axs[1]
ax.hist(shift / DEG, bins=30, color=SERIES[0], alpha=0.6, label="test skies")
ax.axvline((results["joint"][0] - bc) / DEG, color=SERIES[1], lw=1.6, label="SMICA")
ax.set_xlabel(r"$\hat\beta_{\rm main}-\hat\beta_{\rm strict}$ [deg]")
ax.set_title("(b) shift with the stricter window")
ax.legend(fontsize=7)
savefig(fig, "ch13", "biref_checks")
