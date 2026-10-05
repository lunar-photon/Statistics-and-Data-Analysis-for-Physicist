"""05_compare.py -- error bars from the simulations, and our spectrum against Planck's.

Question: with the covariance of the 400 simulated skies, (1) is our estimator unbiased on the
simulations, and does it matter whether the theory is binned with the exact bandpower windows
or with the plain binning operator? (2) Does the Planck best-fit spectrum describe our bandpowers
(chi^2, probability to exceed)?  (3) How do our bandpowers compare with the published Planck
binned TT spectrum, and how large should the difference be, given that both are measured on
almost the same sky?
Writes: figures/ch13/compare.pdf, figures/ch13/covariance.pdf, results/ch13/05_compare.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, SERIES
from camb_fiducial import load_fiducial
import lib_planck as lp

setup()
LC = 1199
z = np.load(lp.DATA / "bandpowers.npz")
REP = z["rep"]
sims = [np.load(f) for f in sorted(lp.DATA.glob("sims_*.npz"))]
S = {k: np.concatenate([s[k] for s in sims]) for k in sims[0].files}
n = S["main_cross"].shape[0]
_, clf = load_fiducial()
clf = clf[: LC + 1]
nums = {"TwelveANsimUsed": n}

lb = z["main_leff"][REP]
X = S["main_cross"][:, REP]
D = z["main_cross"][REP]
p = X.shape[1]
C = np.cov(X, rowvar=False)
sig = np.sqrt(np.diag(C))
hart = (n - p - 2) / (n - 1)
Psi = hart * np.linalg.inv(C)
nums.update({"TwelveAP": p, "TwelveAHartlap": f"{hart:.3f}",
             "TwelveASigVarParam": f"{100 * np.sqrt(2 / (n - p)):.1f}"})

# ---------------------------------------------------------------- (1) bias on the simulations
F = z["main_F"][REP]                              # bandpower windows: <D_b> = sum_l F_bl C_l
P, _, _ = lp.planck_binning(z["edges"], LC)
t_exact = F @ clf
t_naive = P[REP] @ clf
xbar = X.mean(0)
err_mean = sig / np.sqrt(n)
pull_exact = (xbar - t_exact) / err_mean
pull_naive = (xbar - t_naive) / err_mean
chi_exact = float((xbar - t_exact) @ (n * Psi) @ (xbar - t_exact))
chi_naive = float((xbar - t_naive) @ (n * Psi) @ (xbar - t_naive))
nums.update({"TwelveABiasChiExact": f"{chi_exact:.1f}", "TwelveABiasChiNaive": f"{chi_naive:.0f}",
             "TwelveABiasPteExact": f"{stats.chi2.sf(chi_exact, p):.2f}",
             "TwelveAWinMaxPct": f"{100 * np.max(np.abs(t_naive / t_exact - 1)):.1f}",
             "TwelveAWinMaxSig": f"{np.max(np.abs(t_naive - t_exact) / sig):.2f}"})
print(f"mean of sims vs exact windows: chi2 = {chi_exact:.1f}/{p}; vs plain binning: {chi_naive:.1f}")

# correlation of neighbouring bandpowers
R = C / np.outer(sig, sig)
nums["TwelveACorrNext"] = f"{np.mean(np.diag(R, 1)):.3f}"
nums["TwelveACorrNextMin"] = f"{np.min(np.diag(R, 1)):.3f}"

# error bar against the full-sky cosmic variance of the same bins (the price of the mask and noise)
ell = np.arange(LC + 1)
cv = np.array([np.sum(((ell * (ell + 1) / (2 * np.pi)) ** 2 * 2 * clf ** 2 / (2 * ell + 1))[a:b]) / (b - a) ** 2
               for a, b in zip(z["edges"][:-1], z["edges"][1:])])
ratio_cv = sig / np.sqrt(cv[REP])
nums["TwelveAErrOverCvLow"] = f"{ratio_cv[0]:.2f}"
nums["TwelveAErrOverCvMid"] = f"{np.median(ratio_cv[:15]):.2f}"
nums["TwelveAErrOverCvHigh"] = f"{ratio_cv[-1]:.2f}"
nums["TwelveAOneOverSqrtFsky"] = f"{1 / np.sqrt(0.775 * 0.894):.2f}"
# Knox's formula with the effective sky fraction of the window (8b:eq:hivon-nu), first and last bin
W = np.load(lp.DATA / "masks_n512.npz")["main"].astype(float)
fsky_eff = np.mean(W ** 2) ** 2 / np.mean(W ** 4)
for k, tag in ((0, "First"), (-1, "Last")):
    knox = np.sqrt(2 / ((2 * lb[k] + 1) * 30 * fsky_eff)) * t_exact[k]
    nums[f"TwelveASig{tag}"] = f"{sig[k]:.1f}"
    nums[f"TwelveAKnox{tag}"] = f"{knox:.1f}"
nums["TwelveAOneOverSqrtFeff"] = f"{1 / np.sqrt(fsky_eff):.2f}"
nums["TwelveAMcErrSig"] = f"{100 / np.sqrt(2 * (n - 1)):.1f}"

# ---------------------------------------------------------------- (2) chi^2 against the best fit
r = D - t_exact
chi_bf = float(r @ Psi @ r)
chi_sims = np.array([(x - t_exact) @ Psi @ (x - t_exact) for x in X])
nums.update({"TwelveAChiBf": f"{chi_bf:.1f}", "TwelveAPteBf": f"{stats.chi2.sf(chi_bf, p):.2f}",
             "TwelveAPteBfSims": f"{np.mean(chi_sims >= chi_bf):.2f}"})
# one amplitude: D = A t (8b's estimator, now on the real sky)
Fa = t_exact @ Psi @ t_exact
A = float(t_exact @ Psi @ D / Fa)
nums.update({"TwelveAAmp": f"{A:.4f}", "TwelveAAmpErr": f"{1 / np.sqrt(Fa):.4f}"})
print(f"chi2 vs best fit = {chi_bf:.1f} for {p}; A = {A:.4f} +- {1/np.sqrt(Fa):.4f}")

# ---------------------------------------------------------------- (3) against Planck's bandpowers
l_pl, D_pl, s_pl, D_bf = lp.load_planck_binned()
D_pl, s_pl, D_bf, l_pl = D_pl[:p], s_pl[:p], D_bf[:p], l_pl[:p]
diff = D - D_pl
nums.update({"TwelveALeffCheck": f"{np.max(np.abs(lb - l_pl)):.1e}",
             "TwelveADiffRms": f"{np.sqrt(np.mean(diff ** 2)):.0f}",
             "TwelveADiffOverSigPl": f"{np.sqrt(np.mean((diff / s_pl) ** 2)):.2f}",
             "TwelveASigOursOverPl": f"{np.median(sig / s_pl):.2f}",
             "TwelveABfDiffMax": f"{100 * np.max(np.abs(t_exact / D_bf - 1)):.1f}"})
# how different are two estimates of the SAME sky through nested windows? (simulations)
dd = S["main_cross"][:, REP] - S["cut30_cross"][:, REP]
rho = np.array([np.corrcoef(S["main_cross"][:, REP][:, b], S["cut30_cross"][:, REP][:, b])[0, 1] for b in range(p)])
sd_cut = S["cut30_cross"][:, REP].std(0, ddof=1)
nums.update({"TwelveARhoNested": f"{np.median(rho):.2f}",
             "TwelveADiffNestedOverSig": f"{np.median(dd.std(0, ddof=1) / sig):.2f}",
             "TwelveACutOverMain": f"{np.median(sd_cut / sig):.2f}"})
# a 'calibration' ratio: weighted mean of D_ours / D_planck over the reported bins
wts = 1 / sig ** 2
nums["TwelveARatioPl"] = f"{np.sum(wts * D / D_pl) / np.sum(wts):.4f}"
# where do the two differ most (in units of our error bar)?
k = int(np.argmax(np.abs(diff) / sig))
nums.update({"TwelveAWorstL": f"{lb[k]:.0f}", "TwelveAWorstSig": f"{diff[k] / sig[k]:+.1f}"})
hi = lb > 700
nums["TwelveADiffHigh"] = f"{np.mean(diff[hi]):+.0f}"
nums["TwelveADiffHighErr"] = f"{np.sqrt(np.sum(C[np.ix_(hi, hi)])) / hi.sum():.0f}"
print(nums)

# ---------------------------------------------------------------- figures
fig, axs = plt.subplots(3, 1, figsize=(6.8, 7.0), sharex=True, gridspec_kw=dict(height_ratios=[2.0, 1.1, 1.0]))
ax = axs[0]
ax.plot(ell[2:], (ell * (ell + 1) * clf / (2 * np.pi))[2:], color="k", lw=0.7, label="Planck 2018 best fit")
ax.errorbar(l_pl - 4, D_pl, yerr=s_pl, fmt="s", ms=2.5, lw=0.8, color=SERIES[1], label="Planck 2018 binned TT (Plik)")
ax.errorbar(lb + 4, D, yerr=sig, fmt="o", ms=2.5, lw=0.8, color=SERIES[0], label="ours: SMICA half-mission cross, MASTER")
ax.set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
ax.legend(fontsize=7.5)
ax = axs[1]
ax.axhline(0, color="0.4", lw=0.6)
ax.errorbar(l_pl - 4, D_pl - D_bf, yerr=s_pl, fmt="s", ms=2.5, lw=0.8, color=SERIES[1])
ax.errorbar(lb + 4, D - t_exact, yerr=sig, fmt="o", ms=2.5, lw=0.8, color=SERIES[0])
ax.set_ylabel(r"$-$ best fit [$\mu$K$^2$]")
ax = axs[2]
ax.axhline(0, color="0.4", lw=0.6)
ax.fill_between(lb, -1, 1, color="0.9", lw=0)
ax.plot(lb, diff / sig, "o-", ms=2.5, lw=0.7, color=SERIES[2])
ax.set_ylabel(r"(ours$-$Planck)$/\sigma$")
ax.set_xlabel(r"multipole $\ell$")
ax.set_xlim(0, 1040)
fig.tight_layout()
savefig(fig, "ch13", "compare")

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
ax = axs[0]
im = ax.imshow(R, cmap="RdBu_r", vmin=-0.3, vmax=0.3, origin="lower",
               extent=[lb[0] - 15, lb[-1] + 15, lb[0] - 15, lb[-1] + 15])
fig.colorbar(im, ax=ax, fraction=0.046)
ax.set_xlabel(r"$\ell_b$")
ax.set_ylabel(r"$\ell_{b'}$")
ax.set_title("(a) correlation of the bandpowers", fontsize=9)
ax.grid(False)
ax = axs[1]
ax.plot(lb, pull_naive, "s", ms=3, color=SERIES[1], label=r"theory binned with $P_{b\ell}$")
ax.plot(lb, pull_exact, "o", ms=3, color=SERIES[0], label=r"theory with windows $F_{b\ell}$")
ax.axhline(0, color="0.4", lw=0.6)
ax.fill_between(lb, -2, 2, color="0.9", lw=0)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"(mean of sims $-$ theory) / error of mean")
ax.set_title(f"(b) bias test, {n} simulations", fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch13", "covariance")
save_numbers("ch13", "05_compare", nums)
