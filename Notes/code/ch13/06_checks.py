"""06_checks.py -- the checks a real analysis runs before it trusts a spectrum.

Question: is there anything in the SMICA bandpowers that is not a statistically isotropic CMB sky
plus the noise we modelled?  Each check is a difference that should vanish on average if the
answer is no; the simulations, analysed in exactly the same way, give its scatter and the
distribution of its chi^2, so each check ends in a probability to exceed (PTE).
  1. hemispheres: D(north) - D(south)            (statistical isotropy; foregrounds that differ)
  2. sky fraction: D(|b| > 30 deg) - D(main)      (Galactic residuals grow towards the plane)
  3. apodisation: D(binary mask) - D(main)        (leakage and the mode-coupling correction)
  4. half-mission null: spectrum of (hm1 - hm2)/2 against the simulated noise (a consistency
     check of the noise model, which was itself built from this map)
  5. cross minus (auto minus null): the identity C_12 = C_full - C_null for full = (hm1 + hm2)/2
The PTE is the fraction of simulations with a larger chi^2 (and, for comparison, the chi^2_p
tail with the Hartlap-corrected inverse covariance).
Writes: figures/ch13/checks.pdf, results/ch13/06_checks.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, SERIES
import lib_planck as lp

setup()
z = np.load(lp.DATA / "bandpowers.npz")
REP = z["rep"]
sims = [np.load(f) for f in sorted(lp.DATA.glob("sims_*.npz"))]
S = {k: np.concatenate([s[k] for s in sims]) for k in sims[0].files}
n = S["main_cross"].shape[0]
lb = z["main_leff"][REP]
nums = {}


def null_test(d_data, d_sims):
    """chi^2 of (data - mean of sims) with the Hartlap-corrected inverse of the sims' covariance."""
    m = d_sims.mean(0)
    C = np.cov(d_sims, rowvar=False)
    p = C.shape[0]
    Psi = (n - p - 2) / (n - 1) * np.linalg.inv(C)
    chi = lambda v: float((v - m) @ Psi @ (v - m))
    chi_d = chi(d_data)
    chi_s = np.array([chi(v) for v in d_sims])
    return dict(chi=chi_d, pte_sims=np.mean(chi_s >= chi_d), pte_chi2=stats.chi2.sf(chi_d, p),
                pull=(d_data - m) / np.sqrt(np.diag(C)), chi_sims=chi_s, p=p)


tests = {
    "Hemi": (z["north_cross"][REP] - z["south_cross"][REP], S["north_cross"][:, REP] - S["south_cross"][:, REP]),
    "Cut": (z["cut30_cross"][REP] - z["main_cross"][REP], S["cut30_cross"][:, REP] - S["main_cross"][:, REP]),
    "Apo": (z["binary_cross"][REP] - z["main_cross"][REP], S["binary_cross"][:, REP] - S["main_cross"][:, REP]),
    "Null": (z["main_null"][REP], S["main_null"][:, REP]),
}
res = {}
for k, (dd, ds) in tests.items():
    res[k] = null_test(dd, ds)
    r = res[k]
    nums.update({f"TwelveA{k}Chi": f"{r['chi']:.1f}", f"TwelveA{k}PteSims": f"{r['pte_sims']:.2f}",
                 f"TwelveA{k}PteChi": f"{r['pte_chi2']:.2f}",
                 f"TwelveA{k}MaxPull": f"{np.max(np.abs(r['pull'])):.1f}"})
    print(f"{k}: chi2 = {r['chi']:.1f} / {r['p']}, PTE(sims) = {r['pte_sims']:.3f}, PTE(chi2) = {r['pte_chi2']:.3f}")
nums["TwelveAChiSimMean"] = f"{res['Hemi']['chi_sims'].mean():.1f}"

# the scatter of each difference, in units of the main error bar (how sensitive is each test?)
sig_main = S["main_cross"][:, REP].std(0, ddof=1)
for k, (dd, ds) in tests.items():
    if k != "Null":
        nums[f"TwelveA{k}SdOverSig"] = f"{np.median(ds.std(0, ddof=1) / sig_main):.2f}"

# where does the data sit in the hemisphere test at low l (the known asymmetry is at l < 64)?
nums["TwelveAHemiFirst"] = f"{res['Hemi']['pull'][0]:+.1f}"

# 5. the cross-spectrum identity
cross = z["main_cross"][REP]
auto_minus_null = z["main_auto"][REP] - z["main_null"][REP]
nums["TwelveAIdentityMax"] = f"{np.max(np.abs(cross - auto_minus_null)):.2f}"
nums["TwelveANullOverSignal"] = f"{100 * z['main_null'][REP][-1] / cross[-1]:.1f}"
nums["TwelveANullOverSignalMid"] = f"{100 * z['main_null'][REP][15] / cross[15]:.2f}"

# 6. why does the half-mission null fail?  Two candidate explanations:
#    (i) the noise model has the wrong shape in l -> an excess smooth in l;
#    (ii) the halves see the sky with slightly different gains g1 = 1 + e/2, g2 = 1 - e/2 ->
#         the difference keeps (e/2)^2 of the sky: an excess that follows the CMB spectrum.
#    Fit excess_b = a * t_b + c (t_b: fiducial bandpowers) below l = 600 and compare.
from camb_fiducial import load_fiducial
_, clf = load_fiducial()
t = z["main_F"][REP] @ clf[: z["main_F"].shape[1]]
m_null = S["main_null"][:, REP].mean(0)
exc = z["main_null"][REP] - m_null
lo = lb < 600
A = np.vstack([t[lo], np.ones(lo.sum())]).T
(a, c), *_ = np.linalg.lstsq(A, exc[lo], rcond=None)
resid = exc[lo] - A @ np.array([a, c])
k = int(np.argmax(z["main_null"][REP] / m_null))
nums.update({"TwelveANullRatioMax": f"{z['main_null'][REP][k] / m_null[k]:.2f}", "TwelveANullRatioL": f"{lb[k]:.0f}",
             "TwelveANullGainEps": f"{100 * 2 * np.sqrt(max(a, 0)):.1f}",
             "TwelveANullExcessExplained": f"{100 * (1 - np.sum(resid ** 2) / np.sum((exc[lo] - exc[lo].mean()) ** 2)):.0f}",
             "TwelveANullCrossEffect": f"{100 * max(a, 0):.4f}",
             "TwelveANullRatioHigh": f"{np.mean((z['main_null'][REP] / m_null)[lb > 600]):.3f}"})
# root-mean-square pull of each difference test, and the chance of two chi^2 this low among four tests
for k2 in ("Hemi", "Apo", "Cut"):
    nums[f"TwelveA{k2}RmsPull"] = f"{np.sqrt(np.mean(res[k2]['pull'] ** 2)):.2f}"
nums["TwelveALowCorr"] = f"{np.corrcoef(res['Hemi']['chi_sims'], res['Apo']['chi_sims'])[0, 1]:+.2f}"
nums["TwelveAHemiLower"] = int(np.sum(res["Hemi"]["chi_sims"] < res["Hemi"]["chi"]))
nums["TwelveAApoLower"] = int(np.sum(res["Apo"]["chi_sims"] < res["Apo"]["chi"]))
# A count of one or two sims out of n fixes the fraction only roughly, so quote a 95% upper limit:
# the Poisson upper limit on a count k is chi2.ppf(0.95, 2(k + 1)) / 2 (4.74 for k = 1), taken for the
# larger of the two counts; the chance of two of four tests this low is then at most 6 q_up^2.
k_low = max(nums["TwelveAHemiLower"], nums["TwelveAApoLower"])
q_up = stats.chi2.ppf(0.95, 2 * (k_low + 1)) / 2 / n
nums["TwelveALowFracUp"] = f"{q_up:.2f}" if q_up >= 0.01 else f"{q_up:.3f}"
x2 = 6 * q_up ** 2
e2 = int(np.floor(np.log10(x2)))
nums["TwelveALowTwoOfFourUp"] = f"{np.ceil(x2 / 10 ** e2):.0f}\\times10^{{{e2}}}"
print(nums)

fig, axs = plt.subplots(2, 2, figsize=(7.8, 5.2), sharex=True)
titles = {"Hemi": "(a) north $-$ south", "Cut": r"(b) $|b|>30^\circ$ $-$ main",
          "Apo": "(c) binary mask $-$ apodised", "Null": "(d) half-mission difference"}
for ax, (k, r) in zip(axs.flat, res.items()):
    ax.fill_between(lb, -2, 2, color="0.9", lw=0)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.plot(lb, r["pull"], "o-", ms=2.5, lw=0.7, color=SERIES[0])
    ax.set_title(f"{titles[k]}: $\\chi^2={r['chi']:.0f}$/{r['p']}, PTE {r['pte_sims']:.2f}", fontsize=8.5)
    ax.set_ylim(-4, 4) if k != "Null" else ax.set_ylim(np.floor(np.min(r["pull"])) - 2, np.ceil(np.max(r["pull"])) + 2)
for ax in axs[1]:
    ax.set_xlabel(r"multipole $\ell$")
for ax in axs[:, 0]:
    ax.set_ylabel("(data $-$ sims) / scatter")
fig.tight_layout()
savefig(fig, "ch13", "checks")
save_numbers("ch13", "06_checks", nums)
