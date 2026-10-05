"""26_distribution.py -- the shape of the sampling distribution of the cut-sky spectrum.

Question: on the full sky (2l+1) C_hat_l / C_l is chi^2 with 2l+1 degrees of freedom.  What is
the distribution of the spectrum measured through a mask, at low l where few modes exist and at
high l where many do?  Is a scaled chi^2 with an effective number of modes good enough, and
when is a Gaussian good enough?
Computes, from the 2000 skies of 23_mc_masked.py:
  * x = C~_l / <C~_l> (pseudo-spectrum, noise included) at l = 2, 10, 50 for the full sky and
    the 0.3 and 0.1 caps, with the moment-matched chi^2_nu (nu = 2/Var x) and the Gaussian;
  * nu_eff(l) = 2 <C~>^2 / Var C~ from the simulations against (2l+1) fsky w2^2/w4 (Hivon eq. 17);
  * the skewness of x against the chi^2 value sqrt(8/nu_eff);
  * the probability that a sky gives a value below its mean (0.5 for a symmetric law).
Writes: figures/ch08/cl_histograms.pdf, figures/ch08/nu_eff.pdf, results/ch08/26_distribution.tex
"""
import sys, pathlib, importlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, SERIES, theory_line, rng_for
import lib_masks as lm

setup()
L = 767
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"
z = np.load(DATA / "masks.npz")
mc = importlib.import_module("23_mc_masked").load_all()
ell = np.arange(L + 1)
NSIM = mc["sky"].shape[0]
nums = {}
KEYS = ["full", "cap70", "cap30", "cap10", "cap10apo", "band20", "holes"]
NAMES = {"full": "Full", "cap10": "CapTen", "cap10apo": "CapTenApo", "cap30": "CapThirty",
         "cap70": "CapSeventy", "band20": "Band", "band20apo": "BandApo", "holes": "Holes"}
LABEL = {"full": "full sky", "cap10": "cap 0.1", "cap10apo": "cap 0.1 tapered", "cap30": "cap 0.3",
         "cap70": "cap 0.7", "band20": "band", "band20apo": "band tapered", "holes": "holes"}


def moments(x):
    m = x.mean(0)
    v = x.var(0, ddof=1)
    s = stats.skew(x, axis=0, bias=False)
    return m, v, s


def split_ks(v, gauss):
    """KS distance with the model fitted on the first half of v and tested on the second half."""
    h = v.size // 2
    m_f = v[:h].mean()
    y_f, y_t = v[:h] / m_f, v[h:] / m_f
    if gauss:
        return stats.kstest(y_t, stats.norm(1, y_f.std(ddof=1)).cdf).statistic
    nu_f = 2.0 / y_f.var(ddof=1)
    return stats.kstest(y_t * nu_f, stats.chi2(nu_f).cdf).statistic


# ---------------------------------------------------------------- histograms
nu_cases = []                                            # nu_eff of every tested (mask, l)
LS = [2, 10, 50]
fig, axs = plt.subplots(3, 3, figsize=(7.8, 6.2))
for i, k in enumerate(["full", "cap30", "cap10"]):
    x = mc["pcl_" + k].astype(float)
    for j, l in enumerate(LS):
        ax = axs[i, j]
        y = x[:, l] / x[:, l].mean()
        nu = 2.0 / y.var(ddof=1)
        hi = np.quantile(y, 0.995)
        bins = np.linspace(0, max(hi, 1.5), 40)
        ax.hist(y, bins=bins, density=True, color=SERIES[0], alpha=0.45, lw=0)
        g = np.linspace(1e-4, bins[-1], 400)
        ax.plot(g, stats.chi2.pdf(g * nu, nu) * nu, color=SERIES[1], lw=1.3, label=r"$\chi^2_\nu/\nu$")
        ax.plot(g, stats.norm.pdf(g, 1, y.std(ddof=1)), color="k", ls="--", lw=1.0, label="Gaussian")
        ax.set_yticks([])
        ax.text(0.03 if j == 2 else 0.97, 0.92, f"$\\ell={l}$\n$\\nu_{{\\rm eff}}={nu:.1f}$", transform=ax.transAxes,
                ha="left" if j == 2 else "right", va="top", fontsize=8)
        if j == 0:
            ax.set_ylabel(LABEL[k], fontsize=9)
        if i == 2:
            ax.set_xlabel(r"$\widetilde C_\ell/\langle\widetilde C_\ell\rangle$")
        if i == 0 and j == 0:
            ax.legend(fontsize=7, loc="center right")
        below = np.mean(y < 1.0)
        nums[f"EightBbelow{NAMES[k]}{['Two', 'Ten', 'Fifty'][j]}"] = f"{below:.2f}"
        nums[f"EightBnueff{NAMES[k]}{['Two', 'Ten', 'Fifty'][j]}"] = f"{nu:.1f}"
        # KS tests: fit mean and nu_eff on the first half of the skies, test on the other half,
        # so the model is not tuned to the very skies it is tested on
        ks = split_ks(x[:, l], gauss=False)
        ksg = split_ks(x[:, l], gauss=True)
        nu_cases.append(nu)
        nums[f"EightBks{NAMES[k]}{['Two', 'Ten', 'Fifty'][j]}"] = f"{ks:.3f}"
        nums[f"EightBksg{NAMES[k]}{['Two', 'Ten', 'Fifty'][j]}"] = f"{ksg:.3f}"
        print(f"{k:6s} l={l:3d}: nu_eff={nu:6.1f}  P(below mean)={below:.3f}  KS chi2={ks:.3f}  KS gauss={ksg:.3f}")
savefig(fig, "ch08", "cl_histograms")
nums["EightBksNtest"] = NSIM - NSIM // 2
# 5% critical value of the split KS distance. The fitted parameters carry noise of their own, so the
# textbook 1.36/sqrt(n) is too small. Calibrate it by running the same fit-half/test-half procedure on
# NBOOT samples of NSIM draws from the model itself: a scaled chi^2 with each tested nu, and a Gaussian
# (location-scale, so one Gaussian calibration serves all cases). Quote the largest, the safe side.
NBOOT = 300
rng_b = rng_for("ch08", "26_distribution_kscrit")
crit = [np.quantile([split_ks(rng_b.chisquare(nu_c, NSIM), gauss=False) for _ in range(NBOOT)], 0.95)
        for nu_c in nu_cases]
crit.append(np.quantile([split_ks(rng_b.normal(1.0, 0.1, NSIM), gauss=True) for _ in range(NBOOT)], 0.95))
nums["EightBksCrit"] = f"{max(crit):.3f}"
nums["EightBksCritMin"] = f"{min(crit):.3f}"                 # how much the value varies between models
nums["EightBksNboot"] = NBOOT
print("split-KS 5% critical values:", np.round(crit, 3), f"(textbook {1.36 / np.sqrt(NSIM - NSIM // 2):.3f})")

# ---------------------------------------------------------------- nu_eff and skewness against l
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
for k, c in zip(KEYS, SERIES):
    w = z["mask_" + k].astype(float)
    fsky, wi = lm.mask_moments(w)
    x = mc["pcl_" + k].astype(float)
    m, v, s = moments(x)
    nu = 2 * m ** 2 / v
    pred = (2 * ell + 1) * fsky * wi[2] ** 2 / wi[4]
    r = nu / pred
    sel = (ell >= 2) & (ell <= 600)
    axs[0].semilogx(ell[sel], r[sel], color=c, lw=0.8, label=LABEL[k])
    eb = np.unique(np.round(np.geomspace(2, 601, 22)).astype(int))     # log bins: average the noisy skewness
    sk = np.array([np.mean((s / np.sqrt(8 / nu))[a:b]) for a, b in zip(eb[:-1], eb[1:])])
    axs[1].semilogx(np.sqrt(eb[:-1] * (eb[1:] - 1)), sk, "o-", ms=2.5, color=c, lw=0.9)
    n = NAMES[k]
    for l, tag in [(2, "Two"), (10, "Ten"), (100, "Hundred"), (500, "Fivehundred")]:
        nums[f"EightBnuratio{n}{tag}"] = f"{r[l]:.2f}"
    nums[f"EightBnuratioHigh{n}"] = f"{np.median(r[300:601]):.2f}"
    print(f"{k:9s} nu_MC / nu_Hivon at l=2,10,100,500: {r[2]:.2f} {r[10]:.2f} {r[100]:.2f} {r[500]:.2f};"
          f" median 300-600 {np.median(r[300:601]):.3f}")
axs[0].axhline(1, color="0.4", lw=0.6)
axs[0].set_ylim(0, 9.5)
axs[0].set_xlabel(r"multipole $\ell$")
axs[0].set_ylabel(r"$\nu_{\rm eff}\,/\,[(2\ell+1)f_{\rm sky}w_2^2/w_4]$")
axs[0].set_title("(a) effective number of modes", fontsize=9)
leg = axs[0].legend(fontsize=7, ncol=3, loc="upper center")
for ln in leg.get_lines():
    ln.set_linewidth(1.8)
axs[1].axhline(1, color="0.4", lw=0.6)
axs[1].set_ylim(0, 2)
axs[1].set_xlabel(r"multipole $\ell$")
axs[1].set_ylabel(r"skewness / $\sqrt{8/\nu_{\rm eff}}$")
axs[1].set_title(r"(b) skewness against the $\chi^2_\nu$ value", fontsize=9)
fig.subplots_adjust(wspace=0.4)
savefig(fig, "ch08", "nu_eff")
save_numbers("ch08", "26_distribution", nums)
