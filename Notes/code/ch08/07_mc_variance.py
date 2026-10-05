"""07_mc_variance.py -- the scatter and the shape of the sampling distribution with noise.

Question: does the Monte Carlo variance of the debiased estimator equal 2(C_l + N_l/T_l^2)^2/(2l+1),
which part of it is cosmic variance and which is noise, and is the distribution of C_hat_l the
shifted, scaled chi^2 of the derivation (skewed at low l, nearly Gaussian at high l)?
Computes: per-l ratio Var_MC / Var_theory with its expected Monte Carlo spread; the three terms of
the variance; histograms of C_hat_l / C_l at four multipoles with Kolmogorov-Smirnov p-values.
Writes: figures/ch08/variance.pdf, figures/ch08/histograms.pdf, results/ch08/07_mc_variance.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, SERIES, theory_line
from camb_fiducial import load_fiducial
import lib_cmbsim as cs

setup()
ell_all, cl_all = load_fiducial()
exp = cs.Experiment()
sims = cs.cached("mc_fullsky", lambda: (_ for _ in ()).throw(RuntimeError("run 05_mc_run.py first")))
L = exp.lmax
ell = np.arange(L + 1)
nu = 2 * ell + 1
cl = cl_all[: L + 1]
T2 = exp.transfer(L) ** 2
N = exp.nl(L)
nsim = sims["auto"].shape[0]
chat = (sims["auto"].astype(float) - N) / T2
sky = sims["sky"].astype(float)

v_th = cs.var_fullsky(cl_all, exp)
v_mc = chat.var(0, ddof=1)
ratio = v_mc / v_th
# Monte Carlo spread of a sample variance: sqrt(2/(n-1) + kappa/n), kappa = 12/nu for chi^2_nu
spread = np.sqrt(2 / (nsim - 1) + 12 / (nu * nsim))
g = ell >= 2
mean_ratio = ratio[g].mean()
mean_ratio_err = np.sqrt(np.sum(spread[g] ** 2)) / g.sum()
z = (ratio[g] - 1) / spread[g]
chi2 = float(np.sum(z ** 2)); dof = int(g.sum())
frac_out = np.mean(np.abs(z) > 2)
cv_ratio = (sky.var(0, ddof=1)[g] / (2 * cl[g] ** 2 / nu[g])).mean()

# three terms: 2C^2/nu + 4 C (N/T^2)/nu + 2 (N/T^2)^2/nu
x = N / T2
terms = np.stack([2 * cl ** 2, 4 * cl * x, 2 * x ** 2]) / nu
fracs = terms / terms.sum(0)
l_half_noise = int(ell[g][np.argmax((fracs[1:].sum(0))[g] > 0.5)])

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
axs[0].plot(ell[g], ratio[g], color=SERIES[0], lw=0.6, alpha=0.8, label="Monte Carlo")
axs[0].fill_between(ell[g], 1 - 2 * spread[g], 1 + 2 * spread[g], color="0.85", label=r"expected $\pm2\sigma$")
edges = np.r_[np.arange(2, 503, 25), 513]
lc = 0.5 * (edges[:-1] + edges[1:] - 1)
axs[0].plot(lc, cs.bin_spectrum(ratio, edges), "o", ms=3.5, color=SERIES[1], label="bins of 25")
axs[0].axhline(1, color="k", ls="--", lw=1)
axs[0].set_ylim(0.8, 1.2)
axs[0].set_xlabel(r"$\ell$"); axs[0].set_ylabel(r"${\rm Var}_{\rm MC}/{\rm Var}_{\rm theory}$")
axs[0].legend(fontsize=7, loc="upper left", ncol=1)
lab = [r"cosmic $2C_\ell^2$", r"cross $4C_\ell N_\ell/T_\ell^2$", r"noise $2(N_\ell/T_\ell^2)^2$"]
axs[1].stackplot(ell[g], fracs[:, g], colors=[SERIES[0], SERIES[3], SERIES[1]], alpha=0.75, labels=lab)
axs[1].set_xlim(2, L); axs[1].set_ylim(0, 1)
axs[1].set_xlabel(r"$\ell$"); axs[1].set_ylabel("fraction of the variance")
axs[1].legend(fontsize=7, loc="center left")
fig.tight_layout()
savefig(fig, "ch08", "variance")

SHOW = [3, 30, 436, 512]
fig, axs = plt.subplots(1, 4, figsize=(7.8, 2.4))
ks = {}
for ax, l in zip(axs, SHOW):
    n = nu[l]
    s_obs = T2[l] * cl[l] + N[l]
    A = s_obs / (n * T2[l] * cl[l]); B = -N[l] / (T2[l] * cl[l])
    y = chat[:, l] / cl[l]
    ks[l] = stats.kstest((y - B) / A, stats.chi2(n).cdf).pvalue
    lo, hi = np.quantile(y, [0.0005, 0.9995])
    lo = max(lo, B) if l < 10 else lo
    ax.hist(y, bins=50, range=(lo, hi), density=True, color=SERIES[0], alpha=0.5)
    yy = np.linspace(lo, hi, 400)
    theory_line(ax, yy, stats.chi2.pdf((yy - B) / A, n) / A, label=r"shifted $\chi^2$")
    sd = np.sqrt(2 / n) * s_obs / (T2[l] * cl[l])
    ax.plot(yy, stats.norm.pdf(yy, 1, sd), color="0.45", ls=":", lw=1.1, label="Gaussian")
    ax.set_title(rf"$\ell={l}$, KS $p={ks[l]:.2f}$", fontsize=8.5)
    ax.set_xlabel(r"$\widehat C_\ell/C_\ell$"); ax.set_yticks([])
axs[0].legend(fontsize=6.5, loc="upper right")
fig.tight_layout()
savefig(fig, "ch08", "histograms")

skew3 = stats.skew(chat[:, 3] / cl[3])
save_numbers("ch08", "07_mc_variance", {
    "EightAVarRatio": f"{mean_ratio:.3f}",
    "EightAVarRatioErr": f"{mean_ratio_err:.3f}",
    "EightAVarChi": f"{chi2:.0f}",
    "EightAVarDof": dof,
    "EightAVarPval": f"{stats.chi2.sf(chi2, dof):.2f}",
    "EightAVarFracOut": f"{100 * frac_out:.1f}",
    "EightASpreadOne": f"{100 * spread[100]:.1f}",
    "EightACVRatio": f"{cv_ratio:.3f}",
    "EightALHalfNoiseVar": l_half_noise,
    "EightANoiseFracLmax": f"{100 * fracs[2, L]:.0f}",
    "EightACrossFracLmax": f"{100 * fracs[1, L]:.0f}",
    "EightAKSthree": f"{ks[3]:.2f}", "EightAKSthirty": f"{ks[30]:.2f}",
    "EightAKSleq": f"{ks[436]:.2f}", "EightAKSlmax": f"{ks[512]:.2f}",
    "EightASkewThree": f"{skew3:.2f}",
    "EightASkewThreeTh": f"{np.sqrt(8 / 7):.2f}",
})
print("var ratio", mean_ratio, "+-", mean_ratio_err, "chi2", chi2, dof, "out", frac_out, "ks", ks,
      "l half noise", l_half_noise, "cv ratio", cv_ratio, "skew3", skew3)
