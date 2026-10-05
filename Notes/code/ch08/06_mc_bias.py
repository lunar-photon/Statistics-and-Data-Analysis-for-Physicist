"""06_mc_bias.py -- is the debiased, deconvolved estimator unbiased, and what does the cross-spectrum cost?

Question: averaged over the simulated skies of 05_mc_run, does C_hat = (C_obs - N)/T^2 recover the
input C_l?  How large is the bias if the noise is not subtracted, if the pixel window is
forgotten, or if the noise level is wrong by 2 percent?  Does the cross-spectrum of two half
maps need no noise subtraction, and how much larger is its scatter?
Computes: binned fractional biases with Monte Carlo error bars and a chi^2 against zero; the
noise power measured on noise-only maps; the ratio of cross to auto scatter against
sqrt([(S+2N)^2+S^2] / [2(S+N)^2]).
Writes: figures/ch08/bias.pdf, figures/ch08/cross.pdf, results/ch08/06_mc_bias.tex
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
cl = cl_all[: L + 1]
T2 = exp.transfer(L) ** 2
B2 = exp.bl(L) ** 2
N = exp.nl(L)
nsim = sims["auto"].shape[0]
auto = sims["auto"].astype(float)
cross = sims["cross"].astype(float)

est = {
    "debiased": (auto - N) / T2,
    "no noise subtraction": auto / T2,
    "pixel window forgotten": (auto - N) / B2,
    "noise 2% too high": (auto - 1.02 * N) / T2,
    "cross-spectrum": cross / T2,
}
edges = np.r_[np.arange(2, 503, 25), 513]
lc = 0.5 * (edges[:-1] + edges[1:] - 1)
clb = cs.bin_spectrum(cl, edges)


def frac_bias(x):
    xb = cs.bin_spectrum(x, edges)                       # (nsim, nbin)
    m = xb.mean(0)
    e = xb.std(0, ddof=1) / np.sqrt(nsim)
    return m / clb - 1, e / clb


fb = {k: frac_bias(v) for k, v in est.items()}
chi2_auto = float(np.sum((fb["debiased"][0] / fb["debiased"][1]) ** 2))
chi2_cross = float(np.sum((fb["cross-spectrum"][0] / fb["cross-spectrum"][1]) ** 2))
nb = len(lc)
p_auto = stats.chi2.sf(chi2_auto, nb)
p_cross = stats.chi2.sf(chi2_cross, nb)
mis_pred = -0.02 * cs.bin_spectrum(N / T2, edges) / clb    # predicted bias of the 2% error

nl_mc = sims["noise"].astype(float).mean(0)
nl_ratio = nl_mc[2:].mean() / N[0]
nl_ratio_err = sims["noise"][:, 2:].astype(float).mean(1).std(ddof=1) / np.sqrt(nsim) / N[0]

# ------------------------------------------------------------ figure: means and biases
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
d = ell * (ell + 1) / (2 * np.pi)
g = ell >= 2
axs[0].semilogy(ell[g], (d * auto.mean(0))[g], color=SERIES[0], lw=2.2, alpha=0.6, label=r"MC mean of $C^{\rm obs}_\ell$")
theory_line(axs[0], ell[g], (d * (T2 * cl + N))[g], label=r"$T_\ell^2C_\ell+N_\ell$")
axs[0].semilogy(ell[g], (d * T2 * cl)[g], color=SERIES[2], lw=1, label=r"$T_\ell^2C_\ell$")
axs[0].semilogy(ell[g], (d * N)[g], color=SERIES[1], lw=1, label=r"$N_\ell$")
axs[0].set_xlabel(r"$\ell$"); axs[0].set_ylabel(r"$\ell(\ell+1)X_\ell/2\pi$ [$\mu$K$^2$]")
axs[0].set_ylim(1, 1e5)
axs[0].legend(fontsize=7.5, loc="upper right")

for k, c, mk, dx in [("debiased", SERIES[0], "o", -3), ("cross-spectrum", SERIES[2], "s", 3)]:
    b, e = fb[k]
    axs[1].errorbar(lc + dx, 100 * b, 100 * e, fmt=mk, ms=3, color=c, lw=1, label=k)
b, e = fb["noise 2% too high"]
axs[1].errorbar(lc, 100 * b, 100 * e, fmt="^", ms=3, color=SERIES[1], lw=1, label="noise taken 2% too high")
theory_line(axs[1], lc, 100 * mis_pred, label=r"$-0.02\,N_\ell/(T_\ell^2C_\ell)$")
axs[1].axhline(0, color="0.5", lw=0.8)
axs[1].set_xlabel(r"$\ell$ (bins of 25)"); axs[1].set_ylabel(r"$\langle\widehat C_\ell\rangle/C_\ell-1$ [%]")
axs[1].legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "ch08", "bias")

# ------------------------------------------------------------ figure: scatter of cross vs auto
S = T2 * cl
r_th = np.sqrt(((S + 2 * N) ** 2 + S ** 2) / (2 * (S + N) ** 2))
r_mc = est["cross-spectrum"].std(0, ddof=1) / est["debiased"].std(0, ddof=1)
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.plot(ell[g], r_mc[g], color=SERIES[2], lw=0.8, alpha=0.8, label="Monte Carlo")
theory_line(ax, ell[g], r_th[g], label=r"$\sqrt{[(S+2N)^2+S^2]/[2(S+N)^2]}$")
ax.axhline(np.sqrt(2), color="0.6", ls=":", lw=1)
ax.text(10, np.sqrt(2) + 0.02, r"$\sqrt{2}$", fontsize=8, color="0.35")
ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\sigma({\rm cross})/\sigma({\rm auto})$")
ax.set_ylim(0.85, 1.55)
ax.legend(fontsize=7.5, loc="lower right")
savefig(fig, "ch08", "cross")

i512 = -1
save_numbers("ch08", "06_mc_bias", {
    "EightANbins": nb,
    "EightAChiAuto": f"{chi2_auto:.1f}",
    "EightAPAuto": f"{p_auto:.2f}",
    "EightAChiCross": f"{chi2_cross:.1f}",
    "EightAPCross": f"{p_cross:.2f}",
    "EightAMaxAbsBias": f"{100 * np.max(np.abs(fb['debiased'][0])):.2f}",
    "EightATypBiasErr": f"{100 * np.median(fb['debiased'][1]):.2f}",
    "EightANoSubBiasLast": f"{100 * fb['no noise subtraction'][0][i512]:.0f}",
    "EightANoSubBiasFirst": f"{100 * fb['no noise subtraction'][0][0]:.2f}",
    "EightAPixForgotLast": f"{100 * fb['pixel window forgotten'][0][i512]:.0f}",
    "EightAMisLast": f"{100 * fb['noise 2% too high'][0][i512]:.1f}",
    "EightAMisPredLast": f"{100 * mis_pred[i512]:.1f}",
    "EightANlRatio": f"{nl_ratio:.4f}",
    "EightANlRatioErr": f"{nl_ratio_err:.4f}",
    "EightACrossRatioLmax": f"{r_th[L]:.2f}",
    "EightACrossRatioMCLast": f"{r_mc[edges[-2]:].mean():.2f}",
})
print("chi2 auto", chi2_auto, p_auto, "cross", chi2_cross, p_cross, "nl ratio", nl_ratio, nl_ratio_err)
print({k: (100 * v[0][[0, -1]]).round(2) for k, v in fb.items()})
