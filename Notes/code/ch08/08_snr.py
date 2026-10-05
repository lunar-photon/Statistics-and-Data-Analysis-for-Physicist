"""08_snr.py -- signal-to-noise per multipole and cumulative, in theory and in the simulations.

Question: how well does each multipole measure C_l, how much does the whole spectrum up to l_max
say (the signal-to-noise of an overall amplitude A in C_l = A C_l^fid), and where does adding
multipoles stop helping?
Computes: SNR_l = sqrt((2l+1)/2) C_l/(C_l + N_l/T_l^2) and its cosmic-variance limit; the cumulative
SNR up to l_max = 767; the inverse-variance-weighted amplitude estimate A_hat on each of the
simulated skies of 05_mc_run, whose scatter must equal 1/SNR(<= 512).
Writes: figures/ch08/snr.pdf, results/ch08/08_snr.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
from camb_fiducial import load_fiducial
import lib_cmbsim as cs

setup()
ell_all, cl_all = load_fiducial()
exp = cs.Experiment()
LS = exp.lmax_sim
ell = np.arange(LS + 1)
snr_l = cs.snr_per_ell(cl_all, exp, LS)
snr_cum = cs.snr_cumulative(cl_all, exp, LS)
snr_cv = np.sqrt(np.cumsum(np.where(ell >= 2, (2 * ell + 1) / 2.0, 0.0)))
leq = cs.ell_equality(cl_all, exp)
L = exp.lmax

# amplitude estimator on the simulations: A_hat = sum w_l C_hat_l/C_l / sum w_l, w_l = C_l^2/Var_l
sims = cs.cached("mc_fullsky", lambda: (_ for _ in ()).throw(RuntimeError("run 05_mc_run.py first")))
cl = cl_all[: L + 1]
chat = (sims["auto"].astype(float) - exp.nl(L)) / exp.transfer(L) ** 2
w = np.zeros(L + 1); w[2:] = cl[2:] ** 2 / cs.var_fullsky(cl_all, exp)[2:]
A = (chat[:, 2:] / cl[2:] * w[2:]).sum(1) / w[2:].sum()
sigA_mc = A.std(ddof=1)
sigA_th = 1 / snr_cum[L]
nsim = A.size
# running amplitude for several l_max to compare with the cumulative curve
lm_pts = np.array([50, 100, 200, 300, 400, 512])
mc_pts = []
for lm in lm_pts:
    Am = (chat[:, 2:lm + 1] / cl[2:lm + 1] * w[2:lm + 1]).sum(1) / w[2:lm + 1].sum()
    mc_pts.append(1 / Am.std(ddof=1))

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
g = ell >= 2
axs[0].plot(ell[g], np.sqrt((2 * ell[g] + 1) / 2), color=SERIES[0], label="cosmic variance limit")
axs[0].plot(ell[g], snr_l[g], color=SERIES[1], label="with beam and noise")
axs[0].axvline(leq, color="0.5", ls=":", lw=1)
axs[0].axvline(L, color="0.75", lw=0.8)
axs[0].set_xlabel(r"$\ell$"); axs[0].set_ylabel(r"$C_\ell/\sigma(\widehat C_\ell)$")
axs[0].legend(fontsize=7.5, loc="upper left")
axs[1].plot(ell[g], snr_cv[g], color=SERIES[0], label="cosmic variance limit")
axs[1].plot(ell[g], snr_cum[g], color=SERIES[1], label="with beam and noise")
axs[1].plot(lm_pts, mc_pts, "o", ms=4, color=SERIES[2], label=r"$1/\sigma(\hat A)$, simulations")
axs[1].axvline(L, color="0.75", lw=0.8)
axs[1].set_xlabel(r"$\ell_{\max}$"); axs[1].set_ylabel(r"cumulative ${\rm SNR}(\leq\ell_{\max})$")
axs[1].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
savefig(fig, "ch08", "snr")

peak = int(ell[g][np.argmax(snr_l[g])])
l90 = int(ell[np.argmax(snr_cum >= 0.9 * snr_cum[-1])])
save_numbers("ch08", "08_snr", {
    "EightASnrPeakEll": peak,
    "EightASnrPeak": f"{snr_l[peak]:.1f}",
    "EightASnrEq": f"{snr_l[leq]:.1f}",
    "EightASnrCumLmax": f"{snr_cum[L]:.0f}",
    "EightASnrCVLmax": f"{snr_cv[L]:.0f}",
    "EightASnrCumSim": f"{snr_cum[-1]:.0f}",
    "EightASnrCVSim": f"{snr_cv[-1]:.0f}",
    "EightALNinety": l90,
    "EightASigAmc": f"{sigA_mc:.5f}",
    "EightASigAth": f"{sigA_th:.5f}",
    "EightASigAerr": f"{100 / np.sqrt(2 * (nsim - 1)):.1f}",
    "EightAMeanA": f"{A.mean():.5f}",
    "EightAMeanAerr": f"{sigA_mc / np.sqrt(nsim):.5f}",
})
print("peak", peak, snr_l[peak], "cum 512", snr_cum[L], "cv", snr_cv[L], "cum 767", snr_cum[-1], "l90", l90,
      "sigA mc/th", sigA_mc, sigA_th, "meanA", A.mean())
