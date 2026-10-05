"""04_spectra_budget.py -- the signal and noise budget of the toy experiment, and the exact law of C_hat.

Question: at which multipoles is the experiment limited by cosmic variance and at which by
noise, how large is the error bar on C_l, and what does the exact sampling distribution of the
debiased estimator look like (including its chance of coming out negative)?
Computes (analytically, no simulation): C_l, T_l^2 C_l, N_l and N_l/T_l^2 for the experiment of
lib_cmbsim; the fractional error sqrt(Var)/C_l for cosmic variance alone, with noise, and with
Knox's f_sky; the shifted, scaled chi^2 density of C_hat_l / C_l at three multipoles.
Writes: figures/ch08/budget.pdf, figures/ch08/debiased_pdf.pdf, results/ch08/04_spectra_budget.tex
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
L = exp.lmax_sim
ell = np.arange(L + 1)
cl = cl_all[: L + 1]
T2 = exp.transfer(L) ** 2
nl = exp.nl(L)
dfac = ell * (ell + 1) / (2 * np.pi)
leq = cs.ell_equality(cl_all, exp)

# ------------------------------------------------------------ budget figure
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
g = ell >= 2
axs[0].semilogy(ell[g], (dfac * cl)[g], color=SERIES[0], label=r"sky $C_\ell$")
axs[0].semilogy(ell[g], (dfac * T2 * cl)[g], color=SERIES[2], label=r"observed $T_\ell^2C_\ell$")
axs[0].semilogy(ell[g], (dfac * nl)[g], color=SERIES[1], label=r"noise $N_\ell$")
axs[0].semilogy(ell[g], (dfac * nl / T2)[g], color=SERIES[1], ls="--", label=r"$N_\ell/T_\ell^2$")
axs[0].axvline(leq, color="0.5", ls=":", lw=1)
axs[0].axvline(exp.lmax, color="0.75", ls="-", lw=0.8)
axs[0].set_ylim(1, 1e6)
axs[0].set_xlabel(r"$\ell$")
axs[0].set_ylabel(r"$\ell(\ell+1)X_\ell/2\pi$ [$\mu$K$^2$]")
axs[0].legend(loc="upper left", ncol=2, fontsize=7)

l2 = np.arange(2, exp.lmax + 1)
cv = np.sqrt(2 / (2 * l2 + 1))
wn = np.sqrt(cs.var_fullsky(cl_all, exp)[2:]) / cl[2: exp.lmax + 1]
kn = np.sqrt(cs.var_knox(cl_all, exp, fsky=0.5)[2:]) / cl[2: exp.lmax + 1]
axs[1].loglog(l2, cv, color=SERIES[0], label="cosmic variance only")
axs[1].loglog(l2, wn, color=SERIES[1], label="with beam and noise")
axs[1].loglog(l2, kn, color=SERIES[3], ls="-.", label=r"Knox, $f_{\rm sky}=0.5$")
axs[1].axvline(leq, color="0.5", ls=":", lw=1)
axs[1].set_xlabel(r"$\ell$")
axs[1].set_ylabel(r"$\sigma(\widehat C_\ell)/C_\ell$")
axs[1].legend(loc="upper center", fontsize=7.5)
fig.tight_layout()
savefig(fig, "ch08", "budget")

# ------------------------------------------------------------ exact law of the debiased estimator
SHOW = [10, leq, 700]                 # 700 lies beyond the analysed band: a what-if
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6))
pneg = {}
for ax, l in zip(axs, SHOW):
    nu = 2 * l + 1
    s_obs = T2[l] * cl[l] + nl[l]                     # variance of each observed a_lm
    # C_hat/C = (s_obs chi2_nu/nu - N)/(T^2 C)  =  A x + B  with x ~ chi2_nu
    A = s_obs / (nu * T2[l] * cl[l]); B = -nl[l] / (T2[l] * cl[l])
    sd = np.sqrt(2 / nu) * s_obs / (T2[l] * cl[l])
    y = np.linspace(1 - 4.2 * sd, 1 + 4.5 * sd, 600)
    pdf = stats.chi2.pdf((y - B) / A, nu) / A
    pneg[l] = stats.chi2.cdf(-B / A, nu)
    ax.plot(y, pdf, color=SERIES[0], label=r"exact (shifted $\chi^2$)")
    ax.plot(y, stats.norm.pdf(y, 1, sd), color="0.45", ls=":", lw=1.2, label="Gaussian")
    ax.fill_between(y, 0, pdf, where=y < 0, color=SERIES[1], alpha=0.35)
    ax.axvline(1, color="0.6", lw=0.8)
    ax.set_title(rf"$\ell={l}$" + (" (beyond the band)" if l > exp.lmax else ""))
    ax.set_xlabel(r"$\widehat C_\ell/C_\ell$")
    ax.set_yticks([])
axs[0].legend(loc="upper right", fontsize=6.5, bbox_to_anchor=(1.02, 1.0), handlelength=1.2)
fig.tight_layout()
savefig(fig, "ch08", "debiased_pdf")

L5 = exp.lmax
save_numbers("ch08", "04_spectra_budget", {
    "EightALeq": leq,
    "EightANoverSlmax": f"{nl[L5] / (T2[L5] * cl[L5]):.1f}",
    "EightANoverSthree": f"{nl[300] / (T2[300] * cl[300]):.2f}",
    "EightAFracErrCVlmax": f"{100 * np.sqrt(2 / (2 * L5 + 1)):.1f}",
    "EightAFracErrNlmax": f"{100 * wn[-1]:.0f}",
    "EightAFracErrNthree": f"{100 * wn[300 - 2]:.1f}",
    "EightAFracErrCVthree": f"{100 * cv[300 - 2]:.1f}",
    "EightAPnegSeven": f"{100 * pneg[700]:.0f}",
    "EightANoverSseven": f"{nl[700] / (T2[700] * cl[700]):.0f}",
    "EightAFracErrNseven": f"{np.sqrt(2 / 1401) * (1 + nl[700] / (T2[700] * cl[700])):.2f}",
    "EightAPnegLeq": f"{pneg[leq]:.0e}".replace("e-", r"\times10^{-") + "}",
})
print("leq", leq, "N/TC at 512", nl[L5] / (T2[L5] * cl[L5]), "P(neg)", pneg)
