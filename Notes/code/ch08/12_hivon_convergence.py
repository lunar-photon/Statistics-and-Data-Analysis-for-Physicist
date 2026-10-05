"""12_hivon_convergence.py -- Hivon et al. (2002), sections 3 and 4.3, in the full-sky limit.

Question: MASTER calibrates three things by Monte Carlo -- the transfer function F_l from
noise-free skies, the noise power N_l from noise-only maps, and the error bar Delta C_b from
signal+noise skies -- and Hivon et al. give the rate at which each converges (their eqs. 19,
38-42) and the analytic error bar the simulations should reproduce (eqs. 16, 36).  Do our
full-sky simulations obey the same laws, once the sky fraction f_sky = 1.82 % of the Boomerang
test is put back in?
Computes, from the 2000 skies of 05_mc_run (no new simulation), in bins of Delta l = 50:
  * the ratio of the Monte Carlo error bar to Hivon's analytic rule (16) with nu_b = (2 l_b + 1) Delta l;
  * for groups of N = 25 ... 400 simulations: the relative scatter of the estimated transfer
    function, of the estimated noise power, of the estimated error bar, and the error that the
    noise estimate puts on C_hat relative to its statistical error -- each against Hivon's law;
  * the Kolmogorov-Smirnov p-value of a Gaussian (instead of the shifted chi^2) at l = 3.
Writes: figures/ch08/hivon_convergence.pdf, results/ch08/12_hivon_convergence.tex
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
N = exp.nl(L)
FSKY_B = 0.0182                     # the Boomerang region of Hivon et al., sec. 4.1
DL = 50
edges = np.arange(13, L + 2, DL)    # [13,63), ..., [463,513)
lb = 0.5 * (edges[:-1] + edges[1:] - 1)
nsim = sims["auto"].shape[0]

chat = (sims["auto"].astype(float) - N) / T2
chat_b = cs.bin_spectrum(chat, edges)
cl_b = cs.bin_spectrum(cl, edges)
nt_b = cs.bin_spectrum(N / T2, edges)

# ------------------------------------------------ eq. (16)/(36): analytic error bar vs Monte Carlo
nu_b = (2 * lb + 1) * DL
sd_rule = (cl_b + nt_b) * np.sqrt(2 / nu_b)
sd_mc = chat_b.std(0, ddof=1)
r16 = sd_mc / sd_rule
r16_err = 1 / np.sqrt(2 * (nsim - 1))

# ------------------------------------------------ convergence with the number of simulations
# Disjoint groups of n simulations give independent calibrations; their scatter is the Monte Carlo
# error of an n-simulation calibration.  On the full sky different multipoles are independent, so
# the scatter is measured per multipole, divided by its predicted value there, and the ratio is
# pooled over l = 50-512 (the noise-error ratio over the noise-dominated l = 363-512).  The ratio
# is then multiplied by the predicted value for a bin of Delta l = 50 at l ~ 400, the setting of
# Hivon et al.'s eqs. (38)-(42).  (With 2000 skies, pooling over a handful of binned values
# would leave only 5-20 groups per point, too few to measure a scatter reliably.)
i400 = int(np.searchsorted(edges, 400, side="right") - 1)      # bin [363,413) holds l = 400
lu = np.arange(50, L + 1)                                      # multipoles pooled
lnd = np.arange(363, L + 1)                                    # noise-dominated multipoles
sig = sims["sig"].astype(float)                                # noise-free, beamed skies
noi = sims["noise"].astype(float)                              # noise-only maps
nu = 2 * ell + 1
sd_l = chat.std(0, ddof=1)
nd_frac_l = (N / T2) / (cl + N / T2)
Ns = np.array([25, 50, 100, 200, 400])
meas = {k: [] for k in ("F", "N", "sd", "meth")}
errs = {k: [] for k in meas}
for n in Ns:
    G = nsim // n
    grp = lambda x: x[: G * n].reshape(G, n, -1)
    est = {"F": grp(sig[:, lu]).mean(1) / (T2 * cl)[lu],
           "N": grp(noi[:, lu]).mean(1) / N[0],
           "sd": grp(chat[:, lu]).std(1, ddof=1) / sd_l[lu],
           "meth": grp(noi[:, lnd]).mean(1) / T2[lnd] / sd_l[lnd]}
    pred = {"F": np.sqrt(2 / nu[lu] / n), "N": np.sqrt(2 / nu[lu] / n),
            "sd": np.full(lu.size, 1 / np.sqrt(2 * (n - 1))), "meth": nd_frac_l[lnd] / np.sqrt(n)}
    for k in meas:
        q = np.sqrt(np.mean((est[k].std(0, ddof=1) / pred[k]) ** 2))      # measured / predicted
        meas[k].append(q); errs[k].append(q / np.sqrt(2 * (G - 1) * pred[k].size))
meas = {k: np.array(v) for k, v in meas.items()}
errs = {k: np.array(v) for k, v in errs.items()}

# the laws at l ~ 400 (or in the last bin): Hivon's printed coefficients (f_sky = 1.82 %) and ours
l4 = lb[i400]
law_F_full = np.sqrt(2 / ((2 * lb[i400] + 1) * DL)) / np.sqrt(Ns)                                       # eq. (19) with f_sky = 1
law_sd = 1 / np.sqrt(2 * (Ns - 1))                                          # sampling error of a std
nd_frac = (nt_b / (cl_b + nt_b))[-1]
law_meth = nd_frac / np.sqrt(Ns)                                            # binned version of 8a:eq:nsim-noise
hiv_F = 0.75e-2 * np.sqrt(100 / Ns) * np.sqrt(400 / l4)                     # eq. (38), a = 0.6-0.9
hiv_N = 0.6e-2 * np.sqrt(100 / Ns) * np.sqrt(400 / l4) * np.sqrt(50 / DL)   # eq. (39)
hiv_sd = 0.1 * np.sqrt(100 / Ns)                                            # eqs. (40), (42)
resc = np.sqrt(FSKY_B)                                                      # f_sky enters as 1/sqrt(f_sky)
for k, law in [("F", law_F_full), ("N", law_F_full), ("sd", law_sd), ("meth", law_meth)]:
    meas[k + "_abs"] = meas[k] * law; errs[k + "_abs"] = errs[k] * law

# the exact error of a bin average, sqrt(sum Var_l)/Delta l, for comparison with the rule (16)
sd_exact = np.sqrt(cs.bin_spectrum(cs.var_fullsky(cl_all, exp), edges) / DL)
r_exact = sd_mc / sd_exact

fig, axs = plt.subplots(1, 2, figsize=(7.8, 4.0))
for k, c, mk, lab, sh in [("F", SERIES[2], "^", r"transfer $\delta F/F$", 0.92), ("N", SERIES[1], "s", r"noise $\delta N/N$", 1.08)]:
    axs[0].errorbar(Ns * sh, meas[k + "_abs"], errs[k + "_abs"], fmt=mk, ms=4, color=c, lw=1, label=f"{lab}, our MC")
theory_line(axs[0], Ns, law_F_full, label=r"$\sqrt{2/[(2\ell+1)\Delta\ell\,N]}$")
axs[0].loglog(Ns, hiv_N, color=SERIES[1], ls=":", lw=1.4, label=r"Hivon (39), $f_{\rm sky}=1.8\%$")
axs[0].loglog(Ns, hiv_F, color=SERIES[2], ls=":", lw=1.4, label=r"Hivon (38), $f_{\rm sky}=1.8\%$")
axs[0].loglog(Ns, law_F_full / resc, color="0.45", lw=1.0, label=r"ours $/\sqrt{0.018}$")
axs[0].set_xscale("log"); axs[0].set_yscale("log")
axs[0].set_xlabel(r"simulations per estimate $N$"); axs[0].set_ylabel(r"relative error at $\ell\approx400$")
axs[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
axs[1].errorbar(Ns, meas["sd_abs"], errs["sd_abs"], fmt="o", ms=4, color=SERIES[0], lw=1, label=r"$\delta(\Delta C)/\Delta C$, our MC")
axs[1].errorbar(Ns, meas["meth_abs"], errs["meth_abs"], fmt="D", ms=3.5, color=SERIES[3], lw=1, label=r"$\delta C_{\rm noise}/\Delta C$, our MC")
theory_line(axs[1], Ns, law_sd, label=r"$1/\sqrt{2(N-1)}$")
axs[1].loglog(Ns, law_meth, color=SERIES[3], lw=1.0, label=r"$\frac{N_\ell}{S_\ell+N_\ell}\,/\sqrt{N}$")
axs[1].loglog(Ns, hiv_sd, color="0.3", ls=":", lw=1.4, label=r"Hivon (40), (42): $0.1\sqrt{100/N}$")
axs[1].set_xscale("log"); axs[1].set_yscale("log")
axs[1].set_xlabel(r"simulations per estimate $N$"); axs[1].set_ylabel("relative error")
axs[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
for ax in axs:
    ax.xaxis.set_major_locator(FixedLocator(Ns)); ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in Ns]))
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
axs[1].yaxis.set_major_locator(FixedLocator([0.04, 0.07, 0.1, 0.2]))
axs[1].yaxis.set_major_formatter(FixedFormatter(["0.04", "0.07", "0.1", "0.2"]))
fig.tight_layout()
savefig(fig, "ch08", "hivon_convergence")

# Gaussian versus chi^2 at l = 3 (Hivon's Fig. 6 point, in its most extreme full-sky form)
l = 3; nu = 2 * l + 1
s_obs = T2[l] * cl[l] + N[l]
A = s_obs / (nu * T2[l] * cl[l]); B = -N[l] / (T2[l] * cl[l])
y = chat[:, l] / cl[l]
ks_chi2 = stats.kstest((y - B) / A, stats.chi2(nu).cdf).pvalue
ks_gauss = stats.kstest(y, stats.norm(1, np.sqrt(2 / nu) * s_obs / (T2[l] * cl[l])).cdf).pvalue

j100 = int(np.where(Ns == 100)[0][0])
hi = np.arange(1, len(lb))
save_numbers("ch08", "12_hivon_convergence", {
    "EightAHivRfirst": f"{r16[0]:.2f}",
    "EightAHivRsixteenMin": f"{r16[hi].min():.3f}",
    "EightAHivRsixteenMax": f"{r16[hi].max():.3f}",
    "EightAHivRsixteen": f"{r16[hi].mean():.3f}",
    "EightAHivRexact": f"{r_exact.mean():.3f}",
    "EightAHivRexactMin": f"{r_exact.min():.3f}",
    "EightAHivRexactMax": f"{r_exact.max():.3f}",
    "EightAHivRsixteenErr": f"{100 * r16_err:.1f}",
    "EightAHivNbins": len(lb),
    "EightAHivLfour": f"{l4:.0f}",
    "EightAHivFfull": f"{100 * law_F_full[j100]:.3f}",
    "EightAHivFmc": f"{100 * meas['F_abs'][j100]:.3f}",
    "EightAHivNmc": f"{100 * meas['N_abs'][j100]:.3f}",
    "EightAHivFresc": f"{100 * law_F_full[j100] / resc:.2f}",
    "EightAHivFratio": f"{meas['F'][j100]:.2f}", "EightAHivFratioErr": f"{errs['F'][j100]:.2f}",
    "EightAHivNratio": f"{meas['N'][j100]:.2f}", "EightAHivNratioErr": f"{errs['N'][j100]:.2f}",
    "EightAHivSDratio": f"{meas['sd'][j100]:.2f}", "EightAHivSDratioErr": f"{errs['sd'][j100]:.2f}",
    "EightAHivMethratio": f"{meas['meth'][j100]:.2f}", "EightAHivMethratioErr": f"{errs['meth'][j100]:.2f}",
    "EightAHivSDmc": f"{100 * meas['sd_abs'][j100]:.1f}",
    "EightAHivSDth": f"{100 * law_sd[j100]:.1f}",
    "EightAHivMethmc": f"{100 * meas['meth_abs'][j100]:.1f}",
    "EightAHivMethth": f"{100 * law_meth[j100]:.1f}",
    "EightAHivNDfrac": f"{nd_frac:.2f}",
    "EightAHivKSchi": f"{ks_chi2:.2f}",
    "EightAHivKSgauss": f"{ks_gauss:.0e}".replace("e-0", "e-").replace("e-", r"\times10^{-") + "}",
})
print("r16", r16.round(3), "+-", r16_err)
for k in meas:
    print(k, meas[k].round(5), errs[k].round(5))
print("r_exact", r_exact.round(3))
print("laws F", law_F_full.round(5), "sd", law_sd.round(4), "meth", law_meth.round(4), "nd frac", nd_frac)
print("KS l=3 chi2", ks_chi2, "gauss", ks_gauss)
