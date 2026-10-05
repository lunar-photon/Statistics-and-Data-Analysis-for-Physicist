"""45_planck_like.py -- Planck 2018 (ln 10^10 A_s, n_s) from a Planck-like simulated temperature spectrum.

Question: Planck 2018 VI (Table 2, TT+lowE) reports ln(10^10 A_s) = 3.040 +- 0.016 and
n_s = 0.9626 +- 0.0057, with a correlation of +0.03 between them (official chains' covariance).
Can our sampler, our emulator and a Planck-like simulated TT spectrum reproduce these widths,
and why do they change when parameters are fixed?
Data: one simulated spectrum, C_hat_l = (C_l + N_l) chi^2_nu / nu with nu = (2l+1) f_sky,
f_sky = 0.57 (the 143 GHz mask T57), N_l = 143-GHz-like noise (7.22', 33 muK arcmin),
2 <= l <= 2500.  Likelihood: the f_sky-scaled exact form, -2 ln L = sum nu [C_hat/(C+N) + ln(C+N)].
"lowE": a Gaussian prior on tau of width 0.0086, centred on a simulated measurement.
Runs:  P2  (ln A_s, n_s), the other four fixed at the truth;
       P3  (ln A_s, n_s, tau) with the tau prior;
       P6  all six LCDM parameters (H0 in place of theta_MC) with the tau prior (2nd-order emulator).
Also: the bias of the Gaussian approximation with a fixed fiducial covariance at l >= 30
(Planck 2018 V, eq. 20 and footnote 12, "0.1 sigma on n_s") and of L_Q, from 300 simulated spectra.
Writes: figures/ch09/planck_triangle.pdf, figures/ch09/planck_overlay.pdf, data/ch09/planck_chains.npz,
        results/ch09/45_planck_like.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from common import setup, savefig, save_numbers, rng_for, SERIES
import lib_mcmc9c as mc
import lib_cmbemu as ce

setup()
rng = rng_for("ch09", "45_planck_like")
nums = {}
t_start = time.time()
NOTES = pathlib.Path(__file__).resolve().parents[2]
FSKY, SIG_TAU, LMIN, LMAX = 0.57, 0.0086, 2, 2500
l = np.arange(LMAX + 1)
use = (l >= LMIN) & (l <= LMAX)
nu = (2 * l + 1.0) * FSKY
Nl = ce.planck_like_noise(LMAX)
TRUTH6 = ce.FID.copy()
C_true = ce.camb_tt(TRUTH6)                          # the data come from CAMB itself, not the emulator
chat = np.zeros(LMAX + 1)
chat[use] = (C_true[use] + Nl[use]) * rng.chisquare(nu[use]) / nu[use]
tau_obs = TRUTH6[5] + SIG_TAU * rng.standard_normal()   # the simulated "lowE" measurement
nums.update({"NineCPlFsky": FSKY, "NineCPlSigTau": SIG_TAU, "NineCPlTauObs": f"{tau_obs:.4f}",
             "NineCPlTauTrue": f"{TRUTH6[5]:.4f}", "NineCPlLmax": LMAX})

PUB = dict(A=3.040, sA=0.016, N=0.9626, sN=0.0057, rho=0.028, H0=66.88, sH0=0.92, omb=0.02212, somb=0.00022,
           omc=0.1206, somc=0.0021, tau=0.0522, stau=0.0080)
covfile = NOTES / "data" / "ch09" / "planck_covmats" / "base_plikHM_TT_lowl_lowE.covmat"
names_pl = open(covfile).readline()[1:].split()
Cpl = np.loadtxt(covfile)
ipl = [names_pl.index("logA"), names_pl.index("ns")]
Cpl2 = Cpl[np.ix_(ipl, ipl)]
nums["NineCPlPubRho"] = f"{Cpl2[0, 1] / np.sqrt(Cpl2[0, 0] * Cpl2[1, 1]):+.2f}"
nums["NineCPlPubCovSA"] = f"{np.sqrt(Cpl2[0, 0]):.4f}"
nums["NineCPlPubCovSN"] = f"{np.sqrt(Cpl2[1, 1]):.4f}"


def make_lp(names, order):
    em = ce.Emulator(order=order, names=names, lmax=LMAX)
    it = names.index("tau") if "tau" in names else None
    lo = em.fid - np.array([0.5, 0.1, 10, 0.003, 0.02, 0.06])[[ce.NAMES.index(n) for n in names]]
    hi = em.fid + np.array([0.5, 0.1, 10, 0.003, 0.02, 0.06])[[ce.NAMES.index(n) for n in names]]
    lo = np.where(np.array(names) == "tau", 0.01, lo)

    def lp(th):
        th = np.asarray(th)
        if np.any(th < lo) or np.any(th > hi):
            return -np.inf
        c = em.cl(th)[use] + Nl[use]
        out = -0.5 * np.sum(nu[use] * (chat[use] / c + np.log(c)))
        if it is not None:
            out -= 0.5 * ((th[it] - tau_obs) / SIG_TAU) ** 2
        return out
    return lp, em


def fisher(em, names):
    c0 = em.cl(em.fid)
    dC = em.D * c0[None, :]
    w = nu / 2 / (c0 + Nl) ** 2
    F = (dC[:, use] * w[use]) @ dC[:, use].T
    if "tau" in names:
        k = names.index("tau"); F[k, k] += 1 / SIG_TAU ** 2
    return F


RUNS = {"P2": (["lnAs", "ns"], 1), "P3": (["lnAs", "ns", "tau"], 1), "P6": (ce.NAMES, 2)}
NCH, NST, BURN = 4, 40000, 4000
res = {}
for key, (names, order) in RUNS.items():
    lp, em = make_lp(names, order)
    F = fisher(em, names)
    covF = np.linalg.inv(F)
    sd = np.sqrt(np.diag(covF))
    starts = em.fid + 4 * sd * rng.uniform(-1, 1, (NCH, len(names)))
    t0 = time.time()
    ch, _, acc = mc.run_chains(lp, starts, covF, NST, rng)
    secs = time.time() - t0
    post = ch[:, BURN:].reshape(-1, len(names))
    m, s, R = mc.summary(post)
    taus = [mc.tau_int(ch[0, BURN:, k]) for k in range(len(names))]
    tag = {"P2": "Two", "P3": "Three", "P6": "Six"}[key]
    res[key] = dict(names=names, post=post, m=m, s=s, R=R, covF=covF, rhat=mc.rhat(ch[:, BURN:]), acc=acc.mean())
    nums.update({f"NineCPl{tag}A": f"{m[0]:.4f}", f"NineCPl{tag}SA": f"{s[0]:.4f}",
                 f"NineCPl{tag}N": f"{m[1]:.4f}", f"NineCPl{tag}SN": f"{s[1]:.4f}",
                 f"NineCPl{tag}Rho": f"{R[0, 1]:+.2f}", f"NineCPl{tag}Acc": f"{acc.mean():.2f}",
                 f"NineCPl{tag}Rhat": f"{res[key]['rhat'].max():.4f}", f"NineCPl{tag}Tau": f"{max(taus):.0f}",
                 f"NineCPl{tag}Sec": f"{secs:.0f}",
                 f"NineCPl{tag}FSA": f"{np.sqrt(covF[0, 0]):.4f}", f"NineCPl{tag}FSN": f"{np.sqrt(covF[1, 1]):.4f}",
                 f"NineCPl{tag}FRho": f"{covF[0, 1] / np.sqrt(covF[0, 0] * covF[1, 1]):+.2f}"})
    if "tau" in names:
        k = names.index("tau")
        nums[f"NineCPl{tag}Tau"] = f"{max(taus):.0f}"
        nums[f"NineCPl{tag}T"] = f"{m[k]:.4f}"; nums[f"NineCPl{tag}ST"] = f"{s[k]:.4f}"
        nums[f"NineCPl{tag}RhoAT"] = f"{R[0, k]:+.2f}"
    if key == "P6":
        for k, n in enumerate(["A", "N", "H", "B", "C", "T"]):
            nums[f"NineCPlSixM{n}"] = f"{m[k]:.5g}"; nums[f"NineCPlSixS{n}"] = f"{s[k]:.2g}"
nums.update({"NineCPlNch": NCH, "NineCPlNst": NST, "NineCPlBurn": BURN,
             "NineCPlTruthA": f"{TRUTH6[0]:.4f}", "NineCPlTruthN": f"{TRUTH6[1]:.4f}",
             "NineCPlRatioSA": f"{res['P6']['s'][0] / PUB['sA']:.2f}",
             "NineCPlRatioSN": f"{res['P6']['s'][1] / PUB['sN']:.2f}",
             "NineCPlGainA": f"{res['P3']['s'][0] / res['P2']['s'][0]:.0f}",
             "NineCPlGainN": f"{res['P6']['s'][1] / res['P2']['s'][1]:.1f}"})
# the A_s e^{-2 tau} degeneracy: the combination ln A_s - 2 tau is measured much better than either
p3 = res["P3"]["post"]
comb = p3[:, 0] - 2 * p3[:, 2]
nums["NineCPlSdComb"] = f"{comb.std():.4f}"

# ---------------------------------------------------------------- the Gaussian approximation and n_s
NREP = 300
r2 = rng_for("ch09", "45_planck_like", stream=1)
em2 = ce.Emulator(order=1, names=["lnAs", "ns"], lmax=LMAX)
Cf = em2.cl(em2.fid) + Nl                              # fiducial covariance: 2 Cf^2 / nu
low = use & (l < 30)
high = use & (l >= 30)


def m2l_exact(th, ch_, sel=use):
    c = em2.cl(th)[sel] + Nl[sel]
    return np.sum(nu[sel] * (ch_[sel] / c + np.log(c)))


def m2l_gauss_fid(th, ch_):
    c = em2.cl(th)[high] + Nl[high]
    return np.sum(nu[high] / 2 * (ch_[high] - c) ** 2 / Cf[high] ** 2) + m2l_exact(th, ch_, low)


def m2l_Q(th, ch_):
    c = em2.cl(th)[high] + Nl[high]
    return np.sum(nu[high] / 2 * ((ch_[high] - c) / c) ** 2) + m2l_exact(th, ch_, low)


best = {k: np.empty((NREP, 2)) for k in ("exact", "gfid", "Q")}
for i in range(NREP):
    ch_ = np.zeros(LMAX + 1)
    ch_[use] = (C_true[use] + Nl[use]) * r2.chisquare(nu[use]) / nu[use]
    for k, f in (("exact", m2l_exact), ("gfid", m2l_gauss_fid), ("Q", m2l_Q)):
        best[k][i] = optimize.minimize(lambda th: f(th, ch_), em2.fid, method="Nelder-Mead",
                                       options=dict(xatol=1e-7, fatol=1e-9)).x
sN2 = res["P2"]["s"][1]
for k, tag in (("gfid", "Gfid"), ("Q", "Q")):
    d = (best[k][:, 1] - best["exact"][:, 1]) / sN2
    nums[f"NineCPlBias{tag}"] = f"{d.mean():+.3f}"
    nums[f"NineCPlBias{tag}Err"] = f"{d.std() / np.sqrt(NREP):.3f}"
    nums[f"NineCPlBias{tag}Sd"] = f"{d.std():.3f}"
nums["NineCPlBiasN"] = NREP
nums["NineCPlMinutes"] = f"{(time.time() - t_start) / 60:.1f}"
np.savez(NOTES / "data" / "ch09" / "planck_chains.npz", **{f"post_{k}": v["post"] for k, v in res.items()},
         chat=chat, tau_obs=tau_obs)

# ---------------------------------------------------------------- figure 1: triangle P2 / P3 / P6
names = [r"$\ln(10^{10}A_s)$", r"$n_s$"]
sets = [res["P2"]["post"][:, :2], res["P3"]["post"][:, :2], res["P6"]["post"][:, :2]]
fig, axs = mc.triangle(sets, names, [SERIES[0], SERIES[1], SERIES[2]],
                       labels=[r"2 free, others fixed", r"+ $\tau$ (lowE prior)", r"all 6 free"],
                       truths=TRUTH6[:2], size=2.6, filled=[True, False, False],
                       ranges=[(3.00, 3.14), (0.945, 0.985)])
axs[0, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.05, 1.0))
savefig(fig, "ch09", "planck_triangle")

# ---------------------------------------------------------------- figure 2: overlay with Planck 2018
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
ax = axs[0]
post6 = res["P6"]["post"]
shift = np.array([PUB["A"], PUB["N"]]) - res["P6"]["m"][:2]
mc.contour2d(ax, post6[:, 0] + shift[0], post6[:, 1] + shift[1], SERIES[2],
             label="ours, 6 free (moved to Planck's mean)")
mc.gauss_ellipse(ax, [PUB["A"], PUB["N"]], Cpl2, color="k", ls="--", label="Planck 2018 TT+lowE (chains' covariance)")
shift2 = np.array([PUB["A"], PUB["N"]]) - res["P2"]["m"][:2]
mc.contour2d(ax, res["P2"]["post"][:, 0] + shift2[0], res["P2"]["post"][:, 1] + shift2[1], SERIES[0],
             filled=False, label="ours, 2 free (moved likewise)")
ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.legend(fontsize=6.5, loc="lower left")
ax.set_xlim(2.98, 3.10); ax.set_ylim(0.940, 0.985)
ax.set_title(r"(a) $(\ln A_s, n_s)$: ours vs Planck", fontsize=9)
ax = axs[1]
labels = [r"$\ln 10^{10}A_s$", r"$n_s$", r"$H_0$", r"$\omega_b$", r"$\omega_c$", r"$\tau$"]
pub_s = [PUB["sA"], PUB["sN"], PUB["sH0"], PUB["somb"], PUB["somc"], PUB["stau"]]
x = np.arange(6)
ax.bar(x - 0.2, res["P6"]["s"] / pub_s, width=0.4, color=SERIES[2], label="ours, 6 free / Planck")
r2v = np.full(6, np.nan); r2v[:2] = res["P2"]["s"] / pub_s[:2]
ax.bar(x + 0.2, r2v, width=0.4, color=SERIES[0], label="ours, 2 free / Planck")
ax.axhline(1, color="k", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"$\sigma_{\rm ours}/\sigma_{\rm Planck}$"); ax.set_ylim(0, 1.4); ax.legend(fontsize=7, loc="upper left")
ax.set_title("(b) widths relative to Planck 2018 TT+lowE", fontsize=9)
fig.tight_layout()
savefig(fig, "ch09", "planck_overlay")
save_numbers("ch09", "45_planck_like", nums)
print(nums)
