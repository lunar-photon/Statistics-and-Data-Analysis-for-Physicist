"""25_master.py -- undoing the mask: the naive f_sky rescaling, the per-multipole inversion and MASTER.

Question: how biased is the quick estimator  C_l ~ (C~_l - N~_l) / (mean(W^2) T_l^2), why can the
coupling matrix not simply be inverted for a small patch, and does the binned MASTER estimator
(Hivon et al. 2002, eq. 26) remove the bias?  What does a filter on the map add (the transfer
function F_l, Hivon eq. 18) and how well do 500 simulations measure it (Hivon eq. 19)?
Computes, from the 2000 skies of 23_mc_masked.py and the matrices of 22_coupling.py:
  * singular values of M_ll' (l, l' in 2..767) for each mask: the near-null modes of a cut sky;
  * the naive estimator and its mean bias, per mask;
  * the binned MASTER estimator (bins of 20 from l = 2), its mean against the bin average of the
    input D_l and against the exact bandpower-window prediction (Alonso et al. 2019, eq. 19);
  * for the 10 per cent cap with the toy ring filter: F_l from the first 500 signal-only skies,
    then MASTER with and without F on the other 1500 skies.
Writes: figures/ch08/singular.pdf, figures/ch08/naive_master.pdf, figures/ch08/transfer.pdf,
        data/ch08/master.npz, results/ch08/25_master.tex
"""
import sys, pathlib, importlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, theory_line
from camb_fiducial import load_fiducial
import lib_cmbsim as cs
import lib_masks as lm

setup()
L = 767
LREP = 601                                   # report bins below this (well inside 3 nside - 1)
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"
z = np.load(DATA / "masks.npz")
cz = np.load(DATA / "coupling.npz")
mc = importlib.import_module("23_mc_masked").load_all()
NSIM = mc["sky"].shape[0]
ell = np.arange(L + 1)
_, cl = load_fiducial()
cl = cl[: L + 1]
exp = cs.Experiment()
T2 = exp.transfer(L) ** 2
Nw = exp.nl(L)[0]
dl_true = ell * (ell + 1) * cl / (2 * np.pi)
KEYS = ["full", "cap10", "cap10apo", "cap30", "cap70", "band20", "band20apo", "holes"]
NAMES = {"full": "Full", "cap10": "CapTen", "cap10apo": "CapTenApo", "cap30": "CapThirty",
         "cap70": "CapSeventy", "band20": "Band", "band20apo": "BandApo", "holes": "Holes"}
LABEL = {"full": "full sky", "cap10": "cap 0.1", "cap10apo": "cap 0.1 tapered", "cap30": "cap 0.3",
         "cap70": "cap 0.7", "band20": "band", "band20apo": "band tapered", "holes": "holes"}
nums = {"EightBLrep": LREP - 1}

MOM = {k: lm.mask_moments(z["mask_" + k].astype(float)) for k in KEYS}

# ---------------------------------------------------------------- 1. is M invertible?
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for k, c in zip(["cap10", "cap30", "cap70", "band20", "holes"], SERIES):
    fsky, wi = MOM[k]
    # M is not symmetric; G = M/(2l'+1) is (Efstathiou 2004, eq. 11).  Its eigenvalues, scaled by
    # (2l+1) on one side, are those of the symmetrised D^{1/2} G D^{1/2} with D = diag(2l+1).
    d = np.sqrt(2 * ell[2:] + 1.0)
    S = d[:, None] * (cz["M_" + k][2:, 2:] / (2 * ell[2:] + 1.0)[None, :]) * d[None, :]
    ev = np.sort(np.linalg.eigvalsh(0.5 * (S + S.T)))[::-1] / (fsky * wi[2])
    ax.semilogy(np.arange(1, ev.size + 1) / ev.size, np.clip(ev, 1e-18, None), color=c, lw=1.3,
                label=f"{LABEL[k]} ($f_{{\\rm sky}}={fsky:.2f}$)")
    small = np.mean(ev < 1e-3)
    nums[f"EightBnull{NAMES[k]}"] = f"{100 * small:.0f}"
    print(f"{k:8s} fraction of eigenvalues below 1e-3: {small:.3f}  (1 - fsky = {1 - fsky:.3f})")
ax.set_xlabel("eigenvalue rank / number of multipoles")
ax.set_ylabel(r"eigenvalue of $M/(f_{\rm sky}w_2)$")
ax.set_ylim(1e-17, 10)
ax.legend(fontsize=8, loc="lower left")
savefig(fig, "ch08", "singular")

# ---------------------------------------------------------------- 2. naive and MASTER estimators
edges = lm.bin_edges(2, L, 20)
P, Q = lm.binning_operators(edges, L)
lb = 0.5 * (edges[:-1] + edges[1:] - 1)
rep = edges[1:] <= LREP
Dbin_true = P @ cl                                      # bin average of D_l = l(l+1)C_l/2pi
res = {}
for k in KEYS:
    fsky, wi = MOM[k]
    M = cz["M_" + k]
    x = mc["pcl_" + k].astype(float) - Nw * fsky * wi[2]          # noise-debiased pseudo-C_l
    naive = x / (fsky * wi[2] * T2)                                # per sky, per l
    K = lm.master_matrix(M, T2, P, Q)
    Kinv = np.linalg.inv(K)
    Db = (Kinv @ (P @ x.T)).T                                      # MASTER, per sky, per bin
    Fwin = Kinv @ P @ (M * T2[None, :])                             # bandpower windows: <D_b> = F_bl C_l
    Dwin = Fwin @ cl
    sig = Db.std(0, ddof=1)
    mean = Db.mean(0)
    nb = P @ naive.T                                               # naive, binned the same way
    res[k] = dict(naive=nb.T, master=Db, sig=sig, Dwin=Dwin, Fwin=Fwin)
    rel_naive = nb.mean(1) / Dbin_true - 1
    pull_bin = (mean - Dbin_true) / (sig / np.sqrt(NSIM))
    pull_win = (mean - Dwin) / (sig / np.sqrt(NSIM))
    n = NAMES[k]
    nums[f"EightBnaiveBiasLow{n}"] = f"{100 * rel_naive[0]:+.0f}"
    nums[f"EightBnaiveBiasMax{n}"] = f"{100 * np.max(np.abs(rel_naive[rep])):.0f}"
    nums[f"EightBmasterPullBin{n}"] = f"{np.max(np.abs(pull_bin[rep])):.1f}"
    nums[f"EightBmasterPullWin{n}"] = f"{np.max(np.abs(pull_win[rep])):.1f}"
    nums[f"EightBmasterBiasSig{n}"] = f"{np.max(np.abs((mean - Dbin_true) / sig)[rep]):.2f}"
    nums[f"EightBcondK{n}"] = lm.tex_sci(np.linalg.cond(K))
    print(f"{k:10s} naive: first bin {100*rel_naive[0]:+6.1f}%, max |bias| {100*np.max(np.abs(rel_naive[rep])):5.1f}% | "
          f"MASTER: max |pull| vs bin-avg {np.max(np.abs(pull_bin[rep])):5.1f}, vs window {np.max(np.abs(pull_win[rep])):4.1f},"
          f" max |bias|/sigma {np.max(np.abs((mean - Dbin_true) / sig)[rep]):.3f}, cond K {np.linalg.cond(K):.2g}")

np.savez(DATA / "master.npz", edges=edges, lb=lb, Dbin_true=Dbin_true,
         **{f"master_{k}": res[k]["master"].astype(np.float32) for k in KEYS},
         **{f"naive_{k}": res[k]["naive"].astype(np.float32) for k in KEYS},
         **{f"Dwin_{k}": res[k]["Dwin"] for k in KEYS}, **{f"Fwin_{k}": res[k]["Fwin"] for k in KEYS})

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axs[0]
for k, c in zip(["cap10", "cap10apo", "cap30", "band20", "holes"], SERIES):
    ax.plot(lb[rep], 100 * (res[k]["naive"].mean(0)[rep] / Dbin_true[rep] - 1), "o-", ms=2.5, lw=0.9,
            color=c, label=LABEL[k])
ax.axhline(0, color="0.4", lw=0.6)
ax.set_xlabel(r"multipole $\ell$ (bins of 20)")
ax.set_ylabel(r"mean bias of the naive estimator [\%]")
ax.set_title(r"(a) naive: $(\widetilde C_\ell-\widetilde N_\ell)/(f_{\rm sky}w_2T_\ell^2)$", fontsize=9)
ax.legend(fontsize=7.5)
ax = axs[1]
for k, c in zip(["cap10", "cap10apo", "cap30", "band20", "holes"], SERIES):
    r = res[k]
    ax.plot(lb[rep], ((r["master"].mean(0) - Dbin_true) / r["sig"])[rep], "o-", ms=2.5,
            lw=0.9, color=c, label=LABEL[k])
    ax.plot(lb[rep], ((r["master"].mean(0) - r["Dwin"]) / r["sig"])[rep], ":", lw=0.9, color=c)
ax.axhspan(-2 / np.sqrt(NSIM), 2 / np.sqrt(NSIM), color="0.9", zorder=0)
ax.axhline(0, color="0.4", lw=0.6)
ax.set_xlabel(r"multipole $\ell$ (bins of 20)")
ax.set_ylabel(r"mean bias / error of one sky")
ax.set_title("(b) MASTER: against bin average (solid), windows (dotted)", fontsize=8.5)
ax.set_ylim(-0.35, 0.35)
savefig(fig, "ch08", "naive_master")

# ---------------------------------------------------------------- 3. transfer function of the ring filter
k = "cap10"
fsky, wi = MOM[k]
M = cz["M_" + k]
NF = 500                                                    # skies used to calibrate F and the noise
s_mean = mc["filt_s"][:NF].astype(float).mean(0)
S0 = T2 * cl
F0 = s_mean / (fsky * wi[2] * S0)                            # first guess, Hivon below eq. (18)
F0[:2] = 0.0


def running(x, half=25):
    """Running average over 2*half+1 multipoles (Hivon smooth over Delta l = 50)."""
    k_ = np.ones(2 * half + 1) / (2 * half + 1)
    xp = np.pad(x, half, mode="edge")
    return np.convolve(xp, k_, mode="valid")


F1 = F0 + (running(s_mean) - M @ (F0 * S0)) / (S0 * fsky * wi[2])   # one iteration of Hivon eq. (18)
F1[:2] = 0.0


def solve_F(smean):
    """F piecewise constant in the MASTER bins:  P <C~> = P M diag(T^2 C) R F_b  (R: 1 on the bin)."""
    R = (Q > 0).astype(float)
    A = P @ (M * S0[None, :]) @ R
    Fb = np.linalg.solve(A, P @ smean)
    return Fb, R @ Fb


Fb, F = solve_F(s_mean)
# the MC error of F_b from NF skies: scatter of F_b between disjoint groups of NF signal-only skies
NG = NSIM // NF
Fb_groups = np.array([solve_F(mc["filt_s"][g * NF:(g + 1) * NF].astype(float).mean(0))[0] for g in range(NG)])
dF_grp = Fb_groups.std(0, ddof=1)
dF_pred = np.sqrt(2.0 / ((2 * lb + 1) * np.diff(edges) * fsky)) / np.sqrt(NF) * Fb   # Hivon eq. (19)
nf = mc["filt_n"][:NF].astype(float).mean(0)                        # noise on the sky, from simulations
test = slice(NF, NSIM)
x = mc["filt_sn"][test].astype(float) - nf
Kf = lm.master_matrix(M, F * T2, P, Q)
K0 = lm.master_matrix(M, T2, P, Q)
D_with = (np.linalg.inv(Kf) @ (P @ x.T)).T
D_without = (np.linalg.inv(K0) @ (P @ x.T)).T
ntest = D_with.shape[0]
sig_w = D_with.std(0, ddof=1)
Dwin_f = np.linalg.inv(Kf) @ P @ (M * (F * T2)[None, :]) @ cl     # window prediction (with the F model)
pull_w = (D_with.mean(0) - Dbin_true) / (sig_w / np.sqrt(ntest))
rel_wo = D_without.mean(0) / Dbin_true - 1
rel_w = D_with.mean(0) / Dbin_true - 1
hr = (lb >= 100) & rep
ratio_err = np.sqrt(np.mean((dF_grp[hr] / dF_pred[hr]) ** 2))
# its own uncertainty: (NG-1) degrees of freedom per bin, pooled over the bins (correlated bins make it larger)
ratio_err_err = ratio_err / np.sqrt(2 * (NG - 1) * hr.sum())
nums.update({
    "EightBNF": NF, "EightBNtest": ntest,
    "EightBFbinOne": f"{Fb[0]:.2f}", "EightBFbinTwo": f"{Fb[1]:.2f}", "EightBFbinFive": f"{Fb[4]:.3f}",
    "EightBFbinHigh": f"{np.median(Fb[rep][10:]):.3f}",
    "EightBFoneHigh": f"{np.median(F1[300:601]):.2f}", "EightBFzeroHigh": f"{np.median(F0[300:601]):.2f}",
    "EightBdFratio": f"{ratio_err:.2f}", "EightBdFratioErr": f"{ratio_err_err:.2f}", "EightBdFgroups": NG,
    "EightBfiltWithoutLow": f"{100 * rel_wo[0]:+.0f}", "EightBfiltWithoutTwo": f"{100 * rel_wo[1]:+.1f}",
    "EightBfiltWithLow": f"{100 * rel_w[0]:+.1f}",
    "EightBfiltWithPull": f"{np.max(np.abs(pull_w[rep])):.1f}",
    "EightBfiltWithBiasSig": f"{np.max(np.abs((D_with.mean(0) - Dbin_true) / sig_w)[rep]):.2f}",
})
print("F_b (binned solve):", np.round(Fb[:8], 3), "... median high", np.median(Fb[rep][10:]))
print(f"one-step eq.18 F at l 300-600 (median): {np.median(F1[300:601]):.3f}; raw F0: {np.median(F0[300:601]):.3f}")
print(f"MASTER without F: first bins {np.round(100*rel_wo[:3],2)}%; with F: first {np.round(100*rel_w[:3],2)}%,"
      f" max |pull| {np.max(np.abs(pull_w[rep])):.2f}, max |bias|/sigma {np.max(np.abs((D_with.mean(0)-Dbin_true)/sig_w)[rep]):.3f}")
print(f"group scatter of F_b / eq.(19), l >= 100: {ratio_err:.2f} +- {ratio_err_err:.2f} ({NG} groups of {NF})")

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.1))
ax = axs[0]
ax.plot(ell[2:], F0[2:], color="0.75", lw=0.6, label=r"$F^{(0)}_\ell$: raw ratio")
ax.plot(ell[2:], F1[2:], color=SERIES[1], lw=1.0, label=r"one step of eq. (18)")
ax.errorbar(lb[rep], Fb[rep], yerr=10 * dF_pred[rep], fmt="o", ms=3, color=SERIES[0], lw=0.8,
            label=r"solved in bins ($\pm10\times$ eq. 19)")
ax.set_xscale("log")
ax.set_xlim(2, 700)
ax.set_ylim(0, 1.6)
ax.axhline(1, color="0.4", lw=0.5)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"transfer function $F_\ell$")
ax.set_title(f"(a) ring filter on the 10% cap, {NF} skies", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
ax = axs[1]
ax.plot(lb[rep], 100 * rel_wo[rep], "o-", ms=2.5, lw=0.9, color=SERIES[1], label="MASTER without $F_\\ell$")
ax.plot(lb[rep], 100 * rel_w[rep], "o-", ms=2.5, lw=0.9, color=SERIES[0], label="MASTER with $F_\\ell$")
ax.axhline(0, color="0.4", lw=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"multipole $\ell$ (bins of 20)")
ax.set_ylabel(r"mean bias [\%]")
ax.set_title(f"(b) {ntest} independent test skies", fontsize=9)
ax.legend(fontsize=7.5)
fig.subplots_adjust(wspace=0.4)
savefig(fig, "ch08", "transfer")
save_numbers("ch08", "25_master", nums)
