"""30_boomerang.py -- repeating the MASTER tests of Hivon et al. (2002) and comparing with their numbers.

Question: with our own code and the laptop version of their experiment (29_boomerang_mc.py), do
we recover the published results: the apodisation factors w2^2/w4 (their Table 1), the binned
kernel K_bb' (their Fig. 3), the transfer function of a high-pass filter (their Fig. 2 and the
toy model of their App. B), an unbiased binned spectrum (their Fig. 5), Monte Carlo error bars
equal to the analytic ones (their eqs 35-36), and the scaled-chi^2 shape of the estimates
(their Fig. 6, with its nine published values of nu)?
Computes: M_ll' (l, l' <= 1300) of the three windows from 3j symbols; bins of 50 centred on
50, 100, ..., 1250; F_b solved in bins from the 252 signal-only skies; the noise on the sky from
the 450 noise-only skies; MASTER on the 1350 signal+noise skies; error bars, correlations,
distributions; the published numbers side by side.
Writes: data/ch08/boom_coupling.npz, figures/ch08/boom_kernel.pdf, figures/ch08/boom_spectrum.pdf,
        figures/ch08/boom_hist.pdf, figures/ch08/boom_overlay.pdf, results/ch08/30_boomerang.tex
"""
import sys, pathlib, importlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, SERIES, theory_line
import lib_masks as lm

setup()
bm = importlib.import_module("29_boomerang_mc")
DATA = bm.DATA
LAN, LSIM, PROF = bm.LAN, bm.LSIM, bm.PROFILES
ell = np.arange(LAN + 1)
cl = bm.cl[: LAN + 1]
T2 = bm.T[: LAN + 1] ** 2
mc = bm.load_all()
zw = np.load(DATA / "boom_window.npz")
nums = {}

# ---------------------------------------------------------------- published values (Hivon et al. 2002)
PUB_W = {"tophat": 1.0, "cosine": 0.514, "gauss": 0.223}                      # Table 1
PUB_NU = {"tophat": [18.2, 274.6, 1213.0], "cosine": [9.7, 141.7, 626.6],    # Fig. 6
          "gauss": [4.2, 61.4, 272.4]}
PUB_L = [50, 200, 750]
PUB_FSKY, DL = 0.0182, 50

# ---------------------------------------------------------------- windows and coupling
MOM = {p: lm.mask_moments(zw["w_" + p].astype(float)) for p in PROF}
path = DATA / "boom_coupling.npz"
if path.exists():
    MM = {p: np.load(path)["M_" + p] for p in PROF}
else:
    t0 = time.time()
    MM = {p: lm.coupling_matrix(zw["wl_" + p], LAN, LAN) for p in PROF}
    np.savez(path, **{"M_" + p: v for p, v in MM.items()})
    nums["EightBboomCouplingSec"] = f"{(time.time() - t0) / 3:.0f}"
    print(f"coupling matrices: {time.time() - t0:.0f} s")

edges = np.r_[2, np.arange(25, 1276, DL), LAN + 1]
P, Q = lm.binning_operators(edges, LAN)
lb = 0.5 * (edges[:-1] + edges[1:] - 1)
rep = (lb >= 45) & (lb <= 1250)                       # the bins centred on 50, 100, ..., 1250
iL = {l: int(np.argmin(np.abs(lb - l))) for l in PUB_L}
Dth = P @ cl                                                  # bin-averaged input D_b

# ---------------------------------------------------------------- transfer function, from signal-only skies
S0 = T2 * cl
R = (Q > 0).astype(float)
res = {}
for j, p in enumerate(PROF):
    fsky, wi = MOM[p]
    M = MM[p]
    smean = mc["s"][:, j].astype(float).mean(0)
    A = P @ (M * S0[None, :]) @ R
    Fb = np.linalg.lstsq(A[1:, 1:], (P @ smean)[1:], rcond=None)[0]      # first bin (l < 25) is filtered away
    Fb = np.r_[0.0, Fb]
    half = mc["s"].shape[0] // 2
    Fh = [np.r_[0.0, np.linalg.lstsq(A[1:, 1:], (P @ mc["s"][sl, j].astype(float).mean(0))[1:], rcond=None)[0]]
          for sl in (slice(0, half), slice(half, None))]
    # MC error of F_b from all NS skies: scatter between the chunks of the noise-free set
    nc = bm.N_CHUNK
    ns_c = mc["s"].shape[0] // nc
    Fc = np.array([np.r_[0.0, np.linalg.lstsq(A[1:, 1:], (P @ mc["s"][c * ns_c:(c + 1) * ns_c, j].astype(float).mean(0))[1:],
                                              rcond=None)[0]] for c in range(nc)])
    dFb = Fc.std(0, ddof=1) / np.sqrt(nc)
    F = R @ Fb
    nmean = mc["n"][:, j].astype(float).mean(0)                          # noise on the sky
    K = lm.master_matrix(M, F * T2, P, Q)
    Kr = K[1:, 1:]                                                       # drop the empty first bin
    Ki = np.linalg.inv(Kr)
    x = mc["sn"][:, j].astype(float) - nmean
    Db = (Ki @ (P @ x.T)[1:]).T
    Db = np.c_[np.zeros(len(Db)), Db]
    Nb = np.r_[0.0, Ki @ (P @ nmean)[1:]]                                # Hivon eq. (27)
    # the same noise estimate is subtracted from every sky: its own MC error is common to all of them
    Nbs = (Ki @ (P @ mc["n"][:, j].astype(float).T)[1:]).T
    dNb = np.r_[0.0, Nbs.std(0, ddof=1) / np.sqrt(len(Nbs))]
    nu = (2 * lb + 1) * np.diff(edges) * fsky * wi[2] ** 2 / wi[4] * Fb     # Hivon eq. (35)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig36 = (Dth + Nb) * np.sqrt(2.0 / nu)                           # Hivon eq. (36)
    Dwin = np.r_[0.0, Ki @ (P @ (M * (F * T2)[None, :]) @ cl)[1:]]      # bandpower-window prediction
    res[p] = dict(Fb=Fb, Fh=Fh, dFb=dFb, dNb=dNb, K=K, Ki=Ki, Db=Db, Nb=Nb, nu=nu, sig36=sig36, fsky=fsky, wi=wi, Dwin=Dwin)

# ---------------------------------------------------------------- numbers: ours against theirs
NSN = mc["sn"].shape[0]
NS = mc["s"].shape[0]
nums["EightBboomFsky"] = f"{100 * MOM['tophat'][0]:.2f}"
for p, tag in [("tophat", "Top"), ("cosine", "Cos"), ("gauss", "Gau")]:
    r = res[p]
    w = r["wi"][2] ** 2 / r["wi"][4]
    nums[f"EightBboomW{tag}"] = f"{w:.3f}"
    sig = r["Db"].std(0, ddof=1)
    ratio = sig / r["sig36"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (r["Db"].mean(0) - Dth) / (sig / np.sqrt(NSN))
        pullw = (r["Db"].mean(0) - r["Dwin"]) / (sig / np.sqrt(NSN))
    nums[f"EightBboomPullWin{tag}"] = f"{np.max(np.abs(pullw[rep])):.1f}"
    # errors shared by all NSN skies: F (from NS skies) and the noise bias (from NN skies)
    sem = sig / np.sqrt(NSN)
    Fsafe = np.where(r["Fb"] > 0, r["Fb"], 1.0)
    dcomF = r["dFb"] / Fsafe * Dth
    dcom = np.sqrt(dcomF**2 + r["dNb"] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pullt = (r["Db"].mean(0) - r["Dwin"]) / np.sqrt(sem**2 + dcom**2)
        nums[f"EightBboomPullF{tag}"] = f"{np.max((dcomF / sem)[rep]):.1f}"
        nums[f"EightBboomPullN{tag}"] = f"{np.max((r['dNb'] / sem)[rep]):.1f}"
    nums[f"EightBboomPullTot{tag}"] = f"{np.max(np.abs(pullt[rep])):.1f}"
    nums[f"EightBboomBiasSig{tag}"] = f"{np.max(np.abs((r['Db'].mean(0) - Dth) / sig)[rep]):.2f}"
    Rm = np.corrcoef(r["Db"][:, 1:], rowvar=False)
    off = np.abs(Rm - np.eye(len(Rm)))
    off_after = off[3:, 3:].max()                     # skip the first three reported bins
    nums[f"EightBboomRatio{tag}"] = f"{np.median(ratio[rep]):.2f}"
    nums[f"EightBboomRatioLow{tag}"] = f"{ratio[iL[50]]:.2f}"
    nums[f"EightBboomPull{tag}"] = f"{np.max(np.abs(pull[rep])):.1f}"
    nums[f"EightBboomOff{tag}"] = f"{100 * off_after:.0f}"
    nums[f"EightBboomOffFirst{tag}"] = f"{100 * off[:3].max():.0f}"
    for l, ltag in zip(PUB_L, ["Fifty", "Twohundred", "Sevenfifty"]):
        b = iL[l]
        nums[f"EightBboomNu{tag}{ltag}"] = f"{r['nu'][b]:.1f}"
        x = (r["Db"][:, b] - Dth[b]) / r["sig36"][b]
        nu_mc = 2 * (r["Db"][:, b].mean() + r["Nb"][b]) ** 2 / r["Db"][:, b].var(ddof=1)   # signal + noise on the sky
        nums[f"EightBboomNuMC{tag}{ltag}"] = f"{nu_mc:.1f}"
    print(f"{p:7s} w2^2/w4={w:.4f} (pub {PUB_W[p]}), sigma_MC/sigma_36 median {np.median(ratio[rep]):.3f}"
          f" (l=50: {ratio[iL[50]]:.2f}), max|pull| {np.max(np.abs(pull[rep])):.2f},"
          f" max off-diag corr after 3 bins {off_after:.3f}, nu(50,200,750) = "
          f"{[round(r['nu'][iL[l]], 1) for l in PUB_L]} pub {PUB_NU[p]}")
Ft = res["tophat"]["Fb"]
for l, ltag in zip(PUB_L, ["Fifty", "Twohundred", "Sevenfifty"]):
    b = iL[l]
    F_pub = PUB_NU["tophat"][PUB_L.index(l)] / ((2 * l + 1) * DL * PUB_FSKY)   # their F, from their nu
    lc = bm.M_CUT
    ls_ = np.arange(edges[b], edges[b + 1])
    F_b15 = np.mean(np.where(ls_ >= lc, 1 - 2 / np.pi * np.arcsin(np.minimum(lc / ls_, 1)), 0.0))
    nums[f"EightBboomF{ltag}"] = f"{Ft[b]:.3f}"
    nums[f"EightBboomFpub{ltag}"] = f"{F_pub:.2f}"
    nums[f"EightBboomFbfifteen{ltag}"] = f"{F_b15:.3f}"
    # their cosine and Gaussian nu predicted from their tophat nu and OUR w2^2/w4
    for p, tag in [("cosine", "Cos"), ("gauss", "Gau")]:
        wr = res[p]["wi"][2] ** 2 / res[p]["wi"][4]
        nums[f"EightBboomNuPred{tag}{ltag}"] = f"{PUB_NU['tophat'][PUB_L.index(l)] * wr:.1f}"
    print(f"l={l}: F ours {Ft[b]:.3f}, B15 {F_b15:.3f}, theirs (from nu) {F_pub:.3f}")
# MC error of F (their eq. 38: dF/F ~ a 1e-2 sqrt(100/N) sqrt(400/l))
dF = res["tophat"]["dFb"]                                       # sigma of F from all NS skies (chunk scatter)
sel = rep & (lb >= 200)
a_est = np.sqrt(np.mean((dF[sel] / Ft[sel] / (1e-2 * np.sqrt(100 / NS) * np.sqrt(400 / lb[sel]))) ** 2))
nums["EightBboomA"] = f"{a_est:.2f}"
# its own uncertainty: N_CHUNK - 1 degrees of freedom per bin, pooled over the bins
nums["EightBboomAErr"] = f"{a_est / np.sqrt(2 * (bm.N_CHUNK - 1) * sel.sum()):.2f}"
# a common error dF in F shifts every one of the NSN estimates the same way: the pull it should produce
sigT = res["tophat"]["Db"].std(0, ddof=1)
pull_F = (dF / Ft * Dth) / (sigT / np.sqrt(NSN))
nums["EightBboomPullF"] = f"{np.max(pull_F[rep]):.1f}"
nums["EightBboomdFpct"] = f"{100 * np.median((dF / Ft)[sel]):.1f}"
print(f"pull expected from the MC error of F: median {np.median(pull_F[rep]):.1f}, max {np.max(pull_F[rep]):.1f}")
print(f"eq. (38) coefficient a from our F: {a_est:.2f}  (published 0.6-0.9)")
nums["EightBboomNbins"] = int(rep.sum())

# ---------------------------------------------------------------- figure: the binned kernel (their Fig. 3, bottom)
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
r = res["tophat"]
K, Ki = r["K"][1:, 1:], r["Ki"]
lbr = lb[1:]
for ax, A_, ttl in [(axs[0], K, r"(a) $|K_{b_1b_2}|$"), (axs[1], Ki, r"(b) $|K^{-1}_{b_1b_2}|$")]:
    ax.semilogy(lbr, np.abs(np.diag(A_)), "s", ms=3.5, mfc="none", color="k", label=r"$\ell_{b_1}=\ell_{b_2}$")
    for l, c, m in [(200, SERIES[0], "^"), (700, SERIES[1], "D")]:
        b = int(np.argmin(np.abs(lbr - l)))
        row = A_[b]
        ax.semilogy(lbr, np.abs(row), m + "-", ms=3, lw=0.7, mfc="none", color=c, label=f"$\\ell_{{b_1}}={l}$")
        neg = row < 0
        ax.semilogy(lbr[neg], np.abs(row[neg]), m, ms=3, color=SERIES[2])
    ax.set_xlabel(r"$\ell_{b_2}$")
    ax.set_title(ttl, fontsize=9)
axs[0].legend(fontsize=7.5, loc="center right")
savefig(fig, "ch08", "boom_kernel")
nums["EightBboomKdiagTwo"] = lm.tex_sci(np.diag(K)[int(np.argmin(np.abs(lbr - 200)))])
nums["EightBboomKdiagTwelve"] = lm.tex_sci(np.diag(K)[int(np.argmin(np.abs(lbr - 1200)))])

# ---------------------------------------------------------------- figure: recovered spectrum and correlations (their Fig. 5)
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axs[0]
ax.plot(ell[2:], (ell * (ell + 1) * cl / (2 * np.pi))[2:], color="k", lw=0.8, label="input $D_\\ell$")
for j, (p, c) in enumerate(zip(PROF, SERIES)):
    r = res[p]
    off = (j - 1) * 9
    ax.errorbar(lb[rep] + off, r["Db"].mean(0)[rep], yerr=r["Db"].std(0, ddof=1)[rep], fmt="o", ms=2,
                lw=0.8, color=c, label=f"{p}: MC mean $\\pm$ MC rms")
    ax.plot(lb[rep] + off, (Dth + 0 * r["sig36"])[rep] + r["sig36"][rep], "_", color=c, ms=5)
    ax.plot(lb[rep] + off, Dth[rep] - r["sig36"][rep], "_", color=c, ms=5)
ax.plot(ell[30:], (ell * (ell + 1) * bm.NL / T2 / (2 * np.pi))[30:], color="0.5", ls="--", lw=0.8,
        label="noise $N_\\ell/T_\\ell^2$")
ax.set_ylim(0, 7500)
ax.set_xlim(0, 1300)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
ax.set_title(f"(a) MASTER on {NSN} skies (dashes: eq. 36)", fontsize=9)
ax.legend(fontsize=6.5, loc="upper right")
ax = axs[1]
Rm = np.corrcoef(res["tophat"]["Db"][:, 1:], rowvar=False)
im = ax.imshow(np.abs(Rm), origin="lower", cmap="Blues", vmin=0, vmax=0.3,
               extent=[lbr[0] - 25, lbr[-1] + 25, lbr[0] - 25, lbr[-1] + 25])
ax.set_xlabel(r"$\ell_{b_1}$")
ax.set_ylabel(r"$\ell_{b_2}$")
ax.set_title("(b) |correlation|, top-hat\n(diagonal = 1)", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
fig.subplots_adjust(wspace=0.35)
savefig(fig, "ch08", "boom_spectrum")

# ---------------------------------------------------------------- figure: distributions (their Fig. 6)
fig, axs = plt.subplots(3, 3, figsize=(7.8, 5.6), sharex=True)
for i, p in enumerate(PROF):
    r = res[p]
    for j, l in enumerate(PUB_L):
        ax = axs[i, j]
        b = iL[l]
        x = (r["Db"][:, b] - Dth[b]) / r["sig36"][b]
        ax.hist(x, bins=np.linspace(-4, 4, 33), density=True, color="0.75", lw=0)
        nu = r["nu"][b]
        g = np.linspace(-4, 4, 400)
        # x = (D - D_th)/sigma with D ~ (D_th + N_b) chi^2_nu / nu  ->  density in x
        s = r["sig36"][b]
        y = (Dth[b] + g * s) + r["Nb"][b]
        dens = stats.chi2.pdf(y / (Dth[b] + r["Nb"][b]) * nu, nu) * nu * s / (Dth[b] + r["Nb"][b])
        ax.plot(g, dens, color=SERIES[0], lw=1.2, label=r"$\chi^2_\nu$, eq. (35)")
        ax.plot(g, stats.norm.pdf(g, x.mean(), x.std()), color=SERIES[1], ls="--", lw=1.0, label="Gaussian")
        ax.text(0.03, 0.92, f"$\\ell={l}$", transform=ax.transAxes, fontsize=8, va="top")
        ax.text(0.97, 0.92, f"$\\nu={nu:.1f}$\n(pub. {PUB_NU[p][j]})", transform=ax.transAxes, fontsize=7.5,
                va="top", ha="right")
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(p, fontsize=9)
        if i == 2:
            ax.set_xlabel(r"$x=(\hat D_b-D_b^{\rm th})/\Delta \hat D_b$")
axs[0, 0].legend(fontsize=6.5, loc="center right")
savefig(fig, "ch08", "boom_hist")

# ---------------------------------------------------------------- figure: overlay with the published numbers
fig, axs = plt.subplots(1, 3, figsize=(7.9, 2.9))
ax = axs[0]
for p, c in zip(PROF, SERIES):
    ax.loglog(lb[rep], res[p]["nu"][rep], color=c, lw=1.2, label=f"ours, {p}")
    ax.loglog(PUB_L, PUB_NU[p], "o", ms=5, mfc="none", mec=c, mew=1.3)
ax.set_xlabel(r"$\ell_b$")
ax.set_ylabel(r"$\nu_b$ (eq. 35)")
ax.set_title("(a) degrees of freedom (o: published)", fontsize=8.5)
ax.legend(fontsize=6.5)
ax = axs[1]
lsx = np.arange(51, LAN + 1)
theory_line(ax, lsx, 1 - 2 / np.pi * np.arcsin(bm.M_CUT / lsx), label=f"App. B, $\\ell_c={bm.M_CUT}$")
for p, c in zip(PROF, SERIES):
    ax.plot(lb[rep], res[p]["Fb"][rep], "o", ms=2.5, color=c, label=f"MC, {p}")
Fpub = [PUB_NU["tophat"][k] / ((2 * l + 1) * DL * PUB_FSKY) for k, l in enumerate(PUB_L)]
ax.plot(PUB_L, Fpub, "s", ms=6, mfc="none", mec="k", mew=1.3, label="published (from $\\nu$)")
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$F_\ell$")
ax.set_title("(b) filter transfer function", fontsize=8.5)
ax.legend(fontsize=6, loc="lower right")
ax = axs[2]
for p, c in zip(PROF, SERIES):
    r = res[p]
    ax.plot(lb[rep], (r["Db"].std(0, ddof=1) / r["sig36"])[rep], "o-", ms=2.5, lw=0.8, color=c, label=p)
ax.axhline(1, color="k", ls="--", lw=1.0)
ax.axhspan(1 - 2 / np.sqrt(2 * NSN), 1 + 2 / np.sqrt(2 * NSN), color="0.9", zorder=0)
ax.set_ylim(0.6, 1.6)
ax.set_xlabel(r"$\ell_b$")
ax.set_ylabel(r"$\sigma_{\rm MC}/\Delta \hat D_b$ (eq. 36)")
ax.set_title("(c) error bars: MC / analytic", fontsize=8.5)
ax.legend(fontsize=6.5)
fig.tight_layout()
savefig(fig, "ch08", "boom_overlay")
save_numbers("ch08", "30_boomerang", nums)
