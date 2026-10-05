"""27_covariance.py -- the l-l' covariance of the cut-sky spectrum: simulations against the formulas.

Question: how strongly are the errors at different multipoles correlated once a mask is applied,
how does that depend on the sky fraction, the mask shape and l, and how good are the quick
formulas: Knox's 2 C^2 / ((2l+1) Delta_l fsky), Hivon's version with w2^2/w4, and Efstathiou's
(2004) analytic covariance  Cov(C~_l, C~_l') ~ 2 C_l C_l' Xi_ll'(W^2)?
Computes, from the 2000 skies of 23_mc_masked.py and the MASTER estimates of 25_master.py:
  * correlation matrices of C~_l (2 <= l <= 100) for six masks;
  * correlation of neighbouring multipoles r(l, l+1), r(l, l+2) against l;
  * MC variance of the binned MASTER D_b against Knox (fsky) and Hivon (fsky w2^2/w4);
  * Efstathiou's approximation (his eqs 15a and 17) for C~_l and for the MASTER bins;
  * for the band cut (azimuthally symmetric), the exact Gaussian covariance of C~_l at
    l, l' <= 40 (Efstathiou eq. 18), from ring sums of the associated Legendre functions,
    against the simulations and the approximation.
Writes: figures/ch08/corr_maps.pdf, figures/ch08/corr_neighbours.pdf, figures/ch08/knox_mc.pdf,
        figures/ch08/efstathiou.pdf, data/ch08/cov_exact_band.npz, results/ch08/27_covariance.tex
"""
import sys, pathlib, importlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, DIV
from camb_fiducial import load_fiducial
import lib_cmbsim as cs
import lib_masks as lm

setup()
L = 767
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"
z = np.load(DATA / "masks.npz")
cz = np.load(DATA / "coupling.npz")
ms = np.load(DATA / "master.npz")
mc = importlib.import_module("23_mc_masked").load_all()
NSIM = mc["sky"].shape[0]
ell = np.arange(L + 1)
_, cl = load_fiducial()
cl = cl[: L + 1]
exp = cs.Experiment()
T2 = exp.transfer(L) ** 2
Nw = exp.nl(L)[0]
Ctot = T2 * cl + Nw                       # spectrum of the observed (beamed + noisy) field
nums = {}
NAMES = {"full": "Full", "cap10": "CapTen", "cap10apo": "CapTenApo", "cap30": "CapThirty",
         "cap70": "CapSeventy", "band20": "Band", "band20apo": "BandApo", "holes": "Holes"}
LABEL = {"full": "full sky", "cap10": "cap 0.1", "cap10apo": "cap 0.1 tapered", "cap30": "cap 0.3",
         "cap70": "cap 0.7", "band20": "band", "band20apo": "band tapered", "holes": "holes"}
MOM = {k: lm.mask_moments(z["mask_" + k].astype(float)) for k in NAMES}


def corr(c):
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


# ---------------------------------------------------------------- 1. correlation matrices
LC = 100
show = ["cap70", "cap30", "cap10", "cap10apo", "band20", "holes"]
fig, axs = plt.subplots(2, 3, figsize=(7.8, 5.3), sharex=True, sharey=True)
for ax, k in zip(axs.ravel(), show):
    x = mc["pcl_" + k][:, 2:LC + 1].astype(float)
    R = corr(np.cov(x, rowvar=False))
    im = ax.imshow(R, origin="lower", cmap=DIV, vmin=-1, vmax=1, extent=[1.5, LC + 0.5, 1.5, LC + 0.5])
    ax.set_title(f"{LABEL[k]}, $f_{{\\rm sky}}={MOM[k][0]:.2f}$", fontsize=8.5)
    off = R[np.triu_indices_from(R, 1)]
    nums[f"EightBcorrOne{NAMES[k]}"] = f"{np.mean(np.diag(R, 1)[30:]):.2f}"
    nums[f"EightBcorrTwo{NAMES[k]}"] = f"{np.mean(np.diag(R, 2)[30:]):.2f}"
for ax in axs[1]:
    ax.set_xlabel(r"$\ell$")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\ell'$")
fig.colorbar(im, ax=axs, shrink=0.8, label=r"correlation of $\widetilde C_\ell$ and $\widetilde C_{\ell'}$")
savefig(fig, "ch08", "corr_maps")
nums["EightBcorrNoise"] = f"{1/np.sqrt(NSIM):.3f}"

# ---------------------------------------------------------------- 2. neighbour correlations vs l
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.1), sharey=True)
for k, c in zip(["cap70", "cap30", "cap10", "cap10apo", "band20", "holes"], SERIES):
    x = mc["pcl_" + k][:, 2:601].astype(float)
    x = (x - x.mean(0)) / x.std(0)
    r1 = np.mean(x[:, :-1] * x[:, 1:], axis=0)
    r2 = np.mean(x[:, :-2] * x[:, 2:], axis=0)
    l1 = np.arange(2, 600)
    sm = lambda y: np.convolve(y, np.ones(9) / 9, mode="same")
    axs[0].semilogx(l1[4:-4], sm(r1)[4:-4], color=c, lw=1.0, label=LABEL[k])
    axs[1].semilogx(l1[:-1][4:-4], sm(r2)[4:-4], color=c, lw=1.0)
for ax, t in zip(axs, [r"(a) $r(\ell,\ell+1)$", r"(b) $r(\ell,\ell+2)$"]):
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_title(t, fontsize=9)
axs[0].set_ylabel("correlation (running mean over 9)")
axs[0].legend(fontsize=7, ncol=2)
savefig(fig, "ch08", "corr_neighbours")

# ---------------------------------------------------------------- 3. Knox and Hivon against the MC
edges, lb = ms["edges"], ms["lb"]
P, Q = lm.binning_operators(edges, L)
rep = edges[1:] <= 601
nbin = np.diff(edges)
Dfac = (ell * (ell + 1) / (2 * np.pi))
fig, ax = plt.subplots(figsize=(6.4, 3.5))
for k, c in zip(["full", "cap70", "cap30", "cap10", "cap10apo", "band20", "holes"], SERIES):
    fsky, wi = MOM[k]
    D = ms["master_" + k].astype(float)
    var_mc = D.var(0, ddof=1)
    # Knox per l, averaged in the bin: Var D_b = (1/n_b^2) sum_l D-factor^2 2 (C+N/T^2)^2 / ((2l+1) fsky)
    vl = (Dfac ** 2) * 2 * (cl + Nw / T2) ** 2 / (2 * ell + 1)
    vb = np.array([vl[a:b].sum() / (b - a) ** 2 for a, b in zip(edges[:-1], edges[1:])])
    knox = vb / fsky
    hivon = vb / (fsky * wi[2] ** 2 / wi[4])
    r_k = var_mc / knox
    r_h = var_mc / hivon
    ax.semilogx(lb[rep], r_h[rep], "o-", color=c, ms=2.5, lw=0.9, label=LABEL[k])
    if wi[2] ** 2 / wi[4] < 0.99:
        ax.semilogx(lb[rep], r_k[rep], "s:", color=c, ms=2.5, lw=0.9, label=LABEL[k] + r", $f_{\rm sky}$ only")
    n = NAMES[k]
    nums[f"EightBknoxFirst{n}"] = f"{r_h[0]:.2f}"
    nums[f"EightBknoxSecond{n}"] = f"{r_h[1]:.2f}"
    nums[f"EightBknoxHigh{n}"] = f"{np.median(r_h[rep][10:]):.2f}"
    nums[f"EightBknoxOnlyHigh{n}"] = f"{np.median(r_k[rep][10:]):.2f}"
    print(f"{k:9s} Var_MC / Hivon: first bins {r_h[:3]}, median l>200 {np.median(r_h[rep][10:]):.3f};"
          f"  / Knox(fsky): median {np.median(r_k[rep][10:]):.3f}")
ax.axhline(1, color="0.4", lw=0.6)
ax.axhspan(1 - 2 * np.sqrt(2 / NSIM), 1 + 2 * np.sqrt(2 / NSIM), color="0.9", zorder=0)
ax.set_xlabel(r"multipole $\ell$ (MASTER bins of 20)")
ax.set_ylabel(r"$\mathrm{Var}_{\rm MC}(\widehat D_b)\,/\,$formula")
ax.set_ylim(0, 3)
ax.legend(fontsize=7, ncol=2)
savefig(fig, "ch08", "knox_mc")
nums["EightBvarNoise"] = f"{100 * np.sqrt(2 / (NSIM - 1)):.1f}"

# ---------------------------------------------------------------- 4. Efstathiou's approximation
def efstathiou_pcl(k, improved=False):
    """Cov(C~_l, C~_l') ~ 2 C_l C_l' Xi_ll'(W^2), Xi = M(W^2)/(2l'+1)  (his eq. 15a).

    C is the spectrum of the observed field, T^2 C + N.  improved=True replaces it by the
    mode-coupled spectrum <C~>/(fsky w2), which already contains the power leaked from other
    multipoles (a common refinement of the same approximation).
    """
    c = Ctot
    if improved:
        fsky, wi = MOM[k]
        c = cz["M_" + k] @ Ctot / (fsky * wi[2])
    return 2.0 * np.outer(c, c) * cz["Xi2_" + k]


ratio_diag, ratio_imp = {}, {}
for k in ["cap70", "cap30", "cap10", "cap10apo", "band20", "holes"]:
    V = efstathiou_pcl(k)
    x = mc["pcl_" + k].astype(float)
    vmc = x.var(0, ddof=1)
    ratio_diag[k] = np.diag(V) / vmc
    ratio_imp[k] = np.diag(efstathiou_pcl(k, improved=True)) / vmc
    # MASTER bins: Cov(D_b) = Kinv P V P^T Kinv^T  (his eq. 17, binned)
    K = lm.master_matrix(cz["M_" + k], T2, P, Q)
    Ki = np.linalg.inv(K)
    Vb = Ki @ P @ V @ P.T @ Ki.T
    rb = np.diag(Vb) / ms["master_" + k].astype(float).var(0, ddof=1)
    n = NAMES[k]
    nums[f"EightBefsTwo{n}"] = f"{ratio_diag[k][2]:.2f}"
    nums[f"EightBefsTen{n}"] = f"{ratio_diag[k][10]:.2f}"
    nums[f"EightBefsHigh{n}"] = f"{np.median(ratio_diag[k][100:601]):.3f}"
    nums[f"EightBimpTwo{n}"] = f"{ratio_imp[k][2]:.2f}"
    nums[f"EightBimpTen{n}"] = f"{ratio_imp[k][10]:.2f}"
    nums[f"EightBimpHigh{n}"] = f"{np.median(ratio_imp[k][100:601]):.3f}"
    print(f"   improved: l=2 {ratio_imp[k][2]:.3f} l=10 {ratio_imp[k][10]:.3f} median 100-600 {np.median(ratio_imp[k][100:601]):.3f}")
    nums[f"EightBefsBinFirst{n}"] = f"{rb[0]:.2f}"
    nums[f"EightBefsBinHigh{n}"] = f"{np.median(rb[rep][5:]):.3f}"
    print(f"{k:9s} Efstathiou/MC diag C~: l=2 {ratio_diag[k][2]:.3f} l=10 {ratio_diag[k][10]:.3f}"
          f" l=20 {ratio_diag[k][20]:.3f} median 100-600 {np.median(ratio_diag[k][100:601]):.3f};"
          f" MASTER bins first {rb[0]:.3f} median {np.median(rb[rep][5:]):.3f}")

# ---------------------------------------------------------------- 5. exact covariance, band cut
LX = 40
path = DATA / "cov_exact_band.npz"
nside = int(z["nside"])
if path.exists():
    Cx = np.load(path)["cov"]
else:
    t0 = time.time()
    w = z["mask_band20"].astype(float)
    theta_pix, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    th, first, npr = np.unique(theta_pix, return_index=True, return_counts=True)
    wr = w[first]                                    # the mask is constant on each ring
    om = 4 * np.pi / hp.nside2npix(nside)
    Lm = np.arange(L + 1)
    Cx = np.zeros((LX + 1, LX + 1))
    for m in range(LX + 1):
        lam = lm.lambda_lm(L, m, th)                                 # lambda_lm(theta_ring), l >= m
        wgt = om * npr * wr                                          # ring weight: n_phi Omega W
        K = (lam[: LX + 1 - m] * wgt) @ lam.T                        # K^(m)_{l L},  l <= LX, L <= 767
        S = (K * (T2 * cl)[m:][None, :]) @ K.T                       # signal part of <a~ a~*>
        S += Nw * (lam[: LX + 1 - m] * (om * npr * wr ** 2)) @ lam[: LX + 1 - m].T   # pixel noise
        mult = 1.0 if m == 0 else 2.0                                # +m and -m
        Cx[m:, m:] += mult * S ** 2
    lx = np.arange(LX + 1)
    Cx *= 2.0 / np.outer(2 * lx + 1, 2 * lx + 1)
    np.savez(path, cov=Cx)
    print(f"[cache] exact band covariance in {time.time() - t0:.0f} s")
x = mc["pcl_band20"][:, : LX + 1].astype(float)
Cmc = np.cov(x, rowvar=False)
Ve = efstathiou_pcl("band20")[: LX + 1, : LX + 1]
lx = np.arange(LX + 1)
de = np.diag(Cmc)[2:] / np.diag(Cx)[2:]
nums["EightBexactMCmean"] = f"{de.mean():.3f}"
nums["EightBexactMCrms"] = f"{de.std():.3f}"
nums["EightBexactApproxTwo"] = f"{Ve[2, 2] / Cx[2, 2]:.2f}"
nums["EightBexactApproxFour"] = f"{Ve[4, 4] / Cx[4, 4]:.2f}"
nums["EightBexactApproxTwenty"] = f"{Ve[20, 20] / Cx[20, 20]:.2f}"
nums["EightBexactApproxForty"] = f"{Ve[40, 40] / Cx[40, 40]:.2f}"
nums["EightBexactL"] = LX
print(f"band, exact covariance: MC/exact diag mean {de.mean():.3f} rms {de.std():.3f};"
      f" approx/exact at l=2,4,20,40: {Ve[2,2]/Cx[2,2]:.3f} {Ve[4,4]/Cx[4,4]:.3f} {Ve[20,20]/Cx[20,20]:.3f} {Ve[40,40]/Cx[40,40]:.3f}")
off_mc = Cmc[10, 2:LX + 1] / np.sqrt(Cmc[10, 10] * np.diag(Cmc)[2:])
off_ex = Cx[10, 2:LX + 1] / np.sqrt(Cx[10, 10] * np.diag(Cx)[2:])
off_ap = Ve[10, 2:LX + 1] / np.sqrt(Ve[10, 10] * np.diag(Ve)[2:])

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axs[0]
for k, c in zip(["cap70", "cap30", "cap10", "cap10apo", "band20", "holes"], SERIES):
    sm = np.convolve(ratio_diag[k], np.ones(5) / 5, mode="same")
    ax.semilogx(ell[2:601], sm[2:601], color=c, lw=1.0, label=LABEL[k])
    smi = np.convolve(ratio_imp[k], np.ones(5) / 5, mode="same")
    ax.semilogx(ell[2:601], smi[2:601], color=c, lw=0.8, ls=":")
ax.axhline(1, color="0.4", lw=0.6)
ax.set_ylim(0.4, 1.9)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"approximation / MC, $\mathrm{Var}\,\widetilde C_\ell$")
ax.set_title("(a) eq. (15a): with $C_\\ell$ (solid), with coupled $\\widetilde C_\\ell$ (dotted)", fontsize=8)
ax.legend(fontsize=7, ncol=2, loc="upper right")
ax = axs[1]
ax.plot(lx[2:], off_ex, "-", color="k", lw=1.2, label="exact (eq. 18)")
ax.plot(lx[2:], off_mc, "o", ms=3, mfc="none", color=SERIES[0], label=f"MC, {NSIM} skies")
ax.plot(lx[2:], off_ap, "--", color=SERIES[1], lw=1.2, label="approximation (15a)")
ax.set_xlabel(r"$\ell'$")
ax.set_ylabel(r"correlation of $\widetilde C_{10}$ with $\widetilde C_{\ell'}$")
ax.set_title(r"(b) band cut, low $\ell$", fontsize=9)
ax.legend(fontsize=7.5)
savefig(fig, "ch08", "efstathiou")
save_numbers("ch08", "27_covariance", nums)
