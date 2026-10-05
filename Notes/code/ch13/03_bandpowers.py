"""03_bandpowers.py -- the TT power spectrum of the real sky, one correction at a time.

Question: starting from the two SMICA half-mission maps (same sky, independent noise), what are
the MASTER bandpowers D_b in Planck's bins (Delta l = 30, 30 <= l < 1020), and what does each
step of the correction do to them?
Steps:
  0. remove the monopole and dipole fitted on the kept sky (they would leak through the mask);
  1. pseudo-spectra of W x map: the cross-spectrum of the halves, the spectrum of their
     half-difference (sky cancels: noise only), and the auto-spectrum of the full map;
  2. naive correction: divide by fsky w2 and by T_l^2 = (B_l p_l)^2;
  3. MASTER: bin with Planck's weights and solve K D = P C_pseudo with K = P M T^2 Q;
  4. the same for the four other windows of 02_mask.py (for the checks of 06_checks.py);
  5. the bandpower window functions F_bl (<D_b> = sum_l F_bl C_l) for the theory comparison.
Writes: data/ch13/bandpowers.npz, figures/ch13/steps.pdf, results/ch13/03_bandpowers.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
from camb_fiducial import load_fiducial
import lib_planck as lp

setup()
NSIDE, LC = 512, 1199
EDGES = np.concatenate([[2], lp.planck_edges(30, LC, 30)])       # [2,30) is a buffer bin
REP = (EDGES[:-1] >= 30) & (EDGES[1:] <= 1020)                   # reported: 30 <= l < 1020
nums = {}
maps = lp.load_maps(NSIDE)
masks = lp.load_masks(NSIDE)
T2 = (maps["beam"] * maps["pixwin"])[: LC + 1] ** 2
ell = np.arange(LC + 1)

# ---------------------------------------------------------------- 0. monopole and dipole
keep = masks["binary"] > 0
for k in ("full", "hm1", "hm2"):
    m = maps[k].copy()
    m[~keep] = hp.UNSEEN
    mono, dip = hp.fit_dipole(m)                     # least-squares fit on the kept pixels only
    vec = np.array(hp.pix2vec(NSIDE, np.arange(m.size)))
    maps[k] = maps[k] - mono - dip @ vec
    if k == "full":
        nums["TwelveAMono"] = f"{mono:.1f}"
        nums["TwelveADip"] = f"{np.linalg.norm(dip):.1f}"
print("monopole, dipole of the full map:", nums)

# ---------------------------------------------------------------- 1-4. spectra for every window
out = {"edges": EDGES, "rep": REP}
for name, W in masks.items():
    if name in ("gal", "ps"):
        continue
    W = W.astype(np.float64)
    M = lp.coupling(W, LC, name=name)
    mst = lp.Master(M, T2, EDGES)
    a1, a2, af = lp.pseudo_cl(W, [maps["hm1"], maps["hm2"], maps["full"]], LC)
    pcl = {"cross": hp.alm2cl(a1, a2), "null": hp.alm2cl(0.5 * (a1 - a2)), "auto": hp.alm2cl(af)}
    for k, c in pcl.items():
        out[f"{name}_{k}"] = mst(c)
        out[f"{name}_pcl_{k}"] = c
    out[f"{name}_F"] = mst.window()
    out[f"{name}_leff"] = mst.leff
    out[f"{name}_fw2"] = np.mean(W ** 2)
np.savez(lp.DATA / "bandpowers.npz", **out)

# ---------------------------------------------------------------- the steps, for the main window
lb = out["main_leff"]
l_pl, D_pl, s_pl, D_bf = lp.load_planck_binned()
P, Q, _ = lp.planck_binning(EDGES, LC)
pc = out["main_pcl_cross"]
D_raw = P @ pc                                   # binned pseudo-spectrum, no correction at all
D_fsky = P @ (pc / out["main_fw2"])              # divided by the mean of W^2
D_naive = P @ (pc / out["main_fw2"] / T2)        # ... and by the beam and pixel window
D_master = out["main_cross"]
_, clf = load_fiducial()
nb = REP.sum()
ratio_naive = D_naive[REP] / D_pl[:nb]
ratio_master = D_master[REP] / D_pl[:nb]
nums.update({"TwelveANbins": int(nb), "TwelveALlo": int(EDGES[:-1][REP][0]), "TwelveALhi": int(EDGES[1:][REP][-1] - 1),
             "TwelveARawFirst": f"{D_raw[REP][0]:.0f}", "TwelveAPlFirst": f"{D_pl[0]:.0f}",
             "TwelveAFwtwo": f"{out['main_fw2']:.3f}",
             "TwelveANaiveMaxDev": f"{100 * np.max(np.abs(ratio_naive - 1)):.0f}",
             "TwelveAMasterMaxDev": f"{100 * np.max(np.abs(ratio_master - 1)):.1f}",
             "TwelveAMasterRms": f"{100 * np.sqrt(np.mean((ratio_master - 1) ** 2)):.1f}",
             "TwelveABeamLast": f"{np.sqrt(T2[1019]):.3f}"})
print(nums)

fig, axs = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
ax = axs[0]
ax.plot(ell[2:], (ell * (ell + 1) * clf[: LC + 1] / (2 * np.pi))[2:], color="k", lw=0.7, label="Planck best fit")
ax.plot(lb[REP], D_raw[REP], "v", color=SERIES[3], ms=3.5, label="pseudo-spectrum of the cross")
ax.plot(lb[REP], D_fsky[REP], "s", color=SERIES[1], ms=3, label=r"divided by $\langle W^2\rangle$")
ax.plot(lb[REP], D_naive[REP], "^", color=SERIES[2], ms=3.5, label=r"... and by $T_\ell^2$")
ax.plot(lb[REP], D_master[REP], "o", color=SERIES[0], ms=3.5, label="MASTER")
ax.set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
ax.set_xlim(0, 1040)
ax.legend(fontsize=7.5)
ax = axs[1]
ax.axhline(0, color="0.4", lw=0.6)
ax.plot(lb[REP], 100 * (ratio_naive - 1), "^", color=SERIES[2], ms=3.5, label="naive")
ax.plot(lb[REP], 100 * (ratio_master - 1), "o", color=SERIES[0], ms=3.5, label="MASTER")
ax.set_ylabel("vs Planck [%]")
ax.set_xlabel(r"multipole $\ell$")
ax.legend(fontsize=7.5, ncol=2)
fig.tight_layout()
savefig(fig, "ch13", "steps")
save_numbers("ch13", "03_bandpowers", nums)
