"""23_cmass_bao.py -- the acoustic peak in CMASS South: detection, dilation, and D_V/r_d.

Question: does the correlation function of the real CMASS South galaxies show the acoustic peak,
how significant is it, and what distance D_V(z_eff)/r_d does its position give, compared with the
published BOSS measurements?
Computes: the damped acoustic template of chapter 14 (Sigma_nl = 8 Mpc/h) and its no-wiggle twin,
bin-averaged in the 8 Mpc/h bins of the data; for every alpha on a grid, the generalised least
squares fit of B^2 xi_t(alpha s) + a0 + a1/s + a2/s^2 over 30 < s < 180 Mpc/h with the
Hartlap-corrected inverse jackknife covariance; chi^2(alpha), p(alpha), Delta chi^2 between the
templates; D_V/r_d = alpha D_V^fid/r_d^fid. The same fit applied to the published DR12 CMASS
(north + south) monopole and mock covariance of Cuesta et al. (2016). Distances of Anderson et al.
(2012), Cuesta et al. (2016) and Alam et al. (2017) converted to D_V/r_d with CAMB's r_d.
Writes: figures/ch13/cmass_xi_overlay.pdf, figures/ch13/cmass_chi2.pdf, figures/ch13/cmass_errors.pdf,
figures/ch13/cmass_dv.pdf, results/ch13/23_cmass_bao.tex, data/ch13/cmass_bao.npz
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "ch11"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from common import setup, savefig, save_numbers, SERIES, DATA, NOTES
from lib_lss import linear_pk_z0, nowiggle_pk, xi_from_pk

C_KMS = 299792.458
SIG_NL = 8.0                                   # Mpc/h, damping before reconstruction
SMIN, SMAX = 30.0, 180.0                       # fit range [Mpc/h], as Cuesta et al. (2016)
ALPHAS = np.round(np.arange(0.80, 1.2001, 0.002), 4)
AREA_SGC, AREA_ALL = 2524.67, 9376.09          # CMASS DR12 effective areas [deg^2], Cuesta Table 2

cat = np.load(DATA / "ch13" / "cmass_south.npz")
xi = np.load(DATA / "ch13" / "cmass_xi.npz")
h, rd_mpc, z_eff = float(cat["h"]), float(cat["rd_mpc"]), float(cat["z_eff"])
zt, dct, hzt = cat["zt"], cat["dct"], cat["hzt"]


# ---- fiducial distances [Mpc/h]: D_M = D_C (flat), D_H = c/H, D_V = (z D_M^2 D_H)^(1/3)
def dv_fid(z):
    dm = np.interp(z, zt, dct)
    dh = C_KMS / np.interp(z, zt, hzt) * h
    return (z * dm ** 2 * dh) ** (1 / 3)


rd_h = rd_mpc * h                               # template ruler in Mpc/h
DV_FID = dv_fid(z_eff)

# ---- templates in 8 Mpc/h bins, averaged over each bin with weight s^2 ds
k_tab, pk_tab, _, _ = linear_pk_z0()
kk = np.geomspace(1e-4, 10, 4000)
p_nw = nowiggle_pk(kk, k_tab, pk_tab)
p_lin = np.interp(np.log(kk), np.log(k_tab), pk_tab)
p_w = (p_lin - p_nw) * np.exp(-0.5 * (kk * SIG_NL) ** 2) + p_nw
s_fine = np.linspace(1.0, 260.0, 2600)
xi_w_fine = xi_from_pk(kk, p_w, s_fine, damp=1.0)
xi_nw_fine = xi_from_pk(kk, p_nw, s_fine, damp=1.0)


def binned(xi_fine, edges, alpha):
    """Bin average of xi_t(alpha s), weighting each s by s^2 (the number of pairs grows as s^2)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ss = np.linspace(lo, hi, 33)
        out.append(np.trapezoid(np.interp(alpha * ss, s_fine, xi_fine) * ss ** 2, ss) / np.trapezoid(ss ** 2, ss))
    return np.array(out)


def fitter(s, edges, d, C, nmock):
    """chi^2(alpha) for both templates with the linear coefficients profiled out (GLS)."""
    m = (s > SMIN) & (s < SMAX + 1)
    s_, d_, nb = s[m], d[m], int(m.sum())
    Cm = C[np.ix_(m, m)]
    hart = (nmock - nb - 2) / (nmock - 1)
    Ci = hart * np.linalg.inv(Cm)
    e = np.concatenate([edges[:-1][m], [edges[1:][m][-1]]])
    res = {}
    for name, xf in [("w", xi_w_fine), ("nw", xi_nw_fine)]:
        chi2, coefs = [], []
        for a in ALPHAS:
            X = np.column_stack([binned(xf, e, a), np.ones(nb), 1 / s_, 1 / s_ ** 2])
            b = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ d_)
            r = d_ - X @ b
            chi2.append(r @ Ci @ r)
            coefs.append(b)
        chi2 = np.array(chi2)
        i = int(np.argmin(chi2))
        p = np.exp(-0.5 * (chi2 - chi2.min()))
        p /= p.sum()
        mean = np.sum(p * ALPHAS)
        sd = np.sqrt(np.sum(p * ALPHAS ** 2) - mean ** 2)
        res[name] = dict(chi2=chi2, i=i, alpha=ALPHAS[i], mean=mean, sd=sd, coef=coefs[i], chimin=chi2[i])
    res["dchi2"] = res["nw"]["chimin"] - res["w"]["chimin"]
    res["nb"], res["hart"], res["mask"], res["edges"] = nb, hart, m, e
    return res


# ---- 1. our measurement: CMASS South, jackknife covariance
s, edges = xi["s"], xi["edges"]
NJK = int(xi["njk"])
ours = fitter(s, edges, xi["xi_full"], xi["C_jk"], NJK)
a_ours, sa_ours = ours["w"]["mean"], ours["w"]["sd"]
dvrd_ours = a_ours * DV_FID / rd_h
sdv_ours = sa_ours * DV_FID / rd_h

# ---- 2. the published DR12 CMASS monopole (north + south), its mock covariance, the same fit
CDIR = DATA / "ch13" / "cuesta2016"
if not (CDIR / "Cuesta_2016_CMASSDR12_corrfunction_x0_prerecon.dat").exists():
    CDIR.mkdir(parents=True, exist_ok=True)
    url = "https://data.sdss.org/sas/dr12/boss/papers/clustering/Cuesta_2016_CMASSDR12_LOWZDR12_measurements.tar.gz"
    subprocess.run(f"curl -sSL {url} | tar xz -C {CDIR}", shell=True, check=True)
pub = np.loadtxt(CDIR / "Cuesta_2016_CMASSDR12_corrfunction_x0_prerecon.dat")
pub_cov = np.loadtxt(CDIR / "Cuesta_2016_CMASSDR12_corrfunction_cov_x0x2_prerecon.dat")
npb = len(pub)
pub_cov = pub_cov[:npb, :npb] if pub_cov.ndim == 2 and pub_cov.shape[0] >= npb else None
if pub_cov is None:                                           # stored as i j C_ij rows
    raw = np.loadtxt(CDIR / "Cuesta_2016_CMASSDR12_corrfunction_cov_x0x2_prerecon.dat")
    raise SystemExit(f"unexpected covariance format {raw.shape}")
s_pub, xi_pub, e_pub = pub[:, 0], pub[:, 1], pub[:, 2]
edges_pub = np.arange(0.0, 8.0 * npb + 0.1, 8.0)
# their coordinates use the QPM fiducial cosmology (Omega_m = 0.29, h = 0.7)
DV_QPM_MPC, RD_QPM = 2009.55, 147.10                          # Cuesta et al. Table 5, z = 0.57
DV_QPM = DV_QPM_MPC * 0.7                                     # Mpc/h
NMOCK_QPM = 956                                               # QPM CMASS mocks used (their Sec. 4.1)
pubfit = fitter(s_pub, edges_pub, xi_pub, pub_cov, NMOCK_QPM)
a_pub, sa_pub = pubfit["w"]["mean"], pubfit["w"]["sd"]
dvrd_pub = a_pub * DV_QPM / rd_h
sdv_pub = sa_pub * DV_QPM / rd_h
# what Cuesta et al. report: alpha_iso = 1.0153 +- 0.0134 relative to their own template (r_d = 147.10 Mpc)
dvrd_cuesta, sdv_cuesta_stat = 1.0153 * DV_QPM_MPC / RD_QPM, 0.0134 * DV_QPM_MPC / RD_QPM
dvrd_cuesta_quoted, sdv_cuesta_quoted = 13.87, 0.19           # their Sec. 4.4, stat + syst

# ---- 3. Anderson et al. (2012): D_V/r_s with the Eisenstein-Hu r_s; convert with CAMB r_d
import camb
pa = camb.set_params(H0=70.0, ombh2=0.0224, omch2=0.274 * 0.49 - 0.0224, mnu=0.0, ns=0.95, As=2.1e-9,
                     num_massive_neutrinos=0)
rd_and_camb = camb.get_background(pa).get_derived_params()["rdrag"]
RS_EH_AND = 153.19
conv_and = RS_EH_AND / rd_and_camb
dvrd_and = 13.44 * conv_and                    # their xi(r) pre-reconstruction value (Table 2)
sdv_and = 0.22 * conv_and
dvrd_and_cons = 13.67 * conv_and               # their consensus (post-reconstruction)

# ---- 4. Alam et al. (2017) BAO-only consensus -> D_V/r_d with the delta method
alam_z = np.array([0.38, 0.51, 0.61])
alam_dm = np.array([1512.39, 1975.22, 2306.68])                 # D_M r_d,fid/r_d [Mpc]
alam_H = np.array([81.2087, 90.9029, 98.9647])                  # H r_d/r_d,fid [km/s/Mpc]
alam_cov = np.array([
    [624.707, 23.729, 325.332, 8.34963, 157.386, 3.57778],
    [23.729, 5.60873, 11.6429, 2.33996, 6.39263, 0.968056],
    [325.332, 11.6429, 905.777, 29.3392, 515.271, 14.1013],
    [8.34963, 2.33996, 29.3392, 5.42327, 16.1422, 2.85334],
    [157.386, 6.39263, 515.271, 16.1422, 1375.12, 40.4327],
    [3.57778, 0.968056, 14.1013, 2.85334, 40.4327, 6.25936]])   # their BAO_consensus_covtot_dM_Hz.txt
RD_ALAM = 147.78
alam_dv, alam_sdv = [], []
for j, z in enumerate(alam_z):
    dm, H_ = alam_dm[j], alam_H[j]
    dv = (z * dm ** 2 * C_KMS / H_) ** (1 / 3) / RD_ALAM
    cov = alam_cov[2 * j:2 * j + 2, 2 * j:2 * j + 2]
    g = dv * np.array([2 / (3 * dm), -1 / (3 * H_)])          # gradient of D_V/r_d w.r.t. (D_M, H)
    alam_dv.append(dv)
    alam_sdv.append(np.sqrt(g @ cov @ g))
alam_dv, alam_sdv = np.array(alam_dv), np.array(alam_sdv)

# ---- 5. Planck prediction of D_V/r_d (our fiducial = Planck 2018)
pred = lambda z: dv_fid(z) / rd_h
pull_ours = (dvrd_ours - pred(z_eff)) / sdv_ours

# ---- 5b. the shape of chi^2(alpha): local width (connected Delta chi^2 < 1 around the minimum)
#      against the posterior width over the whole prior range 0.8 < alpha < 1.2
def local_interval(chi2):
    i = int(np.argmin(chi2))
    lo = hi = i
    while lo > 0 and chi2[lo - 1] - chi2[i] < 1:
        lo -= 1
    while hi < len(chi2) - 1 and chi2[hi + 1] - chi2[i] < 1:
        hi += 1
    return ALPHAS[lo], ALPHAS[hi]


a_lo, a_hi = local_interval(ours["w"]["chi2"])
sig_loc = 0.5 * (a_hi - a_lo)
plateau = float(np.max(ours["w"]["chi2"] - ours["w"]["chimin"]))
edge_lo = float(ours["w"]["chi2"][0] - ours["w"]["chimin"])
p_lo, p_hi = local_interval(pubfit["w"]["chi2"])
sig_loc_pub = 0.5 * (p_hi - p_lo)

# ---- 6. errors: jackknife (south) against mocks (north + south) scaled by the area
err_jk = np.sqrt(np.diag(xi["C_jk"]))
n_common = min(len(s), npb)
ratio = err_jk[:n_common] / e_pub[:n_common]
mfit = (s[:n_common] > SMIN) & (s[:n_common] < SMAX + 1)
ratio_fit = np.median(ratio[mfit])
area_scale = np.sqrt(AREA_ALL / AREA_SGC)

np.savez(DATA / "ch13" / "cmass_bao.npz", alphas=ALPHAS, chi2_w=ours["w"]["chi2"], chi2_nw=ours["nw"]["chi2"],
         chi2_w_pub=pubfit["w"]["chi2"], chi2_nw_pub=pubfit["nw"]["chi2"])

# ---- figures
setup()
QPM_TO_OURS = DV_FID / DV_QPM * 1.0     # isotropic rescaling of their s to our fiducial (same z_eff assumed)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
m = ours["mask"]
ax.errorbar(s, s ** 2 * xi["xi_full"], yerr=s ** 2 * err_jk, fmt="o", ms=3.5, color=SERIES[0],
            label="CMASS South, our measurement (jackknife errors)")
ss = s_pub * QPM_TO_OURS
ax.errorbar(ss + 1.2, ss ** 2 * xi_pub, yerr=ss ** 2 * e_pub, fmt="s", ms=3, color=SERIES[1], alpha=0.85,
            label="CMASS North+South, Cuesta et al. 2016 (mock errors)")
sm = np.linspace(SMIN, SMAX, 300)
for name, col, ls, lab in [("w", "k", "-", "best fit with the peak"), ("nw", "0.45", "--", "best fit without the peak")]:
    b = ours[name]["coef"]
    a = ALPHAS[ours[name]["i"]]
    xf = xi_w_fine if name == "w" else xi_nw_fine
    model = b[0] * np.interp(a * sm, s_fine, xf) + b[1] + b[2] / sm + b[3] / sm ** 2
    ax.plot(sm, sm ** 2 * model, color=col, ls=ls, lw=1.3, label=lab)
ax.axvspan(0, SMIN, color="0.93", lw=0)
ax.set_xlim(0, 188)
ax.set_ylim(-60, 130)
ax.set_xlabel(r"$s\;[h^{-1}\,{\rm Mpc}]$")
ax.set_ylabel(r"$s^2\,\hat\xi_0(s)\;[h^{-2}\,{\rm Mpc}^2]$")
ax.legend(fontsize=7, loc="lower left")
savefig(fig, "ch13", "cmass_xi_overlay")

fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
for ax, res, title in [(axs[0], ours, "CMASS South (ours)"), (axs[1], pubfit, "CMASS N+S (published data, our fit)")]:
    c0 = res["w"]["chimin"]
    ax.plot(ALPHAS, res["w"]["chi2"] - c0, color=SERIES[0], label="template with the peak")
    ax.plot(ALPHAS, res["nw"]["chi2"] - c0, color=SERIES[1], ls="--", label="without the peak")
    for nsig in [1, 2, 3, 4, 5]:
        ax.axhline(nsig ** 2, color="0.8", lw=0.6)
        ax.text(1.195, nsig ** 2 + 0.3, rf"${nsig}\sigma$", fontsize=7, ha="right", color="0.4")
    ax.set_xlabel(r"$\alpha$")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 30)
axs[0].set_ylabel(r"$\chi^2(\alpha)-\chi^2_{\min}$")
axs[0].legend(fontsize=7, loc="upper left")
savefig(fig, "ch13", "cmass_chi2")

fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), layout="constrained")
axs[0].plot(s[:n_common], ratio, "o-", ms=3, color=SERIES[0], label="jackknife (S) / mocks (N+S)")
axs[0].axhline(area_scale, color="k", ls="--", lw=1, label=r"$\sqrt{A_{\rm N+S}/A_{\rm S}}$")
axs[0].set_xlabel(r"$s\;[h^{-1}\,{\rm Mpc}]$"); axs[0].set_ylabel("ratio of errors")
axs[0].set_ylim(0, 4); axs[0].legend(fontsize=7)
Cjk = xi["C_jk"]
corr = Cjk / np.sqrt(np.outer(np.diag(Cjk), np.diag(Cjk)))
im = axs[1].imshow(corr, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[edges[0], edges[-1], edges[0], edges[-1]])
axs[1].set_xlabel(r"$s\;[h^{-1}\,{\rm Mpc}]$"); axs[1].set_ylabel(r"$s'\;[h^{-1}\,{\rm Mpc}]$")
axs[1].grid(False)
fig.colorbar(im, ax=axs[1], fraction=0.046, label="jackknife correlation")
savefig(fig, "ch13", "cmass_errors")

fig, ax = plt.subplots(figsize=(6.0, 3.3))
zz = np.linspace(0.3, 0.7, 100)
ax.axhline(1.0, color="k", ls="--", lw=1, label="Planck 2018 $\\Lambda$CDM")
ax.errorbar(alam_z, alam_dv / pred(alam_z), yerr=alam_sdv / pred(alam_z), fmt="D", ms=4, color=SERIES[2],
            label="Alam et al. 2017 (DR12, BAO only, after reconstruction)")
ax.errorbar([0.57 - 0.006], [dvrd_cuesta_quoted / pred(0.57)], yerr=[sdv_cuesta_quoted / pred(0.57)], fmt="s", ms=4,
            color=SERIES[1], label="Cuesta et al. 2016 (DR12 CMASS, before reconstruction)")
ax.errorbar([0.57 + 0.006], [dvrd_pub / pred(0.57)], yerr=[sdv_pub / pred(0.57)], fmt="s", ms=4, mfc="white",
            color=SERIES[1], label="same published $\\xi_0$, our fit")
ax.errorbar([0.57 - 0.012], [dvrd_and / pred(0.57)], yerr=[sdv_and / pred(0.57)], fmt="^", ms=4, color=SERIES[3],
            label="Anderson et al. 2012 (DR9 CMASS, before reconstruction)")
ax.errorbar([z_eff], [dvrd_ours / pred(z_eff)], yerr=[sdv_ours / pred(z_eff)], fmt="o", ms=5, color=SERIES[0],
            label="CMASS South, ours")
ax.set_xlabel("$z$"); ax.set_ylabel(r"$(D_V/r_d)\,/\,$Planck prediction")
ax.set_xlim(0.33, 0.66); ax.set_ylim(0.88, 1.12)
ax.legend(fontsize=6.5, loc="upper left", ncol=1)
savefig(fig, "ch13", "cmass_dv")

fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), layout="constrained")
for key, col, lab in [("none", SERIES[1], "no weights"), ("fkp", SERIES[2], "FKP weights only"),
                      ("full", SERIES[0], r"$w_{\rm tot}\,w_{\rm FKP}$ (all weights)")]:
    axs[0].plot(s, s ** 2 * xi[f"xi_{key}"], "o-", ms=2.5, lw=1, color=col, label=lab)
axs[0].set_xlabel(r"$s\;[h^{-1}\,{\rm Mpc}]$"); axs[0].set_ylabel(r"$s^2\,\hat\xi_0(s)$")
axs[0].legend(fontsize=7)
for key, col in [("none", SERIES[1]), ("fkp", SERIES[2])]:
    axs[1].plot(s, s ** 2 * (xi[f"xi_{key}"] - xi["xi_full"]), "o-", ms=2.5, lw=1, color=col)
axs[1].fill_between(s, -s ** 2 * err_jk, s ** 2 * err_jk, color="0.85", lw=0, label=r"$\pm1\sigma$ (jackknife)")
axs[1].axhline(0, color="k", lw=0.6)
axs[1].set_xlabel(r"$s\;[h^{-1}\,{\rm Mpc}]$"); axs[1].set_ylabel(r"$s^2\,(\hat\xi_0-\hat\xi_0^{\rm all})$")
axs[1].legend(fontsize=7)
savefig(fig, "ch13", "cmass_weights")

# each count divided by its total number of weighted pairs (the normalisations of the LS estimator)
nkeep = int(xi["rand_ratio"]) * len(cat["wg"])
wgal, wran = cat["wg"], cat["wr"][:nkeep]
dd_n = xi["DD_full"] / (wgal.sum() ** 2 - (wgal ** 2).sum())
dr_n = xi["DR_full"] / (wgal.sum() * wran.sum())
rr_n = xi["RR_full"] / (wran.sum() ** 2 - (wran ** 2).sum())
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), layout="constrained")
axs[0].plot(s, dd_n, "o", ms=2.5, color=SERIES[0], label="DD")
axs[0].plot(s, dr_n, "-", color=SERIES[1], label="DR")
axs[0].plot(s, rr_n, "--", color="k", lw=1, label="RR")
axs[0].set_yscale("log")
axs[0].set_xlabel(r"$s\;[h^{-1}\,{\rm Mpc}]$"); axs[0].set_ylabel("fraction of all pairs in the bin")
axs[0].legend(fontsize=7)
axs[1].errorbar(s, s ** 2 * xi["xi_full"], yerr=s ** 2 * err_jk, fmt="o", ms=3, color=SERIES[0])
axs[1].axhline(0, color="k", lw=0.6)
axs[1].axvline(rd_h, color=SERIES[3], ls=":", lw=1)
axs[1].set_xlabel(r"$s\;[h^{-1}\,{\rm Mpc}]$"); axs[1].set_ylabel(r"$s^2\,\hat\xi_0(s)\;[h^{-2}\,{\rm Mpc}^2]$")
savefig(fig, "ch13", "cmass_counts")

save_numbers("ch13", "23_cmass_bao", {
    "BaoAlpha": f"{a_ours:.3f}", "BaoSigAlpha": f"{sa_ours:.3f}", "BaoSigAlphaPct": f"{100 * sa_ours:.1f}",
    "BaoAlphaBest": f"{ours['w']['alpha']:.3f}",
    "BaoChiMin": f"{ours['w']['chimin']:.1f}", "BaoNbFit": ours["nb"], "CmFitDof": ours["nb"] - 5,
    "BaoHart": f"{ours['hart']:.3f}",
    "BaoDchi": f"{ours['dchi2']:.1f}", "BaoSignif": f"{np.sqrt(max(ours['dchi2'], 0)):.1f}",
    "BaoChiMinNw": f"{ours['nw']['chimin']:.1f}",
    "BaoDvFid": f"{DV_FID:.1f}", "BaoDvFidMpc": f"{DV_FID / h:.0f}", "BaoRdH": f"{rd_h:.2f}",
    "BaoDvRdFid": f"{DV_FID / rd_h:.2f}", "BaoZeff": f"{z_eff:.3f}",
    "BaoDvRd": f"{dvrd_ours:.2f}", "BaoSigDvRd": f"{sdv_ours:.2f}",
    "BaoPred": f"{pred(z_eff):.2f}", "BaoPredFiftySeven": f"{pred(0.57):.2f}", "BaoPull": f"{pull_ours:.1f}",
    "BaoBsq": f"{ours['w']['coef'][0]:.2f}",
    "PubAlpha": f"{a_pub:.3f}", "PubSigAlpha": f"{sa_pub:.3f}", "PubDvRd": f"{dvrd_pub:.2f}",
    "PubSigDvRd": f"{sdv_pub:.2f}", "PubDchi": f"{pubfit['dchi2']:.1f}",
    "PubSignif": f"{np.sqrt(max(pubfit['dchi2'], 0)):.1f}", "PubChiMin": f"{pubfit['w']['chimin']:.1f}",
    "PubNb": pubfit["nb"], "PubHart": f"{pubfit['hart']:.3f}",
    "CuestaDvRd": f"{dvrd_cuesta:.2f}", "CuestaSigStat": f"{sdv_cuesta_stat:.2f}",
    "DvQpm": f"{DV_QPM:.1f}", "QpmToOurs": f"{QPM_TO_OURS:.4f}",
    "AndRdCamb": f"{rd_and_camb:.2f}", "AndConv": f"{conv_and:.4f}", "AndDvRd": f"{dvrd_and:.2f}",
    "AndSigDvRd": f"{sdv_and:.2f}", "AndDvRdCons": f"{dvrd_and_cons:.2f}",
    "AlamDvA": f"{alam_dv[0]:.2f}", "AlamSdvA": f"{alam_sdv[0]:.2f}",
    "AlamDvB": f"{alam_dv[1]:.2f}", "AlamSdvB": f"{alam_sdv[1]:.2f}",
    "AlamDvC": f"{alam_dv[2]:.2f}", "AlamSdvC": f"{alam_sdv[2]:.2f}",
    "AlamPredB": f"{pred(0.51):.2f}", "AlamPredC": f"{pred(0.61):.2f}",
    "ErrRatio": f"{ratio_fit:.2f}", "AreaScale": f"{area_scale:.2f}",
    "ErrRatioOverArea": f"{ratio_fit / area_scale:.2f}",
    "SigRatioPub": f"{sa_ours / sa_pub:.2f}",
    "WtSumStand": f"{np.sum(cat['wcp'] + cat['wnoz'] - 1):.0f}",
    "WtExtraCp": f"{np.sum(cat['wcp'] - 1):.0f}", "WtExtraNoz": f"{np.sum(cat['wnoz'] - 1):.0f}",
    "WtMeanSys": f"{np.mean(cat['wsys']):.3f}",
    "BaoLocLo": f"{a_lo:.3f}", "BaoLocHi": f"{a_hi:.3f}", "BaoSigLoc": f"{sig_loc:.3f}",
    "BaoSigLocDvRd": f"{sig_loc * DV_FID / rd_h:.2f}", "BaoPlateau": f"{plateau:.1f}",
    "BaoEdgeLo": f"{edge_lo:.1f}", "PubSigLoc": f"{sig_loc_pub:.3f}",
    "BaoProd": f"{sig_loc * np.sqrt(max(ours['dchi2'], 0)):.3f}",
    "PubProd": f"{sig_loc_pub * np.sqrt(max(pubfit['dchi2'], 0)):.3f}",
    "BaoExpSignif": f"{np.sqrt(max(pubfit['dchi2'], 0)) / ratio_fit:.1f}",
    "BaoExpSigAlpha": f"{sig_loc_pub * ratio_fit:.3f}",
    # south is a subset of north + south: Var(a_S - a_NS) = s_S^2 - s_NS^2 (derived in the text)
    "DiffAlpha": f"{ours['w']['alpha'] - pubfit['w']['alpha']:.3f}",
    "SigDiffAlpha": f"{np.sqrt(max(sig_loc ** 2 - sig_loc_pub ** 2, 0)):.3f}",
    "DiffPull": f"{(ours['w']['alpha'] - pubfit['w']['alpha']) / np.sqrt(max(sig_loc ** 2 - sig_loc_pub ** 2, 1e-12)):.1f}",
    "PubAlphaBest": f"{pubfit['w']['alpha']:.3f}",
    # sqrt(Delta chi^2) scatters by one unit between surveys: how far below expectation is the south?
    "BaoShort": f"{np.sqrt(max(pubfit['dchi2'], 0)) / ratio_fit - np.sqrt(max(ours['dchi2'], 0)):.1f}",
    "BaoShortP": f"{100 * norm.cdf(np.sqrt(max(ours['dchi2'], 0)) - np.sqrt(max(pubfit['dchi2'], 0)) / ratio_fit):.0f}",
    "BaoEdgeWeight": f"{np.exp(-0.5 * edge_lo):.2f}",
})
print(f"ours: alpha {a_ours:.4f} +- {sa_ours:.4f}  chi2 {ours['w']['chimin']:.1f}/{ours['nb']-5}  dchi2 {ours['dchi2']:.1f}"
      f"  DV/rd {dvrd_ours:.2f} +- {sdv_ours:.2f}  (Planck {pred(z_eff):.2f})")
print(f"published data, our fit: alpha {a_pub:.4f} +- {sa_pub:.4f} -> DV/rd {dvrd_pub:.2f} +- {sdv_pub:.2f};"
      f" dchi2 {pubfit['dchi2']:.1f}; Cuesta quote {dvrd_cuesta:.2f}")
print(f"Anderson conversion r_s(EH)/r_d(CAMB) = {conv_and:.4f}; error ratio jk/mock {ratio_fit:.2f} vs area {area_scale:.2f}")
