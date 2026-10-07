"""17_biref_fisher.py -- how well can the CMB measure a uniform rotation angle beta?  A Fisher forecast.

Question: an experiment measures the T, E, B maps with some noise, beam and sky fraction.  How
small an angle can it detect, which multipoles carry the information, and what happens when the
instrument's own angle alpha is unknown and only a dusty channel can tell it from beta?

Computes (no random numbers):
  1. sigma(beta) for a Planck-like set-up (HFI 100+143+217 noise, l = 51..1500) from EB, TB and both,
     with the closed-form Fisher information of lib_biref10.info_beta, checked against the numerical
     3x3 trace of the T, E, B covariance;
  2. the information per multipole for the Planck-like and a deep experiment;
  3. the dials: noise depth, beam, l_max (and the lensing floor);
  4. what EE and BB add at beta = 0.35 deg (information that grows like beta^2);
  5. alpha and beta together: one Planck-like channel at a time (100, 143, 217, 353 GHz, with dust),
     then all four channels with one alpha per channel, for several sky fractions and noise levels.
Writes: figures/ch10/biref_info.pdf, figures/ch10/biref_dials.pdf, figures/ch10/biref_alpha_beta.pdf,
        results/ch10/17_biref_fisher.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, theory_line
import lib_biref10 as LB

DEG = LB.DEG
LMIN, LMAX = 51, 1500
FSKY = 0.7
cl = LB.fiducial(3000)
NT, NP = LB.planck_like(3000)
nums = {}


def sci(x, d=1):
    """A small positive number as a LaTeX string a.b x 10^{n} (so that it can carry a degree sign)."""
    m, e = f"{abs(x):.{d}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def sig(NT_, NP_, lmin=LMIN, lmax=LMAX, fsky=FSKY, which="joint", cl_=cl):
    return LB.sigma_beta(cl_, NT_, NP_, lmin, lmax, fsky, which) / DEG


# ---------------------------------------------------------------- 1. Planck-like headline
for w, name in [("EB", "EB"), ("TB", "TB"), ("joint", "Joint")]:
    nums[f"TenCPl{name}"] = round(sig(NT, NP, which=w), 4)
    nums[f"TenCPl{name}Nine"] = round(sig(NT, NP, fsky=0.9, which=w), 4)
print("Planck-like sigma(beta) [deg] EB, TB, joint:", nums["TenCPlEB"], nums["TenCPlTB"], nums["TenCPlJoint"])
nums["TenCPlDepthP"] = round(float(np.sqrt(NP[2]) / LB.ARCMIN), 1)   # muK arcmin at l -> 0
nums["TenCPlDepthT"] = round(float(np.sqrt(NT[2]) / LB.ARCMIN), 1)


# numerical check: 3x3 trace of the (T, E, B) covariance with the exact rotated spectra
def cov_teb(beta, l, NT_=NT, NP_=NP, delens=False):
    tt, ee, bb, te = (cl[k][l] for k in ("TT", "EE", "BB", "TE"))
    if delens:
        bb = 0 * bb
    c2, s2, c4, s4 = np.cos(2 * beta), np.sin(2 * beta), np.cos(4 * beta), np.sin(4 * beta)
    C = np.zeros((len(l), 3, 3))
    C[:, 0, 0] = tt + NT_[l]
    C[:, 1, 1] = c2 ** 2 * ee + s2 ** 2 * bb + NP_[l]
    C[:, 2, 2] = s2 ** 2 * ee + c2 ** 2 * bb + NP_[l]
    C[:, 0, 1] = C[:, 1, 0] = c2 * te
    C[:, 0, 2] = C[:, 2, 0] = s2 * te
    C[:, 1, 2] = C[:, 2, 1] = 0.5 * s4 * (ee - bb)
    return C


l = np.arange(LMIN, LMAX + 1)
Fnum = LB.fisher_numeric(lambda p: cov_teb(p[0], l), [0.0], l, FSKY, h=1e-4)[0, 0]
Fcl = LB.info_beta(cl, NT, NP, LMIN, LMAX, FSKY, "joint")[1].sum()
nums["TenCClosedCheck"] = sci(Fnum / Fcl - 1)
# 4. EE and BB at beta = 0.35 deg: the full trace at beta0 against the first-order (EB, TB) part
b0 = 0.35 * DEG
Ffull = LB.fisher_numeric(lambda p: cov_teb(p[0], l), [b0], l, FSKY, h=1e-5)[0, 0]
nums["TenCRotInvariance"] = sci(Ffull / Fnum - 1)
# the part of it carried by EE and BB alone at beta0: (1/2) sum n [(dEE/EE~)^2 + (dBB/BB~)^2]
ee, bb, s4 = cl["EE"][l], cl["BB"][l], np.sin(4 * b0)
dEE, dBB = -2 * s4 * (ee - bb), 2 * s4 * (ee - bb)
Et = np.cos(2 * b0) ** 2 * ee + np.sin(2 * b0) ** 2 * bb + NP[l]
Bt = np.sin(2 * b0) ** 2 * ee + np.cos(2 * b0) ** 2 * bb + NP[l]
F_eebb = np.sum(0.5 * (2 * l + 1) * FSKY * ((dEE / Et) ** 2 + (dBB / Bt) ** 2))
nums["TenCEEBBshare"] = sci(F_eebb / Fnum)
print("closed-form check", nums["TenCClosedCheck"], " F(0.35deg)/F(0)-1", nums["TenCRotInvariance"],
      " EE+BB share at 0.35 deg", nums["TenCEEBBshare"])

# ---------------------------------------------------------------- 2. information per multipole
NTd, NPd = LB.white(2.0 / np.sqrt(2), 3.0), LB.white(2.0, 3.0)        # deep: 2 muK' in P, 3' beam
nums["TenCDeepJoint"] = round(sig(NTd, NPd, lmax=3000), 4)
nums["TenCDeepJointFifteen"] = round(sig(NTd, NPd, lmax=LMAX), 4)
setup(7.0, 5.4)
fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 1.15]))
L3 = np.arange(2, 3001)
dl = L3 * (L3 + 1) / (2 * np.pi)
ax[0].loglog(L3, dl * cl["EE"][L3], color=SERIES[0], label=r"$EE$ (the lever)")
ax[0].loglog(L3, dl * cl["BB"][L3], color=SERIES[1], label=r"$BB$ from lensing")
ax[0].loglog(L3, dl * NP[L3], color=SERIES[2], ls="--", label=r"noise $N_\ell^P$, Planck-like")
ax[0].loglog(L3, dl * NPd[L3], color=SERIES[3], ls="--", label=r"noise $N_\ell^P$, deep")
ax[0].set_ylim(1e-4, 1e2); ax[0].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
ax[0].legend(fontsize=7.5, ncol=2, loc="lower left")
half = {}
for (nt, np_, lab, col, ls, key) in [(NT, NP, "Planck-like, EB", SERIES[0], ":", "PlEB"),
                                      (NT, NP, "Planck-like, TB", SERIES[1], ":", "PlTB"),
                                      (NT, NP, "Planck-like, EB+TB", SERIES[0], "-", "Pl"),
                                      (NTd, NPd, "deep, EB+TB", SERIES[3], "-", "Deep")]:
    which = "EB" if key.endswith("EB") else ("TB" if key.endswith("TB") else "joint")
    ll, f = LB.info_beta(cl, nt, np_, 2, 3000, 1.0, which)
    _, fj = LB.info_beta(cl, nt, np_, 2, 3000, 1.0, "joint")
    ax[1].semilogx(ll, f / fj.max(), color=col, ls=ls, label=lab)
    if which == "joint":
        c = np.cumsum(f) / f.sum()
        half[key] = int(ll[np.searchsorted(c, 0.5)])
        nums[f"TenCHalf{key}"] = half[key]
        nums[f"TenCLowFrac{key}"] = round(100 * f[ll < 200].sum() / f.sum(), 1)
        nums[f"TenCPeak{key}"] = int(ll[np.argmax(f)])
ax[1].set_xlabel(r"multipole $\ell$"); ax[1].set_ylabel(r"information per $\ell$ (scaled)")
ax[1].legend(fontsize=7.5); ax[1].set_xlim(2, 3000)
savefig(fig, "ch10", "biref_info")

# ---------------------------------------------------------------- 3. the dials
setup(7.2, 2.7)
fig, ax = plt.subplots(1, 3)
depths = np.logspace(-1, np.log10(300), 60)
s15, s30, snl = [], [], []
for d in depths:
    nt, np_ = LB.white(d / np.sqrt(2), 5.0), LB.white(d, 5.0)
    s15.append(sig(nt, np_)); s30.append(sig(nt, np_, lmax=3000))
    cl0 = dict(cl); cl0["BB"] = 0 * cl["BB"]
    snl.append(sig(nt, np_, lmax=3000, cl_=cl0))
ax[0].loglog(depths, s15, color=SERIES[0], label=r"$\ell\leq1500$")
ax[0].loglog(depths, s30, color=SERIES[1], label=r"$\ell\leq3000$")
ax[0].loglog(depths, snl, color=SERIES[2], ls="--", label="no lensing $B$")
ax[0].set_xlabel(r"depth $\Delta_P$ [$\mu$K-arcmin]"); ax[0].set_ylabel(r"$\sigma(\beta)$ [deg]")
ax[0].legend(fontsize=7)
for d, key in [(0.1, "Floor"), (1.0, "One"), (10.0, "Ten"), (60.0, "Sixty")]:
    nt, np_ = LB.white(d / np.sqrt(2), 5.0), LB.white(d, 5.0)
    v_ = sig(nt, np_, lmax=3000)
    nums[f"TenCDial{key}"] = f"{v_:.2g}" if v_ > 1e-3 else sci(v_)
    cl0 = dict(cl); cl0["BB"] = 0 * cl["BB"]
    v_ = sig(nt, np_, lmax=3000, cl_=cl0)
    nums[f"TenCDial{key}NoLens"] = f"{v_:.2g}" if v_ > 1e-3 else sci(v_)
# local slope d ln sigma / d ln depth at 60 and 300 muK'
for d, key in [(3.0, "Three"), (60.0, "Sixty"), (300.0, "ThreeHundred")]:
    a = [sig(LB.white(x / np.sqrt(2), 5.0), LB.white(x, 5.0)) for x in (d / 1.05, d * 1.05)]
    nums[f"TenCSlope{key}"] = round(float(np.log(a[1] / a[0]) / np.log(1.05 ** 2)), 2)
fwhms = np.linspace(1, 60, 60)
for d, col in [(10.0, SERIES[0]), (60.0, SERIES[1])]:
    ax[1].semilogy(fwhms, [sig(LB.white(d / np.sqrt(2), f), LB.white(d, f), lmax=3000) for f in fwhms],
                   color=col, label=rf"{d:g} $\mu$K$'$")
ax[1].set_xlabel("beam FWHM [arcmin]"); ax[1].legend(fontsize=7)
lmaxs = np.arange(100, 3001, 50)
for d, col in [(1.0, SERIES[2]), (10.0, SERIES[0]), (60.0, SERIES[1])]:
    nt, np_ = LB.white(d / np.sqrt(2), 5.0), LB.white(d, 5.0)
    ax[2].semilogy(lmaxs, [sig(nt, np_, lmax=m) for m in lmaxs], color=col, label=rf"{d:g} $\mu$K$'$")
ax[2].set_xlabel(r"$\ell_{\max}$"); ax[2].legend(fontsize=7)
for a in ax:
    a.grid(alpha=0.3)
savefig(fig, "ch10", "biref_dials")
for f in (5.0, 30.0):
    nums["TenCBeam" + ("Five" if f == 5 else "Thirty")] = float(f"{sig(LB.white(60 / np.sqrt(2), f), LB.white(60, f), lmax=3000):.2g}")
for m, key in [(500, "FiveHundred"), (1000, "Thousand"), (1500, "FifteenHundred"), (3000, "ThreeThousand")]:
    nums[f"TenCLmax{key}"] = float(f"{sig(LB.white(60 / np.sqrt(2), 5.0), LB.white(60, 5.0), lmax=m):.2g}")

# ---------------------------------------------------------------- 5. alpha and beta
FREQS = [100, 143, 217, 353]
lab = np.arange(LMIN, LMAX + 1)
cmb = (cl["EE"][lab], cl["BB"][lab])


def chan(nu, scale=1.0):
    fwhm, dT, dP = LB.PLANCK[nu]
    ee, bb = LB.dust(nu, 3000)
    return (ee[lab], bb[lab]), scale * LB.white(60 * dP, fwhm, 3000)[lab]


single = {}
for nu in FREQS:
    fg, nn = chan(nu)
    F = LB.fisher_numeric(lambda p: LB.cov_eb(lab, cmb, [fg], [nn], [p[0]], p[1]), [0.0, 0.0], lab, FSKY)
    C = np.linalg.inv(F)
    sa, sb = np.sqrt(C[0, 0]) / DEG, np.sqrt(C[1, 1]) / DEG
    ssum = np.sqrt(C[0, 0] + C[1, 1] + 2 * C[0, 1]) / DEG
    rho = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
    single[nu] = (F, sa, sb, ssum, rho)
    w = {100: "Hundred", 143: "OneFortyThree", 217: "TwoSeventeen", 353: "ThreeFiftyThree"}[nu]
    nums[f"TenCSA{w}"] = float(f"{sa:.2g}"); nums[f"TenCSB{w}"] = float(f"{sb:.2g}")
    nums[f"TenCSSum{w}"] = float(f"{ssum:.2g}"); nums[f"TenCRho{w}"] = f"{rho:.3f}"
    print(nu, "sigma alpha", sa, "beta", sb, "sum", ssum, "rho", rho)


def multi(fsky=FSKY, scale=1.0, dust_amp=1.0):
    fgs, nns = [], []
    for nu in FREQS:
        fg, nn = chan(nu, scale)
        fgs.append((dust_amp * fg[0], dust_amp * fg[1])); nns.append(nn)
    cov = lambda p: LB.cov_eb(lab, cmb, fgs, nns, p[:4], p[4])
    F = LB.fisher_numeric(cov, np.zeros(5), lab, fsky)
    C = np.linalg.inv(F)
    sb = np.sqrt(C[4, 4]) / DEG
    sb_fixed = 1 / np.sqrt(F[4, 4]) / DEG          # all alpha_nu known
    return F, C, sb, sb_fixed


F4, C4, sb4, sb4fix = multi()
nums["TenCMultiSB"] = float(f"{sb4:.2g}")
nums["TenCMultiSBfixed"] = float(f"{sb4fix:.2g}")
for k, nu in enumerate(FREQS):
    w = {100: "Hundred", 143: "OneFortyThree", 217: "TwoSeventeen", 353: "ThreeFiftyThree"}[nu]
    nums[f"TenCMultiSA{w}"] = float(f"{np.sqrt(C4[k, k]) / DEG:.2g}")
    nums[f"TenCMultiRho{w}"] = round(float(C4[k, 4] / np.sqrt(C4[k, k] * C4[4, 4])), 2)
for fs, key in [(0.5, "Fifty"), (0.9, "Ninety")]:
    nums[f"TenCMultiSBfsky{key}"] = float(f"{multi(fsky=fs)[2]:.2g}")
for sc, key in [(0.1, "Tenth"), (0.3, "Third"), (3.0, "Triple")]:
    nums[f"TenCMultiSBnoise{key}"] = float(f"{multi(scale=sc)[2]:.2g}")
for da, key in [(0.3, "Third"), (3.0, "Triple")]:
    nums[f"TenCMultiSBdust{key}"] = float(f"{multi(dust_amp=da)[2]:.2g}")
# CMB only, four channels: the alpha's and beta are exactly degenerate
F0 = LB.fisher_numeric(lambda p: LB.cov_eb(lab, cmb, None, [chan(nu)[1] for nu in FREQS], p[:4], p[4]),
                       np.zeros(5), lab, FSKY)
ev = np.linalg.eigvalsh(F0)
nums["TenCCondCMBonly"] = sci(ev[0] / ev[-1], 0)
print("multi: sigma beta", sb4, "fixed alphas", sb4fix, "smallest/largest eigenvalue CMB only", ev[0] / ev[-1])

# figure: single-channel sigma(beta) vs frequency at one fixed depth, and the (alpha_143, beta) ellipses
setup(7.2, 3.3)
fig, ax = plt.subplots(1, 2)
nus = np.linspace(60, 450, 60)
for depth, col in [(60.0, SERIES[0]), (10.0, SERIES[1])]:
    sbs = []
    for nu in nus:
        ee, bb = LB.dust(nu, 3000)
        nn = LB.white(depth, 5.0, 3000)[lab]
        F = LB.fisher_numeric(lambda p: LB.cov_eb(lab, cmb, [(ee[lab], bb[lab])], [nn], [p[0]], p[1]),
                              [0.0, 0.0], lab, FSKY)
        sbs.append(np.sqrt(np.linalg.inv(F)[1, 1]) / DEG)
    ax[0].semilogy(nus, sbs, color=col, label=rf"one channel, {depth:g} $\mu$K$'$, 5$'$")
ax[0].semilogy(FREQS, [single[n][2] for n in FREQS], "o", color=SERIES[2], label="Planck channels, alone")
ax[0].axhline(sb4, color=SERIES[3], ls="--", label="Planck, four together")
ax[0].set_xlabel(r"frequency $\nu$ [GHz]"); ax[0].set_ylabel(r"$\sigma(\beta)$ [deg]")
ax[0].legend(fontsize=6.5); ax[0].grid(alpha=0.3)
t = np.linspace(0, 2 * np.pi, 300)


def ell(ax, C2, col, lab_, ls="-"):
    w, v = np.linalg.eigh(C2)
    pts = v @ (np.sqrt(w)[:, None] * np.vstack([np.cos(t), np.sin(t)])) / DEG
    ax.plot(pts[0], pts[1], color=col, ls=ls, label=lab_)


Fs143 = single[143][0]
s_sum143 = 1 / np.sqrt(LB.fisher_numeric(lambda p: LB.cov_eb(lab, cmb, None, [chan(143)[1]], [p[0]], 0.0),
                                         [0.0], lab, FSKY)[0, 0]) / DEG
x = np.linspace(-1.2, 1.2, 2)
ax[1].fill_between(x, -x - s_sum143, -x + s_sum143, color=SERIES[1], alpha=0.2, lw=0,
                   label=r"143 GHz, CMB only: $\alpha+\beta$")
ell(ax[1], np.linalg.inv(Fs143), SERIES[0], "143 GHz with dust")
ell(ax[1], np.linalg.inv(single[353][0]), SERIES[2], "353 GHz with dust", ls=":")
ell(ax[1], C4[np.ix_([1, 4], [1, 4])], SERIES[3], r"four channels: $(\alpha_{143},\beta)$")
ax[1].set_xlim(-1.2, 1.2); ax[1].set_ylim(-1.2, 1.2); ax[1].set_aspect("equal")
ax[1].set_xlabel(r"$\alpha$ [deg]"); ax[1].set_ylabel(r"$\beta$ [deg]"); ax[1].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)  # below the panel, clear of the contours
savefig(fig, "ch10", "biref_alpha_beta")
nums["TenCSumOneFortyThreeCMB"] = float(f"{s_sum143:.2g}")
save_numbers("ch10", "17_biref_fisher", nums)
for k, v in nums.items():
    print(k, v)
