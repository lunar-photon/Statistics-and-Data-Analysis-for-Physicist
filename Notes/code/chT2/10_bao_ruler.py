"""The BAO bump as a standard ruler: fitting the dilation alpha, and what DESI's distances say.

Question: (1) a survey measures the correlation function xi(r) of galaxies with a Gaussian
covariance. If we fit a template B^2 xi_fid(alpha r) plus a smooth polynomial, how well is the
dilation alpha determined, does Delta chi^2 = 1 give the right error bar over many mock surveys,
and how strongly is the bump detected against a template without wiggles (as Eisenstein et al.
2005 did)? (2) The DESI DR1 BAO distances D_M/r_d, D_H/r_d, D_V/r_d depend in flat LCDM only on
Omega_m and h r_d. What do they give, and do they agree with Planck?
Computes: xi(r) from the cached CAMB P(k) (code/chT1/02_pk_xi.py) with an Eisenstein-Hu (1998)
no-wiggle reference; the Gaussian covariance of binned xi; 1000 mock xi vectors fitted on an
alpha grid; a chi^2 grid in (Omega_m, h r_d) for the DESI Table 1 data.
Writes: figures/chT2/bao_fit.pdf, figures/chT2/bao_desi.pdf, results/chT2/10_bao_ruler.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line, DATA

C_KMS = 299792.458
pk = np.load(DATA / "chT1" / "pk.npz")
H = float(pk["h"])
OM_P, H0_P, RD_P = 0.3153, 67.36, 147.09          # Planck 2018 TT,TE,EE+lowE+lensing
OMB_H2 = 0.02237

# ---------------------------------------------------------------- (1) the ruler in xi(r)
k = np.logspace(-4, np.log10(5.0), 6000)
plin = np.interp(np.log(k), np.log(pk["k"]), pk["pk_lin"][0])


def eh_nowiggle(k, om=OM_P, h=H, ob=OMB_H2 / H**2, tcmb=2.7255):
    """Eisenstein & Hu (1998) eqs. (26), (28)-(31): transfer function without acoustic wiggles."""
    omh2, fb = om * h**2, ob / om
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * (ob * h**2) ** 0.75)          # Mpc
    a_gam = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb**2
    gam = om * h * (a_gam + (1 - a_gam) / (1 + (0.43 * k * h * s) ** 4))           # k in h/Mpc
    q = k * (tcmb / 2.7) ** 2 / gam
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


pnw = k**0.9649 * eh_nowiggle(k) ** 2
pnw *= np.mean(plin[k < 2e-3] / pnw[k < 2e-3])       # same amplitude where there are no wiggles
SIG_NL = 8.0                                          # BAO damping before reconstruction [Mpc/h]
B_GAL, NBAR, VOL = 2.0, 3e-4, 1.0e9                   # bias, density [(h/Mpc)^3], volume [(Mpc/h)^3]
pw = pnw + (plin - pnw) * np.exp(-0.5 * k**2 * SIG_NL**2)
damp = np.exp(-(k * 1.0) ** 2)                        # tiny Gaussian cut so the Hankel integral converges
redges = np.arange(50.0, 155.0, 5.0)
rc = 0.5 * (redges[1:] + redges[:-1])
lnk = np.log(k)


def xi_of(r, p):
    kr = np.outer(r, k)
    return np.trapezoid(k**3 * p * damp * np.sinc(kr / np.pi) / (2 * np.pi**2), lnk, axis=1)


rfine = np.linspace(20, 200, 721)
xw_f, xnw_f = xi_of(rfine, pw), xi_of(rfine, pnw)


def template(r, alpha, wig=True):
    return np.interp(alpha * r, rfine, xw_f if wig else xnw_f)


# Gaussian covariance of the shell-averaged xi:  (2/V) int dk k^2/(2pi^2) [b^2 P + 1/n]^2 j0bar_i j0bar_j
def j0bar(kv, r1, r2):
    f = lambda r: (np.sin(kv * r) - kv * r * np.cos(kv * r)) / kv**3
    return 3 * (f(r2) - f(r1)) / (r2**3 - r1**3)


kc = np.linspace(1e-4, 1.5, 15000)
pc = np.interp(kc, k, B_GAL**2 * pw) + 1 / NBAR
J = np.array([j0bar(kc, a, b) for a, b in zip(redges[:-1], redges[1:])])
COV = 2 / VOL * np.einsum("ik,jk,k->ij", J, J, kc**2 * pc**2 / (2 * np.pi**2)) * (kc[1] - kc[0])
CINV = np.linalg.inv(COV)
xi_true = B_GAL**2 * template(rc, 1.0)

alphas = np.linspace(0.8, 1.2, 401)
# design matrices: [xi_fid(alpha r), 1, 1/r, 1/r^2]; the linear parameters are solved exactly at each alpha
basis = lambda a, wig: np.column_stack([template(rc, a, wig), np.ones_like(rc), 1 / rc, 1 / rc**2])
PROJ = {}
for wig in (True, False):
    PROJ[wig] = []
    for a in alphas:
        X = basis(a, wig)
        F = X.T @ CINV @ X
        PROJ[wig].append((X, np.linalg.solve(F, X.T @ CINV)))


def chi2_curve(d, wig=True):
    out = np.empty(alphas.size)
    for i, (X, P) in enumerate(PROJ[wig]):
        res = d - X @ (P @ d)
        out[i] = res @ CINV @ res
    return out


def fit_alpha(d):
    cw, cn = chi2_curve(d, True), chi2_curve(d, False)
    i = np.argmin(cw)
    inside = alphas[cw <= cw[i] + 1]
    return alphas[i], 0.5 * (inside.max() - inside.min()), cw.min(), cn.min(), cw, cn


rng = rng_for("chT2", "10_bao_ruler")
Lc = np.linalg.cholesky(COV)
nmock = 1000
mocks = xi_true + (Lc @ rng.normal(0, 1, (rc.size, nmock))).T
# 9000 more mocks from their own stream, so that the coverage is known to half a per cent
rng_x = rng_for("chT2", "10_bao_ruler_extra")
mocks = np.vstack([mocks, xi_true + (Lc @ rng_x.normal(0, 1, (rc.size, 9000))).T])
nmock = len(mocks)


def fit_many(D):
    """fit_alpha for all rows of D at once: the same profile chi^2, one matrix product per trial alpha."""
    cw, cn = np.empty((len(D), alphas.size)), np.empty((len(D), alphas.size))
    for wig, out in ((True, cw), (False, cn)):
        for i, (X, P) in enumerate(PROJ[wig]):
            R = D - (P @ D.T).T @ X.T
            out[:, i] = np.einsum("ni,ij,nj->n", R, CINV, R)
    i = cw.argmin(1)
    cmin = cw[np.arange(len(D)), i]
    inside = cw <= cmin[:, None] + 1
    lo = np.where(inside, alphas[None, :], np.inf).min(1)
    hi = np.where(inside, alphas[None, :], -np.inf).max(1)
    return alphas[i], 0.5 * (hi - lo), cmin, cn.min(1)


ahat, aerr, cw_min, cn_min = fit_many(mocks)
signif = np.sqrt(np.maximum(cn_min - cw_min, 0))
cover = np.mean(np.abs(ahat - 1) < aerr)
cover_err = np.sqrt(cover * (1 - cover) / nmock)                  # binomial error of the coverage
d0 = mocks[0]
a0, e0, cw0, cn0, cwc, cnc = fit_alpha(d0)

# Fisher error on alpha (other parameters fixed at truth, then marginalised over the linear ones)
eps = 1e-3
dxa = B_GAL**2 * (template(rc, 1 + eps) - template(rc, 1 - eps)) / (2 * eps)
Xfull = np.column_stack([dxa, template(rc, 1.0), np.ones_like(rc), 1 / rc, 1 / rc**2])
sig_fisher = np.sqrt(np.linalg.inv(Xfull.T @ CINV @ Xfull)[0, 0])

setup(7.2, 2.6)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
sd = np.sqrt(np.diag(COV))
ax[0].errorbar(rc, rc**2 * d0, yerr=rc**2 * sd, fmt="o", ms=3, color=SERIES[0], label="one mock survey")
for wig, col, lab in [(True, SERIES[1], "BAO template"), (False, "0.5", "no-wiggle template")]:
    i = np.argmin(cwc if wig else cnc)
    X, P = PROJ[wig][i]
    ax[0].plot(rc, rc**2 * (X @ (P @ d0)), color=col, lw=1.3, label=lab)
ax[0].set(xlabel=r"$r$ [$h^{-1}$Mpc]", ylabel=r"$r^2\xi(r)$ [$h^{-2}$Mpc$^2$]")
ax[0].legend(fontsize=6.5, frameon=False, loc="lower left")
ax[1].plot(alphas, cwc - cwc.min(), color=SERIES[1], label="BAO template")
ax[1].plot(alphas, cnc - cwc.min(), color="0.5", label="no wiggles")
ax[1].axhline(1, color="0.6", lw=0.7, ls=":")
ax[1].set(xlabel=r"dilation $\alpha$", ylabel=r"$\chi^2-\chi^2_{\min}$", ylim=(0, 25))
ax[1].legend(fontsize=6.5, frameon=False)
ax[2].hist(ahat, bins=np.linspace(0.85, 1.15, 41), density=True, color=SERIES[0], alpha=0.7)
aa = np.linspace(0.85, 1.15, 300)
theory_line(ax[2], aa, np.exp(-0.5 * ((aa - 1) / sig_fisher) ** 2) / (np.sqrt(2 * np.pi) * sig_fisher),
            label="Fisher Gaussian")
ax[2].set(xlabel=r"$\hat\alpha$ over mock surveys", ylabel="density")
ax[2].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout()
savefig(fig, "chT2", "bao_fit")

# ---------------------------------------------------------------- (2) DESI DR1 distances (Table 1)
# z_eff, kind, values, errors, correlation r
DESI = [(0.295, "V", 7.93, 0.15),
        (0.510, "MH", (13.62, 20.98), (0.25, 0.61), -0.445),
        (0.706, "MH", (16.85, 20.08), (0.32, 0.60), -0.420),
        (0.930, "MH", (21.71, 17.88), (0.28, 0.35), -0.389),
        (1.317, "MH", (27.79, 13.82), (0.69, 0.42), -0.444),
        (1.491, "V", 26.07, 0.67),
        (2.330, "MH", (39.71, 8.52), (0.94, 0.17), -0.477)]
zg = np.linspace(0, 2.5, 2501)


def dists(om, hrd):
    """D_M/r_d, D_H/r_d on the grid zg, flat LCDM, radiation neglected; depends on Om and h r_d only."""
    E = np.sqrt(om * (1 + zg) ** 3 + 1 - om)
    dh = C_KMS / (100 * hrd) / E
    dm = np.concatenate([[0], np.cumsum(0.5 * (dh[1:] + dh[:-1]) * np.diff(zg))])
    return dm, dh


def chi2_desi(om, hrd):
    dm, dh = dists(om, hrd)
    tot = 0.0
    for row in DESI:
        z = row[0]
        DM, DH = np.interp(z, zg, dm), np.interp(z, zg, dh)
        if row[1] == "V":
            DV = (z * DM**2 * DH) ** (1 / 3)
            tot += ((DV - row[2]) / row[3]) ** 2
        else:
            (vm, vh), (sm, sh), r = row[2], row[3], row[4]
            C = np.array([[sm**2, r * sm * sh], [r * sm * sh, sh**2]])
            dv = np.array([DM - vm, DH - vh])
            tot += dv @ np.linalg.solve(C, dv)
    return tot


oms = np.linspace(0.22, 0.38, 161)
hrds = np.linspace(96, 108, 121)
chi = np.array([[chi2_desi(o, h) for h in hrds] for o in oms])
post = np.exp(-0.5 * (chi - chi.min()))
post /= post.sum()
om_mean = (post.sum(1) * oms).sum(); om_sd = np.sqrt((post.sum(1) * (oms - om_mean) ** 2).sum())
hr_mean = (post.sum(0) * hrds).sum(); hr_sd = np.sqrt((post.sum(0) * (hrds - hr_mean) ** 2).sum())
corr = (post * np.outer(oms - om_mean, hrds - hr_mean)).sum() / (om_sd * hr_sd)
hrd_P = H0_P / 100 * RD_P
chi_P = chi2_desi(OM_P, hrd_P)
H0_from = 100 * hr_mean / RD_P
H0_from_err = 100 * hr_sd / RD_P
ndata = sum(1 if r[1] == "V" else 2 for r in DESI)

setup(7.0, 2.8)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
dmP, dhP = dists(OM_P, hrd_P)
dvP = (zg * dmP**2 * dhP) ** (1 / 3)
for row in DESI:
    z = row[0]
    if row[1] == "V":
        ref = np.interp(z, zg, dvP)
        ax[0].errorbar(z, row[2] / ref, row[3] / ref, fmt="D", color=SERIES[2], ms=4)
    else:
        for v, s, ref, col, sh in [(row[2][0], row[3][0], np.interp(z, zg, dmP), SERIES[0], -0.012),
                                   (row[2][1], row[3][1], np.interp(z, zg, dhP), SERIES[1], 0.012)]:
            ax[0].errorbar(z + sh, v / ref, s / ref, fmt="o", color=col, ms=4)
ax[0].axhline(1, color="k", ls="--", lw=1)
for lab, col, m in [(r"$D_M/r_d$", SERIES[0], "o"), (r"$D_H/r_d$", SERIES[1], "o"), (r"$D_V/r_d$", SERIES[2], "D")]:
    ax[0].plot([], [], m, color=col, label=lab)
ax[0].set(xlabel="effective redshift", ylabel="DESI / Planck $\\Lambda$CDM", ylim=(0.9, 1.1))
ax[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
lv = chi.min() + np.array([2.30, 6.18])
ax[1].contourf(oms, hrds, chi.T, levels=[chi.min(), lv[0], lv[1]], colors=[SERIES[0], "#a9c8ee"])
ax[1].plot(OM_P, hrd_P, "*", color=SERIES[1], ms=9, label="Planck 2018")
ax[1].errorbar(0.295, 101.8, xerr=0.015, yerr=1.3, fmt="x", color="k", ms=5, label="DESI quoted")
ax[1].set(xlabel=r"$\Omega_m$", ylabel=r"$h\,r_d$ [Mpc]", xlim=(0.24, 0.36), ylim=(97, 106))
ax[1].legend(fontsize=7, frameon=False)
fig.tight_layout()
savefig(fig, "chT2", "bao_desi")

save_numbers("chT2", "10_bao_ruler", {
    "TwbaoSigNl": int(SIG_NL), "TwbaoB": f"{B_GAL:.0f}", "TwbaoNbar": "3\\times10^{-4}",
    "TwbaoVol": "1", "TwbaoNbins": rc.size, "TwbaoNmock": nmock,
    "TwbaoAerrMean": f"{aerr.mean():.4f}", "TwbaoAsd": f"{ahat.std(ddof=1):.4f}",
    "TwbaoAmean": f"{ahat.mean():.4f}", "TwbaoFisher": f"{sig_fisher:.4f}",
    "TwbaoCover": f"{100 * cover:.0f}\\%", "TwbaoCoverErr": f"{100 * cover_err:.1f}\\%", "TwbaoSigMed": f"{np.median(signif):.1f}",
    "TwbaoSigLow": f"{np.percentile(signif, 16):.1f}", "TwbaoSigHigh": f"{np.percentile(signif, 84):.1f}",
    "TwbaoAzero": f"{a0:.3f}", "TwbaoEzero": f"{e0:.3f}", "TwbaoDchiZero": f"{cn0 - cw0:.1f}",
    "TwbaoChiZero": f"{cw0:.1f}", "TwbaoDofZero": rc.size - 5,
    "TwdOm": f"{om_mean:.3f}", "TwdOmSd": f"{om_sd:.3f}", "TwdHrd": f"{hr_mean:.1f}", "TwdHrdSd": f"{hr_sd:.1f}",
    "TwdCorr": f"{corr:.2f}", "TwdChiP": f"{chi_P:.1f}", "TwdNdata": ndata, "TwdChiMin": f"{chi.min():.1f}",
    "TwdHrdP": f"{hrd_P:.1f}", "TwdHzero": f"{H0_from:.1f}", "TwdHzeroErr": f"{H0_from_err:.1f}",
})
print(f"alpha: mean {ahat.mean():.4f}, sd {ahat.std():.4f}, mean dchi2=1 err {aerr.mean():.4f}, "
      f"Fisher {sig_fisher:.4f}, cover {cover:.3f}; significance median {np.median(signif):.2f}")
print(f"DESI: Om {om_mean:.4f}+-{om_sd:.4f}, hrd {hr_mean:.2f}+-{hr_sd:.2f}, corr {corr:.2f}; "
      f"chi2(Planck) {chi_P:.2f} for {ndata} points, chi2min {chi.min():.2f}; H0(rd Planck) {H0_from:.2f}+-{H0_from_err:.2f}")
