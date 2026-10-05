"""14_kim2023.py -- reproduce Kim, Lenoci, Stomberg & Xue (2023): an IMRI in a compressed (wave) dark-matter halo.

Setup of the paper (Sec. III, Table I): m1 = 1e4 Msun, m2 = 1 Msun (Mc = 39.8 Msun, q = 1e-4), spike slope 7/3,
rho6 = 25e15 Msun/pc^3 at r6 = 1e-6 pc, D_L = 203 Mpc, last five years before coalescence, band up to
min(1 Hz, f_ISCO), Robson et al. sensitivity, angle-averaged strain sqrt(4/5) A(f) (their eqs. 44-45).
Friction  dE_f/dt = 4 pi (G m2)^2 rho(<v) C / v,  rho(<v) ~ 0.6 rho  (their eq. 39), with
  particle DM:  C_p = (1+L)/L ln[1+L+sqrt(L(2+L))] - sqrt(1+2/L),  L = v^2 r/(G m2)      (eq. 41)
  wave DM:      C_w = Cin(2kr) - 1 + sin(2kr)/(2kr),  kr = sqrt(r/a)                       (eq. 42)
density  rho6 (r6/r)^g (1 - 2R_S/r)^g (particle, eq. 6)  or  rho6 (r6/r)^g (1 - R_c/2r)^g (wave, eq. 40),
R_c = a l_c (l_c + 1), l_c from 2 max_n Gamma_nl t_halo = 1 with the decay rates of eq. (31), and the backreaction
cap dE_f/dt -> (1/Edot_f + 2/Udot)^{-1} (eq. 43).
Questions:
 (1) the time-domain dephasing |Phi - Phi_V| (their eq. 49, Fig. 3 top) for the particle halo and for wave halos
     with Bohr radii a = 1e-11, 1e-10, 1e-9 pc, against their Fig. 3 digitised from the PDF;
 (2) the signal-to-noise ratio of the benchmark (they quote ~15);
 (3) Fisher errors on (Mc, log10 q, rho6, gamma_sp, log10 a), marginalised over (ln A, t_c, phi_c), against the
     widths of their posterior (Fig. 4, 2.5% and 97.5% quantiles).
Writes: figures/ch10/kim_dephasing.pdf, kim_fisher.pdf, results/ch10/14_kim2023.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.special import sici, loggamma, gammaln
from scipy.optimize import brentq
from common import setup, savefig, save_numbers, SERIES
import lib_gw10 as L

setup()
HBARC_EVM = 1.973269804e-7          # hbar c [eV m]
HBAR_EVS = 6.582119569e-16          # hbar [eV s]
T_HALO = 13.6e9 * L.YEAR            # age of a halo formed at z ~ 20
R6 = 1e-6 * L.PC
TRUE = dict(Mc=39.8, lq=-4.0, rho6=25.0, gam=7 / 3, la=-11.0)   # rho6 in 1e15 Msun/pc^3, a in pc
DL = 203 * L.MPC


def masses(Mc, q):
    m1 = Mc * (1 + q) ** 0.2 / q**0.6
    return m1, q * m1


def ell_c(m1, a_pc):
    """Lowest angular momentum that survives absorption for t_halo: 2 Gamma_{n l} t_halo = 1, l continuous.
    Gamma_nl/m = alpha^{4l+5} 2^{4l+3} (n+l)!/(n^{2l+4} (n-l-1)!) [l!/((2l)!(2l+1)!)]^2 prod_k (k^2+(4 alpha)^2),
    n = max(l+1, sqrt((l+1)(l+2)(2l+3))/3); alpha = G m1 m/(hbar c) = sqrt(r_g/a), m c^2 = hbar c/sqrt(a r_g)."""
    rg = L.G * m1 * L.MSUN / L.c**2
    a = a_pc * L.PC
    alpha = np.sqrt(rg / a)
    mc2 = HBARC_EVM / np.sqrt(a * rg)                       # eV
    omega = mc2 / HBAR_EVS                                  # 1/s
    cc = 4 * alpha

    def lnrate(l):
        n = max(l + 1, np.sqrt((l + 1) * (l + 2) * (2 * l + 3)) / 3)
        lp = (np.real(loggamma(l + 1 + 1j * cc) + loggamma(l + 1 - 1j * cc)) - 2 * np.real(loggamma(1 + 1j * cc)))
        lg = ((4 * l + 5) * np.log(alpha) + (4 * l + 3) * np.log(2) + gammaln(n + l + 1) - (2 * l + 4) * np.log(n)
              - gammaln(n - l) + 2 * (gammaln(l + 1) - gammaln(2 * l + 1) - gammaln(2 * l + 2)) + lp)
        return lg + np.log(omega) + np.log(2 * T_HALO)

    if lnrate(0.0) < 0:
        return 0.0, alpha, mc2
    return brentq(lnrate, 0.0, 400.0), alpha, mc2


def Cin(z):
    return np.euler_gamma + np.log(z) - sici(z)[1]


def eps_kim(f, Mc, q, rho6, gam, a_pc=None, cap=True):
    """P_f,eff / P_GW along the inspiral (a_pc = None: particle halo)."""
    m1, m2 = masses(Mc, q)
    M = m1 + m2
    r = L.r_of_f(f, M)
    v = np.sqrt(L.G * M * L.MSUN / r)
    rho = rho6 * 1e15 * L.MSUN / L.PC**3 * (R6 / r) ** gam
    if a_pc is None:
        RS = 2 * L.G * m1 * L.MSUN / L.c**2
        rho = rho * np.clip(1 - 2 * RS / r, 0, None) ** gam
        Lam = v**2 * r / (L.G * m2 * L.MSUN)
        C = (1 + Lam) / Lam * np.log(1 + Lam + np.sqrt(Lam * (2 + Lam))) - np.sqrt(1 + 2 / Lam)
        rin = 2 * RS
    else:
        lc, _, _ = ell_c(m1, a_pc)
        Rc = a_pc * L.PC * lc * (lc + 1)
        rho = rho * np.clip(1 - Rc / (2 * r), 0, None) ** gam
        kr = np.sqrt(r / (a_pc * L.PC))
        C = Cin(2 * kr) - 1 + np.sin(2 * kr) / (2 * kr)
        rin = Rc / 2
    Ef = 4 * np.pi * (L.G * m2 * L.MSUN) ** 2 * 0.6 * rho * C / v
    Pgw = 32 / 5 * L.c**5 / L.G * (L.G * Mc * L.MSUN * np.pi * f / L.c**3) ** (10 / 3)
    if not cap:
        return Ef / Pgw
    # enclosed dark-matter mass (inner edge rin) for the binding energy of a shell
    rr = np.geomspace(max(rin, 1e-3 * r.min()) * 1.0000001, r.max(), 4000)
    rho_r = np.interp(rr, r, rho, left=0.0)
    if a_pc is None:
        rho_r = rho6 * 1e15 * L.MSUN / L.PC**3 * (R6 / rr) ** gam * np.clip(1 - 2 * RS / rr, 0, None) ** gam
    else:
        rho_r = rho6 * 1e15 * L.MSUN / L.PC**3 * (R6 / rr) ** gam * np.clip(1 - Rc / (2 * rr), 0, None) ** gam
    menc_r = cumulative_trapezoid(4 * np.pi * rr**2 * rho_r, rr, initial=0.0)
    menc = np.interp(r, rr, menc_r)
    # Udot = G (m1 + m_enc) 4 pi r rho |rdot|,  |rdot| = 2 r^2 (P_GW + x)/(G m1 m2):  Udot = kappa (P_GW + x)
    kappa = (m1 * L.MSUN + menc) * 4 * np.pi * r * rho * 2 * r**2 / (m1 * m2 * L.MSUN**2)
    B = kappa * (Ef - Pgw) - 2 * Ef
    x = (B + np.sqrt(B**2 + 4 * kappa**2 * Ef * Pgw)) / (2 * np.where(kappa > 0, kappa, 1.0))     # solves 1/x = 1/Ef + 2/(kappa (P_GW + x))
    return np.where(kappa > 0, x, 0.0) / Pgw


def tl_from(f, fdot):
    return -cumulative_trapezoid((1 / fdot)[::-1], f[::-1], initial=0)[::-1]


def two_pi_int(f, g):
    return 2 * np.pi * (-cumulative_trapezoid(g[::-1], f[::-1], initial=0)[::-1])


# ---------------------------------------------------------------- digitise their Fig. 3 (top)
def digitise_fig3():
    import subprocess, tempfile
    from PIL import Image
    pdf = pathlib.Path(__file__).resolve().parents[2] / "references" / "Kim2023.pdf"
    cache = sorted((pathlib.Path(__file__).resolve().parents[2] / "data" / "ch10").glob("kim_p-*.png"))
    if cache:                                   # pre-rendered page (the cluster has no poppler)
        im = np.array(Image.open(cache[0]).convert("RGB")).astype(int)
    else:
      with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-f", "10", "-l", "10", "-r", "220", "-png", str(pdf), tmp + "/p"], check=True)
        im = np.array(Image.open(next(pathlib.Path(tmp).glob("p*.png"))).convert("RGB")).astype(int)
    sub = im[180:1050, 150:950]
    dark = sub.sum(axis=2) < 150
    y0, y1 = np.where(dark.sum(axis=1) > 400)[0][[0, 1]]       # 1e6 (top) and 1e0 (bottom) of the upper panel
    x0, x1 = np.where(dark.sum(axis=0) > 250)[0][[0, -1]]      # 1e-2 and 1e0
    masks = {
        "particle": lambda c: (np.abs(c[:, 0] - c[:, 1]) < 12) & (np.abs(c[:, 1] - c[:, 2]) < 12) & (c[:, 0] > 90) & (c[:, 0] < 180),
        "a11": lambda c: (c[:, 2] > 150) & (c[:, 0] < 80),
        "a10": lambda c: (c[:, 0] > 200) & (c[:, 1] > 90) & (c[:, 1] < 170) & (c[:, 2] < 60),
        "a9": lambda c: (c[:, 1] > 120) & (c[:, 0] < 90) & (c[:, 2] < 90),
    }
    out = {}
    for key, mk in masks.items():
        pts, last = [], None
        for x in range(x0 + 2, x1 - 2, 3):
            ys = np.where(mk(sub[y0 + 2:y1 - 2, x]))[0] + y0 + 2
            if not ys.size:
                continue
            groups = np.split(ys, np.where(np.diff(ys) > 2)[0] + 1)
            cand = [g.mean() for g in groups if g.size <= 6]
            if not cand:
                continue
            y = cand[0] if last is None else min(cand, key=lambda t: abs(t - last))
            if last is not None and abs(y - last) > 12:          # a label, not the curve
                continue
            last = y
            pts.append((10 ** (-2 + 2 * (x - x0) / (x1 - x0)), 10 ** (6 - 6 * (y - y0) / (y1 - y0))))
        out[key] = np.array(pts)
    return out


dig = digitise_fig3()

# ---------------------------------------------------------------- (1) dephasing
m1, m2 = masses(TRUE["Mc"], 10 ** TRUE["lq"])
fI = L.f_isco(m1)                                               # r_ISCO = 6 G m1
f = np.geomspace(2e-3, fI, 30000)
fdV = L.fdot_vac(f, TRUE["Mc"])
nums = dict(TenBkimMone=m1, TenBkimfI=fI)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
cases = [("particle", None, "particle halo", 7), ("a11", 1e-11, r"wave, $a=10^{-11}$ pc", 0),
         ("a10", 1e-10, r"wave, $a=10^{-10}$ pc", 1), ("a9", 1e-9, r"wave, $a=10^{-9}$ pc", 2)]
for key, a_pc, lab, col in cases:
    eps = eps_kim(f, TRUE["Mc"], 10 ** TRUE["lq"], TRUE["rho6"], TRUE["gam"], a_pc)
    dphi = two_pi_int(f, f / (fdV * (1 + eps)) - f / fdV)
    color = "0.45" if col == 7 else SERIES[col]
    ax.loglog(f, np.abs(dphi), color=color, label=lab)
    d = dig[key]
    d = d[(d[:, 1] > 3) & (d[:, 0] < 0.6 * fI)]                       # drop the axis-label pixels near f_ISCO
    ax.plot(d[::3, 0], d[::3, 1], "o", mfc="none", mec=color, ms=3)
    ok = (d[:, 1] > 3) & (d[:, 0] < 0.9 * fI)
    lr = np.log10(np.exp(np.interp(np.log(d[ok, 0]), np.log(f), np.log(np.abs(dphi) + 1e-300))) / d[ok, 1])
    nm = {"particle": "P", "a11": "W", "a10": "X", "a9": "Y"}[key]
    nums[f"TenBkimDevMed{nm}"] = np.median(lr)
    nums[f"TenBkimDevMax{nm}"] = np.max(np.abs(lr))
    nums[f"TenBkimDphiCent{nm}"] = np.interp(np.log(0.01), np.log(f), np.abs(dphi))
    if a_pc is not None:
        lc, al, mc2 = ell_c(m1, a_pc)
        nums[f"TenBkimEll{nm}"] = lc
        nums[f"TenBkimAlpha{nm}"] = al
        nums[f"TenBkimMass{nm}"] = mc2
        nums[f"TenBkimRc{nm}"] = a_pc * lc * (lc + 1)
ax.plot([], [], "o", mfc="none", mec="k", ms=3, label="digitised, Kim et al. Fig. 3")
ax.set_xlim(1e-2, 1); ax.set_ylim(1, 1e6)
ax.set_xlabel("$f$ [Hz]"); ax.set_ylabel(r"$|\Phi-\Phi_V|$ [rad]")
ax.legend(fontsize=7, loc="upper right")
savefig(fig, "ch10", "kim_dephasing")


# ---------------------------------------------------------------- (2) SNR and (3) Fisher
def band(th):
    eps = eps_kim(f, th["Mc"], 10 ** th["lq"], th["rho6"], th["gam"], 10 ** th["la"])
    tl = tl_from(f, L.fdot_vac(f, th["Mc"]) * (1 + eps))
    f5 = np.exp(np.interp(np.log(5 * L.YEAR), np.log(tl[::-1][1:]), np.log(f[::-1][1:])))
    return f5, min(1.0, fI)


f5, fup = band(TRUE)
ff = np.geomspace(f5, fup, 12000)
Sn = L.Sn_robson(ff, "4yr")
nums.update(TenBkimfFive=f5, TenBkimfUp=fup)


def lnh(x, wave=True):
    """ln h(f) up to the overall amplitude: -(1/2) ln(1+eps) - i Psi, Psi = 2 pi f t(f) - Phi(f) (ISCO reference)."""
    Mc, lq, rho6, gam, la = x
    eps = eps_kim(ff, Mc, 10**lq, rho6, gam, 10**la if wave else None)
    fdv = L.fdot_vac(ff, Mc)
    g = 1 / (fdv * (1 + eps)) - 1 / fdv
    # DM part of the Fourier phase: 2 pi int_f^fup (f' - f) g df'  (+ terms linear in f, absorbed by tc, phic)
    dpsi = two_pi_int(ff, ff * g) - ff * two_pi_int(ff, g)
    psi = 3 / 128 * L.u_of(ff, Mc) ** (-5 / 3) + dpsi
    return -0.5 * np.log1p(eps) - 1j * psi


x0 = np.array([TRUE[k] for k in ("Mc", "lq", "rho6", "gam", "la")])
amp = L.amp_avg(ff, TRUE["Mc"], DL) * np.exp(lnh(x0).real)   # |h| including the slower sweep factor
rho = np.sqrt(L.snr2(ff, amp**2, Sn))
nums["TenBkimSNR"] = rho


def dlogh_rows(wave=True, steps=(1e-6, 1e-4, 1e-3, 1e-4, 1e-3)):
    P = 5 if wave else 4
    rows = [np.ones_like(ff) + 0j, -1j * 2 * np.pi * ff, 1j * np.ones_like(ff)]
    for i in range(P):
        xp, xm = x0.copy(), x0.copy()
        hstep = steps[i] * (x0[i] if i == 0 else 1.0)
        xp[i] += hstep; xm[i] -= hstep
        rows.append((lnh(xp, wave) - lnh(xm, wave)) / (2 * hstep))
    return np.array(rows)


def fisher(wave=True, steps=(1e-6, 1e-4, 1e-3, 1e-4, 1e-3)):
    return L.fisher_from_dlogh(ff, amp**2, Sn, dlogh_rows(wave, steps))


Fw = fisher(True)
ew = L.marg_err(Fw)[3:]
Fw2 = fisher(True, steps=(2e-6, 2e-4, 2e-3, 2e-4, 2e-3))
ew2 = L.marg_err(Fw2)[3:]
Fp = fisher(False)
ep = L.marg_err(Fp)[3:]
Rw = L.corr(Fw)[3:, 3:]
# their posterior (Fig. 4): 2.5% and 97.5% quantiles -> half-width / 1.96 as a Gaussian-equivalent sigma
post = dict(Mc=(0.00118 + 0.000599) / 3.92, lq=(0.074 + 0.0637) / 3.92, rho6=(27.8 + 17.4) / 3.92,
            gam=(0.314 + 0.203) / 3.92, la=(0.239 + 0.154) / 3.92)
for i, k in enumerate(("Mc", "lq", "rho6", "gam", "la")):
    K = {"Mc": "Mc", "lq": "Lq", "rho6": "Rho", "gam": "Gam", "la": "La"}[k]
    nums[f"TenBkimPost{K}"] = post[k]
    nums[f"TenBkimFish{K}"] = ew[i]
    nums[f"TenBkimRatio{K}"] = ew[i] / post[k]
    if i < 4:
        nums[f"TenBkimPart{K}"] = ep[i]
nums["TenBkimStepDev"] = np.max(np.abs(ew2 / ew - 1))
nums["TenBkimCorrRhoGam"] = Rw[2, 3]
nums["TenBkimCorrRhoLq"] = Rw[2, 1]
nums["TenBkimCorrMcLq"] = Rw[0, 1]
# Vallisneri's consistency test on the 1 sigma surface of the 8-parameter Fisher matrix:
# |log r| = (1/2)(theta^j h_j - Dh | theta^k h_k - Dh), Dh = h(theta) - h(0); |log r| < 0.1 means the linearised
# signal (hence the Fisher ellipsoid) describes the likelihood there.
rng = np.random.default_rng(2023)
Dr = dlogh_rows(True)
Cw = L.inv_scaled(Fw)
d8 = np.sqrt(np.diag(Cw))
lam, V = np.linalg.eigh(Cw / np.outer(d8, d8))     # (Cholesky fails: rho6 and gamma are correlated at -0.9999)
Lc = (V * np.sqrt(np.clip(lam, 0, None))) * d8[:, None]
nums["TenBkimCond"] = lam.max() / max(lam.min(), 1e-300)
h0 = amp * np.exp(1j * lnh(x0).imag)
logr = []
for _ in range(300):
    z = rng.standard_normal(8)
    th = Lc @ (z / np.linalg.norm(z))
    x = x0 + th[3:]
    if not (x[3] > 0 and x[2] > 0):
        continue
    hx = amp * np.exp(th[0] + np.real(lnh(x)) - np.real(lnh(x0))) * np.exp(
        1j * (np.imag(lnh(x)) - 2 * np.pi * ff * th[1] + th[2]))
    lin = h0 * (th @ Dr)
    res = lin - (hx - h0)
    logr.append(0.5 * 4 * L.integ(ff, np.abs(res) ** 2 / Sn))
logr = np.array(logr)
nums["TenBkimLogrMed"], nums["TenBkimLogrMax"] = np.median(logr), np.max(logr)
nums["TenBkimLogrFrac"] = 100 * np.mean(logr > 0.1)
# the same at higher SNR: the 1 sigma surface shrinks as 1/rho and the norm grows as rho^2
facs = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
med = []
for fac in facs:
    lr2 = []
    for _ in range(80):
        z = rng.standard_normal(8)
        th = Lc @ (z / np.linalg.norm(z)) / fac
        x = x0 + th[3:]
        hx = amp * np.exp(th[0] + np.real(lnh(x)) - np.real(lnh(x0))) * np.exp(
            1j * (np.imag(lnh(x)) - 2 * np.pi * ff * th[1] + th[2]))
        res = h0 * (th @ Dr) - (hx - h0)
        lr2.append(0.5 * 4 * fac**2 * L.integ(ff, np.abs(res) ** 2 / Sn))
    med.append(np.median(lr2))
med = np.array(med)
print("median |log r| vs SNR factor", dict(zip(facs, med)))
# log-log interpolation (or extrapolation along the last segment) to |log r| = 0.1
lf, lm = np.log(facs[-2:]), np.log(med[-2:])
fac01 = np.exp(lf[0] + (np.log(0.1) - lm[0]) * (lf[1] - lf[0]) / (lm[1] - lm[0]))
nums["TenBkimSNRneeded"] = int(round(rho * fac01, -3))
nums["TenBkimLogrHundred"] = med[facs == 100.0][0]
np.savez(pathlib.Path(__file__).resolve().parents[2] / "data" / "ch10" / "kim_logr.npz", facs=facs, med=med, rho=rho)
save_numbers("ch10", "14_kim2023", L.tidy(nums))
print(nums)

fig, ax = plt.subplots(figsize=(5.6, 3.0))
labels = [r"$\mathcal{M}$ [$M_\odot$]", r"$\log_{10}q$", r"$\rho_6$ [$10^{15}M_\odot$pc$^{-3}$]",
          r"$\gamma_{\rm sp}$", r"$\log_{10}a$"]
xs = np.arange(5)
ax.bar(xs - 0.2, [post[k] for k in ("Mc", "lq", "rho6", "gam", "la")], 0.38, color=SERIES[1],
       label="Kim et al. posterior (95% half-width / 1.96)")
ax.bar(xs + 0.2, ew, 0.38, color=SERIES[0], label="our Fisher forecast")
ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"1$\sigma$ error"); ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
fig.tight_layout()
savefig(fig, "ch10", "kim_fisher")
