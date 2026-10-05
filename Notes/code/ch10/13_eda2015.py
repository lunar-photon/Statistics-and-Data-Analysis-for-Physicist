"""13_eda2015.py -- reproduce Eda, Itoh, Kuroyanagi & Silk (2015): dephasing and parameter errors for a DM mini-spike.

Setup of the paper (their Table I and Sec. III): M_BH = 1e3 Msun, a 1 Msun companion, spike rho = rho_sp (r_sp/r)^alpha
with rho_sp = 226 Msun/pc^3, r_sp = 0.54 pc, Coulomb logarithm ln Lambda = 3, Kepler with M_BH, friction power
P_DF = 4 pi G^2 mu^2 rho lnL / v and quadrupole power P_GW, so df/dt = fdot_V (1 + eps), eps = P_DF/P_GW.
Questions:
 (1) the accumulated phase difference  dPhi~(f) = Phi~_DM - Phi~_V,  Phi~(f) = 2 pi int_f^{f_ISCO} (f'-f)/fdot df'
     (their eqs. 28c and 29, Fig. 1) for alpha = 1.5, 2, 7/3, against their Fig. 1 digitised from the PDF;
 (2) the start frequency of a 5-year observation f_ini(alpha) (their Fig. 2);
 (3) the Fisher errors for (ln A, t_c, Phi_c, ln Mc, ln alpha, ln c) with their eLISA noise (eq. 32a), the band
     [f_ini, f_ISCO], normalised to S/N = 10 (their eqs. 41a-f), and Delta alpha/alpha as a function of alpha.
Everything is computed twice: with the Coulomb logarithm ln Lambda = 3 written in the paper, and with
ln Lambda = ln(10^3) = 6.9, which reproduces their figures (Lambda = 10^3, i.e. log10 Lambda = 3).
Writes: figures/ch10/eda_dephasing.pdf, eda_alpha.pdf, results/ch10/13_eda2015.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from common import setup, savefig, save_numbers, SERIES
import lib_gw10 as L

setup()
MBH, MU = 1e3, 1.0
RHOSP, RSP = 226.0, 0.54 * L.PC
LNL_TEXT, LNL_NUM = 3.0, np.log(1e3)          # "ln Lambda ~ 3" as written, and ln(10^3) = 6.91 (see text)
LNL = LNL_TEXT
Mc = MU**0.6 * MBH**0.4                       # their chirp mass mu^{3/5} M^{2/5}
fI = L.f_isco(MBH)


def eps_eda(f, alpha, scale=1.0, lnl=None):
    """eps = P_DF/P_GW = (5 pi/8) c^5 rho(R) lnL R^{11/2} / (G^{5/2} M^{7/2}),  R from Kepler with M_BH."""
    R = L.r_of_f(f, MBH)
    rho = scale * RHOSP * L.MSUN / L.PC**3 * (RSP / R) ** alpha
    lnl = LNL if lnl is None else lnl
    return 5 * np.pi / 8 * L.c**5 * rho * lnl * R**5.5 / (L.G**2.5 * (MBH * L.MSUN) ** 3.5)


def phit(f, fdot):
    """Phi~(f) = 2 pi int_f^{f_ISCO} (f' - f)/fdot df' on the (increasing) grid f, which ends at f_ISCO."""
    A = -cumulative_trapezoid((f / fdot)[::-1], f[::-1], initial=0)[::-1]
    B = -cumulative_trapezoid((1 / fdot)[::-1], f[::-1], initial=0)[::-1]
    return 2 * np.pi * (A - f * B)


def time_left(f, fdot):
    return -cumulative_trapezoid((1 / fdot)[::-1], f[::-1], initial=0)[::-1]


# ---------------------------------------------------------------- digitise their Fig. 1 from the PDF
def digitise_fig1():
    import subprocess, tempfile
    from PIL import Image
    pdf = pathlib.Path(__file__).resolve().parents[2] / "references" / "Eda2015.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-f", "8", "-l", "8", "-r", "220", "-png", str(pdf), tmp + "/p"], check=True)
        im = np.array(Image.open(next(pathlib.Path(tmp).glob("p*.png"))).convert("L")).astype(int)
    sub = im[1000:1580, 500:1300] < 110
    # calibration from the tick marks: 10^6 at row 171, 10^-6 at row 470; 0.01 Hz at col 178.5, 1 Hz at col 706.5
    yv = lambda y: 6 - (y - 171) * 12 / (470 - 171)
    fv = lambda x: -2 + (x - 178.5) * 2 / (706.5 - 178.5)
    grid = {171, 220, 221, 270, 271, 320, 321, 370, 371, 420, 421, 470}
    guide = {7 / 3: (7.5, 0.7), 2.0: (5.6, -2.3), 1.5: (1.8, -6.8)}  # rough log10 values at 0.01 and 1 Hz
    out = {a: [] for a in guide}
    for x in range(192, 696, 3):
        ys = [y for y in np.where(sub[132:492, x])[0] + 132 if y not in grid and not (x < 385 and y > 385)]
        groups = []
        for y in ys:
            if groups and y - groups[-1][-1] <= 2:
                groups[-1].append(y)
            else:
                groups.append([y])
        lf = fv(x)
        for g in groups:
            ly = yv(np.mean(g))
            a = min(guide, key=lambda a: abs(ly - (guide[a][0] + (guide[a][1] - guide[a][0]) * (lf + 2) / 2)))
            if abs(ly - (guide[a][0] + (guide[a][1] - guide[a][0]) * (lf + 2) / 2)) < 1.0:
                out[a].append((10**lf, 10**ly))
    return {a: np.array(v) for a, v in out.items()}


dig = digitise_fig1()

# ---------------------------------------------------------------- (1) dephasing, (2) f_ini
f = np.geomspace(1e-4, fI, 50000)
fdV = L.fdot_vac(f, Mc)
nums = dict(TenBedaMc=Mc, TenBedafI=fI, TenBedaLnLnum=LNL_NUM)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for k, (alpha, nm) in enumerate(((1.5, "A"), (2.0, "B"), (7 / 3, "C"))):
    d = dig[alpha]
    ax.plot(d[::4, 0], d[::4, 1], "o", mfc="none", mec=SERIES[k], ms=3.5)
    for lnl, tag, ls in ((LNL_TEXT, "", "--"), (LNL_NUM, "N", "-")):
        eps = eps_eda(f, alpha, lnl=lnl)
        dphi = phit(f, fdV * (1 + eps)) - phit(f, fdV)
        sel = f >= 0.01
        ax.loglog(f[sel], np.abs(dphi[sel]), color=SERIES[k], ls=ls, lw=1.4 if ls == "-" else 1.0,
                  label=rf"$\alpha={alpha:.3g}$, $\ln\Lambda={lnl:.2g}$")
        ok = d[:, 0] < fI
        ours_d = np.exp(np.interp(np.log(d[ok, 0]), np.log(f), np.log(np.abs(dphi) + 1e-300)))
        lr = np.log10(ours_d / d[ok, 1])
        nums[f"TenBedaDevMed{nm}{tag}"] = np.median(lr)
        nums[f"TenBedaDevRms{nm}{tag}"] = np.sqrt(np.mean((lr - np.median(lr)) ** 2))
        for fx, fn in ((0.01, "One"), (0.1, "Two"), (1.0, "Three")):
            nums[f"TenBedaDphi{nm}{fn}{tag}"] = np.interp(np.log(fx), np.log(f), np.abs(dphi))
        nums[f"TenBedaEpsCent{nm}{tag}"] = eps_eda(np.array([0.01]), alpha, lnl=lnl)[0]
ax.plot([], [], "o", mfc="none", mec="k", ms=3.5, label="digitised, Eda et al. Fig. 1")
ax.set_xlim(0.01, 1); ax.set_ylim(1e-7, 1e8)
ax.set_xlabel("$f$ [Hz]"); ax.set_ylabel(r"$|\Delta\tilde\Phi(f)|$ [rad]")
ax.legend(fontsize=6.5, loc="lower left", ncol=2)
savefig(fig, "ch10", "eda_dephasing")

alphas = np.linspace(0.0, 2.55, 52)
FINI = {}
for lnl, tag in ((LNL_TEXT, ""), (LNL_NUM, "N")):
    fini = []
    for alpha in alphas:
        tl = time_left(f, fdV * (1 + eps_eda(f, alpha, lnl=lnl)))
        fini.append(np.exp(np.interp(np.log(5 * L.YEAR), np.log(tl[::-1][1:]), np.log(f[::-1][1:]))))
    FINI[lnl] = np.array(fini)
    nums.update({f"TenBedaFiniZero{tag}": FINI[lnl][0], f"TenBedaFiniSeven{tag}": np.interp(7 / 3, alphas, FINI[lnl]),
                 f"TenBedaFiniTwoFive{tag}": np.interp(2.5, alphas, FINI[lnl])})


# ---------------------------------------------------------------- (3) Fisher with eLISA noise, S/N = 10
def fisher_eda(alpha, nf=20000, lnl=LNL_TEXT):
    f0 = np.interp(alpha, alphas, FINI[lnl])
    ff = np.geomspace(f0, fI, nf)
    Sn = L.Sn_elisa(ff)
    fp = 0.05                                             # pivot for the amplitude parameter c = eps(fp)
    c0 = eps_eda(np.array([fp]), alpha, lnl=lnl)[0]

    def lnh(lnMc, lna, lnc):
        M_, a_, c_ = np.exp(lnMc), np.exp(lna), np.exp(lnc)
        n = (11 - 2 * a_) / 3
        eps = c_ * (ff / fp) ** (-n)
        fdv = L.fdot_vac(ff, M_)
        dpsi = phit(ff, fdv * (1 + eps)) - phit(ff, fdv)      # DM part of Phi~, referenced at the ISCO
        psi = 3 / 128 * L.u_of(ff, M_) ** (-5 / 3) + dpsi     # Psi_ours = ... + Phi~(f)
        lnamp = 5 / 6 * np.log(M_) - 0.5 * np.log1p(eps)     # |h| ~ Mc^{5/6} f^{-7/6} L^{-1/2}
        return lnamp - 1j * psi

    x0 = np.array([np.log(Mc), np.log(alpha), np.log(c0)])
    steps = [1e-7, 1e-5, 1e-4]
    D = []
    for i, hstep in enumerate(steps):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += hstep; xm[i] -= hstep
        D.append((lnh(*xp) - lnh(*xm)) / (2 * hstep))
    lnMc_row, lna_row, lnc_row = D
    amp2 = np.abs(np.exp(lnh(*x0).real)) ** 2 * ff ** (-7 / 3)
    rows = np.array([np.ones_like(ff) + 0j, -1j * 2 * np.pi * ff, 1j * np.ones_like(ff), lnMc_row, lna_row, lnc_row])
    F = L.fisher_from_dlogh(ff, amp2, Sn, rows)
    F *= 100.0 / F[0, 0]                                    # normalise to S/N = 10 (Gamma_lnA,lnA = rho^2)
    return L.marg_err(F), F


paper = dict(A=0.1, tc=1.0, Phic=1.3, Mc=3.1e-7, alpha=1.2e-6, c=5.9e-5)
for k in paper:
    nums[f"TenBedaPaper{k.capitalize()}"] = paper[k]
al = np.linspace(1.4, 2.5, 23)
DA = {}
for lnl, tag in ((LNL_TEXT, ""), (LNL_NUM, "N")):
    err, F = fisher_eda(7 / 3, lnl=lnl)
    ours = dict(A=err[0], tc=err[1], Phic=err[2], Mc=err[3], alpha=err[4], c=err[5])
    for k in paper:
        nums[f"TenBedaOurs{k.capitalize()}{tag}"] = ours[k]
    print(lnl, {k: (paper[k], ours[k]) for k in paper})
    # The error of an amplitude depends on WHERE it is defined when the exponent is free:
    # eps = c (f/fp)^{-n}, n = (11 - 2 alpha)/3, so the amplitude at another pivot fq is c' = c (fq/fp)^{-n} and
    # d ln c' = d ln c + (2/3) alpha ln(fq/fp) d ln alpha.  Eda et al.'s c_eps multiplies x^{(11-2 alpha)/2} with
    # x = R/R_*, R_* the radius at which the spike holds a mass M_BH (their eqs. 10b, 17, 18, 28d-f): their pivot
    # is the Kepler frequency at R_* (~1 pc), far below the band.
    C = L.inv_scaled(F)
    a73 = 7 / 3
    Rstar = (MBH * L.MSUN * (3 - a73) / (4 * np.pi * RHOSP * L.MSUN / L.PC**3 * RSP**a73)) ** (1 / (3 - a73))
    fstar = np.sqrt(L.G * MBH * L.MSUN / Rstar**3) / np.pi
    def dlnc_at(fq):
        lev = 2 / 3 * a73 * np.log(fq / 0.05)
        return np.sqrt(C[5, 5] + lev**2 * C[4, 4] + 2 * lev * C[4, 5])
    fq = np.geomspace(1e-16, 1.0, 4000)
    dq = np.array([dlnc_at(x) for x in fq])
    nums.update({f"TenBedaRstar{tag}": Rstar / L.PC, f"TenBedaFstar{tag}": fstar,
                 f"TenBedaCstar{tag}": dlnc_at(fstar), f"TenBedaFbest{tag}": fq[np.argmin(dq)],
                 f"TenBedaCbest{tag}": dq.min(), f"TenBedaCorrAC{tag}": C[4, 5] / np.sqrt(C[4, 4] * C[5, 5])})
    DA[lnl] = np.array([fisher_eda(a, nf=12000, lnl=lnl)[0][4] for a in al])
    nums[f"TenBedaAlphaTen{tag}"] = np.interp(-1.0, np.log10(DA[lnl])[::-1], al[::-1])
save_numbers("ch10", "13_eda2015", L.tidy(nums))
print(nums)

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax[0].semilogy(alphas, FINI[LNL_NUM], color=SERIES[0], label=r"$\ln\Lambda=6.9$")
ax[0].semilogy(alphas, FINI[LNL_TEXT], color=SERIES[0], ls="--", label=r"$\ln\Lambda=3$")
ax[0].legend(fontsize=7.5)
ax[0].set_xlabel(r"spike slope $\alpha$"); ax[0].set_ylabel(r"$f_{\rm ini}$ (5 yr before ISCO) [Hz]")
ax[0].set_ylim(5e-4, 5e-2)
ax[1].semilogy(al, DA[LNL_NUM], color=SERIES[0], label=r"ours, $\ln\Lambda=6.9$")
ax[1].semilogy(al, DA[LNL_TEXT], color=SERIES[0], ls="--", label=r"ours, $\ln\Lambda=3$")
ax[1].axhline(0.1, color="k", lw=0.8, ls=":")
ax[1].plot([7 / 3], [1.2e-6], "o", mfc="none", mec="k", label="Eda et al. eq. (41e)")
ax[1].set_xlabel(r"spike slope $\alpha$"); ax[1].set_ylabel(r"$\Delta\alpha/\alpha$ at S/N $=10$")
ax[1].legend(fontsize=7.5)
fig.tight_layout()
savefig(fig, "ch10", "eda_alpha")
