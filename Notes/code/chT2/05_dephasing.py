"""How much phase does dynamical friction in a dark-matter spike steal, and with what power of f?

Question: an IMRI (1e3 + 1.4 Msun) spirals through a static spike rho = rho6 (r6/r)^gamma,
r6 = 1e-6 pc. Dynamical friction adds the power P_DF to the GW power P_GW, so
fdot = fdot_vac (1 + eps(f)) with eps = P_DF/P_GW = c_f f^{-n}, n = (11 - 2 gamma)/3.
How large is the dephasing Delta Phi(f) (GW phase accumulated between f and the ISCO, vacuum
minus spike) for a few slopes gamma, and does it follow the power law f^{-5/3-n} derived in
the text?
Computes:
  * c_f and the equality frequency f_eq = c_f^{1/n} (Coogan et al. 2022 report ~0.015 Hz for
    gamma = 7/3, rho6 = 5.448e15 Msun/pc^3, xi = 0.58, ln Lambda = ln sqrt(m1/m2)),
  * Delta Phi(f) from the closed form with the hypergeometric function, checked against direct
    numerical integration,
  * the local slope d ln Delta Phi / d ln f against the predicted -5/3 - n,
  * the wave-DM friction factor C_w(kr) of Hui et al. 2017 / Kim et al. 2023 and its limits,
  * the ratio of collisionless-accretion to friction power, 4 v^2 / (c^2 ln Lambda).
Writes: figures/chT2/dephasing.pdf, figures/chT2/wave_df.pdf, results/chT2/05_dephasing.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import hyp2f1, sici
from common import setup, savefig, save_numbers, SERIES, theory_line
from lib_lisa import (G, c, MSUN, PC, YEAR, TSUN, chirp_mass, f_isco, fdot_vac, tau_of_f, f_of_tau)

m1, m2 = 1e3, 1.4
M, Mc = m1 + m2, chirp_mass(m1, m2)
rho6, r6 = 5.448e15, 1e-6                     # Msun/pc^3, pc
xi, lnL = 0.58, np.log(np.sqrt(m1 / m2))
fI = f_isco(M)
f4 = f_of_tau(tau_of_f(fI, Mc) + 4 * YEAR, Mc)  # vacuum frequency 4 years before the ISCO

def c_f(gam):
    """eps(f) = P_DF/P_GW = c_f f^{-n} (SI evaluation of the text formula)."""
    rho_SI = rho6 * MSUN / PC**3 * (r6 * PC) ** gam      # rho = rho_SI r^-gam (SI)
    pref = 5*np.pi * c**5 * xi * lnL / (8 * G**2.5 * (M*MSUN)**1.5 * (m1*MSUN)**2)
    n = (11 - 2 * gam) / 3
    return pref * rho_SI * (G * M * MSUN / np.pi**2) ** ((11 - 2 * gam) / 6), n

def phi_vac(f):
    """GW phase left to coalescence, Phi_V = (1/16) (pi G Mc f/c^3)^{-5/3}."""
    return (np.pi * Mc * TSUN * f) ** (-5 / 3) / 16

def dphi_closed(f, gam):
    """Phi_V - Phi_S, static dress: Phi_S = Phi_V 2F1(1,a;1+a;-c_f f^-n), a=5/3n."""
    cf, n = c_f(gam)
    b = 5 / (3 * n)
    return phi_vac(f) * (1 - hyp2f1(1, b, 1 + b, -cf * f ** (-n)))

def dphi_numeric(f, gam, fmax):
    cf, n = c_f(gam)
    integrand = lambda x: 2 * np.pi * x / fdot_vac(x, Mc) * (1 - 1 / (1 + cf * x ** (-n)))
    return quad(integrand, f, fmax, limit=400, epsrel=1e-10)[0]

setup(6.2, 5.2)
gammas = [1.5, 2.0, 7 / 3, 2.5]
labels = [r"$\gamma=1.5$", r"$\gamma=2$", r"$\gamma=7/3$", r"$\gamma=2.5$"]
f = np.logspace(-3, np.log10(fI), 600)
fig, (ax, bx) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
nums = {}
names = {1.5: "A", 2.0: "B", 7 / 3: "C", 2.5: "D"}
for gam, lab, col in zip(gammas, labels, SERIES):
    cf, n = c_f(gam)
    feq = cf ** (1 / n)
    dP = dphi_closed(f, gam) - dphi_closed(fI, gam)          # accumulated between f and the ISCO
    ax.loglog(f, dP, color=col, label=lab)
    # local slope of the to-coalescence dephasing versus the prediction -5/3 - n
    full = dphi_closed(f, gam)
    slope = np.gradient(np.log(full), np.log(f))
    bx.semilogx(f, slope, color=col)
    bx.axhline(-5 / 3 - n, color=col, ls="--", lw=0.8)
    k = names[gam]
    nums[f"feq{k}"] = feq * 1e3
    nums[f"dPhiFour{k}"] = dphi_closed(f4, gam) - dphi_closed(fI, gam)
    nums[f"bexp{k}"] = -5 / 3 - n
    if gam == 7 / 3:
        # power-law asymptote for f >> f_eq: Delta Phi = 5 c_f f^{-n} Phi_V / (2 (8 - gamma))
        asym = 5 * cf * f ** (-n) * phi_vac(f) / (2 * (8 - gam))
        asymI = 5 * cf * fI ** (-n) * phi_vac(fI) / (2 * (8 - gam))
        m = f > 3 * feq
        theory_line(ax, f[m], (asym - asymI)[m], label="power law, $f\\gg f_{\\rm eq}$")
        ax.axvline(feq, color="0.5", lw=0.7, ls=":")
        ax.text(feq * 1.08, 2e7, r"$f_{\rm eq}$", color="0.3")
        test_f = np.array([2e-3, 1e-2, f4, 0.1, 1.0])
        num = np.array([dphi_numeric(x, gam, fI) for x in test_f])
        clo = dphi_closed(test_f, gam) - dphi_closed(fI, gam)
        nums["closedVsNum"] = np.max(np.abs(num / clo - 1))
        nums["PLerrFour"] = 100 * abs((asym - asymI)[np.argmin(abs(f - f4))] / dP[np.argmin(abs(f - f4))] - 1)
        print("closed form vs quad:", num / clo)
ax.axvspan(f4, min(fI, 1.0), color="0.9", zorder=-1)
ax.text(f4 * 1.1, 3e-3, "last 4 yr\n(in LISA band)", fontsize=8, color="0.35")
ax.axhline(1.0, color="0.5", lw=0.7)
ax.set_ylabel(r"$\Delta\Phi(f)$ [rad]")
ax.set_ylim(1e-3, 1e9)
h, l = ax.get_legend_handles_labels()
o = [i for i, t in enumerate(l) if "power law" not in t] + [i for i, t in enumerate(l) if "power law" in t]
ax.legend([h[i] for i in o], [l[i] for i in o], loc="upper right", fontsize=8)
bx.set_ylabel(r"$d\ln\Delta\Phi/d\ln f$")
bx.set_xlabel("$f$ [Hz]")
bx.set_ylim(-5, -1.3)
fig.tight_layout()
savefig(fig, "chT2", "dephasing")

# ---- the wave-DM friction factor  C_w(x) = Cin(2x) - 1 + sin(2x)/(2x),  x = k r
setup(6.0, 3.2)
x = np.logspace(-2, 3, 500)
z = 2 * x
si, ci = sici(z)
Cin = np.euler_gamma + np.log(z) - ci          # Cin(z) = int_0^z (1 - cos t)/t dt
Cw = Cin - 1 + np.sin(z) / z
fig, ax = plt.subplots()
ax.loglog(x, Cw, color=SERIES[0], label=r"$C_w(kr)$")
theory_line(ax, x[x < 1.5], x[x < 1.5] ** 2 / 3, label=r"$(kr)^2/3$")
ax.loglog(x[x > 2], np.log(2 * x[x > 2]) + np.euler_gamma - 1, color="0.3", ls=":",
          label=r"$\ln(2kr)+\gamma_E-1$")
ax.set_xlabel(r"$kr = 2\pi r/\lambda_{\rm dB}$")
ax.set_ylabel("friction factor")
ax.set_ylim(1e-4, 20)
ax.legend()
savefig(fig, "chT2", "wave_df")
nums["CwSmallErr"] = 100 * abs(Cw[np.argmin(abs(x - 0.1))] / (0.1**2 / 3) - 1)

# ---- collisionless accretion vs friction: P_acc/P_DF = 4 v^2/(c^2 ln Lambda), v^2/c^2 = r_h/(2r)
for r_over_rh, k in [(30.0, "Thirty"), (3.0, "Three")]:
    nums[f"accRatio{k}"] = 4 * (1 / (2 * r_over_rh)) / 3.0      # ln Lambda = 3 as in Eda et al. 2015
nums["ffourImri"] = f4 * 1e3
nums["lnLambdaC"] = lnL
for kk, v in nums.items():
    print(f"{kk:12s} {v:.5g}")
save_numbers("chT2", "05_dephasing", {"Tw" + k: v for k, v in nums.items()})   # "Tw" = chapter T2 prefix
