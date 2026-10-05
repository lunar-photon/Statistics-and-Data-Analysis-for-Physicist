"""13_chronometers.py -- H0 from the cosmic-chronometer compilation, with and without systematics.

Question: what do the 32 published chronometer measurements of H(z) say about H0 in flat
LCDM, once the matter density is profiled and the correlated stellar-population-model
systematic is included, and how does that compare with Planck and SH0ES?

Data: Moresco et al. (2022, Living Rev. Relativ. 25, 6), Table 1 (z, H, sigma_H); the errors
are the diagonal (statistical plus uncorrelated systematic) part.
Model: H(z) = H0 E(z; Om), E = sqrt(Om (1+z)^3 + 1 - Om).
Estimators:
  * closed form at fixed Om (generalised least squares): H0_hat = E^T C^-1 H / E^T C^-1 E,
    Var = 1 / E^T C^-1 E;
  * profile over Om on a grid;
  * covariance C = diag(sigma^2) + eta^2 m m^T with m_i = H0_hat E_i, a fully correlated
    fractional error eta (a simplification of the Moresco et al. 2020 recipe, which gives an
    eta(z) per systematic component; eta = 4.5% is their average for the SPS-model term).
Checks: 4000 mock compilations drawn from the fiducial model with the same covariance.
Writes figures/ch11/chronometers.pdf, results/ch11/13_chronometers.tex.
"""
import pathlib
import sys

import numpy as np
from scipy.stats import chi2 as chi2dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import SERIES, rng_for, save_numbers, savefig, setup, theory_line  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CH, NAME = "ch11", "13_chronometers"

# Moresco et al. (2022), Table 1: z, H(z) [km/s/Mpc], sigma_H, method (F full spectrum, D D4000, L Lick)
DATA = [
    (0.07, 69.0, 19.6, "F"), (0.09, 69.0, 12.0, "F"), (0.12, 68.6, 26.2, "F"), (0.17, 83.0, 8.0, "F"),
    (0.179, 75.0, 4.0, "D"), (0.199, 75.0, 5.0, "D"), (0.20, 72.9, 29.6, "F"), (0.27, 77.0, 14.0, "F"),
    (0.28, 88.8, 36.6, "F"), (0.352, 83.0, 14.0, "D"), (0.38, 83.0, 13.5, "D"), (0.4, 95.0, 17.0, "F"),
    (0.4004, 77.0, 10.2, "D"), (0.425, 87.1, 11.2, "D"), (0.445, 92.8, 12.9, "D"), (0.47, 89.0, 49.6, "F"),
    (0.4783, 80.9, 9.0, "D"), (0.48, 97.0, 62.0, "F"), (0.593, 104.0, 13.0, "D"), (0.68, 92.0, 8.0, "D"),
    (0.75, 98.8, 33.6, "L"), (0.781, 105.0, 12.0, "D"), (0.875, 125.0, 17.0, "D"), (0.88, 90.0, 40.0, "F"),
    (0.9, 117.0, 23.0, "F"), (1.037, 154.0, 20.0, "D"), (1.3, 168.0, 17.0, "F"), (1.363, 160.0, 33.6, "D"),
    (1.43, 177.0, 18.0, "F"), (1.53, 140.0, 14.0, "F"), (1.75, 202.0, 40.0, "F"), (1.965, 186.5, 50.4, "D"),
]
z = np.array([d[0] for d in DATA])
H = np.array([d[1] for d in DATA])
sig = np.array([d[2] for d in DATA])
N = len(z)
ETA = 0.045
PLANCK = (67.36, 0.54)
SHOES = (73.04, 1.04)
OM_FID = 0.315


def E(z, om):
    return np.sqrt(om * (1 + z) ** 3 + 1 - om)


def cov(om, H0ref, eta):
    m = H0ref * E(z, om)
    return np.diag(sig ** 2) + eta ** 2 * np.outer(m, m)


def gls(Hd, om, eta=0.0, H0ref=70.0):
    """Closed-form GLS estimate of H0 at fixed Om; returns (H0_hat, sigma, chi2_min)."""
    e = E(z, om)
    for _ in range(3 if eta > 0 else 1):              # m depends weakly on H0: iterate
        Ci = np.linalg.inv(cov(om, H0ref, eta))
        S = e @ Ci @ e
        h = (e @ Ci @ Hd) / S
        H0ref = h
    r = Hd - h * e
    return h, 1 / np.sqrt(S), r @ Ci @ r


def profile(Hd, eta=0.0, omg=np.linspace(0.05, 0.9, 341)):
    res = np.array([gls(Hd, om, eta) for om in omg])
    i = int(np.argmin(res[:, 2]))
    return omg, res, i


def interval_profile(Hd, eta, H0g=np.linspace(50, 90, 801), omg=np.linspace(0.05, 0.9, 171)):
    """Profile chi2 over Om at each H0 on a grid; 1-sigma interval from Delta chi2 = 1."""
    chi = np.empty(len(H0g))
    for j, h in enumerate(H0g):
        best = np.inf
        for om in omg:
            Ci = np.linalg.inv(cov(om, h, eta))
            r = Hd - h * E(z, om)
            best = min(best, r @ Ci @ r)
        chi[j] = best
    chi -= chi.min()
    i0 = int(np.argmin(chi))
    lo = np.interp(1.0, chi[:i0 + 1][::-1], H0g[:i0 + 1][::-1])
    hi = np.interp(1.0, chi[i0:], H0g[i0:])
    return H0g, chi, H0g[i0], lo, hi


def main():
    setup()
    # (1) fixed Om
    hA, sA, chiA = gls(H, OM_FID)
    hB, sB, chiB = gls(H, OM_FID, ETA)
    # (2) profile over Om, statistical only and with systematics
    omg, resS, iS = profile(H)
    omg, resY, iY = profile(H, ETA)
    H0gS, chiS, hS, loS, hiS = interval_profile(H, 0.0)
    H0gY, chiY, hY, loY, hiY = interval_profile(H, ETA)
    # om interval (stat only) from the 1-D profile chi2_p(Om)
    cp = resS[:, 2] - resS[iS, 2]
    om_lo = np.interp(1.0, cp[:iS + 1][::-1], omg[:iS + 1][::-1])
    om_hi = np.interp(1.0, cp[iS:], omg[iS:])
    # analytic: sigma^2 = sigma_stat^2 + (eta H0)^2
    s_analytic = np.sqrt(sA ** 2 + (ETA * hA) ** 2)

    # (3) mocks at fixed Om: verify closed-form scatter, stat-only and with the correlated term
    rng = rng_for(CH, NAME)
    nm = 4000
    efid = E(z, OM_FID)
    est_s, est_y = np.empty(nm), np.empty(nm)
    for i in range(nm):
        Hm = 70 * efid + sig * rng.standard_normal(N)
        est_s[i] = gls(Hm, OM_FID)[0]
        Hy = 70 * efid * (1 + ETA * rng.standard_normal()) + sig * rng.standard_normal(N)
        est_y[i] = gls(Hy, OM_FID, ETA, 70.0)[0]
    # tensions (Gaussian approximation, independent errors)
    def tension(h, s, ref):
        return abs(h - ref[0]) / np.sqrt(s ** 2 + ref[1] ** 2)
    sY = 0.5 * (hiY - loY)
    sS = 0.5 * (hiS - loS)

    # ---------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    cols = {"F": SERIES[0], "D": SERIES[1], "L": SERIES[2]}
    labs = {"F": "full-spectrum fitting", "D": "D4000 calibration", "L": "Lick indices"}
    for m in "FDL":
        s_ = np.array([d[3] == m for d in DATA])
        ax[0].errorbar(z[s_], H[s_], sig[s_], fmt="o", ms=3.5, color=cols[m], label=labs[m])
    zz = np.linspace(0, 2.05, 200)
    ax[0].plot(zz, resS[iS, 0] * E(zz, omg[iS]), color="k", lw=1.4,
               label=rf"best fit: $H_0$={resS[iS,0]:.1f}, $\Omega_m$={omg[iS]:.2f}")
    ax[0].plot(zz, PLANCK[0] * E(zz, 0.3153), color=SERIES[3], lw=1.2, ls="--", label="Planck 2018")
    ax[0].plot(zz, SHOES[0] * E(zz, 0.3153), color=SERIES[4], lw=1.2, ls=":", label=r"SH0ES $H_0$, same $\Omega_m$")
    ax[0].set_xlabel("redshift $z$")
    ax[0].set_ylabel(r"$H(z)$ [km/s/Mpc]")
    ax[0].legend(fontsize=7, loc="upper left")
    ax[1].plot(H0gS, chiS, color=SERIES[0], label=r"statistical, $\Omega_m$ profiled")
    ax[1].plot(H0gY, chiY, color=SERIES[1], label=rf"+ correlated {100*ETA:.1f}% systematic")
    hh = np.linspace(50, 90, 400)
    ax[1].plot(hh, ((hh - hA) / sA) ** 2, color=SERIES[0], ls=":", lw=1.2,
               label=rf"statistical, $\Omega_m$ fixed at {OM_FID}")
    for ref, c, lab in ((PLANCK, SERIES[3], "Planck"), (SHOES, SERIES[4], "SH0ES")):
        ax[1].axvspan(ref[0] - ref[1], ref[0] + ref[1], color=c, alpha=0.25, label=lab)
    ax[1].axhline(1, color="0.5", lw=0.7, ls="--")
    ax[1].set_ylim(0, 6)
    ax[1].set_xlim(55, 85)
    ax[1].set_xlabel(r"$H_0$ [km/s/Mpc]")
    ax[1].set_ylabel(r"$\Delta\chi^2$")
    ax[1].legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, borderaxespad=0.2)
    savefig(fig, CH, "chronometers")

    save_numbers(CH, NAME, {
        "FBCcN": N, "FBCcEta": f"{100*ETA:.1f}", "FBCcOmFid": f"{OM_FID}",
        "FBCcHA": f"{hA:.1f}", "FBCcSA": f"{sA:.1f}", "FBCcChiA": f"{chiA:.1f}", "FBCcDof": N - 1,
        "FBCcHB": f"{hB:.1f}", "FBCcSB": f"{sB:.1f}", "FBCcSAnalytic": f"{s_analytic:.1f}",
        "FBCcHS": f"{hS:.1f}", "FBCcLoS": f"{loS:.1f}", "FBCcHiS": f"{hiS:.1f}", "FBCcSS": f"{sS:.1f}",
        "FBCcOmS": f"{omg[iS]:.2f}", "FBCcOmLo": f"{om_lo:.2f}", "FBCcOmHi": f"{om_hi:.2f}",
        "FBCcChiS": f"{resS[iS,2]:.1f}", "FBCcDofS": N - 2,
        "FBCcPlow": f"{chi2dist.cdf(resS[iS,2], N-2):.3f}",
        "FBCcErrScale": f"{np.sqrt(resS[iS,2]/(N-2)):.2f}",
        "FBCcHY": f"{hY:.1f}", "FBCcLoY": f"{loY:.1f}", "FBCcHiY": f"{hiY:.1f}", "FBCcSY": f"{sY:.1f}",
        "FBCcNmock": nm, "FBCcMockMean": f"{est_s.mean():.2f}", "FBCcMockSd": f"{est_s.std():.2f}",
        "FBCcMockSdTh": f"{gls(70*efid, OM_FID)[1]:.2f}",
        "FBCcMockYMean": f"{est_y.mean():.2f}", "FBCcMockYSd": f"{est_y.std():.2f}",
        "FBCcMockYSdTh": f"{np.sqrt(gls(70*efid, OM_FID)[1]**2 + (ETA*70)**2):.2f}",
        "FBCcTenPlS": f"{tension(hS, sS, PLANCK):.1f}", "FBCcTenShS": f"{tension(hS, sS, SHOES):.1f}",
        "FBCcTenPlY": f"{tension(hY, sY, PLANCK):.1f}", "FBCcTenShY": f"{tension(hY, sY, SHOES):.1f}",
    })
    print(f"fixed Om: {hA:.2f}+-{sA:.2f} chi2 {chiA:.1f}/{N-1}; +sys {hB:.2f}+-{sB:.2f} (analytic {s_analytic:.2f})")
    print(f"profile stat: H0 {hS:.2f} [{loS:.2f},{hiS:.2f}] Om {omg[iS]:.3f} [{om_lo:.2f},{om_hi:.2f}] chi2 {resS[iS,2]:.1f}")
    print(f"profile +sys: H0 {hY:.2f} [{loY:.2f},{hiY:.2f}]")
    print(f"mocks stat {est_s.mean():.2f}+-{est_s.std():.2f} (th {gls(70*efid, OM_FID)[1]:.2f}); "
          f"sys {est_y.mean():.2f}+-{est_y.std():.2f} (th {np.sqrt(gls(70*efid, OM_FID)[1]**2+(ETA*70)**2):.2f})")
    print(f"tension stat: Planck {tension(hS,sS,PLANCK):.1f} SH0ES {tension(hS,sS,SHOES):.1f}; "
          f"sys: Planck {tension(hY,sY,PLANCK):.1f} SH0ES {tension(hY,sY,SHOES):.1f}")


if __name__ == "__main__":
    main()
