"""14_local_research.py -- two error budgets of local-Universe measurements, in numbers.

Question 1: how much does an unmodelled peculiar velocity scatter the distance modulus of
one supernova at redshift z, and why does the SH0ES Hubble-flow sample start at z = 0.0233?
    sigma_mu(z) = (5 / ln 10) * sigma_v / (c z)        (low-redshift Hubble law)
Question 2: how does the error on H0 from cosmic chronometers fall as the number of
measurements grows, when a fully correlated fractional model error eta is present?
    sigma^2(k) = sigma_stat^2 / k + (eta * H0)^2       (k copies of the present compilation;
                                                        eq. 11c:eq:cc-sys)
    checked against the generalised least-squares estimator of 13_chronometers.py run on
    k stacked copies of the data.
Question 3: the naive statistical error on H0 from the Cosmicflows-4 groups
    (log-H scatter 0.091 dex over 35,000 groups, Tully et al. 2023, Sec. 13).
Question 4: how well could 37 Cepheid-host redshifts at cz ~ 2000 km/s fix H0 if their
peculiar velocities (250 km/s) were independent?
Question 5: does the Local Group velocity predicted from the 2M++ density field (540 km/s
towards l=268, b=38) plus the external bulk flow (159 km/s towards l=304, b=6) of
Carrick et al. (2015) add up to the 620 km/s CMB-dipole motion (vector sum)?
Writes figures/ch11/local_research.pdf, results/ch11/14_local_research.tex.
"""
import importlib.util
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import SERIES, save_numbers, savefig, setup  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CH, NAME = "ch11", "14_local_research"
C_KMS = 299792.458
MAG = 5.0 / np.log(10.0)                       # d mu / d ln d

# the chronometer data and estimator are reused from 13_chronometers.py
_spec = importlib.util.spec_from_file_location(
    "cc13", pathlib.Path(__file__).resolve().parent / "13_chronometers.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def sigma_mu(z, sv):
    """Scatter in distance modulus from a line-of-sight velocity scatter sv [km/s]."""
    return MAG * sv / (C_KMS * z)


def main():
    setup(9.0, 3.6)
    out = {}

    # ---- 1. peculiar-velocity scatter in the distance modulus --------------------------
    for sv, tag in [(250.0, "TwoFifty"), (300.0, "Three")]:
        for zz, ztag in [(0.01, "A"), (0.0233, "B"), (0.15, "C")]:
            out[f"FBLrMu{tag}{ztag}"] = f"{sigma_mu(zz, sv):.3f}"
    out["FBLrFracB"] = f"{100 * 250.0 / (C_KMS * 0.0233):.1f}"     # sigma_v / cz in per cent

    # ---- 2. chronometers: error floor from a correlated model error ---------------------
    om, eta = cc.OM_FID, cc.ETA
    h_a, s_a, _ = cc.gls(cc.H, om)                  # diagonal errors only
    floor = eta * h_a
    k = np.logspace(0, 2, 60)
    s_tot = np.sqrt(s_a ** 2 / k + floor ** 2)
    # check the closed form on stacked copies (k identical compilations)
    checks = []
    for kk in (1, 4, 16):
        zs, Hs, ss = cc.z, cc.H, cc.sig
        cc.z, cc.H, cc.sig = np.tile(zs, kk), np.tile(Hs, kk), np.tile(ss, kk)
        _, s_k, _ = cc.gls(cc.H, om, eta=eta)
        cc.z, cc.H, cc.sig = zs, Hs, ss
        checks.append((kk, s_k, np.sqrt(s_a ** 2 / kk + floor ** 2)))
    out["FBLrCcSa"] = f"{s_a:.2f}"
    out["FBLrCcFloor"] = f"{floor:.2f}"
    out["FBLrCcKone"] = f"{checks[0][1]:.2f}"
    out["FBLrCcKfour"] = f"{checks[1][1]:.2f}"
    out["FBLrCcKfourTh"] = f"{checks[1][2]:.2f}"
    out["FBLrCcKsixteen"] = f"{checks[2][1]:.2f}"
    out["FBLrCcKsixteenTh"] = f"{checks[2][2]:.2f}"
    out["FBLrCcHundred"] = f"{np.sqrt(s_a ** 2 / 100 + floor ** 2):.2f}"
    out["FBLrCcFloorPct"] = f"{100 * floor / h_a:.1f}"
    out["FBLrCcIMF"] = f"{0.004 * h_a:.2f}"             # floor if only the <0.4% IMF term remains

    # ---- 3. Cosmicflows-4: naive statistical error -------------------------------------
    scat_dex, ngroups, h_cf4 = 0.091, 35000, 74.6
    s_log = scat_dex / np.sqrt(ngroups)
    out["FBLrCfFrac"] = f"{100 * np.log(10) * scat_dex:.0f}"     # per-group scatter in per cent
    out["FBLrCfSlog"] = f"{s_log * 1e4:.1f}"                      # in units of 1e-4 dex
    out["FBLrCfSH"] = f"{np.log(10) * h_cf4 * s_log:.2f}"
    # ---- 4. two-rung ladder: Cepheid-host redshifts, independent velocities -------------
    nh, czh, sv = 37, 2000.0, 250.0
    out["FBLrHostFrac"] = f"{100 * sv / czh:.1f}"
    out["FBLrHostMean"] = f"{100 * sv / czh / np.sqrt(nh):.1f}"

    # ---- 5. Local Group: 2M++ prediction plus external bulk flow (Carrick et al. 2015) --
    def unit(l, b):
        l, b = np.radians(l), np.radians(b)
        return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])
    v_in = 540.0 * unit(268.0, 38.0)
    v_ext = 159.0 * unit(304.0, 6.0)
    cosang = unit(268.0, 38.0) @ unit(304.0, 6.0)
    v_tot = v_in + v_ext
    sp = np.linalg.norm(v_tot)
    l_tot = np.degrees(np.arctan2(v_tot[1], v_tot[0])) % 360
    b_tot = np.degrees(np.arcsin(v_tot[2] / sp))
    ang_cmb = np.degrees(np.arccos(v_tot / sp @ unit(271.9, 29.6)))
    out["FBLrLgCos"] = f"{cosang:.3f}"
    out["FBLrLgAng"] = f"{np.degrees(np.arccos(cosang)):.0f}"
    out["FBLrLgSpeed"] = f"{sp:.0f}"
    out["FBLrLgL"] = f"{l_tot:.0f}"
    out["FBLrLgB"] = f"{b_tot:.0f}"
    out["FBLrLgAngCMB"] = f"{ang_cmb:.0f}"

    # ---- 6. Cosmicflows-4: what a 0.10 mag modulus error is in velocity at 4000 km/s ---
    out["FBLrCfDist"] = f"{100 * 0.10 * np.log(10) / 5:.1f}"
    out["FBLrCfVerr"] = f"{4000 * 0.10 * np.log(10) / 5:.0f}"
    save_numbers(CH, NAME, out)

    # ---- figure ---------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2)
    zg = np.logspace(np.log10(0.003), np.log10(0.2), 300)
    a1.loglog(zg, sigma_mu(zg, 250.0), color=SERIES[0], label=r"$\sigma_v=250$ km/s")
    a1.loglog(zg, sigma_mu(zg, 300.0), color=SERIES[1], label=r"$\sigma_v=300$ km/s")
    a1.axvspan(0.0233, 0.15, color=SERIES[2], alpha=0.12, lw=0)
    a1.text(0.059, 0.4, "SH0ES\nHubble flow", ha="center", fontsize=8, color=SERIES[2])
    a1.set_xlabel(r"redshift $z$")
    a1.set_ylabel(r"$\sigma_\mu$ from peculiar velocity [mag]")
    a1.set_ylim(5e-3, 1.0)
    a1.legend(loc="lower left")

    a2.loglog(k * cc.N, s_tot, color=SERIES[0], label=r"with $\eta=4.5\%$ correlated")
    a2.loglog(k * cc.N, s_a / np.sqrt(k), color=SERIES[0], ls=":", label="statistical only")
    a2.axhline(floor, color="0.3", lw=0.8, ls="--")
    a2.text(k[-1] * cc.N * 0.9, floor * 0.88, r"floor $\eta H_0$", fontsize=8, color="0.3",
             ha="right", va="top")
    for kk, s_k, _ in checks:
        a2.plot(kk * cc.N, s_k, "o", color=SERIES[1], ms=5)
    from matplotlib.ticker import FormatStrFormatter
    a2.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    from matplotlib.ticker import NullFormatter
    a2.set_yticks([0.2, 0.5, 1, 2, 3])
    a2.yaxis.set_minor_formatter(NullFormatter())
    a2.set_xlabel("number of chronometer measurements")
    a2.set_ylabel(r"error on $H_0$ [km/s/Mpc]")
    a2.legend(loc="lower left")
    fig.tight_layout()
    savefig(fig, CH, "local_research")
    print(out)


if __name__ == "__main__":
    main()
