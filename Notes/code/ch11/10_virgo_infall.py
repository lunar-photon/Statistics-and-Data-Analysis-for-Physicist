"""10_virgo_infall.py -- the local Hubble flow with an infall towards the Virgo cluster.

Question: in a simulated local sample (galaxies within 30 Mpc, distances with lognormal
errors, a small thermal velocity scatter), can H0 and the infall speed v_inf of the Local
Group towards Virgo be fitted together, and what goes wrong if the infall is ignored?

Model (linear theory around a spherical overdensity with mean contrast inside radius R
falling as R^-2, so that the infall speed falls as 1/R):
    u(x) = - v_inf (d_V / R) R_hat,   R = x - x_V,
    v_obs = H0 r + [u(x) - u(0)] . r_hat + eps,   eps ~ N(0, sigma_th^2),
which is linear in (H0, v_inf).  Galaxies closer than R_cut to Virgo are excluded (there
the contrast is not small).  Fit: weighted least squares with variance
sigma_th^2 + (H0 Delta r_obs)^2, iterated on H0.

Also: the motion of the Local Group relative to the CMB, from the Sun-CMB and Sun-LG
vectors of Planck 2018 I (Table 3), and its component towards Virgo.
Writes figures/ch11/virgo_infall.pdf, results/ch11/10_virgo_infall.tex.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import SERIES, rng_for, save_numbers, savefig, setup, theory_line  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CH, NAME = "ch11", "10_virgo_infall"
H0, VINF, SIGTH = 70.0, 200.0, 60.0        # km/s/Mpc, km/s, km/s
DV = 16.5                                  # Mpc, distance of Virgo
LV, BV = 283.8, 74.5                       # Galactic direction of Virgo (M87), degrees
RMIN, RMAX, RCUT = 2.0, 30.0, 5.0          # Mpc
NGAL, NMOCK = 400, 2000


def unit(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


SV = unit(LV, BV)
XV = DV * SV


def infall_g(x):
    """g(x) with [u(x) - u(0)] . r_hat = v_inf * g(x) for the 1/R infall model."""
    r = np.linalg.norm(x, axis=1)
    rhat = x / r[:, None]
    R = x - XV
    Rn = np.linalg.norm(R, axis=1)
    u_dir = -(DV / Rn)[:, None] * (R / Rn[:, None])       # u(x)/v_inf
    u0 = SV                                              # u(0)/v_inf: we fall towards Virgo
    return np.sum((u_dir - u0) * rhat, axis=1)


def draw_positions(rng, n, cone=None):
    """Uniform in volume between RMIN and RMAX, outside RCUT of Virgo; optional cone (deg) on Virgo."""
    out = []
    while len(out) < n:
        m = 4 * n
        u = rng.random(m)
        r = (RMIN ** 3 + u * (RMAX ** 3 - RMIN ** 3)) ** (1 / 3)
        mu = rng.uniform(-1, 1, m)
        ph = rng.uniform(0, 2 * np.pi, m)
        s = np.sqrt(1 - mu ** 2)
        x = r[:, None] * np.column_stack([s * np.cos(ph), s * np.sin(ph), mu])
        keep = np.linalg.norm(x - XV, axis=1) > RCUT
        if cone is not None:
            keep &= (x @ SV) / r > np.cos(np.radians(cone))
        out.extend(x[keep])
    return np.array(out[:n])


def fit(r_obs, g_obs, v, Delta, with_infall=True):
    """Iterated weighted least squares for (H0, v_inf); returns estimates and covariance."""
    A = np.column_stack([r_obs, g_obs]) if with_infall else r_obs[:, None]
    H = 70.0
    for _ in range(5):
        w = 1.0 / (SIGTH ** 2 + (H * Delta * r_obs) ** 2)
        F = A.T @ (w[:, None] * A)
        p = np.linalg.solve(F, A.T @ (w * v))
        H = p[0]
    return p, np.linalg.inv(F)


def mock(rng, Delta, cone=None):
    x = draw_positions(rng, NGAL, cone)
    r = np.linalg.norm(x, axis=1)
    v = H0 * r + VINF * infall_g(x) + rng.normal(0, SIGTH, len(r))
    r_obs = r * np.exp(Delta * rng.standard_normal(len(r)))
    # the observer evaluates g at the position implied by the measured distance
    x_obs = x * (r_obs / r)[:, None]
    return x, r, v, r_obs, infall_g(x_obs)


def main():
    setup()
    rng = rng_for(CH, NAME)
    res = {}
    for label, Delta, cone in (("five", 0.05, None), ("twenty", 0.20, None), ("conefive", 0.05, 40.0)):
        est, est_noinf, errs = [], [], []
        for _ in range(NMOCK):
            x, r, v, r_obs, g_obs = mock(rng, Delta, cone)
            p, C = fit(r_obs, g_obs, v, Delta)
            q, _ = fit(r_obs, g_obs, v, Delta, with_infall=False)
            est.append(p)
            est_noinf.append(q[0])
            errs.append(np.sqrt(np.diag(C)))
        est, est_noinf, errs = np.array(est), np.array(est_noinf), np.array(errs)
        res[label] = dict(H=est[:, 0], V=est[:, 1], Hn=est_noinf, eH=errs[:, 0], eV=errs[:, 1])

    # one mock for the picture (5% distances, full sky)
    rng1 = rng_for(CH, NAME, stream=1)
    x, r, v, r_obs, g_obs = mock(rng1, 0.05)
    p1, C1 = fit(r_obs, g_obs, v, 0.05)
    cosang = (x @ SV) / r

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    sc = ax[0].scatter(r_obs, v - H0 * r_obs, c=cosang, cmap="RdBu_r", vmin=-1, vmax=1, s=10)
    cb = fig.colorbar(sc, ax=ax[0], pad=0.02, fraction=0.05)
    cb.set_label(r"cosine of angle to Virgo")
    ax[0].axhline(0, color="0.5", lw=0.7)
    ax[0].set_xlabel(r"measured distance $r$ [Mpc]")
    ax[0].set_ylabel(r"$v_{\rm obs}-H_0 r$ [km/s]")
    # model along the Virgo direction and opposite
    rr = np.linspace(RMIN, RMAX, 300)
    for sgn, col, lab in ((1, SERIES[7], "towards Virgo"), (-1, SERIES[0], "away from Virgo")):
        xs = (sgn * rr)[:, None] * SV[None, :]
        ok = np.linalg.norm(xs - XV, axis=1) > RCUT
        gg = np.where(ok, infall_g(xs), np.nan)
        ax[0].plot(rr, VINF * gg, color=col, lw=1.5, label=f"model, line of sight {lab}")
    ax[0].legend(fontsize=7.5, loc="lower left")
    ax[0].set_ylim(-1000, 600)

    bins = np.linspace(52, 76, 121)
    for lab, col, txt in (("five", SERIES[0], "5% distances"), ("twenty", SERIES[1], "20% distances"),
                          ("conefive", SERIES[2], r"5%, only within 40$^\circ$ of Virgo")):
        ax[1].hist(res[lab]["H"], bins=bins, density=True, histtype="step", color=col, lw=1.4,
                   label=f"{txt}: joint fit")
        ax[1].hist(res[lab]["Hn"], bins=bins, density=True, histtype="stepfilled", color=col, alpha=0.18,
                   label=f"{txt}: infall ignored")
    ax[1].axvline(H0, color="k", lw=1.0, ls="--")
    ax[1].set_xlabel(r"$\hat H_0$ [km/s/Mpc]")
    ax[1].set_ylabel("density")
    ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.7)
    ax[1].legend(fontsize=6.5, loc="upper left")
    fig.subplots_adjust(wspace=0.38)
    savefig(fig, CH, "virgo_infall")

    # Local Group relative to the CMB (Planck 2018 I, Table 3), Galactic coordinates
    v_sun_cmb = 369.82 * unit(264.021, 48.253)
    v_sun_lg = 299.0 * unit(98.4, -5.9)
    v_lg_cmb = v_sun_cmb - v_sun_lg
    s_lg = np.linalg.norm(v_lg_cmb)
    l_lg = np.degrees(np.arctan2(v_lg_cmb[1], v_lg_cmb[0])) % 360
    b_lg = np.degrees(np.arcsin(v_lg_cmb[2] / s_lg))
    ang_virgo = np.degrees(np.arccos(v_lg_cmb @ SV / s_lg))
    comp_virgo = v_lg_cmb @ SV
    resid = np.linalg.norm(v_lg_cmb - VINF * SV)
    T0, dip = 2.7255, 3362.08e-6
    beta = dip / T0

    out = {"FBViHtrue": f"{H0:.0f}", "FBViVinf": f"{VINF:.0f}", "FBViSigth": f"{SIGTH:.0f}",
           "FBViDv": f"{DV}", "FBViRmax": f"{RMAX:.0f}", "FBViRmin": f"{RMIN:.0f}", "FBViRcut": f"{RCUT:.0f}",
           "FBViNgal": NGAL, "FBViNmock": NMOCK,
           "FBViOneH": f"{p1[0]:.1f}", "FBViOneHe": f"{np.sqrt(C1[0,0]):.1f}",
           "FBViOneV": f"{p1[1]:.0f}", "FBViOneVe": f"{np.sqrt(C1[1,1]):.0f}",
           "FBViRho": f"{C1[0,1]/np.sqrt(C1[0,0]*C1[1,1]):.2f}"}
    names = {"five": "Five", "twenty": "Twenty", "conefive": "Cone"}
    for lab, nm in names.items():
        d = res[lab]
        out.update({f"FBViH{nm}": f"{d['H'].mean():.2f}", f"FBViHsd{nm}": f"{d['H'].std():.2f}",
                    f"FBViHerr{nm}": f"{np.median(d['eH']):.2f}",
                    f"FBViV{nm}": f"{d['V'].mean():.0f}", f"FBViVsd{nm}": f"{d['V'].std():.0f}",
                    f"FBViVerr{nm}": f"{np.median(d['eV']):.0f}",
                    f"FBViHn{nm}": f"{d['Hn'].mean():.2f}", f"FBViHnsd{nm}": f"{d['Hn'].std():.2f}"})
    out.update({"FBViLGspeed": f"{s_lg:.0f}", "FBViLGl": f"{l_lg:.0f}", "FBViLGb": f"{b_lg:.0f}",
                "FBViLGang": f"{ang_virgo:.0f}", "FBViLGcomp": f"{comp_virgo:.0f}",
                "FBViLGresid": f"{resid:.0f}", "FBViBeta": f"{beta*1e3:.4f}",
                "FBViVsunCMB": f"{beta*299792.458:.1f}"})
    save_numbers(CH, NAME, out)
    for lab in names:
        d = res[lab]
        print(lab, f"H {d['H'].mean():.2f}+-{d['H'].std():.2f} (Fisher {np.median(d['eH']):.2f}) "
              f"V {d['V'].mean():.0f}+-{d['V'].std():.0f} (Fisher {np.median(d['eV']):.0f}) "
              f"no-infall H {d['Hn'].mean():.2f}+-{d['Hn'].std():.2f}")
    print(f"one mock: H={p1[0]:.1f}+-{np.sqrt(C1[0,0]):.1f} V={p1[1]:.0f}+-{np.sqrt(C1[1,1]):.0f}; "
          f"LG-CMB {s_lg:.0f} km/s (l={l_lg:.0f}, b={b_lg:.0f}), angle to Virgo {ang_virgo:.0f}, "
          f"component {comp_virgo:.0f}, residual after 200 to Virgo {resid:.0f}; beta={beta:.5e} v={beta*299792.458:.1f}")


if __name__ == "__main__":
    main()
