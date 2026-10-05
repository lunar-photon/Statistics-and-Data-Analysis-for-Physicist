"""08_hubble1929.py -- Hubble's 1929 velocity-distance diagram, refitted.

Question: what slope K do Hubble's own 24 galaxies give, with which estimator, and how
large is the error bar the scatter alone allows?

Computes, from Table 1 of Hubble (1929) (distances r in Mpc, velocities v in km/s):
  (a) least squares through the origin, v = K r  (velocity scatter only);
  (b) Hubble's own model, v = K r + X cos(a)cos(d) + Y sin(a)cos(d) + Z sin(d), where the last
      three terms are the reflex of the Sun's motion (X, Y, Z) projected on each line of sight;
  (c) the inverse regression r = v / K (distance treated as the noisy variable);
  (d) maximum likelihood with velocity scatter sigma_v AND fractional distance errors Delta;
  (e) a bootstrap of (a) and (b).
Writes figures/ch11/hubble1929.pdf, results/ch11/08_hubble1929.tex.
"""
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import SERIES, rng_for, save_numbers, savefig, setup, theory_line  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CH, NAME = "ch11", "08_hubble1929"

# Hubble (1929), Table 1: object, distance r [Mpc], velocity v [km/s].
# Positions: modern J2000 right ascension and declination in degrees (precession since 1900
# moves them by about one degree, far below what the solar-motion fit can resolve).
TABLE = [
    # name        r      v      RA      Dec
    ("SMC",      0.032,  170,  13.2, -72.8),
    ("LMC",      0.034,  290,  80.9, -69.8),
    ("NGC 6822", 0.214, -130, 296.2, -14.8),
    ("NGC 598",  0.263,  -70,  23.5,  30.7),
    ("NGC 221",  0.275, -185,  10.7,  40.9),
    ("NGC 224",  0.275, -220,  10.7,  41.3),
    ("NGC 5457", 0.45,   200, 210.8,  54.3),
    ("NGC 4736", 0.5,    290, 192.7,  41.1),
    ("NGC 5194", 0.5,    270, 202.5,  47.2),
    ("NGC 4449", 0.63,   200, 187.0,  44.1),
    ("NGC 4214", 0.8,    300, 183.9,  36.3),
    ("NGC 3031", 0.9,    -30, 148.9,  69.1),
    ("NGC 3627", 0.9,    650, 170.1,  13.0),
    ("NGC 4826", 0.9,    150, 194.2,  21.7),
    ("NGC 5236", 0.9,    500, 204.3, -29.9),
    ("NGC 1068", 1.0,    920,  40.7,   0.0),
    ("NGC 5055", 1.1,    450, 199.0,  42.0),
    ("NGC 7331", 1.1,    500, 339.3,  34.4),
    ("NGC 4258", 1.4,    500, 184.7,  47.3),
    ("NGC 4151", 1.7,    960, 182.6,  39.4),
    ("NGC 4382", 2.0,    500, 186.4,  18.2),
    ("NGC 4472", 2.0,    850, 187.4,   8.0),
    ("NGC 4486", 2.0,    800, 187.7,  12.4),
    ("NGC 4649", 2.0,   1090, 190.9,  11.6),
]
r = np.array([t[1] for t in TABLE])
v = np.array([t[2] for t in TABLE], float)
ra, dec = np.radians([t[3] for t in TABLE]), np.radians([t[4] for t in TABLE])
N = len(r)


def fit_origin(r, v):
    """(a) v = K r: K = sum(v r)/sum(r^2), error from the residual scatter."""
    K = np.sum(v * r) / np.sum(r ** 2)
    s2 = np.sum((v - K * r) ** 2) / (len(r) - 1)
    return K, np.sqrt(s2 / np.sum(r ** 2)), np.sqrt(s2)


def design_solar(r, ra, dec):
    return np.column_stack([r, np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])


def fit_solar(r, v, ra, dec):
    """(b) linear least squares for (K, X, Y, Z): normal equations A^T A p = A^T v."""
    A = design_solar(r, ra, dec)
    AtA = A.T @ A
    p = np.linalg.solve(AtA, A.T @ v)
    res = v - A @ p
    s2 = res @ res / (len(v) - A.shape[1])
    cov = s2 * np.linalg.inv(AtA)
    return p, np.sqrt(np.diag(cov)), np.sqrt(s2)


def fit_inverse(r, v):
    """(c) r = b v through the origin, K = 1/b = sum(v^2)/sum(v r)."""
    return np.sum(v ** 2) / np.sum(v * r)


def nll_ml(K, sv, Delta, r, v):
    """-ln L for v_i ~ N(K r_i, sv^2 + K^2 Delta^2 r_i^2) (first order in the distance error)."""
    s2 = sv ** 2 + (K * Delta * r) ** 2
    return 0.5 * np.sum((v - K * r) ** 2 / s2 + np.log(s2))


def fit_ml(r, v, Delta):
    """(d) maximise over K and sigma_v at fixed Delta (sigma_v profiled on a grid)."""
    best = (np.inf, None, None)
    for sv in np.linspace(5, 500, 300):
        res = minimize_scalar(lambda K: nll_ml(K, sv, Delta, r, v), bounds=(50, 1500),
                              method="bounded")
        if res.fun < best[0]:
            best = (res.fun, res.x, sv)
    return best[1], best[2]


def main():
    setup()
    Ka, sKa, sva = fit_origin(r, v)
    pb, spb, svb = fit_solar(r, v, ra, dec)
    Kinv = fit_inverse(r, v)
    # ML at fixed fractional distance error
    ml = {D: fit_ml(r, v, D) for D in (0.0, 0.2, 0.4)}

    # bootstrap: resample galaxies with replacement
    rng = rng_for(CH, NAME)
    nb = 4000
    Kb_a, Kb_b = np.empty(nb), np.empty(nb)
    for i in range(nb):
        j = rng.integers(0, N, N)
        Kb_a[i] = fit_origin(r[j], v[j])[0]
        try:
            Kb_b[i] = fit_solar(r[j], v[j], ra[j], dec[j])[0][0]
        except np.linalg.LinAlgError:
            Kb_b[i] = np.nan
    Kb_b = Kb_b[np.isfinite(Kb_b)]

    # solar-motion speed and its direction (equatorial)
    X, Y, Z = pb[1:]
    Vsun = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    ra_sun = np.degrees(np.arctan2(-Y, -X)) % 360    # the Sun moves opposite to the reflex
    dec_sun = np.degrees(np.arcsin(-Z / Vsun))

    # distance factor needed to bring each slope to H0 = 70
    fac_a, fac_b = Ka / 70.0, pb[0] / 70.0
    dmu_b = 5 * np.log10(fac_b)

    # ---------------- figure 1: the diagram
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    vcorr = v - design_solar(r, ra, dec)[:, 1:] @ pb[1:]
    ax[0].scatter(r, v, s=22, color=SERIES[0], label="Hubble's table", zorder=4)
    ax[0].scatter(r, vcorr, s=22, facecolor="none", edgecolor=SERIES[1],
                  label="after removing the solar-motion terms", zorder=4)
    rr = np.linspace(0, 2.2, 50)
    ax[0].plot(rr, Ka * rr, color=SERIES[0], lw=1.3, label=f"(a) through origin, K = {Ka:.0f}")
    ax[0].plot(rr, pb[0] * rr, color=SERIES[1], lw=1.3, label=f"(b) with solar motion, K = {pb[0]:.0f}")
    ax[0].plot(rr, Kinv * rr, color=SERIES[2], lw=1.3, ls=":", label=f"(c) inverse regression, K = {Kinv:.0f}")
    theory_line(ax[0], rr, 70 * rr, label="today's slope, 70")
    ax[0].axhline(0, color="0.6", lw=0.6)
    ax[0].set_xlabel("distance r [Mpc] (Hubble's scale)")
    ax[0].set_ylabel("velocity v [km/s]")
    ax[0].legend(loc="upper left", fontsize=7.5)
    ax[0].set_xlim(0, 2.2)

    bins = np.linspace(250, 750, 51)
    ax[1].hist(Kb_a, bins=bins, density=True, histtype="step", color=SERIES[0], lw=1.4,
               label="bootstrap of (a)")
    ax[1].hist(Kb_b, bins=bins, density=True, histtype="step", color=SERIES[1], lw=1.4,
               label="bootstrap of (b)")
    xx = np.linspace(250, 750, 300)
    theory_line(ax[1], xx, np.exp(-0.5 * ((xx - pb[0]) / spb[0]) ** 2) / (np.sqrt(2 * np.pi) * spb[0]),
                label="Gaussian, least-squares error of (b)")
    ax[1].axvline(465, color=SERIES[3], lw=1.2, label="Hubble: 465 ± 50")
    ax[1].set_xlabel("K [km/s/Mpc]")
    ax[1].set_ylabel("density")
    ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.3)
    ax[1].legend(fontsize=7.5, loc="upper right")
    savefig(fig, CH, "hubble1929")

    save_numbers(CH, NAME, {
        "FBHubN": N,
        "FBHubKa": f"{Ka:.0f}", "FBHubsKa": f"{sKa:.0f}", "FBHubSva": f"{sva:.0f}",
        "FBHubKb": f"{pb[0]:.0f}", "FBHubsKb": f"{spb[0]:.0f}",
        "FBHubX": f"{pb[1]:.0f}", "FBHubsX": f"{spb[1]:.0f}",
        "FBHubY": f"{pb[2]:.0f}", "FBHubsY": f"{spb[2]:.0f}",
        "FBHubZ": f"{pb[3]:.0f}", "FBHubsZ": f"{spb[3]:.0f}",
        "FBHubSvb": f"{svb:.0f}", "FBHubVsun": f"{Vsun:.0f}",
        "FBHubRAsun": f"{ra_sun:.0f}", "FBHubDecsun": f"{dec_sun:.0f}",
        "FBHubKinv": f"{Kinv:.0f}",
        "FBHubKmlZero": f"{ml[0.0][0]:.0f}", "FBHubSvmlZero": f"{ml[0.0][1]:.0f}",
        "FBHubKmlTwo": f"{ml[0.2][0]:.0f}", "FBHubSvmlTwo": f"{ml[0.2][1]:.0f}",
        "FBHubKmlFour": f"{ml[0.4][0]:.0f}", "FBHubSvmlFour": f"{ml[0.4][1]:.0f}",
        "FBHubBootSa": f"{np.std(Kb_a):.0f}", "FBHubBootSb": f"{np.std(Kb_b):.0f}",
        "FBHubBootMb": f"{np.mean(Kb_b):.0f}",
        "FBHubFaca": f"{fac_a:.1f}", "FBHubFacb": f"{fac_b:.1f}", "FBHubDmub": f"{dmu_b:.1f}",
        "FBHubNboot": nb,
        "FBHubSrr": f"{np.sum(r**2):.3f}", "FBHubSvr": f"{np.sum(v*r):.0f}",
    })
    print(f"(a) K={Ka:.1f}+-{sKa:.1f} sv={sva:.0f};  (b) K={pb[0]:.1f}+-{spb[0]:.1f} XYZ={pb[1:]} +- {spb[1:]} "
          f"Vsun={Vsun:.0f} (RA {ra_sun:.0f}, Dec {dec_sun:.0f});  (c) {Kinv:.1f};  ML {ml}; "
          f"boot sd a {np.std(Kb_a):.1f} b {np.std(Kb_b):.1f}")


if __name__ == "__main__":
    main()
