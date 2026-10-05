"""12_malmquist.py -- inhomogeneous Malmquist bias along one line of sight.

Question: galaxies whose true distances r are scattered into inferred distances d by a
lognormal distance indicator, ln d = ln r + Delta*eps, sit on a density peak. What is the
mean true distance of the galaxies found at inferred distance d, and what spurious peculiar
velocities does that produce?

Rebuilds the simulation of Strauss & Willick (1995), Sec. 6.5.2 and Fig. 15: a Gaussian
density peak at 5000 km/s, full width (FWHM) 1200 km/s, peak contrast 25, on a uniform
background; Tully-Fisher scatter 0.35 mag, so Delta = (ln 10/5) 0.35; pure Hubble flow.
Compares the binned mean <r|d> with
  uniform-density Malmquist bias   E(r|d) = d exp(7 Delta^2 / 2)            (SW eq. 184)
  slowly-varying approximation     E(r|d) = d [1 + (7/2 + gamma(d)) Delta^2]  (SW eq. 185)
  the exact Bayes integral         E(r|d) = int r^3 n p(d|r) dr / int r^2 n p(d|r) dr  (SW eq. 183).
Also: the skewness of the naive peculiar velocity cz - H0 d versus the logarithmic estimator.
Writes figures/ch11/malmquist.pdf, results/ch11/12_malmquist.tex.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import SERIES, rng_for, save_numbers, savefig, setup, theory_line  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CH, NAME = "ch11", "12_malmquist"
RP, FWHM, DMAX = 5000.0, 1200.0, 25.0
SP = FWHM / (2 * np.sqrt(2 * np.log(2)))
SIG_TF = 0.35
DELTA = np.log(10) / 5 * SIG_TF
RLO, RHI = 50.0, 16000.0


def n_of_r(r):
    return 1.0 + DMAX * np.exp(-0.5 * ((r - RP) / SP) ** 2)


def gamma_of_r(r):
    """dln n / dln r."""
    pk = DMAX * np.exp(-0.5 * ((r - RP) / SP) ** 2)
    return r * pk * (-(r - RP) / SP ** 2) / (1 + pk)


def exact_mean(d, rr=np.linspace(RLO, RHI, 20000)):
    """E(r|d) from the posterior p(r|d) ~ r^2 n(r) p(d|r), p(d|r) lognormal around r."""
    out = []
    for dd in np.atleast_1d(d):
        lik = np.exp(-0.5 * (np.log(dd / rr) / DELTA) ** 2) / dd
        w = rr ** 2 * n_of_r(rr) * lik
        out.append(np.trapezoid(rr * w, rr) / np.trapezoid(w, rr))
    return np.array(out)


def draw(rng, n_target_peakbin=800, binw=200.0):
    """Draw true distances with density proportional to r^2 n(r) on [RLO, RHI]."""
    rr = np.linspace(RLO, RHI, 200001)
    pdf = rr ** 2 * n_of_r(rr)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    peak_frac = np.interp(RP + binw / 2, rr, cdf) - np.interp(RP - binw / 2, rr, cdf)
    N = int(n_target_peakbin / peak_frac)
    r = np.interp(rng.random(N), cdf, rr)
    return r


def main():
    setup()
    rng = rng_for(CH, NAME)
    r = draw(rng)
    N = len(r)
    d = r * np.exp(DELTA * rng.standard_normal(N))
    edges = np.arange(1000, 8551, 150)
    mid = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(d, edges) - 1
    ok = (idx >= 0) & (idx < len(mid))
    cnt = np.bincount(idx[ok], minlength=len(mid))
    mean_r = np.bincount(idx[ok], weights=r[ok], minlength=len(mid)) / np.maximum(cnt, 1)
    err_r = np.array([r[ok][idx[ok] == i].std() / np.sqrt(max(c, 1)) for i, c in enumerate(cnt)])

    dd = np.linspace(1000, 8500, 400)
    uni = dd * np.exp(3.5 * DELTA ** 2)
    slow = dd * (1 + (3.5 + gamma_of_r(dd)) * DELTA ** 2)
    exact = exact_mean(dd)
    exact_mid = exact_mean(mid)
    good = cnt >= 5
    chi2 = np.sum(((mean_r - exact_mid)[good] / err_r[good]) ** 2)

    # uniform-density check
    rng2 = rng_for(CH, NAME, stream=1)
    ru = np.cbrt(rng2.random(400000)) * 15000
    du = ru * np.exp(DELTA * rng2.standard_normal(len(ru)))
    sel = (du > 3000) & (du < 6000)
    ratio_u = np.mean(ru[sel] / du[sel])

    # skewness of the naive and log peculiar-velocity estimators for a galaxy at r = 5000
    eps = rng2.standard_normal(200000)
    naive = 5000 * (1 - np.exp(DELTA * eps))
    logv = 5000 * np.log(1 / np.exp(DELTA * eps))
    def skew(x):
        return np.mean((x - x.mean()) ** 3) / x.std() ** 3

    # largest spurious velocity in the S-wave (relative to the uniform prediction)
    s_wave = exact - uni
    i_fore = int(np.argmax(s_wave * (dd < RP)))
    i_back = int(np.argmin(s_wave * (dd > RP)))

    fig, ax = plt.subplots(3, 1, figsize=(6.4, 8.2), sharex=False,
                           gridspec_kw=dict(height_ratios=[1, 1.5, 1]))
    ax[0].hist(r, bins=np.arange(0, 9001, 200), histtype="step", color=SERIES[0], lw=1.3)
    ax[0].set_xlim(1000, 8500)
    ax[0].set_xlabel(r"true distance $r$ [km/s]")
    ax[0].set_ylabel("galaxies per bin")
    ax[1].errorbar(mid, mean_r, err_r, fmt="s", ms=3, mfc="white", color=SERIES[0], label="simulation")
    ax[1].plot(dd, dd, color="0.55", lw=1.0, label="Hubble line $r=d$")
    ax[1].plot(dd, uni, color=SERIES[3], lw=1.2, ls="--", label="uniform density, eq. (184)")
    ax[1].plot(dd, slow, color=SERIES[2], lw=1.2, ls=":", label="slowly varying, eq. (185)")
    ax[1].plot(dd, exact, color="k", lw=1.5, label="exact, eq. (183)")
    ax[1].set_xlim(1000, 8500)
    ax[1].set_ylim(500, 10500)
    ax[1].set_xlabel(r"inferred distance $d$ [km/s]")
    ax[1].set_ylabel(r"mean true distance $\langle r\,|\,d\rangle$ [km/s]")
    ax[1].legend(fontsize=7.5, loc="upper left")
    ax[2].errorbar(mid, mean_r - mid, err_r, fmt="s", ms=3, mfc="white", color=SERIES[0])
    ax[2].plot(dd, exact - dd, color="k", lw=1.5)
    ax[2].plot(dd, uni - dd, color=SERIES[3], lw=1.2, ls="--")
    ax[2].axhline(0, color="0.55", lw=0.8)
    ax[2].axvline(RP, color="0.7", lw=0.8, ls=":")
    ax[2].set_xlim(1000, 8500)
    ax[2].set_xlabel(r"inferred distance $d$ [km/s]")
    ax[2].set_ylabel(r"spurious $v_{\rm pec}=cz-d$ [km/s]")
    fig.tight_layout()
    savefig(fig, CH, "malmquist")

    save_numbers(CH, NAME, {
        "FBMbDelta": f"{DELTA:.3f}", "FBMbN": N, "FBMbSp": f"{SP:.0f}",
        "FBMbUniFac": f"{np.exp(3.5*DELTA**2):.3f}", "FBMbUniPct": f"{100*(np.exp(3.5*DELTA**2)-1):.1f}",
        "FBMbUniSim": f"{ratio_u:.3f}",
        "FBMbChi": f"{chi2:.0f}", "FBMbNbins": int(good.sum()), "FBMbChiSd": f"{np.sqrt(2 * good.sum()):.0f}",
        "FBMbForeD": f"{dd[i_fore]:.0f}", "FBMbForeV": f"{s_wave[i_fore]:.0f}",
        "FBMbBackD": f"{dd[i_back]:.0f}", "FBMbBackV": f"{s_wave[i_back]:.0f}",
        "FBMbSkewNaive": f"{skew(naive):.2f}", "FBMbSkewLog": f"{round(skew(logv), 2) + 0.0:.2f}",   # + 0.0: no "-0.00"
        "FBMbMeanNaive": f"{naive.mean():.0f}", "FBMbMeanLog": f"{round(logv.mean()) + 0.0:.0f}",
        "FBMbGammaMax": f"{np.max(np.abs(gamma_of_r(dd))):.0f}",
    })
    print(f"N={N} Delta={DELTA:.4f} uniform factor {np.exp(3.5*DELTA**2):.4f} sim {ratio_u:.4f}; "
          f"chi2 {chi2:.1f}/{good.sum()}; S-wave fore {s_wave[i_fore]:.0f} at {dd[i_fore]:.0f}, "
          f"back {s_wave[i_back]:.0f} at {dd[i_back]:.0f}; skew naive {skew(naive):.2f} log {skew(logv):.2f} "
          f"means {naive.mean():.0f} {logv.mean():.0f}")


if __name__ == "__main__":
    main()
