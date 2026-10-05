"""Scales, covariances and the cost of the simplifications of the weak-lensing forecasts.

Question: (1) at which multipole does the convergence signal fall below the shape noise for a
DES-like, a KiDS-like and a Stage-IV-like galaxy density?  (2) Is the Gaussian band-power variance
2 (C + N)^2 / n_modes right for lognormal maps, and are neighbouring bands correlated?  (3) How
much do a shear-calibration nuisance, a lower l_max, and a Stage-IV survey change sigma(S_8) in
the Gaussian Fisher forecast of the power spectrum?

Computes:
  * l where C_l = sigma_e^2 / n (Limber spectrum of w01);
  * from the 1000 lognormal and 600 Gaussian fiducial maps of w04 (data/ch12/w_beyond_sims.npz): the
    Monte Carlo variance of each band power over the Gaussian prediction, and the band correlations;
  * Gaussian Fisher forecasts of (Omega_m, sigma_8[, m]) for several survey set-ups.

Writes: figures/ch12/w_cov_ratio.pdf, figures/ch12/w_costs.pdf, results/ch12/w05_costs.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, DATA
import lib_wl as wl

ARCMIN_PER_RAD = 180 * 60 / np.pi


def noise_power(n_arcmin2, sigma_e):
    return sigma_e**2 / (n_arcmin2 / wl.ARCMIN**2)


def forecast(sp, area_deg2, n_arcmin2, sigma_e, lmin, lmax, m_prior=None, nbin=13):
    """Gaussian band-power Fisher forecast; returns sigma(S_8) and sigma(Omega_m)."""
    ells = sp["ells"]
    th = np.array([wl.OM_FID, wl.S8_FID])
    h = wl.STEP * th

    def C(name, l):
        return np.exp(np.interp(np.log(l), np.log(ells), np.log(sp[f"cl_{name}"])))

    N = noise_power(n_arcmin2, sigma_e)
    A = area_deg2 * (np.pi / 180) ** 2
    edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin)
    npar = 2 if m_prior is None else 3
    F = np.zeros((npar, npar))
    for lo, hi in zip(edges[:-1], edges[1:]):
        l = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1)
        w = l * A / (2 * np.pi) / 2 / (C("fid", l) + N) ** 2
        d = [(C("om_p", l) - C("om_m", l)) / (2 * h[0]), (C("s8_p", l) - C("s8_m", l)) / (2 * h[1])]
        if m_prior is not None:
            d.append(2 * C("fid", l))                   # C -> (1 + m)^2 C, derivative at m = 0
        for i in range(npar):
            for j in range(npar):
                F[i, j] += np.sum(w * d[i] * d[j])
    if m_prior is not None:
        F[2, 2] += 1 / m_prior**2
    cov = np.linalg.inv(F)[:2, :2]
    S8 = th[1] * np.sqrt(th[0] / 0.3)
    g = np.array([0.5 * S8 / th[0], S8 / th[1]])
    return np.sqrt(g @ cov @ g), np.sqrt(cov[0, 0])


def main():
    setup()
    sp = dict(np.load(DATA / "ch12" / "w_cl.npz"))
    ells = sp["ells"]
    nums = {}

    # ---------------------------------------------------------------- (1) where signal = noise
    for tag, (n, se) in {"DES": (5.59, 0.268), "KiDS": (6.2, 0.27), "Sfour": (30.0, 0.27)}.items():
        N = noise_power(n, se)
        lc = float(np.exp(np.interp(0.0, -np.log(sp["cl_fid"] / N), np.log(ells))))  # C/N falls through 1
        nums[f"WLcross{tag}"] = f"{lc:.0f}"
        nums[f"WThcross{tag}"] = f"{ARCMIN_PER_RAD / lc:.1f}"
        print(tag, "l_cross", lc, "1/l in arcmin", ARCMIN_PER_RAD / lc)

    # ---------------------------------------------------------------- (2) covariance of band powers
    sims = dict(np.load(DATA / "ch12" / "w_beyond_sims.npz"))
    nm = wl.modes_per_bin(wl.CL_EDGES)
    lb = wl.ell_mean(wl.CL_EDGES)
    out = {}
    for kind in ("ln", "g"):
        x = sims[f"{kind}_fid_cl"]
        mu, var = x.mean(0), x.var(0, ddof=1)
        gauss = 2 * mu**2 / nm
        corr = np.corrcoef(x, rowvar=False)
        off = corr[~np.eye(len(mu), dtype=bool)]
        nsim = len(x)
        out[kind] = dict(ratio=var / gauss, corr=corr, err=np.sqrt(2 / (nsim - 1)), offmax=np.abs(off).max(),
                         offmean=off.mean())
        print(kind, "var/gauss", var / gauss, "max |corr|", np.abs(off).max(), "mean corr", off.mean())
    nums.update({
        "WCovRatioLNmax": f"{out['ln']['ratio'].max():.2f}", "WCovRatioLNmin": f"{out['ln']['ratio'].min():.2f}",
        "WCovRatioGmax": f"{out['g']['ratio'].max():.2f}", "WCovRatioGmin": f"{out['g']['ratio'].min():.2f}",
        "WCovRatioErrLN": f"{out['ln']['err']:.3f}", "WCovRatioErrG": f"{out['g']['err']:.3f}",
        "WCorrLNmax": f"{out['ln']['offmax']:.2f}", "WCorrGmax": f"{out['g']['offmax']:.2f}",
        "WCorrLNmean": f"{out['ln']['offmean']:.2f}", "WCorrGmean": f"{out['g']['offmean']:.3f}",
        "WModesLow": int(nm[0]), "WModesHigh": int(nm[-1]),
    })
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    ax[0].errorbar(lb, out["ln"]["ratio"], yerr=out["ln"]["err"] * out["ln"]["ratio"], fmt="o", color=SERIES[1],
                   label="lognormal maps")
    ax[0].errorbar(lb * 1.04, out["g"]["ratio"], yerr=out["g"]["err"] * out["g"]["ratio"], fmt="s", mfc="none",
                   color=SERIES[0], label="Gaussian maps")
    ax[0].axhline(1, color="k", lw=0.8, ls="--")
    ax[0].set_xscale("log")
    ax[0].set_xticks([100, 300, 1000, 3000]); ax[0].set_xticklabels(["100", "300", "1000", "3000"])
    ax[0].minorticks_off()
    ax[0].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(r"Var$_{\rm MC}$ / $[2(C+N)^2/n_{\rm modes}]$")
    ax[0].legend(fontsize=8)
    im = ax[1].imshow(out["ln"]["corr"], cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    ax[1].set_title("band correlations, lognormal", fontsize=9)
    ax[1].set_xlabel("band"); ax[1].set_ylabel("band")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    savefig(fig, "ch12", "w_cov_ratio")

    # ---------------------------------------------------------------- (3) what the simplifications cost
    cases = [
        ("WCostBase", "ours: 1000 deg$^2$, $n=6.2$, $\\ell\\leq3000$", (1000, 6.2, 0.27, 100, 3000, None)),
        ("WCostLow", "$\\ell\\leq1500$", (1000, 6.2, 0.27, 100, 1500, None)),
        ("WCostM", "+ shear bias $m$, prior $0.02$", (1000, 6.2, 0.27, 100, 3000, 0.02)),
        ("WCostLowM", "$\\ell\\leq1500$ + $m$", (1000, 6.2, 0.27, 100, 1500, 0.02)),
        ("WCostDES", "DES-like: 4143 deg$^2$, $n=5.59$, $\\ell\\leq1500$ + $m$", (4143, 5.59, 0.268, 100, 1500, 0.02)),
        ("WCostSfour", "Stage IV: 15000 deg$^2$, $n=30$, $\\ell\\leq5000$ + $m$", (15000, 30.0, 0.27, 100, 5000, 0.02)),
    ]
    vals = []
    for key, label, args in cases:
        sS8, sOm = forecast(sp, *args)
        nums[key] = f"{sS8:.4f}"
        nums[key + "Om"] = f"{sOm:.3f}"
        vals.append((label, sS8))
        print(key, sS8, sOm)
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, [v[1] for v in vals], color=SERIES[0])
    ax.set_yticks(y)
    ax.set_yticklabels([v[0] for v in vals], fontsize=8)
    ax.axvline(0.0225, color=SERIES[1], ls="--", lw=1)
    ax.text(0.0230, y[-1] - 0.3, "KiDS-1000 published", color=SERIES[1], fontsize=7.5)
    ax.set_xlabel(r"forecast $\sigma(S_8)$, power spectrum only")
    savefig(fig, "ch12", "w_costs")
    save_numbers("ch12", "w05_costs", nums)
    print(nums)


if __name__ == "__main__":
    main()
