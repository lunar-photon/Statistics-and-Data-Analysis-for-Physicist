"""From convergence to shear and back: the Kaiser-Squires inversion, shape noise and a mask.

Question: a survey measures shear, not convergence.  How well does the Kaiser-Squires
inversion return the convergence map (a) on a full periodic patch without noise, (b) with the
shape noise of a KiDS-like survey, (c) when some of the patch is masked?  And which smoothing
scale gives the mass map closest to the truth (the test of Jeffrey et al. 2021, their Fig. 6)?

Computes, for lognormal convergence maps of the fiducial cosmology:
  * the round trip kappa -> gamma -> (kappa_E, kappa_B): differences at machine precision;
  * pure shape noise -> kappa_E, kappa_B power against sigma_e^2/n, and the smoothed-noise rms
    against the lattice sum and the continuum formula sigma_e^2/(4 pi theta^2 n);
  * a mask (star holes and a 0.4-deg edge strip): E-mode errors and B-mode leakage, and how they
    fall with the distance from the mask;
  * the Pearson coefficient between the true map and the smoothed noisy kappa_E, as a function of
    the smoothing scale, for n = 5.59 (DES Y3), 6.2 (KiDS-1000) and 30 (Stage IV) per arcmin^2.

Writes: figures/ch12/w_ks_maps.pdf, figures/ch12/w_ks_pearson.pdf, results/ch12/w02_ks.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA
import lib_wl as wl


def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean())


def main():
    setup()
    sp = dict(np.load(DATA / "ch12" / "w_cl.npz"))
    P = wl.cl_on_grid(sp["ells"], sp["cl_fid"])
    lam = float(sp["lam_fid"])
    PG, _ = wl.lognormal_spectrum(P, lam)
    rng = rng_for("ch12", "w02_ks")

    # ---------------------------------------------------------------- (a) round trip
    kap = wl.lognormal_map(PG, lam, rng.standard_normal((wl.N, wl.N)))
    g1, g2 = wl.shear_from_kappa(kap)
    kE, kB = wl.kaiser_squires(g1, g2)
    rt_E = np.abs(kE - kap).max() / kap.std()
    rt_B = np.abs(kB).max() / kap.std()
    gam_rms = np.sqrt((g1**2 + g2**2).mean() / 2)

    # ---------------------------------------------------------------- (b) noise
    sp_ = wl.sigma_pix()
    edges = np.logspace(np.log10(200), np.log10(6000), 8)
    nE, nB = [], []
    for _ in range(50):
        n1, n2 = sp_ * rng.standard_normal((2, wl.N, wl.N))
        e, b = wl.kaiser_squires(n1, n2)
        nE.append(wl.band_power(e, edges)); nB.append(wl.band_power(b, edges))
    Nexp = wl.SIGMA_E**2 / (wl.NGAL / wl.ARCMIN**2)                  # sigma_e^2 / n, n per sr
    ratioE = np.mean(nE) / Nexp
    ratioB = np.mean(nB) / Nexp
    lat, cont = wl.sigma_noise_smoothed()
    sm = []
    for _ in range(50):
        n1, n2 = sp_ * rng.standard_normal((2, wl.N, wl.N))
        e, _ = wl.kaiser_squires(n1, n2)
        sm.append(wl.smooth(e, wl.THETA_G).std())
    sig_meas = np.mean(sm)

    # ---------------------------------------------------------------- (c) mask
    mask = wl.make_mask(rng)
    fobs = mask.mean()
    kEm, kBm = wl.kaiser_squires(g1 * mask, g2 * mask)
    # distance (arcmin) of every observed pixel from the nearest masked pixel (periodic tiling)
    big = np.tile(~mask, (3, 3))
    dist = ndimage.distance_transform_edt(~big)[wl.N:2 * wl.N, wl.N:2 * wl.N] * wl.PIX_ARCMIN
    err = (kEm - kap)
    near = mask & (dist <= 3)
    far = mask & (dist > 30)
    err_near = err[near].std() / kap.std()
    err_far = err[far].std() / kap.std()
    b_near = kBm[near].std() / kap.std()
    b_far = kBm[far].std() / kap.std()
    # power: leakage of E into B on the masked map, relative to the E power
    pEm = wl.band_power(kEm, edges)
    pBm = wl.band_power(kBm, edges)
    pE = wl.band_power(kap, edges)
    leak = pBm / pE
    supp = pEm / pE

    # ---------------------------------------------------------------- figure: maps
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 6.2), constrained_layout=True)
    ext = [0, wl.SIDE_DEG, 0, wl.SIDE_DEG]
    v = np.percentile(np.abs(wl.smooth(kap, 1.5)), 99.5)
    noisy1, noisy2 = g1 + sp_ * rng.standard_normal((wl.N, wl.N)), g2 + sp_ * rng.standard_normal((wl.N, wl.N))
    kEn, _ = wl.kaiser_squires(noisy1, noisy2)
    panels = [(wl.smooth(kap, 1.5), r"(a) true $\kappa$"),
              (np.where(mask, wl.smooth(kEm, 1.5), np.nan), r"(b) $\kappa_E$ from masked shear"),
              (np.where(mask, wl.smooth(kBm, 1.5), np.nan), r"(c) $\kappa_B$ from masked shear"),
              (wl.smooth(kEn, wl.THETA_G), r"(d) $\kappa_E$ with shape noise, $\theta_G=3'$")]
    for a, (m, t) in zip(ax.ravel(), panels):
        im = a.imshow(m.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, extent=ext)
        a.set_title(t)
        a.grid(False)
        a.set_xticks([0, 2, 4, 6]); a.set_yticks([0, 2, 4, 6])
    for a in ax[1]:
        a.set_xlabel(r"$\theta_1$ [deg]")
    for a in ax[:, 0]:
        a.set_ylabel(r"$\theta_2$ [deg]")
    fig.colorbar(im, ax=ax, shrink=0.6, label=r"$\kappa$")
    savefig(fig, "ch12", "w_ks_maps")

    # ---------------------------------------------------------------- Pearson coefficient vs smoothing
    thetas = np.array([0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    surveys = {"DES Y3 ($n=5.59$, $\\sigma_e=0.268$)": (5.59, 0.268),
               "KiDS-like ($n=6.2$, $\\sigma_e=0.27$)": (6.2, 0.27),
               "Stage IV ($n=30$, $\\sigma_e=0.27$)": (30.0, 0.27)}
    nmap = 20
    res = {k: np.zeros((nmap, len(thetas))) for k in surveys}
    res_hp = {k: np.zeros((nmap, len(thetas))) for k in surveys}
    for i in range(nmap):
        kt = wl.lognormal_map(PG, lam, rng.standard_normal((wl.N, wl.N)))
        t1, t2 = wl.shear_from_kappa(kt)
        # the truth at a HEALPix-like resolution (N_side = 1024 pixels are 3.4 arcmin wide)
        kt_hp = wl.smooth(kt, 3.4 / np.sqrt(12))
        white = rng.standard_normal((2, wl.N, wl.N))
        for name, (n, se) in surveys.items():
            s = wl.sigma_pix(n, se)
            e, _ = wl.kaiser_squires(t1 + s * white[0], t2 + s * white[1])
            for j, th in enumerate(thetas):
                es = wl.smooth(e, th)
                res[name][i, j] = pearson(es, kt)
                res_hp[name][i, j] = pearson(es, kt_hp)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    best = {}
    for c, name in zip(SERIES, surveys):
        r = res[name].mean(0)
        rh = res_hp[name].mean(0)
        ax.plot(thetas, r, "-o", color=c, ms=3, label=name)
        ax.plot(thetas, rh, ":", color=c)
        # optimum from a parabola through the best point and its neighbours (in log theta)
        best[name] = (thetas[np.argmax(r)], r.max(), thetas[np.argmax(rh)], rh.max())
    ax.set_xscale("log")
    ax.set_xlabel(r"smoothing scale $\theta_G$ [arcmin]")
    ax.set_ylabel(r"Pearson $r$ (true $\kappa$, smoothed $\kappa_E$)")
    ax.set_ylim(top=0.82)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"])
    savefig(fig, "ch12", "w_ks_pearson")

    names = list(surveys)
    save_numbers("ch12", "w02_ks", {
        "WRoundE": sci(rt_E), "WRoundB": sci(rt_B),
        "WGamRms": f"{gam_rms:.4f}", "WKapRms": f"{kap.std():.4f}",
        "WSigPixNoise": f"{sp_:.4f}",
        "WNoiseRatioE": f"{ratioE:.3f}", "WNoiseRatioB": f"{ratioB:.3f}",
        "WNoisePow": sci(Nexp, 2),
        "WSigNLat": f"{lat:.4f}", "WSigNCont": f"{cont:.4f}", "WSigNMeas": f"{sig_meas:.4f}",
        "WFobs": f"{fobs:.2f}",
        "WErrNear": f"{err_near:.2f}", "WErrFar": f"{err_far:.3f}",
        "WBNear": f"{b_near:.2f}", "WBFar": f"{b_far:.3f}",
        "WLeakLow": f"{100 * leak[0]:.0f}", "WLeakHigh": f"{100 * leak[-1]:.1f}",
        "WSuppLow": f"{supp[0]:.2f}", "WSuppHigh": f"{supp[-1]:.2f}",
        "WBestDES": f"{best[names[0]][0]:g}", "WBestRDES": f"{best[names[0]][1]:.2f}",
        "WBestKiDS": f"{best[names[1]][0]:g}", "WBestRKiDS": f"{best[names[1]][1]:.2f}",
        "WBestSfour": f"{best[names[2]][0]:g}", "WBestRSfour": f"{best[names[2]][1]:.2f}",
        "WBestDEShp": f"{best[names[0]][2]:g}", "WBestRDEShp": f"{best[names[0]][3]:.2f}",
        "WBestSfourhp": f"{best[names[2]][2]:g}",
    })
    print("round trip", rt_E, rt_B, "noise ratios", ratioE, ratioB, "sigN", lat, cont, sig_meas)
    print("mask fobs", fobs, "err near/far", err_near, err_far, "B near/far", b_near, b_far)
    print("leak", leak, "supp", supp)
    for k in best:
        print(k, best[k])
    print(res[names[1]].mean(0))


if __name__ == "__main__":
    main()
