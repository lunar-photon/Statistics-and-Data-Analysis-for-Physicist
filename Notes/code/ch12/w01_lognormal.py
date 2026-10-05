"""Gaussian and lognormal convergence maps with the same power spectrum.

Question: a convergence map cannot go below the value of an empty line of sight, but it can go
far above zero inside a cluster.  A Gaussian field with the right C_ell ignores that.  Does a
shifted lognormal field (Xavier, Abdalla and Joachimi 2016) with the same C_ell fix the one-point
distribution without spoiling the power spectrum?

Computes:
  * C_ell^{kappa kappa} (own Limber integral over CAMB's halofit P(k)) and the empty-beam
    convergence kappa_empty for the fiducial cosmology and for Omega_m, sigma_8 shifted by +/-5%
    (cached in data/ch12/w_cl.npz for the later scripts);
  * the Gaussian spectrum P_G that makes kappa = lam (e^{G - s^2/2} - 1) have the target spectrum;
  * 100 Gaussian and 100 lognormal maps (6.4 x 6.4 deg, 1.5 arcmin pixels) from the same white
    noise: one-point distribution, skewness against the lognormal formula, recovered spectrum.

Writes: data/ch12/w_cl.npz, figures/ch12/w_lognormal_maps.pdf, figures/ch12/w_lognormal_pdf.pdf,
results/ch12/w01_lognormal.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA, theory_line
import lib_wl as wl

CACHE = DATA / "ch12" / "w_cl.npz"
ELLS = np.logspace(np.log10(5.0), np.log10(11500.0), 160)
COSMOS = {
    "fid": (wl.OM_FID, wl.S8_FID),
    "om_p": (wl.OM_FID * (1 + wl.STEP), wl.S8_FID),
    "om_m": (wl.OM_FID * (1 - wl.STEP), wl.S8_FID),
    "s8_p": (wl.OM_FID, wl.S8_FID * (1 + wl.STEP)),
    "s8_m": (wl.OM_FID, wl.S8_FID * (1 - wl.STEP)),
}


def spectra(force=False):
    if CACHE.exists() and not force:
        return dict(np.load(CACHE))
    out = {"ells": ELLS}
    for name, (om, s8) in COSMOS.items():
        cl, ke = wl.limber_and_shift(om, s8, ELLS)
        out[f"cl_{name}"], out[f"lam_{name}"] = cl, ke
        out[f"theta_{name}"] = np.array([om, s8])
        print(name, om, s8, "kappa_empty", ke)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def skew(x):
    x = x - x.mean()
    return (x**3).mean() / (x**2).mean() ** 1.5


def main():
    setup()
    sp = spectra()
    P = wl.cl_on_grid(sp["ells"], sp["cl_fid"])
    lam = float(sp["lam_fid"])
    PG, sG2 = wl.lognormal_spectrum(P, lam)
    xi0 = np.fft.ifft2(P).real[0, 0] / wl.A_PIX                 # variance of the pixel field
    r = np.sqrt(xi0) / lam
    skew_theory = r * (3 + r**2)                                   # lognormal skewness, alpha = lam
    rng = rng_for("ch12", "w01_lognormal")
    edges = np.logspace(np.log10(60), np.log10(9000), 15)
    nsim = 100
    pg, pl, sk_g, sk_l, sk_ls, sk_gs = [], [], [], [], [], []
    pix_g, pix_l, pix_ls, pix_gs = [], [], [], []
    for i in range(nsim):
        w = rng.standard_normal((wl.N, wl.N))
        kg = wl.gaussian_map(P, w)
        kl = wl.lognormal_map(PG, lam, w)
        pg.append(wl.band_power(kg, edges))
        pl.append(wl.band_power(kl, edges))
        sk_g.append(skew(kg)); sk_l.append(skew(kl))
        kls, kgs = wl.smooth(kl, wl.THETA_G), wl.smooth(kg, wl.THETA_G)
        sk_ls.append(skew(kls)); sk_gs.append(skew(kgs))
        if i < 20:
            pix_g.append(kg.ravel()); pix_l.append(kl.ravel())
            pix_ls.append(kls.ravel()); pix_gs.append(kgs.ravel())
        if i == 0:
            map_g, map_l = kgs, kls
    pg, pl = np.array(pg), np.array(pl)
    lb = wl.ell_mean(edges)
    target = np.array([P.ravel()[(wl.LMAG.ravel() >= a) & (wl.LMAG.ravel() < b)].mean()
                                              for a, b in zip(edges[:-1], edges[1:])])
    ratio_l = pl.mean(0) / target
    ratio_g = pg.mean(0) / target
    err_l = pl.std(0) / np.sqrt(nsim) / target
    # fraction of P_G modes that had to be clipped (negative before clipping)
    xi = np.fft.ifft2(P).real / wl.A_PIX
    PGraw = np.fft.fft2(np.log1p(xi / lam**2)).real * wl.A_PIX
    clipped = float((PGraw[wl.LMAG > 0] < 0).mean())

    # ---------------------------------------------------------------- figure 1: the two maps
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    vmax = np.percentile(np.abs(map_l), 99.7)
    ext = [0, wl.SIDE_DEG, 0, wl.SIDE_DEG]
    for a, m, t in zip(ax, (map_g, map_l), ("Gaussian", "lognormal")):
        im = a.imshow(m.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext)
        a.set_title(f"{t}: min {m.min():.3f}, max {m.max():.3f}")
        a.set_xlabel(r"$\theta_1$ [deg]")
        a.grid(False)
    ax[0].set_ylabel(r"$\theta_2$ [deg]")
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\kappa$ (smoothed, $\theta_G=3'$)")
    savefig(fig, "ch12", "w_lognormal_maps")

    # ---------------------------------------------------------------- figure 2: one-point PDF and spectrum
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    xg, xl = np.concatenate(pix_g), np.concatenate(pix_l)
    bins = np.linspace(-0.06, 0.10, 100)
    ax[0].hist(xg, bins=bins, density=True, histtype="step", color=SERIES[0], label="Gaussian")
    ax[0].hist(xl, bins=bins, density=True, histtype="step", color=SERIES[1], label="lognormal")
    # lognormal density: kappa = lam (e^{G - s2/2} - 1), G ~ N(0, s2)
    s2 = np.log1p(xi0 / lam**2)
    xx = np.linspace(-lam + 1e-4, 0.10, 600)
    y = np.log(xx / lam + 1) + s2 / 2
    dens = np.exp(-y**2 / (2 * s2)) / np.sqrt(2 * np.pi * s2) / (xx + lam)
    theory_line(ax[0], xx, dens, label="lognormal formula")
    ax[0].axvline(-lam, color=SERIES[3], lw=1, ls=":")
    ax[0].text(-lam + 0.002, 2e-2, r"$-\lambda$", color=SERIES[3])
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-2, 60)
    ax[0].set_xlabel(r"pixel $\kappa$ (1.5$'$ pixels)")
    ax[0].set_ylabel("density")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].errorbar(lb, ratio_l, yerr=err_l, fmt="o", color=SERIES[1], label="lognormal")
    ax[1].plot(lb, ratio_g, "s", mfc="none", color=SERIES[0], label="Gaussian")
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0.9, 1.1)
    ax[1].set_xlabel(r"$\ell$")
    ax[1].set_ylabel(r"$\langle\hat P\rangle / P_{\rm target}$ (100 maps)")
    ax[1].legend(fontsize=8)
    savefig(fig, "ch12", "w_lognormal_pdf")

    lams = {k: float(sp[f"lam_{k}"]) for k in COSMOS}
    save_numbers("ch12", "w01_lognormal", {
        "WLam": f"{lam:.4f}",
        "WLamOmP": f"{lams['om_p']:.4f}", "WLamOmM": f"{lams['om_m']:.4f}",
        "WLamRatio": f"{(np.log(lams['om_p']) - np.log(lams['om_m'])) / (np.log(1.05) - np.log(0.95)):.2f}",
        "WSigPix": f"{np.sqrt(xi0):.4f}",
        "WRatioR": f"{r:.2f}",
        "WSkewTheory": f"{skew_theory:.2f}",
        "WSkewLog": f"{np.mean(sk_l):.2f}", "WSkewLogErr": f"{np.std(sk_l) / np.sqrt(nsim):.3f}",
        "WSkewGauss": f"{np.mean(sk_g):.3f}",
        "WSkewLogSm": f"{np.mean(sk_ls):.2f}", "WSkewGaussSm": f"{np.mean(sk_gs):.3f}",
        "WSigGTwo": f"{s2:.3f}",
        "WPowDev": f"{100 * np.max(np.abs(ratio_l - 1)[lb > 300]):.1f}", "WPowDevLow": f"{100 * np.max(np.abs(ratio_l - 1)):.1f}",
        "WPowDevG": f"{100 * np.max(np.abs(ratio_g - 1)):.1f}",
        "WClipped": f"{100 * clipped:.2f}",
        "WMinLog": f"{min(np.concatenate(pix_l)):.4f}",
        "WSigSm": f"{np.concatenate(pix_ls).std():.4f}",
        "WNsimOne": nsim,
    })
    print("lam", lam, "sigma_pix", np.sqrt(xi0), "skew theory", skew_theory, "measured", np.mean(sk_l),
          "smoothed", np.mean(sk_ls), "ratio", ratio_l, "clipped", clipped)


if __name__ == "__main__":
    main()
