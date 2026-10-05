"""Matter power spectrum, linear versus halofit, and the BAO bump in xi(r).

Question: what do the matter power spectrum P(k) and its Fourier partner, the
correlation function xi(r), look like for the Planck 2018 cosmology, where does
linear theory fail, and where is the baryon acoustic peak?

Computes: CAMB linear and halofit (Takahashi 2012) P(k) at z = 0 and z = 1,
sigma_8, the drag-epoch sound horizon r_drag; xi(r) from the linear P(k) via
    xi(r) = int_0^inf dk/(2 pi^2) k^2 P(k) sin(kr)/(kr)
(the Fourier pair in the cosmology notes), with a Gaussian damping exp(-k^2 s^2), s = 1 Mpc/h,
only to make the oscillatory integral converge (it does not move the BAO peak).

Writes: data/chT1/pk.npz (cache), figures/chT1/pk_lin_nl.pdf,
figures/chT1/xi_bao.pdf, results/chT1/02_pk_xi.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, DATA
from camb_fiducial import FIDUCIAL

CACHE = DATA / "chT1" / "pk.npz"
ZS = [0.0, 1.0]


def compute():
    import camb
    p = camb.set_params(**FIDUCIAL, halofit_version="takahashi")
    p.set_matter_power(redshifts=ZS, kmax=60.0, nonlinear=True)
    r = camb.get_results(p)
    # CAMB returns redshifts in increasing order; units h/Mpc and (Mpc/h)^3
    k, z, pk_nl = r.get_matter_power_spectrum(minkh=1e-4, maxkh=50, npoints=2000)
    p.NonLinear = camb.model.NonLinear_none
    r_lin = camb.get_results(p)
    _, _, pk_lin = r_lin.get_matter_power_spectrum(minkh=1e-4, maxkh=50, npoints=2000)
    der = r_lin.get_derived_params()
    s8 = r_lin.get_sigma8_0()  # sigma_8 at z = 0
    out = dict(k=k, z=np.array(z), pk_nl=pk_nl, pk_lin=pk_lin,
               sigma8=s8, rdrag=der["rdrag"], h=FIDUCIAL["H0"] / 100)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def xi_of_r(k, pk, r, s=1.0):
    """xi(r) as in the cosmology notes, on a dense log-k grid with Gaussian damping."""
    kk = np.logspace(-4, np.log10(50), 20000)
    pp = np.exp(np.interp(np.log(kk), np.log(k), np.log(pk))) * np.exp(-(kk * s) ** 2)
    kr = np.outer(r, kk)
    integrand = kk ** 3 * pp * np.sinc(kr / np.pi) / (2 * np.pi ** 2)  # d ln k form
    return np.trapezoid(integrand, np.log(kk), axis=1)


def tophat_sigma(k, pk, R):
    """sigma_R from the linear P(k) and a spherical top hat of radius R (Mpc/h)."""
    x = k * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.sqrt(np.trapezoid(k ** 3 * pk * W ** 2 / (2 * np.pi ** 2), np.log(k)))


def main():
    z = dict(np.load(CACHE)) if CACHE.exists() else compute()
    k, pk_lin, pk_nl = z["k"], z["pk_lin"], z["pk_nl"]
    i0 = int(np.argmin(np.abs(z["z"] - 0.0)))
    i1 = int(np.argmin(np.abs(z["z"] - 1.0)))

    setup()
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.2, 1]))
    for c, i, lab in ((SERIES[0], i0, "z=0"), (SERIES[1], i1, "z=1")):
        ax.loglog(k, pk_lin[i], color=c, lw=1.2, ls="--", label=f"linear, {lab}")
        ax.loglog(k, pk_nl[i], color=c, lw=1.6, label=f"halofit, {lab}")
        axr.semilogx(k, pk_nl[i] / pk_lin[i], color=c, lw=1.4)
    ax.set_ylabel(r"$P(k)$ [$(h^{-1}\mathrm{Mpc})^3$]")
    ax.set_ylim(1e-1, 5e4)
    ax.legend(ncol=2)
    axr.axhline(1, color="0.4", lw=0.6)
    axr.set_yscale("log")
    axr.set_ylabel(r"$P_{\rm halofit}/P_{\rm lin}$")
    axr.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    savefig(fig, "chT1", "pk_lin_nl")

    # xi(r) with the BAO bump (linear, z=0)
    r = np.linspace(20, 180, 321)
    xi = xi_of_r(k, pk_lin[i0], r)
    rd_h = z["rdrag"] * z["h"]
    win = (r > 80) & (r < 130)
    r_peak = r[win][np.argmax((r ** 2 * xi)[win])]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.plot(r, r ** 2 * xi, color=SERIES[0])
    ax.axvline(rd_h, color=SERIES[2], ls=":", lw=1.2, label=r"$r_{\rm drag}$ from CAMB")
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xlabel(r"$r$ [$h^{-1}\mathrm{Mpc}$]")
    ax.set_ylabel(r"$r^2\xi(r)$ [$(h^{-1}\mathrm{Mpc})^2$]")
    ax.legend(loc="upper left")
    savefig(fig, "chT1", "xi_bao")

    # where does halofit exceed linear by 10%?  (z = 0)
    ratio = pk_nl[i0] / pk_lin[i0]
    k_nl = k[np.argmax(ratio > 1.1)]
    k_nl1 = k[np.argmax(pk_nl[i1] / pk_lin[i1] > 1.1)]   # same threshold at z = 1
    k_peak = k[np.argmax(pk_lin[i0])]                     # turnover of the linear P(k)
    s8_camb = float(z["sigma8"])
    s8_ours = tophat_sigma(k, pk_lin[i0], 8.0)
    save_numbers("chT1", "02_pk_xi", {
        "TOneSigmaEight": f"{s8_camb:.4f}",
        "TOneSigmaEightOurs": f"{s8_ours:.4f}",
        "TOneSigmaEightZone": f"{tophat_sigma(k, pk_lin[i1], 8.0):.4f}",
        "TOneRdrag": f"{z['rdrag']:.2f}",
        "TOneRdragH": f"{rd_h:.1f}",
        "TOneRpeak": f"{r_peak:.1f}",
        "TOneKnl": f"{k_nl:.2f}",
        "TOneKnlZone": f"{k_nl1:.2f}",
        "TOneKpeak": f"{k_peak:.3f}",
        "TOneRatioOne": f"{np.interp(1.0, k, ratio):.1f}",
    })


if __name__ == "__main__":
    main()
