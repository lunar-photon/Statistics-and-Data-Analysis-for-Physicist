"""Which combination of Omega_m and sigma_8 does weak lensing measure?

Question: the convergence spectrum C_ell^{kappa kappa} of a Stage-III-like galaxy
sample depends on the matter density Omega_m and the clustering amplitude sigma_8.
Do models with the same S_8 = sigma_8 (Omega_m/0.3)^0.5 give the same spectrum,
and models with a different S_8 a different one?  And is the exponent 0.5 special?

Computes (CAMB, halofit non-linear P(k), Limber):
  * C_ell^{kappa kappa} for five (Omega_m, sigma_8) points: three on the fiducial
    S_8 line and two off it (S_8 = 0.76 and 0.90 at the fiducial Omega_m);
  * the same spectrum at the fiducial point from our own Limber integral over
    CAMB's non-linear P(k), as a check of the library call;
  * the local degeneracy exponent alpha(ell) = (dlnC/dlnOmega_m)/(dlnC/dlnsigma_8),
    so that sigma_8 Omega_m^alpha is what C_ell holds fixed, for three source
    samples of different depth (median redshift about 0.5, 0.7 and 1.0).

Writes: data/chT1/kappa_s8_samples.npz (cache), figures/chT1/kappa_s8_spectra.pdf,
figures/chT1/kappa_s8_alpha.pdf, results/chT1/03_kappa_s8.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, trapezoid
from common import setup, savefig, save_numbers, theory_line, SERIES, DATA
from camb_fiducial import FIDUCIAL

LMAX = 3000
CACHE = DATA / "chT1" / "kappa_s8_samples.npz"
H = FIDUCIAL["H0"] / 100.0
C_KMS = 299792.458


def source_nz(z, z0=0.5):
    """Stage-III-like source redshift distribution p_z(z), normalised to 1."""
    nz = z**2 * np.exp(-(z / z0) ** 1.5)
    return nz / trapezoid(nz, z)


ZGRID = np.linspace(0.0, 3.5, 701)
Z0S = (0.35, 0.5, 0.7)            # shallow, fiducial (Stage-III-like), deep
NZS = [source_nz(ZGRID, z0) for z0 in Z0S]
NZ = NZS[1]


def camb_params(om, s8, windows=True, as_ref=FIDUCIAL["As"]):
    """CAMB parameters for flat LCDM at (Omega_m, sigma_8), other parameters fiducial.
    Omega_m includes the massive neutrino; sigma_8 is set by rescaling A_s
    (linear sigma_8 is proportional to sqrt(A_s))."""
    import camb
    from camb.sources import SplinedSourceWindow
    par = dict(FIDUCIAL)
    p0 = camb.set_params(**par)
    par["omch2"] = om * H**2 - par["ombh2"] - p0.omnuh2
    # first a cheap linear run to find sigma_8 at the reference A_s
    plin = camb.set_params(**{**par, "As": as_ref}, WantTransfer=True)
    plin.set_matter_power(redshifts=[0.0], kmax=2.0)
    s8_ref = camb.get_results(plin).get_sigma8_0()
    par["As"] = as_ref * (s8 / s8_ref) ** 2
    p = camb.set_params(**par, lmax=LMAX + 200, lens_potential_accuracy=1,
                        NonLinear=camb.model.NonLinear_both, halofit_version="takahashi")
    if windows:
        p.SourceWindows = [SplinedSourceWindow(z=ZGRID[1:], W=nz[1:], source_type="lensing")
                           for nz in NZS]
    return p, par


def kappa_spectra(om, s8):
    """Raw C_ell^{kappa kappa} (ell = 0..LMAX) of the three source samples, as rows."""
    import camb
    p, _ = camb_params(om, s8)
    cls = camb.get_results(p).get_source_cls_dict(raw_cl=True, lmax=LMAX)
    return np.array([cls[f"W{i}xW{i}"] for i in (1, 2, 3)])


def limber_by_hand(om, s8):
    """C_ell^{kappa kappa} = (9/4)(H0/c)^4 Om^2 int dchi [g/a]^2 P_delta(l/chi, chi)."""
    import camb
    p, _ = camb_params(om, s8, windows=False)
    res = camb.get_background(p)
    pk = camb.get_matter_power_interpolator(p, nonlinear=True, hubble_units=False,
                                            k_hunit=False, kmax=60.0, zmax=ZGRID[-1])
    z = ZGRID[1:]
    chi = res.comoving_radial_distance(z)                        # Mpc
    # lensing efficiency g(chi) = int_chi dchi' p(chi') (chi' - chi)/chi'  (flat), done in z
    g = np.array([trapezoid(NZ[1:][i:] * (1.0 - chi[i] / chi[i:]), z[i:]) for i in range(len(z))])
    dchi_dz = C_KMS / res.hubble_parameter(z)
    pref = 2.25 * (FIDUCIAL["H0"] / C_KMS) ** 4 * om**2
    ell = np.arange(2, LMAX + 1)
    cl = np.zeros(LMAX + 1)
    a = 1.0 / (1.0 + z)
    for l in ell:
        k = (l + 0.5) / chi
        ok = k < 60.0
        cl[l] = pref * trapezoid(((g / a) ** 2 * pk.P(z, k, grid=False) * dchi_dz)[ok], z[ok])
    return cl


# (Omega_m, sigma_8) points: fiducial, two along the S_8 line, two across it
OM_FID = 0.3153
S8_FID = 0.8111 * np.sqrt(OM_FID / 0.3)
POINTS = {
    "fid": (OM_FID, 0.8111),
    "along_lo": (0.25, S8_FID / np.sqrt(0.25 / 0.3)),
    "along_hi": (0.40, S8_FID / np.sqrt(0.40 / 0.3)),
    "across_lo": (OM_FID, 0.76 / np.sqrt(OM_FID / 0.3)),
    "across_hi": (OM_FID, 0.90 / np.sqrt(OM_FID / 0.3)),
}
STEP = 0.05   # +/- 5% in Omega_m and in sigma_8 for the log-derivatives


def compute():
    out = {}
    for name, (om, s8) in POINTS.items():
        out[f"gal_{name}"] = kappa_spectra(om, s8)
        print("done", name, om, s8)
    om, s8 = POINTS["fid"]
    for tag, (o, s) in dict(om_p=(om * (1 + STEP), s8), om_m=(om * (1 - STEP), s8),
                            s8_p=(om, s8 * (1 + STEP)), s8_m=(om, s8 * (1 - STEP))).items():
        out[f"gal_{tag}"] = kappa_spectra(o, s)
        print("done", tag)
    out["limber"] = limber_by_hand(om, s8)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def zmedian(nz):
    return ZGRID[np.searchsorted(cumulative_trapezoid(nz, ZGRID, initial=0), 0.5)]


def main():
    d = dict(np.load(CACHE)) if CACHE.exists() else compute()
    gal = {k[4:]: v for k, v in d.items() if k.startswith("gal_")}   # each: rows = 3 samples
    fid = {k: v[1] for k, v in gal.items()}                           # the Stage-III-like sample
    ell = np.arange(LMAX + 1)
    L = ell[2:]
    dl = L * (L + 1) / (2 * np.pi)
    setup()

    # ---- figure 1: the (Omega_m, sigma_8) plane, spectra and ratios --------------------
    fig = plt.figure(figsize=(9.0, 3.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[2, 1.2],
                          wspace=0.32, hspace=0.08)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    style = {"fid": (SERIES[0], "-", r"fiducial, $S_8=%.3f$" % S8_FID),
             "along_lo": (SERIES[1], "-", r"along: $\Omega_m=0.25$"),
             "along_hi": (SERIES[2], "-", r"along: $\Omega_m=0.40$"),
             "across_lo": (SERIES[3], "--", r"across: $S_8=0.76$"),
             "across_hi": (SERIES[4], "--", r"across: $S_8=0.90$")}
    om_grid = np.linspace(0.2, 0.45, 100)
    for s8line, ls in ((S8_FID, "-"), (0.76, ":"), (0.90, ":")):
        ax0.plot(om_grid, s8line / np.sqrt(om_grid / 0.3), color="0.55", ls=ls, lw=1.0)
        ax0.text(0.447, s8line / np.sqrt(0.447 / 0.3) + 0.02, r"$S_8=%.2f$" % s8line,
                 ha="right", va="bottom", fontsize=7, color="0.35", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    for name, (om, s8) in POINTS.items():
        ax0.plot(om, s8, "o", color=style[name][0], ms=6, zorder=5)
    ax0.set_xlabel(r"$\Omega_m$")
    ax0.set_ylabel(r"$\sigma_8$")
    ax0.set_xlim(0.2, 0.45)
    for name in POINTS:
        c, ls, lab = style[name]
        ax1.loglog(L, dl * fid[name][2:], color=c, ls=ls, label=lab)
        ax2.semilogx(L, fid[name][2:] / fid["fid"][2:], color=c, ls=ls)
    theory_line(ax1, L[::20], (dl * d["limber"][2:])[::20], label="our Limber integral")
    ax1.set_ylabel(r"$\ell(\ell+1)C_\ell^{\kappa\kappa}/2\pi$")
    ax1.legend(fontsize=7, loc="lower right", ncol=1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.axhline(1, color="0.4", lw=0.6)
    ax2.set_ylim(0.7, 1.35)
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel("ratio to fid.")
    ax2.set_xlim(10, LMAX)
    savefig(fig, "chT1", "kappa_s8_spectra")

    # ---- figure 2: degeneracy exponent alpha(ell) for three source depths ---------------------
    lnstep = np.log((1 + STEP) / (1 - STEP))
    a = np.log(gal["om_p"][:, 2:] / gal["om_m"][:, 2:]) / lnstep   # dlnC/dlnOm at fixed s8
    b = np.log(gal["s8_p"][:, 2:] / gal["s8_m"][:, 2:]) / lnstep   # dlnC/dlns8 at fixed Om
    al = a / b
    zmeds = [zmedian(nz) for nz in NZS]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for r, c in zip(range(3), (SERIES[1], SERIES[0], SERIES[2])):
        ax.semilogx(L, al[r], color=c, label=r"sources with $z_{\rm med}=%.2f$" % zmeds[r])
    ax.axhline(0.5, color="0.45", ls=":", lw=1.0)
    ax.text(LMAX * 0.95, 0.49, r"$S_8$ convention, $\alpha=0.5$", fontsize=8, color="0.35", ha="right", va="top")
    ax.set_xlim(10, LMAX)
    ax.set_ylim(0.2, 0.9)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\alpha(\ell)$")
    ax.legend(loc="upper right", fontsize=8)
    savefig(fig, "chT1", "kappa_s8_alpha")

    # ---- numbers quoted in the text ------------------------------------------------------
    band = (L >= 100) & (L <= 2000)
    rat = lambda n: fid[n][2:][band] / fid["fid"][2:][band] - 1  # noqa: E731
    i = lambda l: l - 2  # noqa: E731  index into L
    save_numbers("chT1", "03_kappa_s8", {
        "TOneSEightFid": f"{S8_FID:.3f}",
        "TOneZmed": f"{zmeds[1]:.2f}",
        "TOneZmedLo": f"{zmeds[0]:.2f}",
        "TOneZmedHi": f"{zmeds[2]:.2f}",
        "TOneSigEightLo": f"{POINTS['along_lo'][1]:.3f}",
        "TOneSigEightHi": f"{POINTS['along_hi'][1]:.3f}",
        "TOneAlongMax": f"{100 * max(np.max(np.abs(rat('along_lo'))), np.max(np.abs(rat('along_hi')))):.0f}",
        "TOneAlongMedLo": f"{100 * np.median(rat('along_lo')):+.0f}",
        "TOneAlongMedHi": f"{100 * np.median(rat('along_hi')):+.0f}",
        "TOneAcrossLo": f"{100 * np.median(rat('across_lo')):.0f}",
        "TOneAcrossHi": f"{100 * np.median(rat('across_hi')):+.0f}",
        "TOneLimberErr": f"{100 * np.median(np.abs(d['limber'][2:][band] / fid['fid'][2:][band] - 1)):.1f}",
        "TOneDlnCdlnSEight": f"{b[1, i(500)]:.1f}",
        "TOneDlnCdlnOm": f"{a[1, i(500)]:.1f}",
        "TOneAlphaGalTwo": f"{al[1, i(200)]:.2f}",
        "TOneAlphaGalTwoK": f"{al[1, i(2000)]:.2f}",
        "TOneAlphaLoFive": f"{al[0, i(500)]:.2f}",
        "TOneAlphaFive": f"{al[1, i(500)]:.2f}",
        "TOneAlphaHiFive": f"{al[2, i(500)]:.2f}",
    })


if __name__ == "__main__":
    main()
