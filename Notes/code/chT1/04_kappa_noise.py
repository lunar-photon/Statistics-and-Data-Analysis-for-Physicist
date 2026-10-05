"""Signal and shape noise for a Stage-III-like cosmic-shear survey.

Question: a galaxy's intrinsic ellipticity (rms 0.27 per component) is some
thirty times larger than the shear it carries.  On which angular scales does the
lensing signal C_ell^{kappa kappa} still beat the shape noise sigma_e^2/n, how
does that change from the nearest to the farthest photometric-redshift bin, and
how well could such a survey measure the amplitude sigma_8 if nothing else were
uncertain?

Computes:
  * five photometric-redshift bins (equal galaxy numbers, Gaussian photo-z scatter
    0.05(1+z)) cut from the source distribution of 03_kappa_s8.py, and their
    lensing weights q_i(z), next to the CMB-lensing weight;
  * all auto and cross spectra C_ell^{ij} from CAMB (halofit, Limber), the noise
    N_i = sigma_e^2 / n_i, and Gaussian error bars for a 1000 deg^2 survey;
  * the multipole where signal = noise in each bin, the total signal-to-noise for
    ell <= 2000, and the Fisher error on ln sigma_8 with every other parameter fixed.

Writes: data/chT1/kappa_noise_bins.npz (cache), figures/chT1/kappa_bins.pdf,
figures/chT1/kappa_noise.pdf, results/chT1/04_kappa_noise.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.special import ndtr
from common import setup, savefig, save_numbers, SERIES, DATA
from camb_fiducial import FIDUCIAL

LMAX = 3000
CACHE = DATA / "chT1" / "kappa_noise_bins.npz"
NBIN = 5
SIGMA_E = 0.27            # intrinsic ellipticity rms per component (KiDS-1000: 0.25-0.27)
N_EFF = 6.2               # galaxies per arcmin^2, all bins together (KiDS-1000: 6.2)
AREA = 1000.0             # deg^2
F_SKY = AREA / (4 * np.pi * (180 / np.pi) ** 2)
ARCMIN2_PER_SR = (180 * 60 / np.pi) ** 2
STEP = 0.02               # +/- 2% in sigma_8 for the Fisher derivative

Z = np.linspace(0.0, 3.5, 701)


def source_nz(z, z0=0.5):
    nz = z**2 * np.exp(-(z / z0) ** 1.5)
    return nz / trapezoid(nz, z)


def tomographic_bins(sig0=0.05):
    """n_i(z): true-redshift distribution of the galaxies whose photo-z falls in bin i."""
    nz = source_nz(Z)
    zp = np.linspace(0.0, 3.5, 1401)
    sig = sig0 * (1 + Z)
    # distribution of photometric redshifts, to place edges with equal numbers per bin
    pzp = trapezoid(nz[None, :] * np.exp(-0.5 * ((zp[:, None] - Z[None, :]) / sig) ** 2)
                    / (np.sqrt(2 * np.pi) * sig), Z, axis=1)
    cdf = cumulative_trapezoid(pzp, zp, initial=0) / trapezoid(pzp, zp)
    edges = np.interp(np.linspace(0, 1, NBIN + 1), cdf, zp)
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.array([nz * (ndtr((edges[i + 1] - Z) / sig) - ndtr((edges[i] - Z) / sig))
                     for i in range(NBIN)])
    return nz, bins, edges[1:-1]


def spectra(nz_list, s8_factor=1.0):
    """Raw C_ell^{ij} (shape nwin x nwin x LMAX+1) from CAMB lensing windows."""
    import camb
    from camb.sources import SplinedSourceWindow
    par = dict(FIDUCIAL)
    par["As"] = FIDUCIAL["As"] * s8_factor**2          # linear sigma_8 is proportional to sqrt(A_s)
    p = camb.set_params(**par, lmax=LMAX + 200, lens_potential_accuracy=1,
                        NonLinear=camb.model.NonLinear_both, halofit_version="takahashi")
    p.SourceWindows = [SplinedSourceWindow(z=Z[1:], W=w[1:], source_type="lensing") for w in nz_list]
    r = camb.get_results(p)
    cls = r.get_source_cls_dict(raw_cl=True, lmax=LMAX)
    n = len(nz_list)
    out = np.zeros((n, n, LMAX + 1))
    for i in range(n):
        for j in range(n):
            out[i, j] = cls[f"W{i + 1}xW{j + 1}"]
    chi = r.comoving_radial_distance(Z)
    zc = np.linspace(0.0, 8.0, 801)                         # for the CMB weight
    chi_c = r.comoving_radial_distance(zc)
    chi_star = r.comoving_radial_distance(r.get_derived_params()["zstar"])
    return out, chi, np.array([zc, chi_c]), chi_star


def compute():
    nz, bins, edges = tomographic_bins()
    wins = list(bins) + [nz]                                   # 5 bins + the full sample
    c0, chi, cmbz, chi_star = spectra(wins)
    cp = spectra(list(bins), 1 + STEP)[0]
    cm = spectra(list(bins), 1 - STEP)[0]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, nz=nz, bins=bins, edges=edges, c0=c0, cp=cp, cm=cm, chi=chi, cmbz=cmbz,
             chi_star=chi_star)
    return dict(np.load(CACHE))


def lensing_weight(p_z, chi):
    """Weight per unit z: q(chi) dchi/dz, with q proportional to g(chi) chi / a and
    g(chi) = int p(chi') (chi'-chi)/chi' dchi' (flat)."""
    p_z = p_z / trapezoid(p_z, Z)
    g = np.array([trapezoid(p_z[k:] * (1 - chi[k] / np.maximum(chi[k:], 1e-9)), Z[k:])
                  for k in range(len(Z))])
    return g * chi * (1 + Z) * np.gradient(chi, Z)


def main():
    d = dict(np.load(CACHE)) if CACHE.exists() else compute()
    bins, chi, c0 = d["bins"], d["chi"], d["c0"]
    frac = trapezoid(bins, Z, axis=1)                          # fraction of galaxies per bin
    n_i = N_EFF * frac * ARCMIN2_PER_SR                        # per steradian
    noise = SIGMA_E**2 / n_i
    ell = np.arange(LMAX + 1)
    setup()

    # ---- figure 1: the bins and their lensing weights ----------------------------------------
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.0, 4.6), sharex=True)
    for i in range(NBIN):
        ax0.plot(Z, bins[i], color=SERIES[i], label=f"bin {i + 1}")
        q = lensing_weight(bins[i], chi)
        ax1.plot(Z, q / q.max(), color=SERIES[i])
    ax0.plot(Z, d["nz"] / NBIN, color="0.4", ls="--", lw=1.0, label="all / 5")
    zc, chic = d["cmbz"]
    qc = chic * (d["chi_star"] - chic) / d["chi_star"] * (1 + zc) * np.gradient(chic, zc)  # CMB
    ax1.plot(zc, qc / qc.max(), color="0.3", ls="--", lw=1.2, label="CMB lensing")
    ax0.set_ylabel(r"$p_i(z)$")
    ax0.legend(ncol=3, fontsize=8)
    ax1.set_ylabel("lensing weight per unit $z$\n(peak = 1)")
    ax1.set_xlabel(r"$z$")
    ax1.set_xlim(0, 6.0)
    ax1.legend(loc="upper right", fontsize=8)
    savefig(fig, "chT1", "kappa_bins")

    # ---- figure 2: signal versus noise ---------------------------------------------------------
    edges = np.unique(np.geomspace(20, LMAX, 19).astype(int))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    L = ell[2:]
    dl = L * (L + 1) / (2 * np.pi)
    for i, c in zip((0, 2, 4), (SERIES[0], SERIES[2], SERIES[4])):
        cl = c0[i, i]
        ax.loglog(L, dl * cl[2:], color=c, label=f"bin {i + 1}")
        lc, err, mid = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sl = slice(lo, hi)
            m = np.mean(cl[sl])
            nmodes = np.sum(2 * ell[sl] + 1) * F_SKY
            lc.append(m)
            err.append(np.sqrt(2.0 / nmodes) * (m + noise[i]))   # Gaussian band-power error
            mid.append(np.sqrt(lo * hi))
        mid, lc, err = map(np.array, (mid, lc, err))
        f = mid * (mid + 1) / (2 * np.pi)
        ax.errorbar(mid, f * lc, yerr=f * err, fmt="o", ms=3, color=c, capsize=0, lw=1)
    ax.loglog(L, dl * noise[0], color="0.3", ls="--", lw=1.2,
              label=r"noise $\sigma_e^2/n_i$")             # equal galaxy numbers: same in every bin
    ax.set_xlim(15, LMAX)
    ax.set_ylim(1e-7, 3e-3)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    savefig(fig, "chT1", "kappa_noise")

    # ---- numbers --------------------------------------------------------------------------------
    def ell_equal(i):
        return int(ell[2:][np.argmax(c0[i, i][2:] < noise[i])])

    lmax_sn = 2000
    sl = slice(10, lmax_sn + 1)
    cmat = c0[:NBIN, :NBIN, sl] + np.eye(NBIN)[:, :, None] * noise[:, None, None]
    dcl = (d["cp"] - d["cm"])[:, :, sl] / np.log((1 + STEP) / (1 - STEP))   # dC/dln sigma_8
    sn2, fish = 0.0, 0.0
    for k, l in enumerate(ell[sl]):
        ci = np.linalg.inv(cmat[:, :, k])
        s = c0[:NBIN, :NBIN, sl][:, :, k]
        w = F_SKY * (2 * l + 1) / 2
        sn2 += w * np.trace(ci @ s @ ci @ s)
        fish += w * np.trace(ci @ dcl[:, :, k] @ ci @ dcl[:, :, k])
    # single (non-tomographic) sample: the sixth window, noise sigma_e^2 / n_total
    call, nall = c0[NBIN, NBIN, sl], SIGMA_E**2 / (N_EFF * ARCMIN2_PER_SR)
    sn2_one = np.sum(F_SKY * (2 * ell[sl] + 1) / 2 * (call / (call + nall)) ** 2)
    save_numbers("chT1", "04_kappa_noise", {
        "TOneNoiseOne": float(noise[0]),
        "TOneFsky": f"{F_SKY:.4f}",
        "TOneEdgeOne": f"{d['edges'][0]:.2f}",
        "TOneEdgeFour": f"{d['edges'][-1]:.2f}",
        "TOneLeqOne": ell_equal(0),
        "TOneLeqThree": ell_equal(2),
        "TOneLeqFive": ell_equal(4),
        "TOneSNtomo": f"{np.sqrt(sn2):.0f}",
        "TOneSNone": f"{np.sqrt(sn2_one):.0f}",
        "TOneSigLnSEight": f"{100 / np.sqrt(fish):.1f}",
        "TOneLmaxSN": lmax_sn,
        "TOneSigmaE": SIGMA_E,
        "TOneNeff": N_EFF,
        "TOneArea": int(AREA),
    })


if __name__ == "__main__":
    main()
