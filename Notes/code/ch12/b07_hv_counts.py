"""b07_hv_counts.py -- how many polarization singularities a Gaussian sky has, as a function of l_max.

Question: Huterer and Vachaspati (2005) counted about 6000 singularities on the full sky for
maps with power up to l_max = 100 and about 150 000 for l_max = 500, "roughly as l_max^2".
Does the density formula n = sigma_1^2 / (4 pi sigma_0^2) reproduce these numbers, with
their cosmology and with ours, and do direct counts on simulated maps agree with the formula?
Computes: C_l^EE from CAMB for their model (flat LCDM, Omega_m = 0.3, Omega_m h^2 = 0.127,
Omega_b h^2 = 0.021, n_s = 1, no tensors, unlensed) and for the book's fiducial model;
N_sky(l_max) = sum (2l+1) l(l+1) C_l / sum (2l+1) C_l; direct counts on periodic flat patches
(E modes only, sharp cut at l_max), scaled to the full sky; the power-law slope of N_sky(l_max).
Writes: data/ch12/b_hv_spectra.npz, figures/ch12/b_hv_counts.pdf, results/ch12/b07_hv_counts.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA, theory_line
from camb_fiducial import load_fiducial
from lib_persist import flat_qu, winding

rng = rng_for("ch12", "b07_hv_counts")
CACHE = DATA / "ch12" / "b_hv_spectra.npz"
LMAX = 2500


def hv_spectrum():
    """Unlensed EE for the Huterer-Vachaspati model.  Their optical depth is not stated, and it
    only shapes l < 20 (the reionization bump), so we compute two versions: tau = 0.06 and no
    reionization at all."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["EE"], z["EE_noreion"]
    import camb
    h = np.sqrt(0.127 / 0.3)
    out = []
    for reion in (True, False):
        p = camb.set_params(H0=100 * h, ombh2=0.021, omch2=0.127 - 0.021, ns=1.0, As=2.1e-9,
                            tau=0.06, mnu=0.0, lmax=LMAX + 200, WantTensors=False)
        p.Reion.Reionization = reion
        r = camb.get_results(p)
        cl = r.get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=True)["unlensed_scalar"]
        out.append(cl[: LMAX + 1, 1])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ell=np.arange(LMAX + 1), EE=out[0], EE_noreion=out[1], h=h)
    return out[0], out[1]


def n_sky(cl, lmax):
    l = np.arange(2, lmax + 1)
    w = (2 * l + 1) * cl[2: lmax + 1]
    return (w * l * (l + 1)).sum() / w.sum()


ee_hv, ee_hv0 = hv_spectrum()
_, ee_fid = load_fiducial("EE")
lmaxes = np.array([50, 100, 150, 200, 300, 400, 500])
N_hv = np.array([n_sky(ee_hv, L) for L in lmaxes])
N_fid = np.array([n_sky(ee_fid, L) for L in lmaxes])
N_hv0 = np.array([n_sky(ee_hv0, L) for L in lmaxes])
print("N_sky (HV, no reionization):", np.round(N_hv0))
slope = np.polyfit(np.log(lmaxes[1:]), np.log(N_hv[1:]), 1)[0]
print("N_sky (HV cosmology):", np.round(N_hv))
print("N_sky (fiducial)    :", np.round(N_fid))
print("slope dlnN/dln lmax (100-500):", slope)

# ---------------------------------------------------------------- direct counts on flat patches
SIDE, NSIM = 30.0, 12
counts, errs, ratio_grid = [], [], []
for L in lmaxes:
    npix = int(2 ** np.ceil(np.log2(SIDE * L / 45.0)))      # >= ~4 pixels per half-wavelength
    c = []
    for s in range(NSIM):
        Q, U, ell2 = flat_qu(ee_hv, npix, SIDE, rng, lmax=L)
        c.append(np.count_nonzero(winding(Q, U)))
    area = np.deg2rad(SIDE) ** 2
    c = np.array(c) / area * 4 * np.pi                        # scaled to the full sky
    counts.append(c.mean())
    errs.append(c.std() / np.sqrt(NSIM))
    ratio_grid.append(c.mean() / (ell2 / (4 * np.pi) * 4 * np.pi))
    print(f"l_max={L}: N={npix}, flat count {c.mean():.0f} +- {c.std()/np.sqrt(NSIM):.0f}, "
          f"grid formula {ell2:.0f}")
counts, errs, ratio_grid = map(np.array, (counts, errs, ratio_grid))

setup(4.6, 3.3)
fig, ax = plt.subplots()
theory_line(ax, lmaxes, N_hv, label=r"$\langle\ell(\ell+1)\rangle_E$, their cosmology")
ax.plot(lmaxes, N_hv0, color=SERIES[3], lw=1.2, label=r"their cosmology, no reionization")
ax.plot(lmaxes, N_fid, color=SERIES[2], lw=1.2, label=r"fiducial cosmology")
ax.errorbar(lmaxes, counts, errs, fmt="o", ms=4, color=SERIES[0], label="counts on simulated patches")
ax.plot([100, 500], [6000, 150000], "s", ms=7, mfc="none", mec=SERIES[1], mew=1.5,
        label="Huterer & Vachaspati (2005)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(lmaxes)
ax.set_xticklabels([str(int(v)) for v in lmaxes])
ax.minorticks_off()
ax.set_xlabel(r"$\ell_{\max}$")
ax.set_ylabel("singularities on the full sky")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch12", "b_hv_counts")

i100, i500 = list(lmaxes).index(100), list(lmaxes).index(500)
save_numbers("ch12", "b07_hv_counts", {
    "PbHvh": float(np.sqrt(0.127 / 0.3)),
    "PbHvFormHundred": int(round(N_hv[i100])), "PbHvFormFiveHundred": int(round(N_hv[i500])),
    "PbHvNoreionHundred": int(round(N_hv0[i100])), "PbHvNoreionFiveHundred": int(round(N_hv0[i500])),
    "PbHvNoreionFifty": int(round(N_hv0[0])), "PbHvFormFifty": int(round(N_hv[0])),
    "PbFidFormHundred": int(round(N_fid[i100])), "PbFidFormFiveHundred": int(round(N_fid[i500])),
    "PbHvSimHundred": int(round(counts[i100])), "PbHvSimFiveHundred": int(round(counts[i500])),
    "PbHvSimErrHundred": int(round(errs[i100])), "PbHvSimErrFiveHundred": int(round(errs[i500])),
    "PbHvRatioHundred": 6000 / N_hv[i100], "PbHvRatioFiveHundred": 150000 / N_hv[i500],
    "PbHvSlope": slope, "PbHvSide": SIDE, "PbHvNsim": NSIM,
    "PbHvGridDev": 100 * np.abs(ratio_grid - 1).max(),
    "PbHvGridPull": f"{(np.abs(ratio_grid - 1) / (errs / counts)).max():.1f}",   # same, in Monte Carlo errors
})
