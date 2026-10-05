"""07_planck_mf.py -- the Gaussian-simulation Minkowski-functional curves of Planck 2018 VII, Fig. 10.

Question: Planck compares the area, perimeter and genus curves of the real temperature map
with those of Gaussian simulations (their Sect. 5.3, eqs. 13-19, Fig. 10).  At their
N_side = 256, 40 arcmin resolution, what do Gaussian skies with the fiducial spectrum give,
in which normalisation are the published curves drawn, and what does the combined chi^2 of
the 36 numbers (3 functionals x 12 thresholds) look like for a Gaussian sky?

Computes: NSIM full-sky Gaussian maps (fiducial C_l, 40' beam, pixel window) at nside 256;
          per map: v0, V1 = length/(4 area), V2 = chi density (sphere curvature removed), at
          Planck's 12 thresholds and on a fine grid; the normalisation V_k / (sigma1/sigma0)^k
          with sigma0, sigma1 measured on the map; the leave-one-out chi^2 of each map
          against the others (Hartlap-corrected); 40 maps at nside 512 for the pixel check.
Writes:   figures/ch12/planck_mf.pdf, results/ch12/07_planck_mf.tex, data/ch12/planck_mf.npz
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import erfc
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES, DATA
from camb_fiducial import load_fiducial
from lib_fields import gaussian_field_sphere
from lib_sphere import mesh, chi_sphere

setup()
rng = rng_for("ch12", "07_planck_mf")
NSIDE, FWHM, NSIM = 256, 40.0, 300
LMAX = 3 * NSIDE - 1
NU12 = np.linspace(-3, 3, 12)                          # Planck: 12 thresholds between -3 and 3


def spectrum(nside, lmax):
    ell, cl = load_fiducial("TT")                      # C_l in microK^2
    cl = cl[:lmax + 1].copy()
    cl[:2] = 0.0
    b = hp.gauss_beam(np.radians(FWHM / 60), lmax) * hp.pixwin(nside)[:lmax + 1]
    return cl * b ** 2


def measure(alm, nside, nus, msh, lmax, dnu=0.1):
    """V0, V1/(s1/s0), V2/(s1/s0)^2 per steradian, and s1/s0 of the map (rad^-1)."""
    t, dth, dph = hp.alm2map_der1(alm, nside, lmax=lmax)
    t = t - t.mean()
    s0 = t.std()
    g = np.hypot(dth, dph)
    r = np.sqrt((g ** 2).mean()) / s0                  # sigma1 / sigma0, measured on this map
    u, gu = t / s0, g / s0
    v0 = np.array([(u > nu).mean() for nu in nus])
    v1 = np.array([0.25 * (gu * (np.abs(u - nu) < dnu / 2)).mean() / dnu for nu in nus])
    chi = np.array([chi_sphere(u > nu, msh) for nu in nus])
    v2 = (chi - 2 * v0) / (4 * np.pi)                  # remove the sphere's own curvature term
    return v0, v1 / r, v2 / r ** 2, r


out = DATA / "ch12" / "planck_mf.npz"
nus = np.round(np.linspace(-3.5, 3.5, 29), 4)
if out.exists():
    z = dict(np.load(out))
else:
    cl = spectrum(NSIDE, LMAX)
    msh = mesh(NSIDE)
    z = {k: np.zeros((NSIM, len(nus))) for k in ("v0", "v1", "v2")}
    z.update({k + "_12": np.zeros((NSIM, 12)) for k in ("v0", "v1", "v2")})
    z["ratio"] = np.zeros(NSIM)
    for s in range(NSIM):
        _, alm = gaussian_field_sphere(cl, NSIDE, rng, lmax=LMAX)
        a = measure(alm, NSIDE, np.concatenate([nus, NU12]), msh, LMAX)
        for k, arr in zip(("v0", "v1", "v2"), a[:3]):
            z[k][s], z[k + "_12"][s] = arr[:len(nus)], arr[len(nus):]
        z["ratio"][s] = a[3]
        if s % 50 == 0:
            print("maps", s, flush=True)
    # pixel check: the same band-limited skies sampled on 4x smaller pixels
    cl5 = spectrum(NSIDE, LMAX)                        # the same band-limited sky
    msh5 = mesh(512)
    z["v2_512"] = np.zeros((40, 12))
    z["v2_256b"] = np.zeros((40, 12))
    for s in range(40):
        _, alm = gaussian_field_sphere(cl5, 512, rng, lmax=LMAX)
        z["v2_512"][s] = measure(alm, 512, NU12, msh5, LMAX)[2]
        z["v2_256b"][s] = measure(alm, NSIDE, NU12, msh, LMAX)[2]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **z)

# ---------------------------------------------------------------- the analytic normalised curves
e = lambda x: np.exp(-np.asarray(x) ** 2 / 2)
th = {"v0": lambda x: 0.5 * erfc(np.asarray(x) / np.sqrt(2)),
      "v1": lambda x: e(x) / (8 * np.sqrt(2)),                      # sqrt(lam)/8 with sigma1/sigma0 = 1
      "v2": lambda x: np.asarray(x) * e(x) / (2 * (2 * np.pi) ** 1.5)}   # lam/(2pi)^{3/2} with sigma1/sigma0 = 1
# values read off Planck 2018 VII Fig. 10, N_side = 256 row (reading accuracy ~0.002)
READ = {"v1": ([-0.273, 0.273], [0.081, 0.081]), "v2": ([-0.818, 0.818], [-0.020, 0.021])}

# ---------------------------------------------------------------- combined chi^2 (their eqs. 20-22)
Y = np.concatenate([z["v0_12"], z["v1_12"], z["v2_12"]], axis=1)       # NSIM x 36
n, p = Y.shape
chi2 = np.zeros(n)
for i in range(n):                                     # each map against the other n - 1 maps
    rest = np.delete(Y, i, axis=0)
    C = np.cov(rest, rowvar=False)
    h = (n - 1 - p - 2) / (n - 2)                      # Hartlap factor for n - 1 simulations
    d = Y[i] - rest.mean(0)
    chi2[i] = h * d @ np.linalg.solve(C, d)

i0 = int(np.argmin(np.abs(NU12 - 0.273)))
i8 = int(np.argmin(np.abs(NU12 - 0.818)))
r_th = np.sqrt(np.sum((2 * np.arange(LMAX + 1) + 1) * spectrum(NSIDE, LMAX) * np.arange(LMAX + 1) * (np.arange(LMAX + 1) + 1))
               / np.sum((2 * np.arange(LMAX + 1) + 1) * spectrum(NSIDE, LMAX)))
nums = {"PlNside": NSIDE, "PlFwhm": int(FWHM), "PlNsim": NSIM, "PlLmax": LMAX,
        "PlRatioTh": round(float(r_th), 1), "PlRatioMC": round(float(z["ratio"].mean()), 1),
        "PlRatioSd": round(float(z["ratio"].std(ddof=1)), 1),
        "PlThetaC": round(float(np.degrees(1 / r_th) * 60), 1),
        "PlPixArcmin": round(float(np.degrees(hp.nside2resol(NSIDE)) * 60), 1),
        "PlVonePeakTh": round(float(1 / (8 * np.sqrt(2))), 4),
        "PlVoneTh": round(float(th["v1"](NU12[i0])), 4), "PlVoneMC": round(float(z["v1_12"][:, i0].mean()), 4),
        "PlVoneSd": round(float(z["v1_12"][:, i0].std(ddof=1)), 4), "PlVoneRead": 0.081,
        "PlVtwoTh": round(float(th["v2"](NU12[i8])), 4), "PlVtwoMC": round(float(z["v2_12"][:, i8].mean()), 4),
        "PlVtwoSd": round(float(z["v2_12"][:, i8].std(ddof=1)), 4), "PlVtwoRead": 0.021,
        "PlVtwoTwoFiveSix": round(float(z["v2_256b"][:, i8].mean()), 4),
        "PlVtwoFiveOneTwo": round(float(z["v2_512"][:, i8].mean()), 4),
        "PlPixDef": round(float(100 * (1 - z["v2_256b"][:, i8].mean() / z["v2_512"][:, i8].mean())), 1),
        "PlChiMean": round(float(chi2.mean() / p), 2), "PlChiMed": round(float(np.median(chi2) / p), 2),
        "PlNdof": p, "PlHartlap": round((n - 1 - p - 2) / (n - 2), 3)}
# full-sky chi at the median: the formula gives 2 Psi(0) = 1; pixel counting shifts the curve
iz = int(np.argmin(np.abs(nus)))
chi0 = z["v2"][:, iz] * z["ratio"] ** 2 * 4 * np.pi + 2 * z["v0"][:, iz]
slope = 4 * np.pi * (r_th ** 2 / 2) / (2 * np.pi) ** 1.5           # d E[chi]/d nu at nu = 0
nums.update({"PlChiZero": round(float(chi0.mean()), 0), "PlChiZeroSe": round(float(chi0.std(ddof=1) / np.sqrt(n)), 0),
             "PlChiSlope": round(float(slope), 0), "PlShift": round(float(-chi0.mean() / slope), 3)})
save_numbers("ch12", "07_planck_mf", nums)
for k, v in nums.items():
    print(k, v)

# ---------------------------------------------------------------- figure
xx = np.linspace(-3.5, 3.5, 300)
fig, axs = plt.subplots(1, 4, figsize=(11.0, 2.9))
for ax, k, lab in zip(axs[:3], ("v0", "v1", "v2"), ("Area", "Perimeter length", "Genus")):
    D = z[k]
    lo, hi = np.percentile(D, [0.5, 99.5], axis=0)
    ax.fill_between(nus, lo, hi, color="0.8", lw=0, label="99% of our Gaussian skies")
    ax.plot(nus, D.mean(0), color=SERIES[0], lw=1.4, label=f"mean of {NSIM} skies")
    theory_line(ax, xx, th[k](xx), label=r"$A_k v_k$ with $\sigma_1/\sigma_0=1$")
    if k in READ:
        ax.plot(*READ[k], "*", ms=9, color=SERIES[3], label="read off Planck Fig. 10")
    ax.set_title(lab, fontsize=9)
    ax.set_xlabel(r"$\nu$")
axs[0].legend(fontsize=6.5, loc="lower left")
axs[2].legend(fontsize=6.5, loc="upper left")
ax = axs[3]
ax.hist(chi2 / p, bins=np.linspace(0, 2.5, 21), color=SERIES[0], alpha=0.6, label="each sky vs the rest")
from scipy.stats import chi2 as chi2dist
g = np.linspace(0.01, 2.5, 200)
ax.plot(g, chi2dist.pdf(g * p, p) * p * n * (2.5 / 20), color="k", lw=1, label=r"$\chi^2_{36}$")
ax.set_xlabel(r"$\chi^2/N_{\rm dof}$")
ax.set_title("Combined", fontsize=9)
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.legend(fontsize=6.5, loc="upper right")
fig.tight_layout()
savefig(fig, "ch12", "planck_mf")
