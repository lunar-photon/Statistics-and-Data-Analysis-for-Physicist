"""01_excursion_mfs.py -- the three Minkowski functionals of a Gaussian CMB temperature patch.

Question: does the morphology of a simulated Gaussian CMB sky (area, boundary length and Euler
characteristic of its hot regions, as functions of the threshold) follow the Gaussian formulas,
which depend on the power spectrum only through sigma_0 and sigma_1?

Computes: 300 flat-sky patches (12.8 x 12.8 deg, 512^2 pixels of 1.5 arcmin) from the fiducial
lensed C_ell^TT smoothed by a 10 arcmin beam; the functionals at 41 thresholds; the Gaussian
predictions from the spectral moments; a cross-check of our Euler counter against scikit-image;
the expected number of hot spots on the full sky.
Writes: figures/chT3/patch_excursion.pdf, figures/chT3/mf_gaussian.pdf, results/chT3/01_excursion_mfs.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from camb_fiducial import load_fiducial
from lib_t3 import ARCMIN, beam, gaussian_flat, moments_flat, moments_sphere, minkowski, mf_gauss, euler_torus

setup()
rng = rng_for("chT3", "01_excursion_mfs")
N, dx, FWHM, NSIM = 512, 1.5 * ARCMIN, 10.0, 300
ell, TT = load_fiducial("TT")
cl = TT * beam(ell, FWHM) ** 2                      # what a 10' beam leaves of the sky

# ---- the two numbers the Gaussian formulas need: sigma_0^2 = <T^2>, sigma_1^2 = <|grad T|^2>
s0sq, s1sq = moments_flat(cl, N, dx)
s0, s1 = np.sqrt(s0sq), np.sqrt(s1sq)
S0sq, S1sq = moments_sphere(cl)                     # the same sums over the whole sky
nus = np.linspace(-4, 4, 41)

# ---- Monte Carlo: maps in units of sigma_0, functionals at every threshold
mf = np.array([minkowski(gaussian_flat(cl, N, dx, rng) / s0, nus, dx) for _ in range(NSIM)])
mean, std = mf.mean(0), mf.std(0, ddof=1)
th = mf_gauss(nus, 1.0, s1 / s0)                    # map is in sigma units: s0 -> 1

# ---- cross-check of the Euler counter with scikit-image on one map (non-periodic, so we
#      compare on a version of the map whose excursion set does not touch the edges)
from skimage.measure import euler_number
u = gaussian_flat(cl, N, dx, rng) / s0
pad = np.zeros((N + 2, N + 2), bool)
pad[1:-1, 1:-1] = u > 1.0
ours, sk = euler_torus(pad), euler_number(pad, connectivity=2)

# ---- full-sky expectation of chi at nu = 2 (Gaussian kinematic formula on the sphere)
nu2 = 2.0
rho2 = (S1sq / (2 * S0sq)) / (2 * np.pi) ** 1.5 * nu2 * np.exp(-nu2 ** 2 / 2)
chi_sky = 4 * np.pi * rho2 + 2 * 0.5 * erfc(nu2 / np.sqrt(2))

# ---- figure 1: one patch and two excursion sets
fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.2), sharey=True)
ext = [0, N * dx / ARCMIN / 60, 0, N * dx / ARCMIN / 60]
ax[0].imshow(u.T, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, extent=ext)
ax[0].set_title(r"$T/\sigma_0$")
for a, n, col in [(ax[1], 1.0, SERIES[1]), (ax[2], -1.0, SERIES[0])]:
    a.imshow((u > n).T, origin="lower", cmap="Greys", vmin=0, vmax=1.6, extent=ext)
    a.set_title(rf"excursion set $T/\sigma_0>{n:+.0f}$: $\chi={euler_torus(u > n)}$")
for a in ax:
    a.grid(False); a.set_xlabel("deg")
ax[0].set_ylabel("deg")
savefig(fig, "chT3", "patch_excursion")

# ---- figure 2: the three functionals, simulation mean +- 1 sigma vs the Gaussian formulas
fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
labels = [r"$v_0$: area fraction", r"$v_1$ [arcmin$^{-1}$]", r"$v_2$ [deg$^{-2}$]"]
scale = [1.0, ARCMIN, (np.pi / 180) ** 2]
for k in range(3):
    ax[k].fill_between(nus, (mean[k] - std[k]) * scale[k], (mean[k] + std[k]) * scale[k],
                       color=SERIES[0], alpha=0.3, lw=0, label=r"300 sims, $\pm1\sigma$")
    ax[k].plot(nus, mean[k] * scale[k], color=SERIES[0], label="mean of sims")
    theory_line(ax[k], nus, th[k] * scale[k], label="Gaussian formula")
    ax[k].set_xlabel(r"threshold $\nu$"); ax[k].set_title(labels[k])
ax[0].legend(loc="lower left")
savefig(fig, "chT3", "mf_gaussian")

# ---- numbers for the text
dev = np.max(np.abs(mean[2] - th[2])) / np.max(np.abs(th[2]))
err = np.max(std[2] / np.sqrt(NSIM)) / np.max(np.abs(th[2]))
area_deg = (N * dx * 180 / np.pi) ** 2
save_numbers("chT3", "01_excursion_mfs", {
    "TaNsimMF": NSIM, "TaPatchDeg": f"{N * dx * 180 / np.pi:.1f}", "TaPixArcmin": "1.5",
    "TaSigZero": f"{s0:.1f}", "TaSigZeroSky": f"{np.sqrt(S0sq):.1f}",
    "TaThetaC": f"{s0 / s1 / ARCMIN:.1f}", "TaThetaCSky": f"{np.sqrt(S0sq / S1sq) / ARCMIN:.1f}",
    "TaVzeroOne": f"{th[0][nus == 1.0][0]:.4f}", "TaVzeroOneSim": f"{mean[0][nus == 1.0][0]:.4f}",
    "TaChiDevPct": f"{100 * dev:.1f}", "TaChiErrPct": f"{100 * err:.1f}",
    "TaChiPeakPatch": f"{np.max(th[2]) * N ** 2 * dx ** 2:.0f}",
    "TaChiOne": int(euler_torus(u > 1.0)), "TaChiMinusOne": int(euler_torus(u > -1.0)),
    "TaEulerOurs": int(ours), "TaEulerSkimage": int(sk),
    "TaHotSpotsSky": f"{chi_sky:.0f}", "TaPatchArea": f"{area_deg:.0f}",
})
print(f"sigma0={s0:.1f} (sky {np.sqrt(S0sq):.1f}) muK, theta_c={s0/s1/ARCMIN:.2f}', chi dev {100*dev:.2f}% "
      f"(MC err {100*err:.2f}%), euler ours={ours} skimage={sk}, chi_sky(nu=2)={chi_sky:.0f}")
