"""06_sphere_sg98.py -- the Gaussian kinematic formula on the sphere; Schmalzing & Gorski (1998), Fig. 2.

Question: Schmalzing & Gorski (SG98) predicted the mean Minkowski functionals of a Gaussian
sky (their eq. 14) and checked them on 1000 simulated COBE-DMR-like skies (their Fig. 2).
Starting from their power spectrum (eqs. 17-20), do we get their curves, including the
full-sky Euler characteristic chi = 4 pi v2 + 2 v0, whose "+2 v0" is the sphere's own
topology?  And do the two ways of counting chi on HEALPix pixels (vertices - edges + faces of
the pixel mesh, and components of the hot set minus components of the cold set plus one)
agree map by map?

Computes: 1000 skies at nside 32 from C_l = g_smooth^2 (C_l^HZ g_beam^2 + C_noise);
          v0, v1, v2, chi at 41 thresholds; the analytic curves from sigma and tau of the
          correct eq. (16) (with its missing 1/4pi restored); the curves the misprinted
          eq. (16) would give.
Writes:   figures/ch12/sg98_mfs.pdf, results/ch12/06_sphere_sg98.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES
from lib_fields import gaussian_field_sphere
from lib_sphere import mesh, mfs_sphere, betti_sphere, gauss_mfs_sphere, sigma_tau

setup()
rng = rng_for("ch12", "06_sphere_sg98")
NSIDE, LMAX, NSIM = 32, 95, 1000

# ---------------------------------------------------------------- SG98's spectrum, eqs. (17)-(20), in mK^2
ell = np.arange(LMAX + 1)
cl_hz = np.zeros(LMAX + 1)
cl_hz[2:] = 6.0 / (ell[2:] * (ell[2:] + 1))          # eq. (17) with n = 1: Gamma ratios -> 6 / l(l+1)
cl_hz *= (7e-3) ** 2 / cl_hz[10]                      # normalised to C_10 = (7 microK)^2
sig_b = np.radians(7.0) / np.sqrt(8 * np.log(2))      # 7 deg FWHM beam
g_beam = np.exp(-0.5 * sig_b ** 2 * ell * (ell + 1))
g_smooth = np.exp(-0.5 * np.radians(3.0) ** 2 * ell * (ell + 1))   # eq. (19), omega = 3 deg
cl_noise = (3e-3) ** 2 * np.ones(LMAX + 1)            # eq. (18)
cl_noise[:2] = 0.0
cl = g_smooth ** 2 * (cl_hz * g_beam ** 2 + cl_noise)  # eq. (20)

s2, tau = sigma_tau(cl)                               # correct eq. (16): sums divided by 4 pi
lam = tau / s2
s2_bad, tau_bad = 4 * np.pi * s2, 4 * np.pi * tau     # eq. (16) as printed
sig = np.sqrt(s2)

# ---------------------------------------------------------------- 1000 skies at two pixel sizes
nus = np.linspace(-3, 3, 41)                          # thresholds in units of each map's rms


def run(nside):
    """MFs of NSIM skies at this nside; plus a map-by-map check of the two chi counters."""
    msh = mesh(nside)
    out = np.zeros((4, NSIM, len(nus)))
    bad = 0
    for s in range(NSIM):
        _, alm = gaussian_field_sphere(cl, nside, rng, lmax=LMAX)
        out[:, s] = mfs_sphere(alm, nside, nus, msh, lmax=LMAX, dnu=0.15)
        if s < 50:
            u = hp.alm2map(alm, nside, lmax=LMAX)
            u = (u - u.mean()) / u.std()
            for j, nu in enumerate(nus):
                m = u > nu
                if m.all() or not m.any():
                    continue
                hot, cold = betti_sphere(m, nside)
                bad += int(hot - cold + 1 != out[3, s, j])
    return out, bad


state = rng.bit_generator.state                      # the same 1000 skies at both pixel sizes (common random numbers)
(V0, V1, V2, CHI), betti_bad = run(128)              # pixels 0.46 deg: well below the 5.7 deg feature size
rng.bit_generator.state = state
(_, _, _, CHI32), betti_bad32 = run(NSIDE)           # pixels 1.8 deg, close to the DMR pixels (2.6 deg)

# analytic curves (thresholds in sigma units; SG98 plot them in mK)
a0, a1, a2 = gauss_mfs_sphere(nus, 1.0, lam)
achi = 4 * np.pi * a2 + 2 * a0
i1 = int(np.argmin(np.abs(nus - 1.0)))
im3 = 0
ipk = int(np.argmax(CHI.mean(0)))
nums = {"SgNside": NSIDE, "SgNsim": NSIM, "SgNpix": hp.nside2npix(NSIDE), "SgNpixFine": hp.nside2npix(128),
        "SgSigma": round(1e3 * sig, 1),                        # microK
        "SgSigmaBad": round(1e3 * np.sqrt(s2_bad), 1),
        "SgLam": round(lam, 1), "SgThetaC": round(np.degrees(1 / np.sqrt(2 * lam)), 2),
        "SgVoneMaxTh": round(np.sqrt(lam) / 8, 3), "SgVoneMaxMC": round(V1.mean(0)[len(nus) // 2], 3),
        "SgVtwoMaxTh": round(lam * np.exp(-0.5) / (2 * np.pi) ** 1.5, 2),
        "SgVtwoMaxMC": round(V2.mean(0)[i1], 2),
        "SgChiOneTh": round(achi[i1], 1), "SgChiOneMC": round(CHI[:, i1].mean(), 1),
        "SgChiOneSd": round(CHI[:, i1].std(ddof=1), 1), "SgChiOneCoarse": round(CHI32[:, i1].mean(), 1),
        # deficit of the coarse pixels against the fine ones, sky by sky (same skies), and its error
        "SgCoarseDef": round(100 * (1 - CHI32[:, i1].mean() / CHI[:, i1].mean()), 1),
        "SgCoarseDefSe": round(100 * (CHI32[:, i1] - CHI[:, i1]).std(ddof=1) / np.sqrt(NSIM) / CHI[:, i1].mean(), 1),
        "SgChiMthreeTh": round(achi[im3], 2), "SgChiMthreeMC": round(CHI[:, im3].mean(), 2),
        "SgChiPeakMC": round(CHI.mean(0)[ipk], 1), "SgChiPeakNu": round(nus[ipk], 2),
        "SgChiPeakMK": round(nus[ipk] * sig, 3),
        "SgBettiChecks": 2 * 50 * len(nus), "SgBettiBad": betti_bad + betti_bad32,
        "SgMaxPull": round(float(np.max(np.abs((CHI.mean(0) - achi) / (CHI.std(0, ddof=1) / np.sqrt(NSIM))))), 1),
        "SgMaxPullNu": round(float(nus[np.argmax(np.abs((CHI.mean(0) - achi) / (CHI.std(0, ddof=1) / np.sqrt(NSIM))))]), 2),
        "SgMaxDiff": round(float(np.max(np.abs(CHI.mean(0) - achi))), 2)}
save_numbers("ch12", "06_sphere_sg98", nums)
for k, v in nums.items():
    print(k, v)

# ---------------------------------------------------------------- figure: their Fig. 2, our skies
x = nus * sig                                          # threshold in mK, as SG98 plot it
xx = np.linspace(-3, 3, 300)
b0, b1, b2 = gauss_mfs_sphere(xx, 1.0, lam)
# values read off SG98 Fig. 2 (central lines; reading accuracy about 5 per cent)
READ = {"v1": (0.0, 0.9), "v2": (0.04, 2.1), "chi": (0.04, 26.0)}
fig, axs = plt.subplots(2, 2, figsize=(8.4, 5.6))
panels = [(V0, b0, r"$v_0$ (area fraction)", None), (V1, b1, r"$v_1$ (quarter boundary length per sr)", "v1"),
          (V2, b2, r"$v_2$ [sr$^{-1}$]", "v2"), (CHI, 4 * np.pi * b2 + 2 * b0, r"$\chi$ of the full sky", "chi")]
for ax, (D, th, lab, key) in zip(axs.flat, panels):
    m, sd = D.mean(0), D.std(0, ddof=1)
    ax.fill_between(x, m - sd, m + sd, color=SERIES[0], alpha=0.3, lw=0,
                    label=f"{NSIM} skies (nside 128), mean $\\pm1$ s.d.")
    ax.plot(x, m, color=SERIES[0], lw=1.2)
    theory_line(ax, xx * sig, th, label="SG98 eq. (14), corrected eq. (16)")
    if key:
        ax.plot(*READ[key], marker="*", ms=9, color=SERIES[3], ls="none", label="read off SG98 Fig. 2")
    ax.set_title(lab, fontsize=9)
    ax.set_xlabel("threshold [mK]")
    ax.set_xlim(-0.27, 0.27)
ax = axs[1, 1]
ax.plot(x, CHI32.mean(0), "o", ms=2.5, color=SERIES[1], label="mean, nside 32 pixels")
ax.plot(xx * sig, 4 * np.pi * b2, color=SERIES[2], ls=":", lw=1.2, label=r"without the $2v_0$ term")
ax.plot(xx * np.sqrt(s2_bad), 4 * np.pi * b2 + 2 * b0, color=SERIES[4], lw=1.0, label="eq. (16) as printed")
ax.legend(fontsize=6.3, loc="upper left")
axs[0, 0].legend(fontsize=6.3, loc="lower left")
fig.tight_layout()
savefig(fig, "ch12", "sg98_mfs")
