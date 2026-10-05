"""02_nongaussian_mfs.py -- what a skewed sky does to the Minkowski functionals.

Question: when the temperature map is a mildly non-Gaussian field, how do its Minkowski
functionals move away from the Gaussian curves, and how much of that change is only a relabelling
of the threshold?

Two toy skies with the same smoothed CMB spectrum shape as 01 (exaggerated amplitudes, so the
effect is visible in 300 patches):
  local    : f = u + a (u^2 - 1), u Gaussian (a pointwise, monotone change of the values);
  nonlocal : the same quadratic term applied to a scale-invariant "potential" phi, then the
             result filtered with a transfer W(ell) that turns phi's spectrum into the CMB one
             (the Sachs-Wolfe-plus-acoustic step is a filter, i.e. non-local).
Computes: v0, v1, v2 at fixed threshold nu = f/sigma_f and at fixed area-fraction threshold nu_A;
the three skewness parameters S^(0), S^(1), S^(2) measured on the maps; Matsubara's first-order
predictions (Matsubara 2003, eqs. 3.48 and 3.54, d = 2).
Writes: figures/chT3/ng_mfs.pdf, results/chT3/02_nongaussian_mfs.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erfcinv
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from camb_fiducial import load_fiducial
from lib_t3 import (ARCMIN, beam, gaussian_flat, ell_grid, cl_on_grid, gradient, euler_torus,
                    mf_gauss)

setup()
rng = rng_for("chT3", "02_nongaussian_mfs")
N, dx, NSIM, A_LOC, B_NL = 512, 1.5 * ARCMIN, 300, 0.05, 0.10
ell, TT = load_fiducial("TT")
cl = TT * beam(ell, 10.0) ** 2
L = np.arange(cl.size, dtype=float)
clphi = np.where(L > 1, 1.0 / np.maximum(L, 1.0) ** 2, 0.0)        # ell^2 C_ell = const: scale invariant
W = np.sqrt(np.where(clphi > 0, cl / np.where(clphi > 0, clphi, 1.0), 0.0))
ellg, _, _ = ell_grid(N, dx)
Wg = cl_on_grid(W, ellg)
area = N * N * dx * dx
nus = np.linspace(-3.5, 3.5, 29)
H = {0: lambda x: np.ones_like(x), 1: lambda x: x, 2: lambda x: x**2 - 1,
     3: lambda x: x**3 - 3*x, 4: lambda x: x**4 - 6*x**2 + 3}         # probabilists' Hermite


def skewness_params(f):
    """S^(0) s0, S^(1) s0, S^(2) s0 (Matsubara 2003, eqs. 2.61-2.63 with d = 2)."""
    s0 = f.std()
    gx, gy = gradient(f, dx)
    g2 = gx**2 + gy**2
    s1 = np.sqrt(g2.mean())
    lap = np.fft.ifft2(-ellg**2 * np.fft.fft2(f)).real
    return np.array([np.mean(f**3) / s0**3,
                     -0.75 * np.mean(f**2 * lap) / (s0 * s1**2),
                     -3.0 * np.mean(g2 * lap) * s0 / s1**4])


def mfs_at(f, thresholds, widths):
    """v0, v1, v2 at the given thresholds (in the units of f) for a map f of unit rms."""
    gx, gy = gradient(f, dx)
    g = np.hypot(gx, gy)
    v0 = np.array([(f > t).mean() for t in thresholds])
    v1 = np.array([0.25 * np.mean(g * (np.abs(f - t) < w / 2)) / w for t, w in zip(thresholds, widths)])
    v2 = np.array([euler_torus(f > t) for t in thresholds]) / area
    return np.array([v0, v1, v2])


def both(f):
    """Functionals at fixed nu and at fixed area-fraction threshold nu_A."""
    f = (f - f.mean()) / f.std()
    w = np.full(nus.size, nus[1] - nus[0])
    fixed = mfs_at(f, nus, w)
    # area threshold: t(nu_A) is the value with a fraction (1/2) erfc(nu_A/sqrt2) of pixels above it
    q = lambda n: np.quantile(f, 1 - 0.5 * erfc(n / np.sqrt(2)))
    t = np.array([q(n) for n in nus])
    h = 0.5 * (nus[1] - nus[0])
    wA = np.array([q(n + h) - q(n - h) for n in nus])
    return fixed, mfs_at(f, t, wA), skewness_params(f)


res = {"gauss": [], "local": [], "nonlocal": []}
s1s0_all = []                                # sigma_1/sigma_0 of every unit Gaussian map
for _ in range(NSIM):
    u = gaussian_flat(cl, N, dx, rng); u /= u.std()
    gx, gy = gradient(u, dx)
    s1s0_all.append(np.sqrt(np.mean(gx**2 + gy**2)))
    res["gauss"].append(both(u))
    res["local"].append(both(u + A_LOC * (u**2 - 1)))
    phi = gaussian_flat(clphi, N, dx, rng); phi /= phi.std()
    res["nonlocal"].append(both(np.fft.ifft2(np.fft.fft2(phi + B_NL * (phi**2 - 1)) * Wg).real))

mean = {k: [np.mean([r[i] for r in v], 0) for i in range(3)] for k, v in res.items()}
err = {k: [np.std([r[i] for r in v], 0, ddof=1) / np.sqrt(NSIM) for i in range(2)] for k, v in res.items()}

# ---- first-order (Matsubara) predictions for v2, with the measured skewness parameters
s1s0 = np.mean(s1s0_all)                     # sigma_1/sigma_0, averaged over all Gaussian maps
amp2 = (s1s0**2 / 2) / (2 * np.pi) ** 1.5
e = np.exp(-nus**2 / 2)
g2 = amp2 * e * H[1](nus)


def v2_first_order(S, area_thr=False):
    S0, S1, S2 = S
    if area_thr:
        return amp2 * e * (2/3 * (S1 - S0) * H[2](nus) + 1/3 * (S2 - S0))
    return amp2 * e * (S0/6 * H[4](nus) + 2/3 * S1 * H[2](nus) + 1/3 * S2)


S = {k: mean[k][2] for k in mean}
peak = g2.max()

fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.3), sharey=True)
for k, col, lab in [("local", SERIES[1], "local (pointwise) skewness"),
                    ("nonlocal", SERIES[2], "non-local skewness")]:
    for j, a in enumerate(ax):
        d = (mean[k][j][2] - mean["gauss"][j][2]) / peak
        de = np.hypot(err[k][j][2], err["gauss"][j][2]) / peak
        a.errorbar(nus, d, de, fmt="o", ms=3, color=col, label=lab)
        a.plot(nus, v2_first_order(S[k], area_thr=(j == 1)) / peak, color=col, lw=1.2)
for a in ax:
    a.axhline(0, color="k", lw=0.6)
ax[0].set_xlabel(r"threshold $\nu=f/\sigma_0$"); ax[1].set_xlabel(r"area threshold $\nu_A$")
ax[0].set_ylabel(r"$(v_2-v_2^{\rm Gauss})/\max v_2^{\rm Gauss}$")
ax[0].set_title("at fixed threshold"); ax[1].set_title("at fixed area fraction")
ax[0].legend(loc="lower left", fontsize=8)
savefig(fig, "chT3", "ng_mfs")

dl = (mean["local"][0][2] - mean["gauss"][0][2]) / peak
dlA = (mean["local"][1][2] - mean["gauss"][1][2]) / peak
dnA = (mean["nonlocal"][1][2] - mean["gauss"][1][2]) / peak
# the non-local and Gaussian maps are independent draws, so the error of their difference
# combines both Monte Carlo errors
eNA = np.hypot(err["nonlocal"][1][2], err["gauss"][1][2]) / peak
ePk = eNA.max()
# what the first-order formula leaves over near nu_A = -2, in units of that error
i2 = np.argmin(np.abs(nus + 2.0))
resA = dnA - v2_first_order(S["nonlocal"], area_thr=True) / peak
print(f"nonlocal residual at nu_A={nus[i2]:.2f}: {100*resA[i2]:.2f}% = {resA[i2]/eNA[i2]:.1f} sigma")
save_numbers("chT3", "02_nongaussian_mfs", {
    "TbNsim": NSIM, "TbA": f"{A_LOC}", "TbB": f"{B_NL}", "TbSixA": f"{6 * A_LOC:.2f}",
    "TbLocSzero": f"{S['local'][0]:.3f}", "TbLocSone": f"{S['local'][1]:.3f}", "TbLocStwo": f"{S['local'][2]:.3f}",
    "TbNlSzero": f"{S['nonlocal'][0]:.3f}", "TbNlSone": f"{S['nonlocal'][1]:.3f}", "TbNlStwo": f"{S['nonlocal'][2]:.3f}",
    "TbLocMaxShiftPct": f"{100 * np.abs(dl).max():.0f}",
    "TbLocAreaMaxPct": f"{100 * np.abs(dlA).max():.1f}",
    "TbNlAreaMaxPct": f"{100 * np.abs(dnA).max():.1f}",
    "TbErrPct": f"{100 * ePk:.1f}",
    "TbNlResNu": f"{nus[i2]:.1f}",
    "TbNlResPct": f"{100 * resA[i2]:.1f}",
    "TbNlResSig": f"{abs(resA[i2]) / eNA[i2]:.1f}",
})
print("S local", S["local"], "S nonlocal", S["nonlocal"])
print(f"max shift local {100*np.abs(dl).max():.1f}%, at nu_A: local {100*np.abs(dlA).max():.2f}%, "
      f"nonlocal {100*np.abs(dnA).max():.2f}% (MC err {100*ePk:.2f}%)")
