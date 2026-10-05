"""03_polarization.py -- polarization patterns of a simulated CMB patch and their singularities.

Question: where does the polarization of a Gaussian CMB sky vanish, with which index, how many
such points are there per square degree, and how are they arranged?

Computes:
  (a) a 6 x 6 deg patch with correlated T, E (C_ell^TT, TE, EE, BB; 30 arcmin beam): the
      temperature map, its nu = +1 excursion set, the headless polarization sticks and the
      singularities (index +1/2 and -1/2) found by the plaquette winding rule;
  (b) 200 patches of 12.8 x 12.8 deg (1.5 arcmin pixels, 10 arcmin beam): the number of
      singularities against the density formula n = sigma_1^2 / (4 pi sigma_0^2), the net index,
      and the distances to the nearest singularity of the same and of the opposite index;
  (c) the full-sky count N = sum (2l+1) l(l+1) C_l^P / sum (2l+1) C_l^P for a sharp cut at
      ell_max = 100 and 500 (the two maps of Huterer & Vachaspati 2005), with a flat-patch check
      of the ell_max = 500 density.
Writes: figures/chT3/pol_patch.pdf, figures/chT3/pol_stats.pdf, results/chT3/03_polarization.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from common import setup, savefig, save_numbers, rng_for, SERIES
from camb_fiducial import load_fiducial
from lib_t3 import (ARCMIN, beam, ell_grid, cl_on_grid, gaussian_flat, polarization_flat,
                    moments_flat, singularities, euler_torus)

setup()
rng = rng_for("chT3", "03_polarization")
ell, TT = load_fiducial("TT")
EE, BB, TE = (load_fiducial(s)[1] for s in ("EE", "BB", "TE"))

# ---------------------------------------------------------------- (a) the picture
N, dx, F = 256, 6.0 * 60 / 256 * ARCMIN, 30.0
b2 = beam(ell, F) ** 2
lg, lx, ly = ell_grid(N, dx)
ph = np.arctan2(ly, lx)
g1 = np.fft.fft2(rng.standard_normal((N, N)))
g2 = np.fft.fft2(rng.standard_normal((N, N)))
ctt, cte, cee = (cl_on_grid(c * b2, lg) for c in (TT, TE, EE))
with np.errstate(divide="ignore", invalid="ignore"):
    aT = np.sqrt(ctt) * g1
    aE = np.where(ctt > 0, cte / np.sqrt(ctt), 0) * g1 + np.sqrt(np.clip(cee - np.where(ctt > 0, cte**2 / ctt, 0), 0, None)) * g2
aT[0, 0] = aE[0, 0] = 0
aB = np.sqrt(cl_on_grid(BB * b2, lg)) * np.fft.fft2(rng.standard_normal((N, N)))
T = np.fft.ifft2(aT).real / dx
Q = np.fft.ifft2(aE * np.cos(2 * ph) - aB * np.sin(2 * ph)).real / dx
U = np.fft.ifft2(aE * np.sin(2 * ph) + aB * np.cos(2 * ph)).real / dx
i, j, q = singularities(Q, U)

side = N * dx / ARCMIN / 60
x = (np.arange(N) + 0.5) * dx / ARCMIN / 60
fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.imshow((T / T.std()).T, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, extent=[0, side, 0, side], alpha=0.85)
ax.contour(x, x, (T / T.std()).T, levels=[1.0], colors="k", linewidths=0.8)
st = 6                                                  # one stick every 6 pixels
P = np.hypot(Q, U)
alpha = 0.5 * np.arctan2(U, Q)                          # polarization angle, headless
X, Y = np.meshgrid(x[::st], x[::st], indexing="ij")
L = 0.85 * st * dx / ARCMIN / 60 * np.sqrt(P[::st, ::st] / P.max())
ax.quiver(X, Y, np.cos(alpha[::st, ::st]) * L, np.sin(alpha[::st, ::st]) * L, angles="xy",
          scale_units="xy", scale=1, headwidth=0, headlength=0, headaxislength=0, pivot="middle",
          width=0.0025, color="0.15")
xs, ys = (i + 1) * dx / ARCMIN / 60, (j + 1) * dx / ARCMIN / 60
ax.plot(xs[q > 0], ys[q > 0], "o", ms=6, mfc="none", mec=SERIES[1], mew=1.6, label=r"index $+\frac{1}{2}$")
ax.plot(xs[q < 0], ys[q < 0], "s", ms=6, mfc="none", mec=SERIES[6], mew=1.6, label=r"index $-\frac{1}{2}$")
ax.set_xlim(0, side); ax.set_ylim(0, side); ax.grid(False)
ax.set_xlabel("deg"); ax.set_ylabel("deg")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
savefig(fig, "chT3", "pol_patch")
n_patch, plus_patch = q.size, int((q > 0).sum())

# ---------------------------------------------------------------- (b) counts and neighbours
N, dx, NSIM = 512, 1.5 * ARCMIN, 200
cle, clb = EE * beam(ell, 10.0) ** 2, BB * beam(ell, 10.0) ** 2
s0sq, s1sq = moments_flat(cle + clb, N, dx)
area = (N * dx) ** 2
pred = s1sq / s0sq / (4 * np.pi) * area
counts, net, d_same, d_opp = [], [], [], []
box = N
for k in range(NSIM):
    Qm, Um = polarization_flat(cle, clb, N, dx, rng)
    i, j, q = singularities(Qm, Um)
    counts.append(q.size); net.append(q.sum())
    if k < 40:                                           # neighbour statistics from 40 patches
        pts = np.c_[i, j] + 0.5
        for sgn in (+1, -1):
            a, b = pts[q * sgn > 0], pts[q * sgn < 0]
            ta, tb = cKDTree(a, boxsize=box), cKDTree(b, boxsize=box)
            d_same.append(ta.query(a, k=2)[0][:, 1]); d_opp.append(tb.query(a, k=1)[0])
counts = np.array(counts)
d_same = np.concatenate(d_same) * dx / ARCMIN
d_opp = np.concatenate(d_opp) * dx / ARCMIN
n_arcmin = pred / (area / ARCMIN**2)                     # all singularities per arcmin^2
poisson_same = 0.5 / np.sqrt(n_arcmin / 2)               # Poisson mean NN distance, density n/2

fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2))
ax[0].hist(counts, bins=20, color=SERIES[0], alpha=0.7, label=f"{NSIM} simulated patches")
ax[0].axvline(pred, ymax=0.7, color="k", ls="--", label="density formula")
ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.45)
ax[0].set_xlabel("number of singularities in the patch"); ax[0].legend(loc="upper left", fontsize=8)
bins = np.arange(0, 43.5, 3.0)            # 2-pixel bins smooth the pixel grid
ax[1].hist(d_same, bins=bins, density=True, histtype="step", lw=1.6, color=SERIES[1], label="same index")
ax[1].hist(d_opp, bins=bins, density=True, histtype="step", lw=1.6, color=SERIES[6], label="opposite index")
r = np.linspace(0, 40, 200)
lam = n_arcmin / 2
ax[1].plot(r, 2 * np.pi * lam * r * np.exp(-np.pi * lam * r**2), "k--", lw=1.2, label="Poisson, same density")
ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.45)
ax[1].set_xlabel("distance to nearest neighbour [arcmin]"); ax[1].legend(fontsize=8, loc="upper right")
savefig(fig, "chT3", "pol_stats")

# ---------------------------------------------------------------- (c) Huterer-Vachaspati counts
def n_sky(lmax):
    l = np.arange(lmax + 1)
    c = (EE + BB)[: lmax + 1] * (2 * l + 1)
    return (c * l * (l + 1)).sum() / c.sum()

N3, dx3 = 256, 20 * 60 / 256 * ARCMIN                   # 20 deg patch, sharp cut at ell = 500
cut = (ell <= 500).astype(float)
s0c, s1c = moments_flat((EE + BB) * cut, N3, dx3)
c3 = [singularities(*polarization_flat(EE * cut, BB * cut, N3, dx3, rng))[2].size for _ in range(30)]
sky500_patch = np.mean(c3) / (N3 * dx3) ** 2 * 4 * np.pi

save_numbers("chT3", "03_polarization", {
    "TcPatchCount": n_patch, "TcPatchPlus": plus_patch, "TcPatchMinus": n_patch - plus_patch,
    "TcNsim": NSIM, "TcPred": f"{pred:.0f}", "TcMean": f"{counts.mean():.0f}",
    "TcSd": f"{counts.std(ddof=1):.0f}", "TcSdPoisson": f"{np.sqrt(pred):.0f}",
    "TcSdErr": f"{counts.std(ddof=1) / np.sqrt(2 * (NSIM - 1)):.0f}",          # error of the sd itself
    "TcMeanDevPct": f"{100 * (counts.mean() - pred) / pred:.1f}",             # signed offset of the mean
    "TcMeanDevSig": f"{abs(counts.mean() - pred) / (counts.std(ddof=1) / np.sqrt(NSIM)):.1f}",
    "TcNetMax": int(np.max(np.abs(net))),
    "TcPerDeg": f"{pred / (area * (180 / np.pi) ** 2):.1f}",
    "TcSame": f"{d_same.mean():.1f}", "TcOpp": f"{d_opp.mean():.1f}", "TcPoisson": f"{poisson_same:.1f}",
    "TcSkyHundred": f"{n_sky(100):.0f}", "TcSkyFiveHundred": f"{n_sky(500):.0f}",
    "TcSkyFiveHundredPatch": f"{sky500_patch:.0f}",
})
print(f"patch: {n_patch} singularities ({plus_patch} positive); counts {counts.mean():.1f}+-{counts.std():.1f} vs {pred:.1f}; "
      f"net max {np.max(np.abs(net))}; NN same {d_same.mean():.2f}' opp {d_opp.mean():.2f}' Poisson {poisson_same:.2f}'")
print(f"sky counts lmax=100: {n_sky(100):.0f}, lmax=500: {n_sky(500):.0f}; patch-scaled lmax=500: {sky500_patch:.0f}")
