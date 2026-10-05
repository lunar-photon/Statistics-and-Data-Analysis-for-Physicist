"""05_hgw3d.py -- reproduce the genus curve of Hamilton, Gott & Weinberg (1986), their Fig. 4.

Question: HGW predicted the integrated Gaussian curvature per unit volume of the isodensity
surfaces of a smoothed Gaussian field,
    C(nu) = -(1/pi) (<k^2>/3)^{3/2} (1 - nu^2) e^{-nu^2/2},
and checked it on 8 simulations: Poisson noise, 5 x 32^3 points in a periodic 32^3 box,
smoothed with W(r) = exp(-r^2/r0^2), r0 = 2 cells.  Do we get their curve from their setup,
and what changes with many more simulations and a finer grid?

Computes: chi of the solid excursion set by V - E + F - C of the voxel complex (C = 4 pi chi);
          their setup with 8 and with 400 simulations; a finer 64^3, r0 = 4 version (200 sims).
Writes:   figures/ch12/hgw_genus.pdf, results/ch12/05_hgw3d.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import euler_number
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES
from lib_topo import euler_cubical_3d

setup()
rng = rng_for("ch12", "05_hgw3d")
nus = np.round(np.linspace(-3, 3, 25), 3)


def kvec(N):
    k = 2 * np.pi * np.fft.fftfreq(N)
    kz = 2 * np.pi * np.fft.rfftfreq(N)
    return np.meshgrid(k, k, kz, indexing="ij")


def smoothed_poisson(N, r0, mean_per_cell=5.0):
    """HGW's field: Poisson counts in cells, smoothed with exp(-r^2/r0^2) (FT: exp(-k^2 r0^2/4))."""
    n = rng.poisson(mean_per_cell, size=(N, N, N)).astype(float)
    kx, ky, kz = kvec(N)
    W = np.exp(-(kx ** 2 + ky ** 2 + kz ** 2) * r0 ** 2 / 4)
    return np.fft.irfftn(np.fft.rfftn(n - n.mean()) * W, s=(N, N, N), axes=(0, 1, 2))


def lam3(N, r0):
    """<k^2>/3 of the smoothed white-noise field on the lattice: sum k^2 W^2 / (3 sum W^2)."""
    k = 2 * np.pi * np.fft.fftfreq(N)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    w2 = np.exp(-k2 * r0 ** 2 / 2)
    w2.flat[0] = 0.0
    return (k2 * w2).sum() / (3 * w2.sum())


def run(N, r0, nsim):
    C = np.zeros((nsim, len(nus)))
    for s in range(nsim):
        f = smoothed_poisson(N, r0)
        u = (f - f.mean()) / f.std()
        C[s] = [4 * np.pi * euler_cubical_3d(u > nu, periodic=True) / N ** 3 for nu in nus]
    return C / lam3(N, r0) ** 1.5          # HGW's normalisation: -xi''(0)/xi(0) = <k^2>/3 = 1


theory = lambda x: -(1 / np.pi) * (1 - x ** 2) * np.exp(-x ** 2 / 2)

# a check of the voxel counting against scikit-image on one non-periodic block
f = smoothed_poisson(32, 2.0)
u = (f - f.mean()) / f.std()
ok = all(euler_cubical_3d(u > nu, periodic=False) == euler_number(u > nu, connectivity=3) for nu in nus)
print("voxel chi agrees with skimage (26-connectivity):", ok)

C8 = run(32, 2.0, 8)
C400 = run(32, 2.0, 400)
Cfine = run(64, 4.0, 200)
i0 = int(np.where(nus == 0.0)[0][0])
ir3 = int(np.argmin(np.abs(nus - 1.75)))
irm = int(np.argmin(np.abs(nus + 1.75)))
nums = {"HgwSkOk": "yes" if ok else "no", "HgwLamThree": round(lam3(32, 2.0), 4),
        "HgwThZero": round(theory(0.0), 4), "HgwThMax": round(theory(np.sqrt(3)), 4),
        "HgwEightZero": round(C8[:, i0].mean(), 3), "HgwEightZeroSe": round(C8[:, i0].std(ddof=1) / np.sqrt(8), 3),
        "HgwFourZero": round(C400[:, i0].mean(), 4), "HgwFourZeroSe": round(C400[:, i0].std(ddof=1) / 20, 4),
        "HgwFineZero": round(Cfine[:, i0].mean(), 4), "HgwFineZeroSe": round(Cfine[:, i0].std(ddof=1) / np.sqrt(200), 4),
        "HgwEightPeak": round(C8[:, ir3].mean(), 3), "HgwFourPeak": round(C400[:, ir3].mean(), 4),
        "HgwFinePeak": round(Cfine[:, ir3].mean(), 4),
        "HgwFourPeakNeg": round(C400[:, irm].mean(), 4), "HgwFinePeakNeg": round(Cfine[:, irm].mean(), 4),
        "HgwPeakSe": round(C400[:, ir3].std(ddof=1) / 20, 4), "HgwThPeak": round(theory(1.75), 4),
        "HgwFourRel": round(100 * (C400[:, i0].mean() / theory(0.0) - 1), 1),
        "HgwFineRel": round(100 * (Cfine[:, i0].mean() / theory(0.0) - 1), 1),
        # the same offsets in units of their standard errors
        "HgwFourPull": round(abs(C400[:, i0].mean() - theory(0.0)) / (C400[:, i0].std(ddof=1) / 20), 1),
        "HgwFinePull": round(abs(Cfine[:, i0].mean() - theory(0.0)) / (Cfine[:, i0].std(ddof=1) / np.sqrt(200)), 1)}
save_numbers("ch12", "05_hgw3d", nums)
for k, v in nums.items():
    print(k, v)

fig, ax = plt.subplots(figsize=(6.0, 3.6))
nn = np.linspace(-3.2, 3.2, 300)
ax.errorbar(nus - 0.04, C8.mean(0), yerr=C8.std(0, ddof=1), fmt="s", ms=3, color=SERIES[1],
            label=r"HGW setup: $32^3$, $r_0=2$, 8 sims (bar: one sim)")
ax.errorbar(nus, C400.mean(0), yerr=C400.std(0, ddof=1) / 20, fmt="o", ms=3, color=SERIES[0],
            label=r"same setup, 400 sims (bar: error of mean)")
ax.errorbar(nus + 0.04, Cfine.mean(0), yerr=Cfine.std(0, ddof=1) / np.sqrt(200), fmt="^", ms=3,
            color=SERIES[2], label=r"finer: $64^3$, $r_0=4$, 200 sims")
theory_line(ax, nn, theory(nn), label=r"HGW eq. (3): $-\pi^{-1}(1-\nu^2)e^{-\nu^2/2}$")
ax.axhline(0, color="0.6", lw=0.6)
ax.set_xlabel(r"$\nu$ (standard deviations)")
ax.set_ylabel(r"$C$ per unit volume, $(\langle k^2\rangle/3)=1$")
ax.set_ylim(-0.6, 0.2)
ax.legend(fontsize=7, loc="lower right")
savefig(fig, "ch12", "hgw_genus")
