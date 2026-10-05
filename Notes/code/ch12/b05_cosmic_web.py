"""b05_cosmic_web.py -- Betti curves in three dimensions: the Gaussian baseline and the Poisson baseline.

Question: for a three-dimensional field, how do the numbers of clusters (beta_0), tunnels
(beta_1) and voids (beta_2) depend on the density threshold, do they add up to the Euler
characteristic predicted for a Gaussian field, and what does pure shot noise (a Poisson
point process) look like in the same summary?
Computes: (A) superlevel persistence of periodic 3D Gaussian fields with gudhi's
PeriodicCubicalComplex, mean Betti curves, their alternating sum against the Gaussian genus
formula, peak positions; beta_0 cross-checked on one box with scipy.ndimage.label plus gluing across the periodic faces.
(B) Poisson points in a periodic box at three mean separations, a k-nearest-neighbour density
on a grid (an adaptive estimator, standing in for the Delaunay estimator of Pranav et al.),
Betti curves against density / std, peak positions.
Writes: figures/ch12/b_web_grf.pdf, figures/ch12/b_web_poisson.pdf, results/ch12/b05_cosmic_web.tex
"""
import sys
import itertools
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_fields import gaussian_field, kgrid, rfft_weights
import gudhi

rng = rng_for("ch12", "b05_cosmic_web")


def betti_curves_3d(f, nus):
    """Betti curves of the superlevel sets {f >= nu} on the 3-torus (gudhi, sublevel of -f)."""
    cc = gudhi.PeriodicCubicalComplex(top_dimensional_cells=-f, periodic_dimensions=[True] * 3)
    cc.persistence()
    out = []
    for dim in range(4):
        p = cc.persistence_intervals_in_dimension(dim)
        b, d = -p[:, 0], -p[:, 1]                        # back to levels of f: alive for d < nu <= b
        out.append(((d[:, None] < nus[None]) & (nus[None] <= b[:, None])).sum(0))
    return np.array(out)


def periodic_components(mask):
    """Components of a 3D mask on the torus: scipy.ndimage.label, then glue labels that touch
    across the faces (all 26 neighbour offsets, with wrap-around)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    a, b = [], []
    for off in itertools.product([-1, 0, 1], repeat=3):
        r = np.roll(lab, off, axis=(0, 1, 2))
        m = (lab > 0) & (r > 0) & (lab != r)
        a.append(lab[m])
        b.append(r[m])
    a, b = np.concatenate(a), np.concatenate(b)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n + 1, n + 1))
    return connected_components(g)[0] - 1                  # minus the background label 0


# ---------------------------------------------------------------- (A) Gaussian fields
N, R, NS, NBOX = 80, 2.5, -1.5, 24
P = lambda k: k ** NS * np.exp(-(k * R) ** 2)
kmag, _ = kgrid(N, float(N), 3)
w = rfft_weights(N, 3)
with np.errstate(divide="ignore"):
    Pk = np.where(kmag > 0, P(np.where(kmag > 0, kmag, 1.0)), 0.0)
k2 = (w * kmag ** 2 * Pk).sum() / (w * Pk).sum()           # <k^2> of the realised grid modes
nus = np.linspace(-3.5, 3.5, 71)
B = []
for i in range(NBOX):
    f = gaussian_field(P, N, float(N), d=3, rng=rng)
    f /= f.std()
    B.append(betti_curves_3d(f, nus))
    if i == 0:
        lab0 = [periodic_components(f >= nu) for nu in (1.0, 2.0)]
        check = int(np.abs(B[0][0][[45, 55]] - lab0).max())
        print("box 0, beta0 at nu=1,2 (gudhi):", B[0][0][[45, 55]], " ndimage + seams:", lab0)
B = np.array(B)                                            # (box, dim, nu)
mean, sd = B.mean(0), B.std(0)
chi = (B[:, 0] - B[:, 1] + B[:, 2] - B[:, 3]).mean(0)
chi_th = -N ** 3 * (1 / (4 * np.pi ** 2)) * (k2 / 3) ** 1.5 * (1 - nus ** 2) * np.exp(-nus ** 2 / 2)
peaks = [nus[np.argmax(mean[k])] for k in range(3)]
amp = [mean[k].max() for k in range(3)]
dev = np.abs(chi - chi_th).max() / np.abs(chi_th).max()
err = (B[:, 0] - B[:, 1] + B[:, 2] - B[:, 3]).std(0).max() / np.sqrt(NBOX) / np.abs(chi_th).max()
# the deviation threshold by threshold in units of its own Monte Carlo error (a max over many thresholds)
se_chi = (B[:, 0] - B[:, 1] + B[:, 2] - B[:, 3]).std(0, ddof=1) / np.sqrt(NBOX)
pull = np.abs(chi - chi_th)[se_chi > 0] / se_chi[se_chi > 0]
max_pull, n_pull = pull.max(), pull.size
print("GRF peaks", peaks, "amplitudes", amp, "chi deviation", dev, "MC error", err)

# ---------------------------------------------------------------- (B) Poisson point process
NG, KNN = 64, 5
g1 = np.arange(NG) + 0.5
grid = np.stack(np.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
lams = [2.0, 3.0, 4.0]                                     # mean separations, in grid cells
nuP = np.linspace(0, 10, 201)
pois, ppeaks = [], []
for lam in lams:
    curves = []
    for r in range(3):
        X = rng.uniform(0, NG, (int((NG / lam) ** 3), 3))
        dist, _ = cKDTree(X, boxsize=NG).query(grid, k=KNN)
        rho = (KNN / (4 / 3 * np.pi * dist[:, -1] ** 3)).reshape(NG, NG, NG)
        curves.append(betti_curves_3d(rho / rho.std(), nuP))
    c = np.mean(curves, 0)
    pois.append(c)
    ppeaks.append([nuP[np.argmax(c[k])] for k in range(3)])
ppeaks = np.array(ppeaks)
print("Poisson peaks (nu/sigma) per mean separation:", ppeaks)

# ---------------------------------------------------------------- figures
names = [r"$\beta_0$ clusters", r"$\beta_1$ tunnels", r"$\beta_2$ voids"]
setup(4.6, 3.2)
fig, ax = plt.subplots()
for k in range(3):
    ax.plot(nus, mean[k] / N ** 3 * 1e3, color=SERIES[k], label=names[k])
    ax.fill_between(nus, (mean[k] - sd[k]) / N ** 3 * 1e3, (mean[k] + sd[k]) / N ** 3 * 1e3,
                    color=SERIES[k], alpha=0.2, lw=0)
ax.plot(nus, chi / N ** 3 * 1e3, color="0.35", lw=1.2, label=r"$\beta_0-\beta_1+\beta_2-\beta_3$")
theory_line(ax, nus, chi_th / N ** 3 * 1e3, label="Gaussian formula for $\\chi$")
ax.set_xlabel(r"threshold $\nu$ (units of $\sigma$)")
ax.set_ylabel(r"number per $10^3$ cells")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "ch12", "b_web_grf")

setup(4.6, 3.2)
fig, ax = plt.subplots()
for j, lam in enumerate(lams):
    for k in range(3):
        ax.plot(nuP, pois[j][k] / NG ** 3 * lam ** 3, color=SERIES[k], ls=["-", "--", ":"][j], lw=1.2,
                label=(names[k] if j == 0 else None))
for k, v in enumerate([1.8, 0.6, 0.3]):
    ax.axvline(v, color=SERIES[k], lw=0.8, alpha=0.6)
ax.set_xlabel(r"density / its standard deviation")
ax.set_ylabel("number per particle")
ax.set_xlim(0, 7)
ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch12", "b_web_poisson")

save_numbers("ch12", "b05_cosmic_web", {
    "PbWebN": N, "PbWebCheck": check, "PbWebR": R, "PbWebBoxes": NBOX,
    "PbWebPeakZero": peaks[0], "PbWebPeakOne": peaks[1], "PbWebPeakTwo": peaks[2],
    "PbWebAmpZero": amp[0] / N ** 3 * 1e3, "PbWebAmpOne": amp[1] / N ** 3 * 1e3,
    "PbWebAmpTwo": amp[2] / N ** 3 * 1e3,
    "PbWebChiZero": chi[35] / N ** 3 * 1e3, "PbWebChiPeak": chi[51] / N ** 3 * 1e3,
    "PbWebBzeroAtZero": mean[0][35] / N ** 3 * 1e3, "PbWebBtwoAtZero": mean[2][35] / N ** 3 * 1e3,
    "PbWebChiDev": 100 * dev, "PbWebChiErr": 100 * err,
    "PbWebChiPull": f"{max_pull:.1f}", "PbWebChiNpull": n_pull,
    "PbPoisNg": NG, "PbPoisK": KNN,
    "PbPoisPeakZero": ppeaks[:, 0].mean(), "PbPoisPeakOne": ppeaks[:, 1].mean(),
    "PbPoisPeakTwo": ppeaks[:, 2].mean(),
    "PbPoisSpread": float(np.max(ppeaks.max(0) - ppeaks.min(0))),
})
