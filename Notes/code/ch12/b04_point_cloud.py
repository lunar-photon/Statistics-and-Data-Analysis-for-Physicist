"""b04_point_cloud.py -- persistence of point clouds: Vietoris-Rips, Kruskal, boundary-matrix reduction.

Question: when the data are points (galaxies, haloes) rather than a field, how do we build
shapes from them at every scale r, and does the minimum spanning tree give the H0 diagram?
Computes: the six-point example worked by hand (Kruskal); the unit square (one loop born at 1,
filled at sqrt 2); a noisy circle with a cluster beside it: H0 by Kruskal, by scipy's single
linkage and by ripser; H1 by our boundary-matrix reduction and by ripser; gudhi's RipsComplex
as a third check; timing of Kruskal and ripser against the number of points.
Writes: figures/ch12/b_rips.pdf, figures/ch12/b_cloud_diagram.pdf, results/ch12/b04_point_cloud.tex
"""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.cluster.hierarchy import linkage
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2
from lib_persist import kruskal_h0, rips_bruteforce
import ripser
import persim
import gudhi

rng = rng_for("ch12", "b04_point_cloud")

# ---------------------------------------------------------------- the six points by hand
six = np.array([[0, 0], [1, 0], [0, 1], [4, 0], [5, 0], [4, 2]], float)
print("six points, Kruskal deaths:", kruskal_h0(six))
print("ripser H0:", ripser.ripser(six)["dgms"][0][:, 1])
sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
print("unit square, ours:", rips_bruteforce(sq)[1], " ripser:", ripser.ripser(sq)["dgms"][1])

# ---------------------------------------------------------------- noisy circle + a small cluster
n_c, n_k = 40, 12
th = rng.uniform(0, 2 * np.pi, n_c)
circle = np.column_stack([np.cos(th), np.sin(th)]) + 0.07 * rng.standard_normal((n_c, 2))
blob = np.array([2.6, 0.3]) + 0.12 * rng.standard_normal((n_k, 2))
X = np.vstack([circle, blob])

t0 = time.perf_counter()
k0 = kruskal_h0(X)
t_kr = time.perf_counter() - t0
sl = np.sort(linkage(X, method="single")[:, 2])            # single-linkage merge heights
rp = ripser.ripser(X, maxdim=1)["dgms"]
t0 = time.perf_counter()
bf = rips_bruteforce(X, maxdim=1)
t_bf = time.perf_counter() - t0
st = gudhi.RipsComplex(points=X, max_edge_length=4.0).create_simplex_tree(max_dimension=2)
st.compute_persistence()
gd1 = st.persistence_intervals_in_dimension(1)
print("Kruskal = single linkage:", np.allclose(np.sort(k0), sl))
print("Kruskal = ripser H0:", np.allclose(np.sort(k0), np.sort(rp[0][:-1, 1]), atol=1e-6))
dB_bf = persim.bottleneck(bf[1], rp[1])
# gudhi's Rips uses the edge length itself as filtration value (same as ripser)
dB_gd = persim.bottleneck(gd1, rp[1])
print("H1 ours vs ripser d_B =", dB_bf, " gudhi vs ripser d_B =", dB_gd)
loop = rp[1][np.argmax(rp[1][:, 1] - rp[1][:, 0])]
second = np.sort(rp[1][:, 1] - rp[1][:, 0])[-2] if len(rp[1]) > 1 else 0.0
gap = np.sort(k0)[-1]                                      # the edge that joins blob to circle

# ---------------------------------------------------------------- figure: Rips complex at three r
setup(7.4, 2.6)
fig, axes = plt.subplots(1, 3)
D = np.linalg.norm(X[:, None] - X[None], axis=-1)
for ax, r in zip(axes, [0.25, 0.6, 1.9]):
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if D[i, j] <= r:
                for k in range(j + 1, len(X)):
                    if D[i, k] <= r and D[j, k] <= r:
                        ax.add_patch(Polygon(X[[i, j, k]], closed=True, color=SERIES[0], alpha=0.06,
                                             lw=0))
                ax.plot(X[[i, j], 0], X[[i, j], 1], color=SERIES[0], lw=0.5)
    ax.plot(X[:, 0], X[:, 1], "o", ms=2.5, color="k")
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 3.2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(rf"$r={r}$", fontsize=9)
fig.tight_layout()
savefig(fig, "ch12", "b_rips")

setup(7.0, 2.8)
fig, ax = plt.subplots(1, 2)
y = np.arange(len(k0))
ax[0].hlines(y, 0, np.sort(k0)[::-1], color=SERIES[0], lw=1.2)
ax[0].hlines(len(k0), 0, 3.4, color=SERIES[0], lw=1.2)
for i, (b, d) in enumerate(rp[1][np.argsort(rp[1][:, 1] - rp[1][:, 0])]):
    ax[0].hlines(len(k0) + 2 + i, b, d, color=SERIES[1], lw=2.4)
ax[0].set_xlabel(r"scale $r$")
ax[0].set_yticks([])
ax[0].set_title(r"barcode ($H_0$ blue, $H_1$ orange)", fontsize=9)
ax[1].plot([0, 3.4], [0, 3.4], color=INK2, ls=":", lw=0.8)
ax[1].scatter(np.zeros_like(k0), k0, s=8, color=SERIES[0], label=r"$H_0$ (Kruskal)")
ax[1].scatter(rp[1][:, 0], rp[1][:, 1], s=10, color=SERIES[1], marker="s", label=r"$H_1$ (ripser)")
ax[1].scatter(bf[1][:, 0], bf[1][:, 1], s=40, facecolor="none", edgecolor="k", lw=0.6,
              label=r"$H_1$ (our reduction)")
ax[1].set_xlabel(r"birth $r$")
ax[1].set_ylabel(r"death $r$")
ax[1].legend(fontsize=7, loc="lower right")
fig.tight_layout()
savefig(fig, "ch12", "b_cloud_diagram")

# ---------------------------------------------------------------- timing
sizes = [100, 200, 400, 800, 1600]
tk, tr = [], []
for n in sizes:
    Y = rng.uniform(size=(n, 2))
    t0 = time.perf_counter()
    kruskal_h0(Y)
    tk.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    ripser.ripser(Y, maxdim=1)
    tr.append(time.perf_counter() - t0)
    print(f"n={n}: Kruskal {tk[-1]:.3f} s, ripser (H0+H1) {tr[-1]:.3f} s")

save_numbers("ch12", "b04_point_cloud", {
    "PbCloudNc": n_c, "PbCloudNk": n_k, "PbCloudN": len(X),
    "PbLoopBirth": loop[0], "PbLoopDeath": loop[1], "PbLoopSecond": second,
    "PbGapEdge": gap, "PbTimeKruskal": t_kr, "PbTimeBrute": t_bf,
    "PbdBbrute": dB_bf, "PbdBgudhi": dB_gd, "PbSimplices": sum(1 for _ in st.get_simplices()),
    "PbTimeKruskalBig": tk[-1], "PbTimeRipserBig": tr[-1], "PbCloudBig": sizes[-1],
})
