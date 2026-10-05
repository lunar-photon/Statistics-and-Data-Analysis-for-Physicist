"""05_web_genus.py -- the genus curve of a three-dimensional Gaussian density field.

Question: is a Gaussian density field "sponge-like" at the median density, as Hamilton, Gott &
Weinberg (1986) predicted, and does its genus per unit volume follow
    g(nu) = (1/4 pi^2) (<k^2>/3)^{3/2} (1 - nu^2) exp(-nu^2/2) ?

Computes: 50 periodic boxes, 128^3 cells of side 2 Mpc/h, power spectrum P(k) ~ k^{-1.5} smoothed
by a Gaussian of radius 6 Mpc/h (a rough stand-in for the galaxy-scale matter spectrum); the Euler
characteristic of each excursion set {delta > nu sigma} by counting vertices, edges, faces and
cubes of the voxel complex on the 3-torus; genus = -chi of the set; the formula from the moments.
Cross-check of the counter with scikit-image on one box.
Writes: figures/chT3/web_genus.pdf, results/chT3/05_web_genus.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_fields import gaussian_field, kgrid, rfft_weights

setup()
rng = rng_for("chT3", "05_web_genus")
N, Lbox, R, NS = 128, 256.0, 6.0, 50
P = lambda k: k ** -1.5 * np.exp(-(k * R) ** 2)          # smoothed spectrum (arbitrary units)
nus = np.linspace(-3, 3, 25)


def euler_3torus(m):
    """chi = V - E + F - C of the union of closed voxels in m, on the periodic grid."""
    r = lambda a, s: np.roll(a, 1, axis=s)
    C = m.sum()
    F = sum((m | r(m, ax)).sum() for ax in range(3))                       # faces between neighbours
    E = 0
    for a1, a2 in [(0, 1), (0, 2), (1, 2)]:                                 # edges shared by 4 voxels
        E += (m | r(m, a1) | r(m, a2) | r(r(m, a1), a2)).sum()
    V = m.copy()
    for ax in range(3):
        V = V | r(V, ax)                                                    # vertex shared by 8 voxels
    return int(V.sum() - E + F - C)


# moments from the grid sums: sigma^2 = <delta^2>, <k^2> = <|grad delta|^2>/<delta^2>
k, _ = kgrid(N, Lbox, 3)
w = rfft_weights(N, 3)
Pk = np.where(k > 0, P(np.where(k > 0, k, 1.0)), 0.0)
k2mean = (w * Pk * k ** 2).sum() / (w * Pk).sum()
vol = Lbox ** 3
G = []
for s in range(NS):
    d = gaussian_field(P, N, Lbox, d=3, rng=rng)
    d /= d.std()
    G.append([-euler_3torus(d > n) / vol for n in nus])
G = np.array(G)
th = (1 / (4 * np.pi ** 2)) * (k2mean / 3) ** 1.5 * (1 - nus ** 2) * np.exp(-nus ** 2 / 2)

from skimage.measure import euler_number
pad = np.zeros((N + 2,) * 3, bool)
pad[1:-1, 1:-1, 1:-1] = d > 1.5
ours, sk = euler_3torus(pad), euler_number(pad, connectivity=3)

fig, ax = plt.subplots(figsize=(5.2, 3.3))
m, e = G.mean(0) * 1e5, G.std(0, ddof=1) * 1e5
ax.fill_between(nus, m - e, m + e, color=SERIES[0], alpha=0.3, lw=0, label=rf"{NS} boxes, $\pm1\sigma$")
ax.plot(nus, m, color=SERIES[0], label="mean")
theory_line(ax, nus, th * 1e5, label="Gaussian genus formula")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(r"threshold $\nu=\delta/\sigma$"); ax.set_ylabel(r"genus per volume [$10^{-5}\,h^3$Mpc$^{-3}$]")
ax.legend(fontsize=8)
savefig(fig, "chT3", "web_genus")

i0 = np.argmin(np.abs(nus))
save_numbers("chT3", "05_web_genus", {
    "TeNs": NS, "TeN": N, "TeCell": f"{Lbox / N:.0f}", "TeR": f"{R:.0f}",
    "TeGzeroSim": f"{G.mean(0)[i0] * vol:.0f}", "TeGzeroTh": f"{th[i0] * vol:.0f}",
    "TeDevPct": f"{100 * np.max(np.abs(G.mean(0) - th)) / th.max():.1f}",
    "TeErrPct": f"{100 * np.max(G.std(0, ddof=1)) / np.sqrt(NS) / th.max():.1f}",
    "TeEulerOurs": ours, "TeEulerSkimage": int(sk),
})
print(f"genus at nu=0 per box: sim {G.mean(0)[i0]*vol:.0f}, theory {th[i0]*vol:.0f}; max dev "
      f"{100*np.max(np.abs(G.mean(0)-th))/th.max():.1f}%; euler ours {ours} skimage {sk}")
