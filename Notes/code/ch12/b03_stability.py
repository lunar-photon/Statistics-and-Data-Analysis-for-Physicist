"""b03_stability.py -- the stability theorem tested: noise moves the diagram by at most its size.

Question: if a field f is perturbed into g with max|f - g| = eps, how far apart are the two
persistence diagrams in the bottleneck distance, and where do the extra points of the noisy
diagram sit?
Computes: a smooth 128x128 Gaussian field f; g = f + white or smooth noise at several
amplitudes; the bottleneck distance d_B between their H0 (and H1) diagrams (persim) against
eps = max|f - g|; the persistence of the points of the noisy diagram; the same after
smoothing the noisy field.
Writes: figures/ch12/b_stability.pdf, results/ch12/b03_stability.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2
from lib_fields import gaussian_field
from lib_persist import islands, lakes, to_sublevel, finite
import persim
import gudhi

rng = rng_for("ch12", "b03_stability")
N, R = 128, 4.0
P = lambda k: np.exp(-(k * R) ** 2)


def clean_field():
    f = gaussian_field(P, N, float(N), d=2, rng=rng)
    return f / f.std()


def dgm(f):
    return to_sublevel(finite(islands(f))), to_sublevel(lakes(f))


# ---------------------------------------------------------------- d_B against eps
amps = [0.01, 0.03, 0.1, 0.2, 0.4]
rows = []
for real in range(6):
    f = clean_field()
    D0, D1 = dgm(f)
    for a in amps:
        for kind in ("white", "smooth"):
            n = rng.standard_normal((N, N))
            if kind == "smooth":
                n = ndimage.gaussian_filter(n, 1.5, mode="wrap")
                n /= n.std()
            g = f + a * n
            eps = np.abs(g - f).max()
            E0, E1 = dgm(g)
            rows.append((eps, gudhi.bottleneck_distance(D0, E0), gudhi.bottleneck_distance(D1, E1),
                         kind == "white"))
rows = np.array(rows)
ratio = np.maximum(rows[:, 1], rows[:, 2]) / rows[:, 0]
print("max d_B / eps over all trials:", ratio.max())
# persim (Python) on the smallest perturbation, as a check on gudhi's C++ bottleneck distance
f = clean_field()
g = f + 0.01 * rng.standard_normal((N, N))
check = abs(persim.bottleneck(dgm(f)[0], dgm(g)[0]) - gudhi.bottleneck_distance(dgm(f)[0], dgm(g)[0]))
print("persim - gudhi bottleneck:", check)

# ---------------------------------------------------------------- one example in detail
f = clean_field()
a = 0.2
g = f + a * rng.standard_normal((N, N))
eps = np.abs(g - f).max()
hf, hg = finite(islands(f)), finite(islands(g))
pf, pg = hf[:, 0] - hf[:, 1], hg[:, 0] - hg[:, 1]
n_short = int(np.sum(pg < 2 * eps))
gs = ndimage.gaussian_filter(g, 1.5)                       # smoothing the noisy field
hs = finite(islands(gs))
print(f"clean: {len(hf)} islands; noisy: {len(hg)}, of which {n_short} have persistence < 2 eps;"
      f" smoothed: {len(hs)}")
# the long bars of the clean field and their partners in the noisy field
long_f = np.sort(pf)[::-1][:10]
long_g = np.sort(pg)[::-1][:10]

setup(7.2, 2.9)
fig, ax = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.1]))
w = rows[:, 3] == 1
ax[0].loglog(rows[w, 0], rows[w, 1], "o", ms=3, color=SERIES[0], label=r"$H_0$, white noise")
ax[0].loglog(rows[~w, 0], rows[~w, 1], "^", ms=3, color=SERIES[2], label=r"$H_0$, smooth noise")
ax[0].loglog(rows[:, 0], rows[:, 2], "s", ms=2.5, mfc="none", color=SERIES[1], label=r"$H_1$")
x = np.array([rows[:, 0].min() * 0.8, rows[:, 0].max() * 1.2])
ax[0].loglog(x, x, "k--", lw=1, label=r"$d_B=\|f-g\|_\infty$")
ax[0].set_xlabel(r"size of the perturbation $\|f-g\|_\infty$")
ax[0].set_ylabel(r"bottleneck distance $d_B$")
ax[0].legend(fontsize=7)
lim = (-3.2, 4)
ax[1].fill_between([lim[0], lim[1]], [lim[0], lim[1]], [lim[0] - 2 * eps, lim[1] - 2 * eps],
                   color="#dddddd", lw=0, label=rf"band $b-d<2\epsilon$, $\epsilon={eps:.2f}$")
ax[1].plot(lim, lim, color=INK2, lw=0.8, ls=":")
ax[1].scatter(hg[:, 0], hg[:, 1], s=4, color=SERIES[1], label="noisy field", zorder=2)
ax[1].scatter(hf[:, 0], hf[:, 1], s=14, facecolor="none", edgecolor=SERIES[0], label="clean field",
              zorder=3)
ax[1].set_xlim(*lim)
ax[1].set_ylim(lim[0] - 0.5, lim[1])
ax[1].set_xlabel(r"birth $b$")
ax[1].set_ylabel(r"death $d$")
ax[1].legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch12", "b_stability")

save_numbers("ch12", "b03_stability", {
    "PbStabTrials": len(rows), "PbStabCheck": check, "PbStabMaxRatio": ratio.max(),
    "PbStabEps": eps, "PbStabAmp": a,
    "PbStabNclean": len(hf), "PbStabNnoisy": len(hg), "PbStabNshort": n_short,
    "PbStabNsmooth": len(hs),
    "PbStabLongClean": long_f[0], "PbStabLongNoisy": long_g[0],
    "PbStabTenthClean": long_f[9], "PbStabTenthNoisy": long_g[9],
})
