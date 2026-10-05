"""02_grid_counting.py -- counting components, holes and chi on a pixel grid, three ways.

Question: do the from-scratch counts (scipy.ndimage.label for beta0 and for the holes, and
V - E + F of the cubical complex for chi) agree with the standard libraries
(scikit-image euler_number, gudhi CubicalComplex), and how much does the choice of pixel
connectivity matter?

Computes: (1) the 5x5 hand example; (2) all methods on 200 random fields x 9 thresholds;
          (3) the 8- versus 4-connectivity difference as a function of smoothing scale.
Writes:   figures/ch12/grid_components.pdf, results/ch12/02_grid_counting.tex
"""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import euler_number
import gudhi
from common import setup, savefig, save_numbers, rng_for, SERIES
from lib_fields import gaussian_field
from lib_topo import betti_plane, euler_cubical, EIGHT

setup()
rng = rng_for("ch12", "02_grid_counting")
nums = {}

# ---------------------------------------------------------------- (1) the 5x5 hand example
F5 = np.array([[2, 2, 2, 0, 0],
               [2, 0, 2, 0, 1],
               [2, 2, 3, 0, 0],
               [0, 0, 0, 1, 0],
               [1, 0, 0, 0, 0]])
for nu, tag in [(0.5, "A"), (1.5, "B"), (2.5, "C")]:
    m = F5 > nu
    b0, b1 = betti_plane(m, 8)
    c0, c1 = betti_plane(m, 4)
    print(f"5x5 nu={nu}: 8-conn b0={b0} b1={b1} chi={euler_cubical(m, 8)} | "
          f"4-conn b0={c0} b1={c1} chi={euler_cubical(m, 4)}")
    nums.update({f"GridHand{tag}Bzero": b0, f"GridHand{tag}Bone": b1, f"GridHand{tag}Chi": b0 - b1,
                 f"GridHand{tag}FourChi": c0 - c1})
    assert b0 - b1 == euler_cubical(m, 8) and c0 - c1 == euler_cubical(m, 4)

# ---------------------------------------------------------------- (2) three methods, many fields
N, R, NSIM = 128, 4.0, 200
nus = np.linspace(-2, 2, 9)
fields = gaussian_field(lambda k: np.exp(-(k * R) ** 2), N, N, 2, rng, nsim=NSIM)
mism = {"betti": 0, "skimage": 0, "gudhi": 0}
tim = {"label": 0.0, "cells": 0.0, "skimage": 0.0, "gudhi": 0.0}
for f in fields:
    u = (f - f.mean()) / f.std()
    t = time.perf_counter()
    cc = gudhi.CubicalComplex(top_dimensional_cells=-u)     # sublevel sets of -u = superlevel sets of u
    cc.compute_persistence()
    tim["gudhi"] += time.perf_counter() - t
    for nu in nus:
        m = u > nu
        t = time.perf_counter(); b0, b1 = betti_plane(m); tim["label"] += time.perf_counter() - t
        t = time.perf_counter(); chi = euler_cubical(m, 8); tim["cells"] += time.perf_counter() - t
        t = time.perf_counter(); sk = euler_number(m, connectivity=2); tim["skimage"] += time.perf_counter() - t
        g = cc.persistent_betti_numbers(-nu - 1e-12, -nu - 1e-12)
        mism["betti"] += int(b0 - b1 != chi)
        mism["skimage"] += int(sk != chi)
        mism["gudhi"] += int(g[0] - g[1] != chi or g[0] != b0 or g[1] != b1)
ncase = NSIM * len(nus)
print("cases", ncase, "mismatches", mism)
nums.update({"GridNcase": ncase, "GridNsim": NSIM, "GridN": N, "GridR": int(R),
             "GridMisBetti": mism["betti"], "GridMisSk": mism["skimage"], "GridMisGudhi": mism["gudhi"]})
for k, v in tim.items():
    nums[f"GridTime{k.capitalize()}"] = round(1e3 * v / (ncase if k != "gudhi" else NSIM), 2)
print("ms per call", {k: nums[f'GridTime{k.capitalize()}'] for k in tim})

# ---------------------------------------------------------------- (3) does the connectivity convention matter?
for R2 in (1.0, 2.0, 4.0, 8.0):
    fs = gaussian_field(lambda k: np.exp(-(k * R2) ** 2), N, N, 2, rng, nsim=50)
    d, tot = 0, 0
    for f in fs:
        u = (f - f.mean()) / f.std()
        for nu in nus:
            m = u > nu
            d += abs(euler_cubical(m, 8) - euler_cubical(m, 4))
            tot += abs(euler_cubical(m, 8))
    print(f"R={R2}: mean |chi8 - chi4| / mean |chi8| = {d / tot:.3f}")
    nums[f"GridConnDiff{['One','Two','Four','Eight'][[1.0,2.0,4.0,8.0].index(R2)]}"] = round(100 * d / tot, 1)

save_numbers("ch12", "02_grid_counting", nums)

# ---------------------------------------------------------------- figure: components and holes of one small map
f = gaussian_field(lambda k: np.exp(-(k * 3.0) ** 2), 64, 64, 2, rng)
u = (f - f.mean()) / f.std()
m = u > 0.3
lab, n = ndimage.label(m, structure=EIGHT)
hol, nh = ndimage.label(~np.pad(m, 1), structure=ndimage.generate_binary_structure(2, 1))
hol = hol[1:-1, 1:-1]
outside = hol[0, 0] if hol[0, 0] > 0 else None
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
axs[0].imshow(u, cmap="RdBu_r", origin="lower", vmin=-3, vmax=3)
axs[0].contour(u, levels=[0.3], colors="k", linewidths=0.8)
axs[0].set_title(r"field $u=\phi/\sigma$ and the contour $u=0.3$")
img = np.ones(m.shape + (3,))
cols = [np.array(plt.matplotlib.colors.to_rgb(c)) for c in SERIES]
for i in range(1, n + 1):
    img[lab == i] = cols[(i - 1) % len(cols)] * 0.75 + 0.25
pad = np.pad(m, 1)
hl, _ = ndimage.label(~pad, structure=ndimage.generate_binary_structure(2, 1))
outer = hl[0, 0]
holes = (hl[1:-1, 1:-1] > 0) & (hl[1:-1, 1:-1] != outer)
img[holes] = [0.1, 0.1, 0.1]
b0, b1 = betti_plane(m)
axs[1].imshow(img, origin="lower", interpolation="nearest")
axs[1].set_title(rf"components coloured ($\beta_0={b0}$), holes black ($\beta_1={b1}$)")
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
savefig(fig, "ch12", "grid_components")
save_numbers("ch12", "02_grid_counting_fig", {"GridFigBzero": b0, "GridFigBone": b1, "GridFigChi": b0 - b1})
