"""RANDU's triples lie on 15 planes (Marsaglia 1968), reproduced.

Question: consecutive triples (u_k, u_{k+1}, u_{k+2}) of a good generator fill
the unit cube.  Where do RANDU's triples go?

Computes: 30000 consecutive triples from RANDU (seed 1) and from PCG64; the
integer k = 9u_k - 6u_{k+1} + u_{k+2} for every RANDU triple (it takes only the
15 values -5..9); the 3-D scatter seen from a generic angle and from an angle
that looks along the planes.
Writes: figures/ch06/randu_planes.pdf, results/ch06/04_randu_planes.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_prng import randu

n = 30_000
u = randu(1, n + 2)
T = np.column_stack([u[:-2], u[1:-1], u[2:]])            # overlapping triples
k = 9 * T[:, 0] - 6 * T[:, 1] + T[:, 2]
kint = np.rint(k).astype(int)
assert np.all(k == kint), "9u1 - 6u2 + u3 is an exact integer for RANDU"
vals, counts = np.unique(kint, return_counts=True)
print("plane labels:", vals, "counts:", counts)

rng = rng_for("ch06", "04_randu_planes")
P = rng.random((n, 3))

setup(6.6, 2.5)
fig = plt.figure()
# viewing direction (1, 1.5, 0) is perpendicular to the normal (9, -6, 1): planes seen edge-on
az_edge = np.degrees(np.arctan2(1.5, 1.0))
panels = [(P, 30, -60, "PCG64"), (T, 30, -60, "RANDU, generic view"),
          (T, 0, az_edge, "RANDU, looking along the planes")]
for j, (X, el, az, title) in enumerate(panels):
    ax = fig.add_subplot(1, 3, j + 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=0.15, color=SERIES[0 if j == 0 else 1],
               lw=0, depthshade=False)
    ax.view_init(elev=el, azim=az)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=8)
savefig(fig, "ch06", "randu_planes")

save_numbers("ch06", "04_randu_planes", {
    "SixARanduNplanes": len(vals), "SixARanduKmin": int(vals.min()), "SixARanduKmax": int(vals.max()),
    "SixARanduAzEdge": f"{az_edge:.1f}",
})
