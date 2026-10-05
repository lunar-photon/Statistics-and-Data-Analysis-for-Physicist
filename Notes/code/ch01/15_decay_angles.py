"""Isotropic two-body decays: flat cos(theta), sin(theta)/2 in theta, and the Jacobian peak in p_T.

Question: a particle at rest decays isotropically into two daughters of momentum p* each.
'Isotropic' means the daughter direction is uniform on the sphere.  Which variables are flat,
and which are not?  Change of variables predicts
    cos(theta) ~ Uniform(-1, 1),   p(theta) = sin(theta)/2,   p_z = p* cos(theta) ~ Uniform(-p*, p*),
    p(p_T) = p_T / (p* sqrt(p*^2 - p_T^2)),   0 < p_T < p*      (the Jacobian peak at p_T = p*),
where p_T = p* sin(theta) is the momentum transverse to the beam (z) axis.
Physics: W -> e nu at rest has p* = m_W / 2, so the electron p_T spectrum has an edge at ~40 GeV.
Writes: figures/ch01/decay_angles.pdf, results/ch01/15_decay_angles.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "15_decay_angles")
setup(7.0, 3.0)

m_W = 80.4                       # GeV
pstar = m_W / 2
N = 400_000
# uniform direction on the sphere: normalise a 3-D standard Gaussian vector
v = rng.normal(size=(N, 3))
v /= np.linalg.norm(v, axis=1, keepdims=True)
cos_t = v[:, 2]
theta = np.arccos(cos_t)
pT = pstar * np.sqrt(v[:, 0]**2 + v[:, 1]**2)

fig, axes = plt.subplots(1, 3)
ax = axes[0]
ax.hist(cos_t, bins=40, range=(-1, 1), density=True, color=SERIES[0], alpha=0.6)
theory_line(ax, np.array([-1, 1]), np.array([0.5, 0.5]), label="$1/2$")
ax.set_ylim(0, 0.8)
ax.set_xlabel("$\\cos\\theta$"); ax.set_ylabel("density")
ax.legend(fontsize=7, loc="upper right")

ax = axes[1]
th = np.linspace(0, np.pi, 300)
ax.hist(theta, bins=40, range=(0, np.pi), density=True, color=SERIES[1], alpha=0.6)
theory_line(ax, th, 0.5 * np.sin(th), label="$\\sin\\theta/2$")
ax.set_xlabel("$\\theta$ [rad]")
ax.legend(fontsize=7, loc="upper right")

ax = axes[2]
pp = np.linspace(0.2, pstar * 0.9995, 800)
ax.hist(pT, bins=80, range=(0, pstar), density=True, color=SERIES[2], alpha=0.6)
theory_line(ax, pp, pp / (pstar * np.sqrt(pstar**2 - pp**2)), label="Jacobian")
ax.set_ylim(0, 0.2)
ax.set_xlabel("$p_T$ [GeV]")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch01", "decay_angles")

save_numbers("ch01", "15_decay_angles", {
    "OneBDecN": "4\\times10^{5}",
    "OneBDecPstar": f"{pstar:.1f}",
    "OneBDecMeanPtSim": f"{pT.mean():.3f}",
    "OneBDecMeanPtExact": f"{np.pi * pstar / 4:.3f}",
    "OneBDecFracSim": f"{np.mean(pT > 0.9 * pstar):.4f}",
    "OneBDecFracExact": f"{np.sqrt(1 - 0.81):.4f}",
    "OneBDecMeanCos": f"{cos_t.mean():.4f}",
})
