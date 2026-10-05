"""What does a passing gravitational wave do to a ring of free test masses?

Question: a plane wave travels along z through a ring of free masses in the x-y plane. Where is
each mass, measured by proper distance from the centre, at each phase of the wave?
Computes: the first-order displacement xi = xi0 + (1/2) h xi0 for a pure h_+ and a pure h_x
wave at five phases (amplitude exaggerated to 0.35 so the picture shows it), and checks that the
h_x pattern is the h_+ pattern turned by 45 degrees.
Writes: figures/chT2/ring_masses.pdf, results/chT2/12_ring_masses.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES

setup(7.2, 3.3)


def h_matrix(hp, hx):
    """The 2x2 transverse block of the TT strain for a wave along z."""
    return np.array([[hp, hx], [hx, -hp]])


# ---- (1) the picture: 16 masses on a unit ring, amplitude 0.35 (exaggerated)
n_mass = 16
theta = 2 * np.pi * np.arange(n_mass) / n_mass
ring0 = np.stack([np.cos(theta), np.sin(theta)])          # shape (2, n_mass)
phases = np.array([0, 0.5, 1.0, 1.5, 2.0]) * np.pi
labels = [r"$\omega t=0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
h0_pic = 0.35
circ = np.linspace(0, 2 * np.pi, 200)

fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, gridspec_kw=dict(hspace=0.05, wspace=0.05))
for row, (name, colour) in enumerate([("plus", SERIES[0]), ("cross", SERIES[1])]):
    for col, ph in enumerate(phases):
        h = h0_pic * np.cos(ph)
        H = h_matrix(h, 0.0) if name == "plus" else h_matrix(0.0, h)
        ring = ring0 + 0.5 * H @ ring0                       # xi = xi0 + (1/2) h xi0
        ax = axes[row, col]
        ax.plot(np.cos(circ), np.sin(circ), color="0.75", lw=0.8, ls="--")
        ax.plot(ring[0], ring[1], "o", color=colour, ms=3.2)
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if row == 0:
            ax.set_title(labels[col])
    axes[row, 0].set_ylabel(r"$h_+$ only" if name == "plus" else r"$h_\times$ only")
savefig(fig, "chT2", "ring_masses")

# ---- (2) one number for the text: the two polarisations are the same pattern turned by 45 degrees
rot = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
h = 0.01
plus_turned = rot @ (0.5 * h_matrix(h, 0.0)) @ rot.T     # the h_+ strain pattern rotated by 45 degrees
cross = 0.5 * h_matrix(0.0, h)
diff = np.max(np.abs(plus_turned - cross))
print(f"max |R(45) h_+ R^T - h_x| = {diff:.1e}")
save_numbers("chT2", "12_ring_masses", {"TwRingAmp": h0_pic})
