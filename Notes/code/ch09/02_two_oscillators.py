"""When the dynamics does NOT sample for us: two oscillators that never share their energy fairly.

Question: does one long trajectory of a mechanical system visit its energy shell the way the
microcanonical ensemble (all shell states equally likely) says it should?
Two unit oscillators (m = omega = 1) with a weak linear spring between them,
    H = (p1^2 + p2^2 + x1^2 + x2^2)/2 + kappa (x1 - x2)^2 / 2,
start with all the energy in oscillator 1.  We compare the fraction u = E1/E of the energy in
oscillator 1, E1 = (p1^2 + x1^2)/2,
  * along the trajectory (time average: histogram of u(t) at uniformly random instants), and
  * over the energy shell H = E (ensemble average: uniform points on the shell, drawn exactly
    because H is a sum of squares in scaled normal-mode coordinates).
Writes figures/ch09/two_oscillators.pdf and results/ch09/02_two_oscillators.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

KAPPA, E, N = 0.05, 1.0, 200_000
rng = rng_for("ch09", "02_two_oscillators")
# normal modes: q_+ = (x1 + x2)/sqrt2 with frequency 1, q_- = (x1 - x2)/sqrt2 with sqrt(1 + 2 kappa)
w_plus, w_minus = 1.0, np.sqrt(1.0 + 2.0 * KAPPA)


def energy_fraction(qp, pp, qm, pm):
    x1 = (qp + qm) / np.sqrt(2)
    p1 = (pp + pm) / np.sqrt(2)
    return 0.5 * (p1**2 + x1**2) / E


# --- time average: one trajectory, looked at at uniformly random instants ------------------
# start: x1 = A, x2 = 0, p1 = p2 = 0 ; then q_+(0) = q_-(0) = A/sqrt2, both modes at rest
A = np.sqrt(2 * E / (1 + KAPPA))                 # H(x1=A, x2=0) = A^2/2 + kappa A^2/2 = E
T_beat = 2 * np.pi / (w_minus - w_plus)          # energy sloshes back and forth with this period
t = rng.uniform(0, 200 * T_beat, size=N)
q0 = A / np.sqrt(2)
u_time = energy_fraction(q0 * np.cos(w_plus * t), -q0 * w_plus * np.sin(w_plus * t),
                         q0 * np.cos(w_minus * t), -q0 * w_minus * np.sin(w_minus * t))

# --- ensemble average: uniform on the shell H = E -------------------------------------------
# In scaled coordinates (w q, p) for each mode H = |z|^2 / 2 with z in R^4, so the shell is a
# 3-sphere and uniform points on it are normalised Gaussian 4-vectors (the Jacobian is constant).
z = rng.standard_normal((N, 4))
z *= np.sqrt(2 * E) / np.linalg.norm(z, axis=1, keepdims=True)
u_ens = energy_fraction(z[:, 0] / w_plus, z[:, 1], z[:, 2] / w_minus, z[:, 3])

# --- summaries --------------------------------------------------------------------------------
stats = {}
for name, u in (("Time", u_time), ("Ens", u_ens)):
    stats[f"NineATwoMean{name}"] = round(float(np.mean(u)), 3)
    stats[f"NineATwoSq{name}"] = round(float(np.mean(u**2)), 3)
    stats[f"NineATwoLow{name}"] = round(float(np.mean(u < 0.1)), 3)
stats["NineATwoKappa"] = KAPPA

setup(7.0, 3.0)
fig, (ax1, ax2) = plt.subplots(1, 2)
tt = np.linspace(0, 2 * T_beat, 4000)
ax1.plot(tt / T_beat, energy_fraction(q0 * np.cos(tt), -q0 * np.sin(tt), q0 * np.cos(w_minus * tt),
                                      -q0 * w_minus * np.sin(w_minus * tt)), color=SERIES[0], lw=0.8)
ax1.set_xlabel(r"time $t/T_{\rm beat}$")
ax1.set_ylabel(r"$u=E_1/E$")
ax1.set_title("(a) the energy sloshes back and forth")
bins = np.linspace(0, 1, 41)
ax2.hist(u_time, bins=bins, density=True, color=SERIES[0], alpha=0.55, label="time along the trajectory")
ax2.hist(u_ens, bins=bins, density=True, histtype="step", color=SERIES[1], lw=1.6, label="energy shell (ensemble)")
uu = np.linspace(0.003, 0.997, 400)
theory_line(ax2, uu, 1 / (np.pi * np.sqrt(uu * (1 - uu))), label=r"$1/\pi\sqrt{u(1-u)}$")
ax2.set_ylim(0, 4)
ax2.set_xlabel(r"$u=E_1/E$")
ax2.set_ylabel("density")
ax2.set_title("(b) time average $\\neq$ ensemble average")
ax2.legend(loc="upper center", fontsize=8)
fig.tight_layout()
savefig(fig, "ch09", "two_oscillators")
save_numbers("ch09", "02_two_oscillators", stats)
print(stats)
