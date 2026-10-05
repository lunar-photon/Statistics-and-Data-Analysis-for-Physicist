"""Two relatives of Metropolis on the correlated Gaussian of script 14 (rho = 0.95).

Question 1 (Gibbs): drawing each coordinate in turn from its exact conditional,
theta_1 | theta_2 ~ N(rho theta_2, 1 - rho^2) and vice versa, every proposal is accepted.
How correlated is the chain?  Theory: theta_1 is then an AR(1) chain with coefficient rho^2, so
tau = (1 + rho^2) / (1 - rho^2).
Question 2 (Hamiltonian Monte Carlo): give the point a momentum p ~ N(0, 1), follow Hamilton's
equations for H = -ln p(theta) + |p|^2/2 with the leapfrog integrator, accept with
min(1, exp(-dH)).  How long are the moves, how often are they accepted, and what is tau?
Writes figures/ch09/gibbs_hmc.pdf and results/ch09/21_gibbs_hmc.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES
from lib_mcmc import tau_int

rng = rng_for("ch09", "21_gibbs_hmc")
rho = 0.95
C = np.array([[1.0, rho], [rho, 1.0]])
Ci = np.linalg.inv(C)
U = lambda th: 0.5 * th @ Ci @ th             # potential energy = -ln p(theta) + const
gradU = lambda th: Ci @ th


def gibbs(n, th0, rng):
    th = np.array(th0, float)
    out = np.empty((n + 1, 2)); out[0] = th
    path = [th.copy()]                          # every half-step, to draw the staircase
    sd = np.sqrt(1 - rho**2)
    for t in range(n):
        th[0] = rho * th[1] + sd * rng.standard_normal()     # draw theta_1 | theta_2
        path.append(th.copy())
        th[1] = rho * th[0] + sd * rng.standard_normal()     # draw theta_2 | theta_1
        path.append(th.copy())
        out[t + 1] = th
    return out, np.array(path)


def hmc(n, th0, eps, n_leap, rng):
    th = np.array(th0, float)
    out = np.empty((n + 1, 2)); out[0] = th
    acc, dHs = 0, []
    for t in range(n):
        p = rng.standard_normal(2)                           # fresh momentum: a canonical draw
        H0 = U(th) + 0.5 * p @ p
        q, pp = th.copy(), p.copy()
        pp -= 0.5 * eps * gradU(q)                           # leapfrog: half kick,
        for k in range(n_leap):
            q += eps * pp                                    # drift,
            if k < n_leap - 1:
                pp -= eps * gradU(q)                         # full kicks,
        pp -= 0.5 * eps * gradU(q)                           # final half kick
        dH = U(q) + 0.5 * pp @ pp - H0
        dHs.append(dH)
        if np.log(rng.random()) < -dH:                       # Metropolis on the total energy
            th = q; acc += 1
        out[t + 1] = th
    return out, acc / n, np.array(dHs)


n = 50_000
g, gpath = gibbs(n, [2.5, -2.5], rng)
eps, n_leap = 0.15, 15
h, acc_h, dH = hmc(n, [2.5, -2.5], eps, n_leap, rng)
burn = 500
nums = {"NineBGhRho": rho, "NineBGhN": f"{n:,}".replace(",", r"\,"),
        "NineBGhGibbsTau": tau_int(g[burn:, 0])[0],
        "NineBGhGibbsTauExact": (1 + rho**2) / (1 - rho**2),
        "NineBGhHmcTau": max(tau_int(h[burn:, i])[0] for i in range(2)),
        "NineBGhHmcAcc": acc_h, "NineBGhEps": eps, "NineBGhLeap": n_leap,
        "NineBGhHmcJump": np.mean(np.linalg.norm(np.diff(h[burn:], axis=0), axis=1)),
        "NineBGhHmcDH": np.mean(np.abs(dH)),
        "NineBGhGibbsVar": g[burn:, 0].var(), "NineBGhHmcVar": h[burn:, 0].var()}
save_numbers("ch09", "21_gibbs_hmc", nums)

setup(7.2, 2.9)
fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.3]})
gg = np.linspace(-3.5, 3.5, 200)
G1, G2 = np.meshgrid(gg, gg)
Z = np.exp(-0.5 * (Ci[0, 0] * G1**2 + 2 * Ci[0, 1] * G1 * G2 + Ci[1, 1] * G2**2))
for ax, title in zip(axes[:2], ("(a) Gibbs: 40 sweeps", "(b) HMC: 40 trajectories")):
    ax.contour(G1, G2, Z, levels=[np.exp(-2), np.exp(-0.5)], colors="0.6", linewidths=0.8)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$"); ax.set_title(title, fontsize=9)
axes[0].plot(gpath[:81, 0], gpath[:81, 1], "-", color=SERIES[0], lw=0.7)
axes[0].plot(g[:41, 0], g[:41, 1], "o", color=SERIES[0], ms=2)
axes[0].set_ylabel(r"$\theta_2$")
axes[1].plot(h[:41, 0], h[:41, 1], "-o", color=SERIES[1], lw=0.7, ms=2)
ax = axes[2]
ax.plot(g[burn:burn + 1500, 0], color=SERIES[0], lw=0.5, label="Gibbs")
ax.plot(h[burn:burn + 1500, 0] - 7, color=SERIES[1], lw=0.5, label="HMC (shifted by $-7$)")
ax.set_xlabel("step"); ax.set_ylabel(r"$\theta_1$")
ax.set_title("(c) traces", fontsize=9, loc="left")
ax.legend(fontsize=7, loc="upper right")
ax.set_ylim(-11.5, 5.5)
fig.tight_layout()
savefig(fig, "ch09", "gibbs_hmc")
