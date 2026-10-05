"""Quasi-random (Sobol) points against pseudorandom points for integration.

Question: for estimating an integral we do not need points that *look* random,
we need points that cover the domain evenly.  How much better do low-discrepancy
(Sobol) points do than independent uniform points, and for which integrands?

Computes: (1) the van der Corput sequence (radical inverse in base 2) for
n = 1..8; (2) 256 points in the unit square from PCG64 and from a scrambled
Sobol sequence, with the number of empty cells in a 16 x 16 grid; (3) the RMS
error over R = 40 independent randomisations of the estimate of
 I_smooth = int_[0,1]^5 prod_i (pi/2) sin(pi x_i) dx = 1   (smooth)
 I_ball   = volume of {|x - c| < 0.5} in [0,1]^5 = pi^(5/2)/Gamma(7/2) / 2^5  (a jump)
for N = 2^4 ... 2^16 points, with fitted slopes of log error against log N.
Writes: figures/ch06/sobol_points.pdf, figures/ch06/sobol_error.pdf,
        results/ch06/09_sobol.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.special import gamma


def van_der_corput(n, base=2):
    q, denom = 0.0, 1.0
    while n:
        n, r = divmod(n, base)
        denom *= base
        q += r / denom
    return q


vdc = [van_der_corput(n) for n in range(1, 9)]
print("van der Corput:", vdc)

rng = rng_for("ch06", "09_sobol")

# ---------------- (2) points in the square ----------------
P = rng.random((256, 2))
Q = qmc.Sobol(2, scramble=True, rng=rng).random(256)


def empty_cells(X, k=16):
    idx = np.minimum((X * k).astype(int), k - 1)
    return k * k - len(set(map(tuple, idx)))


eP, eQ = empty_cells(P), empty_cells(Q)
print("empty cells of 256: PCG64", eP, " Sobol", eQ, " expected for random", 256 * (1 - 1 / 256) ** 256)

setup(6.0, 3.0)
fig, axes = plt.subplots(1, 2)
for ax, X, t, c in [(axes[0], P, f"PCG64: {eP} empty cells", SERIES[0]),
                    (axes[1], Q, f"scrambled Sobol: {eQ} empty cells", SERIES[1])]:
    for g in np.linspace(0, 1, 17):
        ax.axhline(g, color="0.85", lw=0.5); ax.axvline(g, color="0.85", lw=0.5)
    ax.scatter(X[:, 0], X[:, 1], s=5, color=c, lw=0, zorder=3)
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.grid(False)
    ax.set_title(t, fontsize=8)
savefig(fig, "ch06", "sobol_points")

# ---------------- (3) integration error ----------------
d = 5
f_smooth = lambda X: np.prod(np.pi / 2 * np.sin(np.pi * X), axis=1)
f_ball = lambda X: (np.sum((X - 0.5) ** 2, axis=1) < 0.25).astype(float)
I_ball = np.pi ** (d / 2) / gamma(d / 2 + 1) * 0.5**d
ms = np.arange(4, 17)
Ns = 2**ms
R = 40
err = {k: np.zeros((2, len(ms))) for k in ("smooth", "ball")}
for r in range(R):
    sob = qmc.Sobol(d, scramble=True, rng=rng).random(Ns[-1])
    mc = rng.random((Ns[-1], d))
    for j, n in enumerate(Ns):
        for k, f, I in (("smooth", f_smooth, 1.0), ("ball", f_ball, I_ball)):
            err[k][0, j] += (f(mc[:n]).mean() - I) ** 2
            err[k][1, j] += (f(sob[:n]).mean() - I) ** 2
for k in err:
    err[k] = np.sqrt(err[k] / R)
slopes = {k: [np.polyfit(np.log(Ns[4:]), np.log(err[k][i, 4:]), 1)[0] for i in (0, 1)] for k in err}
print("slopes (MC, Sobol):", slopes)
gain = {k: err[k][0, -1] / err[k][1, -1] for k in err}
print("error ratio MC/Sobol at N=2^16:", gain)

setup(6.4, 3.0)
fig, axes = plt.subplots(1, 2, sharey=True)
for ax, k, t in [(axes[0], "smooth", r"smooth: $\prod_i (\pi/2)\sin \pi x_i$"),
                 (axes[1], "ball", r"with a jump: inside a ball")]:
    ax.loglog(Ns, err[k][0], "o-", ms=3, color=SERIES[0], label=f"PCG64 (slope {slopes[k][0]:.2f})")
    ax.loglog(Ns, err[k][1], "s-", ms=3, color=SERIES[1], label=f"scrambled Sobol (slope {slopes[k][1]:.2f})")
    ax.loglog(Ns, err[k][0][0] * (Ns / Ns[0]) ** -0.5, "k--", lw=1, label=r"$\propto N^{-1/2}$")
    ax.loglog(Ns, err[k][1][0] * (Ns / Ns[0]) ** -1.0, "k:", lw=1, label=r"$\propto N^{-1}$")
    ax.set_title(t, fontsize=8)
    ax.set_xlabel("number of points $N$")
    ax.legend(fontsize=7, loc="lower left")
axes[0].set_ylabel("RMS error, $d=5$")
savefig(fig, "ch06", "sobol_error")

save_numbers("ch06", "09_sobol", {
    "SixAEmptyPcg": eP, "SixAEmptySobol": eQ, "SixAEmptyExpected": f"{256*(1-1/256)**256:.0f}",
    "SixASlopeSmoothMc": f"{slopes['smooth'][0]:.2f}", "SixASlopeSmoothSob": f"{slopes['smooth'][1]:.2f}",
    "SixASlopeBallMc": f"{slopes['ball'][0]:.2f}", "SixASlopeBallSob": f"{slopes['ball'][1]:.2f}",
    "SixAGainSmooth": f"{gain['smooth']:.0f}", "SixAGainBall": f"{gain['ball']:.1f}",
    "SixASobolR": R, "SixABallVol": f"{I_ball:.4f}",
})
