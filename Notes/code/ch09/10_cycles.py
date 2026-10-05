"""Stationary is not the same as convergent, and convergent does not need detailed balance.

Question: two chains on three states arranged in a ring, both with the uniform distribution
(1/3, 1/3, 1/3) as their stationary distribution.
  (i)  the deterministic cycle 1 -> 2 -> 3 -> 1;
  (ii) a biased cycle: one step clockwise with probability 0.6, anticlockwise 0.2, stay 0.2.
Started in state 1, does p_t converge to the uniform distribution?  What probability current
flows around the ring?
Writes figures/ch09/cycles.pdf and results/ch09/10_cycles.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
from lib_markov import stationary, evolve, tv

C = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
B = np.array([[0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.6, 0.2, 0.2]])
T = 20
pC = evolve(np.array([1.0, 0, 0]), C, T)
pB = evolve(np.array([1.0, 0, 0]), B, T)
piB = stationary(B)
J = piB[0] * B[0, 1] - piB[1] * B[1, 0]          # net clockwise current on the edge 1 -> 2
lamB = np.linalg.eigvals(B)
lam2 = np.sort(np.abs(lamB))[-2]

setup(7.0, 2.9)
fig, (ax1, ax2) = plt.subplots(1, 2)
t = np.arange(T + 1)
ax1.plot(t, pC[:, 0], "-o", ms=3, color=SERIES[0], label="deterministic cycle")
ax1.plot(t, pB[:, 0], "-o", ms=3, color=SERIES[1], label="biased cycle")
ax1.axhline(1 / 3, color="k", ls="--", lw=1.2)
ax1.set_xlabel("step $t$")
ax1.set_ylabel(r"$p_t(1)$")
ax1.set_title("(a) probability of state 1")
ax1.set_ylim(-0.05, 1.3)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.legend(fontsize=7, loc="upper center", ncol=2)
ax2.semilogy(t, np.maximum(tv(pB, piB), 1e-17), "-o", ms=3, color=SERIES[1], label="biased cycle")
ax2.semilogy(t, tv(pB[0], piB) * lam2**t, "--", color="k", lw=1.2, label=fr"$|\lambda_2|^t$, $|\lambda_2|={lam2:.3f}$")
ax2.semilogy(t, tv(pC, 1 / 3 * np.ones(3)), "-", color=SERIES[0], label="deterministic cycle")
ax2.set_ylim(1e-9, 2)
ax2.set_xlabel("step $t$")
ax2.set_ylabel(r"$\|p_t-\pi\|_{\rm TV}$")
ax2.set_title("(b) distance to the uniform $\\pi$")
ax2.legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "ch09", "cycles")

save_numbers("ch09", "10_cycles", {
    "NineACycJ": f"{J:.4f}", "NineACycLam": round(float(lam2), 4),
    "NineACycTVTen": f"{tv(pB[10], piB):.1e}".replace("e-0", r"\times10^{-") + "}",
})
print("pi", piB, "J", J, "eig", lamB, "TV10", tv(pB[10], piB))
