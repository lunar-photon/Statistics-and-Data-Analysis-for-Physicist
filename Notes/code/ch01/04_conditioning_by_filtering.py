"""Conditioning is selecting a subset of the simulated events.

Question: how do we estimate P(A|B) from a simulation, and why does the
estimate get noisier than the estimate of P(A)?

Computes: two fair dice, A = "sum is 8", B = "at least one die shows 6".
P(A|B) is estimated by keeping only the rows where B happened and asking what
fraction of those also have A; the same for P(B|A). Exact values: P(A|B)=2/11,
P(B|A)=2/5. The running estimates are plotted against the total number of
simulated rolls N, together with the +-2 sigma band computed from the
*surviving* sample size N P(B).
Writes: figures/ch01/1a_conditioning.pdf, results/ch01/1a_conditioning.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "1a_conditioning")
N = 200_000
d1 = rng.integers(1, 7, size=N)
d2 = rng.integers(1, 7, size=N)
A = (d1 + d2 == 8)
B = (d1 == 6) | (d2 == 6)

# --- conditioning = filtering: keep only the rows in B, then count A among them
A_given_B = A[B].mean()
B_given_A = B[A].mean()
# the same thing written as the ratio of frequencies, P(AB)/P(B)
ratio = (A & B).mean() / B.mean()

exact_AgB, exact_BgA = 2 / 11, 2 / 5
exact_A, exact_B = 5 / 36, 11 / 36

# running estimates as the simulation grows
n = np.arange(1, N + 1)
cumB = np.cumsum(B)
cumAB = np.cumsum(A & B)
cumA = np.cumsum(A)
with np.errstate(invalid="ignore", divide="ignore"):
    run_AgB = cumAB / cumB
    run_A = cumA / n

# plot on a log-spaced subset of N so the vector figure stays small
idx = np.unique(np.logspace(0, np.log10(N), 1500).astype(int)) - 1
n, run_A, run_AgB = n[idx], run_A[idx], run_AgB[idx]

setup(6.4, 3.6)
fig, ax = plt.subplots()
ax.plot(n, run_A, color=SERIES[0], lw=1.0, label=r"estimate of $P(A)$ (all rolls)")
ax.plot(n, run_AgB, color=SERIES[1], lw=1.0, label=r"estimate of $P(A\mid B)$ (rolls in $B$)")
theory_line(ax, n, np.full(n.size, exact_A), label="exact values")
ax.plot(n, np.full(n.size, exact_AgB), color="k", ls="--", lw=1.4, zorder=5)
bandB = 2 * np.sqrt(exact_AgB * (1 - exact_AgB) / (n * exact_B))
bandA = 2 * np.sqrt(exact_A * (1 - exact_A) / n)
ax.fill_between(n, exact_AgB - bandB, exact_AgB + bandB, color=SERIES[1], alpha=0.15, lw=0)
ax.fill_between(n, exact_A - bandA, exact_A + bandA, color=SERIES[0], alpha=0.15, lw=0)
ax.set_xscale("log")
ax.set_xlim(10, N)
ax.set_ylim(0, 0.45)
ax.set_xlabel("total number of simulated rolls $N$")
ax.set_ylabel("estimated probability")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
savefig(fig, "ch01", "1a_conditioning")

save_numbers("ch01", "1a_conditioning", {
    "OneACondN": N,
    "OneACondNB": int(B.sum()),
    "OneACondNA": int(A.sum()),
    "OneACondNAB": int((A & B).sum()),
    "OneACondAgB": f"{A_given_B:.4f}",
    "OneACondBgA": f"{B_given_A:.4f}",
    "OneACondRatio": f"{ratio:.4f}",
    "OneACondExAgB": f"{exact_AgB:.4f}",
    "OneACondExBgA": f"{exact_BgA:.4f}",
})
print(A_given_B, B_given_A, ratio, B.sum(), A.sum())
