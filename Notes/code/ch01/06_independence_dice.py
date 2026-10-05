"""Independence checked by simulation (AoS Ch.1, exercise 23).

Question: for a fair die, A = {2,4,6} and B = {1,2,3,4} are independent,
P(AB) = P(A)P(B). A = {2,4,6} and C = {4,5,6} are not. Do the simulated
frequencies show this, and how large is the random scatter of the difference
Phat(AB) - Phat(A)Phat(B) when the events ARE independent?

Computes: frequencies from N rolls; the product test for both pairs; and the
distribution of the difference over many repeated simulations.
Writes: figures/ch01/1a_independence.pdf, results/ch01/1a_independence.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "1a_independence")
N = 10_000
roll = rng.integers(1, 7, size=N)
A = np.isin(roll, [2, 4, 6])
B = np.isin(roll, [1, 2, 3, 4])
C = np.isin(roll, [4, 5, 6])

fA, fB, fC = A.mean(), B.mean(), C.mean()
fAB, fAC = (A & B).mean(), (A & C).mean()

# repeat the whole N-roll experiment many times to see the scatter of the test
reps = 4000
rolls = rng.integers(1, 7, size=(reps, N), dtype=np.int8)
Ar, Br, Cr = np.isin(rolls, [2, 4, 6]), np.isin(rolls, [1, 2, 3, 4]), np.isin(rolls, [4, 5, 6])
diff_AB = (Ar & Br).mean(1) - Ar.mean(1) * Br.mean(1)
diff_AC = (Ar & Cr).mean(1) - Ar.mean(1) * Cr.mean(1)

setup(6.4, 3.4)
fig, ax = plt.subplots()
bins = np.linspace(-0.02, 0.10, 121)
ax.hist(diff_AB, bins=bins, color=SERIES[0], alpha=0.8,
        label=r"$A=\{2,4,6\}$, $B=\{1,2,3,4\}$ (independent)")
ax.hist(diff_AC, bins=bins, color=SERIES[1], alpha=0.8,
        label=r"$A=\{2,4,6\}$, $C=\{4,5,6\}$ (dependent)")
ax.axvline(0.0, color="k", ls="--", lw=1.2)
ax.axvline(1 / 3 - 1 / 4, color="k", ls="--", lw=1.2)
ax.set_xlabel(r"$\hat P(\text{both}) - \hat P(\text{first})\,\hat P(\text{second})$")
ax.set_ylabel(f"number of repeats (of {reps})")
ax.set_title("each repeat: $N=" + f"{N:,}".replace(",", "{,}") + "$ die rolls")
ax.legend(loc="upper center", fontsize=8)
fig.tight_layout()
savefig(fig, "ch01", "1a_independence")

save_numbers("ch01", "1a_independence", {
    "OneAIndN": N,
    "OneAIndFA": f"{fA:.4f}", "OneAIndFB": f"{fB:.4f}", "OneAIndFC": f"{fC:.4f}",
    "OneAIndFAB": f"{fAB:.4f}", "OneAIndProdAB": f"{fA*fB:.4f}",
    "OneAIndFAC": f"{fAC:.4f}", "OneAIndProdAC": f"{fA*fC:.4f}",
    "OneAIndSdAB": f"{diff_AB.std():.4f}", "OneAIndMeanAC": f"{diff_AC.mean():.4f}",
    "OneAIndReps": reps,
})
print(fAB, fA * fB, fAC, fA * fC, diff_AB.std(), diff_AC.mean())
