"""Periods, pair plots and low bits of linear congruential generators.

Question: an LCG x -> (a x + c) mod m is fully determined by (a, c, m) and the
seed.  How long is its period, what do consecutive pairs (u_n, u_{n+1}) look
like, and how random are its individual bits?

Computes: the periods of the small LCGs used in the text (by walking the
sequence until a value repeats); the pair plots of two generators from Lambert's
problem 12.7 and of RANDU; the first 64 outputs of the 32-bit LCG
x -> (1664525 x + 1013904223) mod 2^32 split into bits, and the period of each
of its low bits.
Writes: figures/ch06/lcg_pairs.pdf, figures/ch06/lcg_lowbits.pdf,
        results/ch06/03_lcg_structure.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_prng import lcg_ints, lcg_ints_vec, randu


def rho(a, c, m, seed):
    """Walk until a value repeats: return (tail length, cycle length)."""
    seen, x, k = {}, seed % m, 0
    while x not in seen:
        seen[x] = k
        x = (a * x + c) % m
        k += 1
    return seen[x], k - seen[x]


cases = {
    "toy": (2, 3, 10, 5), "toyseven": (2, 3, 10, 7), "toyzero": (2, 3, 10, 0),
    "hd": (5, 3, 16, 0), "mult": (3, 0, 7, 1), "multtwo": (2, 0, 7, 1),
    "lamA": (1229, 1, 2048, 1), "lamB": (1597, 51749, 244944, 1),
}
per = {k: rho(*v) for k, v in cases.items()}
for k, v in per.items():
    print(k, cases[k], "tail, period =", v)
print("5x+3 mod 16 from 0:", lcg_ints(5, 3, 16, 0, 16))

# ---------------- pair plots ----------------
setup(6.6, 2.4)
fig, axes = plt.subplots(1, 3)
uA = lcg_ints(1229, 1, 2048, 1, 2048) / 2048
uB = lcg_ints_vec(1597, 51749, 244944, 1, 10001) / 244944
uR = randu(1, 10001)
for ax, u, title in [(axes[0], uA, r"$(1229x+1)\ \mathrm{mod}\ 2048$"),
                     (axes[1], uB, r"$(1597x+51749)\ \mathrm{mod}\ 244944$"),
                     (axes[2], uR, r"RANDU, $65539x\ \mathrm{mod}\ 2^{31}$")]:
    ax.scatter(u[:-1], u[1:], s=0.4, color=SERIES[0], lw=0)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("$u_n$", labelpad=-6)
axes[0].set_ylabel("$u_{n+1}$", labelpad=-6)
savefig(fig, "ch06", "lcg_pairs")

# ---------------- low bits of a 32-bit LCG ----------------
x = lcg_ints_vec(1664525, 1013904223, 2**32, 2026, 16384).astype(np.uint64)
bits = np.array([(x >> np.uint64(b)) & np.uint64(1) for b in range(32)], dtype=int)  # bits[b, n]
bitper = []
for b in range(12):
    s = bits[b]
    for p in range(1, 8193):
        if np.all(s[p:] == s[:-p]):
            bitper.append(p)
            break
print("periods of bits 0..11:", bitper)
setup(6.4, 2.6)
fig, ax = plt.subplots()
ax.imshow(bits[::-1, :64], cmap="Greys", aspect="auto", interpolation="nearest",
          extent=[0.5, 64.5, -0.5, 31.5])
ax.set_yticks([0, 4, 8, 16, 24, 31])
ax.set_ylabel("bit number (0 = lowest)")
ax.set_xlabel("output number $n$")
ax.grid(False)
savefig(fig, "ch06", "lcg_lowbits")

save_numbers("ch06", "03_lcg_structure", {
    "SixALamAPeriod": per["lamA"][1], "SixALamBPeriod": per["lamB"][1],
    "SixABitPerZero": bitper[0], "SixABitPerOne": bitper[1], "SixABitPerTwo": bitper[2],
    "SixABitPerThree": bitper[3], "SixABitPerEleven": bitper[11],
})
