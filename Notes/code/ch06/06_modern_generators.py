"""Inside xorshift, the Mersenne Twister and PCG64, checked against numpy.

Question: what do the state and the output function of the modern generators
actually do, and can a few lines of Python reproduce numpy's own bits?

Computes:
 (1) PCG64 written from scratch (lib_prng.PCG64) against numpy.random.PCG64:
     number of identical 64-bit outputs out of 100000.
 (2) MT19937 from scratch against numpy's MT19937 (legacy seeding, seed 5489);
     then the state rebuilt from 624 consecutive outputs by undoing the
     tempering, and the next 1000 outputs predicted.
 (3) xorshift64 as a 64x64 matrix T over the bits (arithmetic mod 2): check
     T x = xorshift(x), check linearity of whole streams, and check that the
     period is 2^64 - 1 by verifying T^(2^64-1) = 1 and T^((2^64-1)/p) != 1 for
     each prime p dividing 2^64 - 1.
 (4) a toy 16-bit LCG whose top 8 bits are the output, against a toy PCG that
     feeds the same 16-bit state through PCG's xorshift-and-rotate output
     (XSH-RR 16 -> 8): pair plots of all 65536 consecutive pairs.
Writes: figures/ch06/pcg_toy.pdf, results/ch06/06_modern_generators.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_prng import PCG64, MT19937, xorshift64, M32, M64

# ---------------- (1) PCG64 ----------------
bg = np.random.PCG64(20261001)
mine = PCG64.like_numpy(bg)
theirs = bg.random_raw(100_000)
same_pcg = sum(int(t) == mine.next64() for t in theirs)
print("PCG64 identical outputs:", same_pcg)

# ---------------- (2) MT19937 ----------------
mt_np = np.random.MT19937()
mt_np._legacy_seeding(5489)
ref = mt_np.random_raw(10_000)
mt_py = MT19937(5489)
same_mt = sum(int(r) == mt_py.next32() for r in ref)
print("MT19937 identical outputs:", same_mt)


def untemper(y):
    """Invert the four tempering steps, last one first."""
    y ^= y >> 18
    y ^= (y << 15) & 0xEFC60000
    t = y                                   # undo y ^= (y << 7) & 0x9D2C5680, 7 bits at a time
    for _ in range(5):
        t = y ^ ((t << 7) & 0x9D2C5680)
    y = t & M32
    t = y                                   # undo y ^= y >> 11
    for _ in range(3):
        t = y ^ (t >> 11)
    return t & M32


observed = [int(v) for v in ref[1000:1624]]          # 624 consecutive outputs from the middle
clone = MT19937(0)
clone.mt = [untemper(v) for v in observed]
clone.i = 624
pred = [clone.next32() for _ in range(1000)]
same_pred = sum(p == int(r) for p, r in zip(pred, ref[1624:2624]))
print("MT19937 predicted outputs:", same_pred, "of 1000")

# ---------------- (3) xorshift64 as a bit matrix ----------------
def xs_step(x):
    x ^= (x << 13) & M64
    x ^= x >> 7
    x ^= (x << 17) & M64
    return x


def bits(x):
    return np.array([(x >> i) & 1 for i in range(64)], dtype=np.uint8)


T = np.zeros((64, 64), dtype=np.uint8)
for j in range(64):
    T[:, j] = bits(xs_step(1 << j))                  # column j = image of the j-th unit vector
rng_np = np.random.Generator(np.random.PCG64(7))
ok_matrix = all(np.array_equal(T @ bits(int(x)) % 2, bits(xs_step(int(x))))
                for x in rng_np.integers(1, 2**63, 20))
s1, s2 = 0x9E3779B97F4A7C15, 0x0123456789ABCDEF
lin = all((a ^ b) == c for a, b, c in zip(xorshift64(s1, 50), xorshift64(s2, 50), xorshift64(s1 ^ s2, 50)))


def matpow(A, e):
    R = np.eye(64, dtype=np.int64)
    A = A.astype(np.int64)
    while e:
        if e & 1:
            R = (R @ A) % 2
        A = (A @ A) % 2
        e >>= 1
    return R


N = 2**64 - 1
primes = [3, 5, 17, 257, 641, 65537, 6700417]
assert np.prod([float(p) for p in primes]) and eval("*".join(map(str, primes))) == N
full = np.array_equal(matpow(T, N), np.eye(64, dtype=np.int64))
proper = all(not np.array_equal(matpow(T, N // p), np.eye(64, dtype=np.int64)) for p in primes)
print("xorshift: matrix ok", ok_matrix, " linear streams", lin, " T^(2^64-1)=1", full,
      " no smaller divisor", proper)

# ---------------- (4) toy LCG top byte vs toy PCG ----------------
A16, C16 = 25173, 13849


def rotr8(x, r):
    return ((x >> r) | (x << ((8 - r) % 8))) & 0xFF


s = 1
lcg8, pcg8 = [], []
for _ in range(2**16 + 1):
    s = (A16 * s + C16) & 0xFFFF
    lcg8.append(s >> 8)                                   # plain truncation: keep the top byte
    pcg8.append(rotr8((((s >> 5) ^ s) >> 5) & 0xFF, s >> 13))   # PCG XSH-RR: xorshift, then rotate
lcg8, pcg8 = np.array(lcg8), np.array(pcg8)
pairs_lcg = len(set(zip(lcg8[:-1], lcg8[1:])))
pairs_pcg = len(set(zip(pcg8[:-1], pcg8[1:])))
print("distinct pairs out of 65536: LCG top byte", pairs_lcg, " toy PCG", pairs_pcg)

setup(6.2, 3.0)
fig, axes = plt.subplots(1, 2)
for ax, v, t in [(axes[0], lcg8, "16-bit LCG, top 8 bits"), (axes[1], pcg8, "same state, PCG output (XSH-RR)")]:
    ax.scatter(v[:-1], v[1:], s=2.0, lw=0, color=SERIES[0])
    ax.set_aspect("equal")
    ax.set_xlim(0, 64); ax.set_ylim(0, 64)
    ax.set_xticks([0, 32, 64]); ax.set_yticks([0, 32, 64])
    ax.set_title(t, fontsize=8)
    ax.set_xlabel("output $n$")
axes[0].set_ylabel("output $n+1$")
savefig(fig, "ch06", "pcg_toy")

save_numbers("ch06", "06_modern_generators", {
    "SixAPcgSame": f"{same_pcg:,}".replace(",", "{,}"),
    "SixAMtSame": f"{same_mt:,}".replace(",", "{,}"),
    "SixAMtPred": f"{same_pred:,}".replace(",", "{,}"),
    "SixAPairsLcg": f"{pairs_lcg:,}".replace(",", "{,}"),
    "SixAPairsPcg": f"{pairs_pcg:,}".replace(",", "{,}"),
    "SixAPairsLcgPct": f"{100*pairs_lcg/65536:.1f}", "SixAPairsPcgPct": f"{100*pairs_pcg/65536:.1f}",
})
