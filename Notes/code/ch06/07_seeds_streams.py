"""Seeds, streams and what rng_for does with a script's name.

Question: when many simulations run side by side, each needs its own stream of
random numbers.  How do we make the streams reproducible, different from each
other, and free of hidden relations?  What exactly does common.rng_for do?

Computes:
 (1) RANDU started from seeds 1 and 3: the second stream is 3 x the first mod 1.
 (2) the path of rng_for("ch06", "07_seeds_streams"): SHA-256 of the name ->
     64-bit integer -> SeedSequence([seed, stream]) -> four 64-bit words ->
     PCG64 (state, increment), rebuilt by hand and compared with numpy.
 (3) eight streams (stream = 0..7) of 10^6 numbers: the 28 correlations between
     pairs of them, scaled by sqrt(N), compared with N(0,1) (KS test).
 (4) the chance that two of n jobs seeded with random 32-bit integers share a
     seed (birthday problem), exact and approximate.
 (5) how numpy turns 64 random bits into a double: k/2^53.
Writes: figures/ch06/streams.pdf, results/ch06/07_seeds_streams.tex
"""
import sys, pathlib, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_prng import randu, PCG64, M128

# ---------------- (1) RANDU from seeds 1 and 3 ----------------
n = 5000
u1, u3 = randu(1, n), randu(3, n)
frac_check = np.max(np.abs(u3 - np.mod(3 * u1, 1.0)))
print("RANDU seed 3 = frac(3 x seed 1): max deviation", frac_check)

# ---------------- (2) rng_for by hand ----------------
name = "ch06/07_seeds_streams"
seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
ss = np.random.SeedSequence([seed, 0])
w = ss.generate_state(4, np.uint64)                    # four 64-bit words of mixed entropy
initstate = (int(w[0]) << 64) | int(w[1])
initseq = (int(w[2]) << 64) | int(w[3])
inc = ((initseq << 1) | 1) & M128                      # the increment must be odd
st = 0
st = (st * 0x2360ED051FC65DA44385DF649FCCF645 + inc) & M128   # PCG's seeding: step, add, step
st = (st + initstate) & M128
st = (st * 0x2360ED051FC65DA44385DF649FCCF645 + inc) & M128
bg = rng_for("ch06", "07_seeds_streams").bit_generator
match_state = (st == int(bg.state["state"]["state"])) and (inc == int(bg.state["state"]["inc"]))
by_hand = PCG64(st, inc)
match_out = all(by_hand.next64() == int(v) for v in bg.random_raw(1000))
print("seed integer:", seed, " state rebuilt:", match_state, " outputs match:", match_out)

# ---------------- (3) eight streams ----------------
N = 1_000_000
S = np.array([rng_for("ch06", "07_seeds_streams", stream=k).random(N) for k in range(8)])
C = np.corrcoef(S)
off = np.abs(C[np.triu_indices(8, 1)])
zc = C[np.triu_indices(8, 1)] * np.sqrt(N)          # r sqrt(N) is N(0,1) under independence
from scipy import stats
ks_z = stats.kstest(zc, "norm").pvalue
print("max |corr| between streams:", off.max(), " 1/sqrt(N) =", 1 / np.sqrt(N),
      " std of r sqrt(N):", zc.std(), " KS p:", ks_z)

# ---------------- (4) birthday problem for 32-bit seeds ----------------
def p_collision(njobs, space=2**32):
    k = np.arange(njobs)
    return 1 - np.exp(np.sum(np.log1p(-k / space)))


jobs = [10**3, 10**4, 10**5]
pc = [p_collision(j) for j in jobs]
approx = [1 - np.exp(-j * (j - 1) / 2 / 2**32) for j in jobs]
pc128 = 1e5 * (1e5 - 1) / 2 / 2.0**128
print("P(shared 32-bit seed):", dict(zip(jobs, np.round(pc, 4))), "approx", np.round(approx, 4))

# ---------------- (5) 64 bits -> double ----------------
g = np.random.Generator(np.random.PCG64(99))
raw = np.random.PCG64(99).random_raw(5)
dbl = g.random(5)
match_dbl = np.array_equal(dbl, (raw >> np.uint64(11)).astype(float) * 2.0**-53)
print("double = (x >> 11) * 2^-53:", match_dbl)

# ---------------- figure ----------------
setup(6.2, 2.9)
fig, axes = plt.subplots(1, 2)
axes[0].scatter(u1, u3, s=0.6, lw=0, color=SERIES[1])
axes[0].set_title("RANDU: seed 1 against seed 3", fontsize=8)
axes[0].set_xlabel("$u_n$ from seed 1"); axes[0].set_ylabel("$u_n$ from seed 3")
axes[1].scatter(S[0, :n], S[1, :n], s=0.6, lw=0, color=SERIES[0])
axes[1].set_title("rng_for: stream 0 against stream 1", fontsize=8)
axes[1].set_xlabel("stream 0"); axes[1].set_ylabel("stream 1")
for ax in axes:
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
savefig(fig, "ch06", "streams")

save_numbers("ch06", "07_seeds_streams", {
    "SixASeedInt": str(seed), "SixAStreamMaxCorr": f"{off.max():.4f}",
    "SixAStreamInvSqrtN": f"{1/np.sqrt(N):.4f}",
    "SixAStreamZstd": f"{zc.std():.2f}", "SixAStreamKsP": f"{ks_z:.2f}",
    "SixABdayThree": f"{pc[0]:.4f}", "SixABdayFour": f"{pc[1]:.3f}", "SixABdayFive": f"{pc[2]:.3f}",
    "SixABdayFiveApprox": f"{approx[2]:.3f}", "SixABdayOneTwoEight": pc128,
})
