"""A loop over numbers versus one operation on a whole array: how much faster is NumPy?

Question:  the chi-square of N data points, chi2 = sum_i ((d_i - m_i) / s_i)^2, can be computed
           with a Python for loop, one element at a time, or with a single NumPy expression on
           whole arrays.  Do the two give the same number, and how do their run times grow with N?
Computes:  both versions for N = 10 ... 10^6 (loop) and N = 10 ... 10^7 (arrays), timing each with
           time.perf_counter (best of several repeats); the ratio of the two times.
Writes:    figures/chT0/02_vectorise.pdf, results/chT0/02_vectorise.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import time
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES

rng = rng_for("chT0", "02_vectorise")


def chi2_loop(d, m, s):
    """One element at a time, the way one would write it on paper."""
    total = 0.0
    for i in range(len(d)):
        total += ((d[i] - m[i]) / s[i]) ** 2
    return total


def chi2_vec(d, m, s):
    """The same sum as one expression on whole arrays."""
    return np.sum(((d - m) / s) ** 2)


def best_time(f, *args, repeats=5):
    """Shortest of several runs: the least disturbed by whatever else the machine is doing."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


Ns_loop = np.logspace(1, 6, 11).astype(int)      # 10, 31, 100, ..., 10^6
Ns_vec = np.logspace(1, 7, 13).astype(int)       # 10, 31, 100, ..., 10^7
t_loop, t_vec = [], []
for N in Ns_vec:
    d = rng.normal(0.0, 1.0, N)                  # data
    m = np.zeros(N)                              # model prediction
    s = np.full(N, 1.0)                          # error bars
    t_vec.append(best_time(chi2_vec, d, m, s, repeats=7))
    if N in Ns_loop:
        t_loop.append(best_time(chi2_loop, d, m, s, repeats=3 if N < 10**5 else 1))
t_loop, t_vec = np.array(t_loop), np.array(t_vec)

# the same answer?  (summation order differs, so equality is only to rounding)
N = 10**6
d, m, s = rng.normal(0.0, 1.0, N), np.zeros(N), np.full(N, 1.0)
a, b = chi2_loop(d, m, s), chi2_vec(d, m, s)
rel = abs(a - b) / b

setup(8.0, 3.2)
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.loglog(Ns_loop, t_loop, "o-", color=SERIES[1], label="Python for loop")
ax0.loglog(Ns_vec, t_vec, "o-", color=SERIES[0], label="NumPy on whole arrays")
ax0.set_xlabel("number of data points $N$")
ax0.set_ylabel("time for one $\\chi^2$ (s)")
ax0.legend()
ratio = t_loop / t_vec[: len(t_loop)]
ax1.semilogx(Ns_loop, ratio, "o-", color=SERIES[2])
ax1.set_xlabel("number of data points $N$")
ax1.set_ylabel("loop time / array time")
fig.tight_layout()
savefig(fig, "chT0", "02_vectorise")

i6 = list(Ns_loop).index(10**6)
save_numbers("chT0", "02_vectorise", {
    "TzLoopMillion": t_loop[i6],                       # seconds
    "TzVecMillionMs": 1e3 * t_vec[list(Ns_vec).index(10**6)],
    "TzSpeedup": f"{ratio[i6]:.0f}",
    "TzSpeedupSmall": f"{ratio[0]:.0f}",
    "TzLoopNsPerElem": f"{1e9 * t_loop[i6] / 10**6:.0f}",
    "TzVecNsPerElem": f"{1e9 * t_vec[list(Ns_vec).index(10**6)] / 10**6:.1f}",
    "TzVecRelDiff": rel,
})
print(f"N = 10^6: loop {t_loop[i6]:.3f} s, arrays {1e3 * t_vec[list(Ns_vec).index(10**6)]:.2f} ms, "
      f"ratio {ratio[i6]:.0f}; relative difference of the two answers {rel:.1e}")
