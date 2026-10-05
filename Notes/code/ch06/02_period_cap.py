"""What a finite state does to a Monte Carlo estimate.

Question: a generator with finitely many states must eventually repeat.  What
happens to a Monte Carlo estimate once the sequence has gone round its cycle,
and does the reported error bar notice?  And how quickly does the very first
proposal for a generator, von Neumann's middle-square rule, fall into a cycle?

Computes: (1) pi estimated as 4 x (fraction of pairs (u1,u2) with u1^2+u2^2<1)
for N up to 10^6 pairs, with pairs taken from the LCG x -> (1229 x + 1) mod 2048
(period 2048) and from PCG64; the true error and the reported standard error
4 sqrt(p(1-p)/N) against N.  (2) For every 4-digit seed of the middle-square
rule: the number of steps before a value repeats, and where it ends.
Writes: figures/ch06/period_cap.pdf, results/ch06/02_period_cap.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_prng import lcg_ints_vec, middle_square

NMAX = 1_000_000
rng = rng_for("ch06", "02_period_cap")

sources = {
    "LCG $(1229x+1)\\,\\mathrm{mod}\\,2048$": lcg_ints_vec(1229, 1, 2048, 1, 2 * NMAX) / 2048.0,
    "PCG64": rng.random(2 * NMAX),
}
Ns = np.unique(np.logspace(1, 6, 60).astype(int))
curves = {}
for name, u in sources.items():
    hit = (u[0::2] ** 2 + u[1::2] ** 2 < 1.0).astype(float)
    csum = np.cumsum(hit)
    p = csum[Ns - 1] / Ns
    est = 4 * p
    se = 4 * np.sqrt(p * (1 - p) / Ns)
    curves[name] = (np.abs(est - np.pi), se, est[-1], se[-1])

# ---------------- middle-square census ----------------
steps, ends_zero, cyc_len = [], 0, {}
for s in range(10000):
    seen, x, k = {s: 0}, s, 0
    while True:
        x = (x * x // 100) % 10000
        k += 1
        if x in seen:
            break
        seen[x] = k
    steps.append(k)                      # number of steps until the first repeated value
    L = k - seen[x]
    if x == 0:
        ends_zero += 1
    cyc_len[L] = cyc_len.get(L, 0) + 1
steps = np.array(steps)
print("middle-square: max steps before repeat", steps.max(), "mean", steps.mean(),
      "ends at 0:", ends_zero, "cycle lengths:", sorted(cyc_len.items()))

# ---------------- figure ----------------
setup(6.4, 3.0)
fig, ax = plt.subplots()
for j, (name, (err, se, _, _)) in enumerate(curves.items()):
    ax.loglog(Ns, err, color=SERIES[j], lw=1.2, label=f"{name}: true error")
    ax.loglog(Ns, se, color=SERIES[j], lw=1.0, ls=":", label=f"{name}: reported s.e.")
ax.axvline(1024, color="0.5", lw=0.8)
ax.text(1150, 0.6, "one full period\n(1024 pairs)", fontsize=8, color="0.3", va="top")
ax.set_xlabel("number of pairs $N$")
ax.set_ylabel(r"$|\hat\pi-\pi|$ and standard error")
ax.set_ylim(1e-5, 40)
ax.legend(loc="upper center", ncol=2, fontsize=7, columnspacing=1.0)
savefig(fig, "ch06", "period_cap")

(eL, sL, estL, seL), (eP, sP, estP, seP) = curves.values()
save_numbers("ch06", "02_period_cap", {
    "SixAPiLcgEst": f"{estL:.4f}", "SixAPiLcgSe": f"{seL:.4f}", "SixAPiLcgErr": f"{estL-np.pi:.4f}",
    "SixAPiPcgEst": f"{estP:.4f}", "SixAPiPcgSe": f"{seP:.4f}", "SixAPiPcgErr": f"{estP-np.pi:.4f}",
    "SixAMidSqMaxSteps": int(steps.max()), "SixAMidSqMeanSteps": f"{steps.mean():.1f}",
    "SixAMidSqZero": int(ends_zero), "SixAMidSqZeroPct": f"{100*ends_zero/10000:.0f}",
})
