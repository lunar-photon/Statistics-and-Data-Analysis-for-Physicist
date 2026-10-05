"""Does a random walker on the integers come home?

Question: a walker on ..., -2, -1, 0, 1, 2, ... steps +1 with probability p and -1 with q = 1 - p.
Starting at 0, is a return to 0 certain?  The exact return probability at step 2n is
p_00(2n) = C(2n, n) p^n q^n ~ (4pq)^n / sqrt(pi n); the expected number of returns is the sum.
We compare exact values with Stirling's approximation and simulate the fraction of walkers that
have returned at least once by step t (the exact limit is 1 - |p - q|).
Writes figures/ch09/return_probability.pdf and results/ch09/09_return_probability.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

rng = rng_for("ch09", "09_return_probability")
n = np.arange(1, 2001)
W, T = 10_000, 10_000                              # walkers, steps (10^8 coin flips, a few s)


def p00(p, n):
    return np.exp(gammaln(2 * n + 1) - 2 * gammaln(n + 1) + n * np.log(p) + n * np.log(1 - p))


returned = {}
for p in (0.5, 0.6):
    x = np.zeros(W, dtype=np.int32)
    back = np.zeros(W, dtype=bool)
    frac = np.empty(T)
    for t in range(T):
        x += np.where(rng.random(W) < p, 1, -1).astype(np.int32)
        back |= x == 0
        frac[t] = back.mean()
    returned[p] = frac

setup(7.0, 3.0)
fig, (ax1, ax2) = plt.subplots(1, 2)
for c, p in zip(SERIES, (0.5, 0.6)):
    ax1.loglog(2 * n, p00(p, n), "o", ms=2, color=c, label=f"exact, $p={p}$")
    theory_line(ax1, 2 * n, (4 * p * (1 - p)) ** n / np.sqrt(np.pi * n), label=None)
ax1.set_ylim(1e-12, 1)
ax1.set_xlabel("step $2n$")
ax1.set_ylabel(r"$p_{00}(2n)$")
ax1.set_title(r"(a) exact vs Stirling $(4pq)^n/\sqrt{\pi n}$")
ax1.legend(fontsize=8)
tt = np.arange(1, T + 1)
for c, p in zip(SERIES, (0.5, 0.6)):
    ax2.semilogx(tt, returned[p], color=c, label=f"$p={p}$")
ax2.axhline(0.8, color="k", ls="--", lw=1.2)
ax2.axhline(1.0, color="k", ls=":", lw=1.2)
ax2.set_ylim(0.4, 1.02)
ax2.set_xlabel("step $t$")
ax2.set_ylabel("fraction returned to 0")
ax2.set_title("(b) symmetric walkers all come home")
ax2.legend(fontsize=8, loc="lower right")
fig.tight_layout()
savefig(fig, "ch09", "return_probability")

save_numbers("ch09", "09_return_probability", {
    "NineARetHalf": round(returned[0.5][-1], 4), "NineARetSix": round(returned[0.6][-1], 4),
    "NineARetHalfPred": round(1 - np.sqrt(2 / (np.pi * T)) * 1.0, 4),
    "NineARetSumHalf": round(float(p00(0.5, n).sum()), 1), "NineARetSumSix": round(float(p00(0.6, n).sum()), 4),
    "NineARetWalkers": f"{W:,}".replace(",", r"\,"), "NineARetSteps": f"{T:,}".replace(",", r"\,"),
    "NineARetStirTen": f"{(p00(0.5, 10) * np.sqrt(np.pi * 10)):.4f}",
})
print("returned", returned[0.5][-1], returned[0.6][-1], "sum", p00(0.5, n).sum(), p00(0.6, n).sum(),
      "ratio exact/Stirling n=10", p00(0.5, 10) * np.sqrt(np.pi * 10))
