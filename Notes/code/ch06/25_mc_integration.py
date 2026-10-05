"""25_mc_integration.py -- Monte Carlo integration and its sigma / sqrt(N) error.

Question: an integral is an expectation, I = int h(x) f(x) dx = E_f[h(X)], so the
average of h over N draws from f estimates it.  How big is the error, and how
does it compare with a grid rule as the dimension grows?

  1. Wasserman Ex 24.1: I = int_0^1 x^3 dx = 1/4 from N = 10^4 uniforms, with the
     standard error s / sqrt(N); repeated 2000 times at each N to measure the RMS
     error, compared with sigma / sqrt(N), sigma^2 = 1/7 - 1/16 = 9/112.
  2. Wasserman Ex 24.2: Phi(2) as the fraction of N(0,1) draws below 2.
  3. Dimension: I_d = int_[0,1]^d prod_i (pi/2) sin(pi x_i) dx = 1, by the
     midpoint rule on n^d points and by Monte Carlo with the same N = n^d
     points; grid error ~ N^(-2/d), Monte Carlo ~ sqrt(((pi^2/8)^d - 1) / N).
  4. Wasserman Ex 24.3: posterior of delta = p2 - p1 for two binomials,
     n = m = 10, X = 8, Y = 6, flat priors: p1 ~ Beta(9, 3), p2 ~ Beta(7, 5).

Writes: figures/ch06/mc_error.pdf, figures/ch06/mc_two_binomials.pdf,
        results/ch06/25_mc_integration.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup(9.0, 3.4)
rng = rng_for("ch06", "25_mc_integration")
out = {}

# ---------------------------------------------------------------- 1. int x^3
N = 10_000
y = rng.random(N) ** 3
out.update(SixBMcCubeN=N, SixBMcCube=y.mean(), SixBMcCubeSe=y.std(ddof=1) / np.sqrt(N),
           SixBMcCubeSig=np.sqrt(9 / 112))
Ns = np.unique(np.logspace(1, 5, 13).astype(int))
rms = []
for n in Ns:
    est = np.array([np.mean(rng.random(n) ** 3) for _ in range(2000)])
    rms.append(np.sqrt(np.mean((est - 0.25) ** 2)))
rms = np.array(rms)
slope = np.polyfit(np.log(Ns), np.log(rms), 1)[0]
out.update(SixBMcSlope=slope, SixBMcRmsRatio=rms[-1] / (np.sqrt(9 / 112) / np.sqrt(Ns[-1])))

# ---------------------------------------------------------------- 2. Phi(2) by counting
for n, tag in [(10_000, "Four"), (100_000, "Five")]:
    z = rng.standard_normal(n)
    ph = np.mean(z <= 2.0)
    out[f"SixBMcPhi{tag}"] = ph
    out[f"SixBMcPhiSe{tag}"] = np.sqrt(ph * (1 - ph) / n)
out["SixBMcPhiExact"] = stats.norm.cdf(2.0)

# ---------------------------------------------------------------- 3. grid versus Monte Carlo in d dimensions
def h(x):                                         # product of (pi/2) sin(pi x_i); integral over the cube = 1
    return np.prod(0.5 * np.pi * np.sin(np.pi * x), axis=-1)


dims = [1, 2, 4, 8]
res = {}
for d in dims:
    ns = np.unique(np.round(np.logspace(np.log10(2), np.log10(2e6) / d, 9)).astype(int))
    ns = ns[ns ** d <= 2_000_000]
    grid_err, mc_err, Npts = [], [], []
    for n in ns:
        g1 = (np.arange(n) + 0.5) / n            # midpoints in one dimension
        # the integrand factorises, so the d-dim midpoint sum is the 1-d sum to the power d
        grid = (np.mean(0.5 * np.pi * np.sin(np.pi * g1))) ** d
        grid_err.append(abs(grid - 1.0))
        Npts.append(n ** d)
        reps = [abs(np.mean(h(rng.random((n ** d, d)))) - 1.0) for _ in range(8 if n ** d < 3e5 else 2)]
        mc_err.append(np.sqrt(np.mean(np.square(reps))))
    res[d] = (np.array(Npts), np.array(grid_err), np.array(mc_err))
sig = {d: np.sqrt((np.pi**2 / 8) ** d - 1) for d in dims}
out.update(SixBMcSigOne=sig[1], SixBMcSigEight=sig[8])
# error of each method at (about) one million points
for d, tag in zip(dims, ["One", "Two", "Four", "Eight"]):
    Np, ge, me = res[d]
    k = np.argmin(np.abs(np.log(Np) - np.log(1e6)))
    out[f"SixBMcGridErr{tag}"] = ge[k]
    out[f"SixBMcMcErr{tag}"] = sig[d] / np.sqrt(Np[k])
    out[f"SixBMcNpts{tag}"] = int(Np[k])

fig, ax = plt.subplots(1, 2)
ax[0].loglog(Ns, rms, "o", color=SERIES[0], label="RMS error, 2000 repeats")
theory_line(ax[0], Ns, np.sqrt(9 / 112) / np.sqrt(Ns), label=r"$\sigma/\sqrt{N}$")
ax[0].set(xlabel=r"number of draws $N$", ylabel=r"error of $\hat I$", title=r"$\int_0^1 x^3\,dx$")
ax[0].legend()
for i, d in enumerate(dims):
    Np, ge, me = res[d]
    ax[1].loglog(Np, ge, "-", color=SERIES[i], lw=1.4, label=f"grid, $d={d}$")
    ax[1].loglog(Np, sig[d] / np.sqrt(Np), "--", color=SERIES[i], lw=1.0)
    ax[1].loglog(Np, me, "o", color=SERIES[i], ms=3)
ax[1].set(xlabel=r"number of points $N$", ylabel="absolute error",
          title=r"grid (solid) vs Monte Carlo (dashed, dots)", ylim=(1e-13, 3))
ax[1].legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "ch06", "mc_error")

# ---------------------------------------------------------------- 4. two binomials
nB = 100_000
p1 = rng.beta(9, 3, nB)
p2 = rng.beta(7, 5, nB)
delta = p2 - p1
lo, hi = np.quantile(delta, [0.025, 0.975])
out.update(SixBMcDeltaN=nB, SixBMcDeltaMean=delta.mean(), SixBMcDeltaLo=lo, SixBMcDeltaHi=hi,
           SixBMcDeltaPneg=np.mean(delta < 0),
           SixBMcDeltaSe=delta.std(ddof=1) / np.sqrt(nB))
setup(5.0, 3.0)
fig, ax = plt.subplots()
ax.hist(delta, bins=80, density=True, color=SERIES[0], alpha=0.6)
for q in (lo, hi):
    ax.axvline(q, color=SERIES[1], lw=1.2, ls=":")
ax.axvline(delta.mean(), color="k", lw=1.0)
ax.set(xlabel=r"$\delta=p_2-p_1$", ylabel="posterior density")
fig.tight_layout()
savefig(fig, "ch06", "mc_two_binomials")
save_numbers("ch06", "25_mc_integration", out)
print(out)
