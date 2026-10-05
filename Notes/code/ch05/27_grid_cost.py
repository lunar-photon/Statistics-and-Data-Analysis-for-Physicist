"""27_grid_cost.py -- why the evidence integral cannot be done by brute force.

Question: how expensive is the denominator of Bayes' theorem when the
parameters are many?

Computes
  (1) the bump-model evidence of 22_line_or_bump.py by a brute-force grid over
      the whole 4-D prior box (a, b, A, mu), n points per axis: wall time and
      ln B_10 against n, compared with the exact value;
  (2) for a D-dimensional Gaussian likelihood of width 0.05 in the unit cube:
      (a) the evidence from 10^5 draws of the prior (Z = mean likelihood), its
          relative error over 20 repetitions, and the fraction of draws that
          land where the posterior is,
      (b) the posterior mean of theta_1 from 10^5 posterior draws: its error
          does not grow with D.

Writes: figures/ch05/grid_cost.pdf, figures/ch05/dimension_curse.pdf,
        results/ch05/27_grid_cost.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
import lib_bump as lb

setup()
y = lb.make_data(rng_for("ch05", "22_line_or_bump", 0), True)
lnB_exact = lb.ln_bayes_factor(y)
lnZ0 = lb.ln_evidence_line(y)


def ln_Z_bump_bruteforce(y, n):
    """Midpoint rule over the prior box, n points per axis: Z = mean of the likelihood over the box."""
    edges = lambda lo, hi: lo + (np.arange(n) + 0.5) * (hi - lo) / n
    a, b = edges(*lb.A_RANGE), edges(*lb.B_RANGE)
    A, mu = edges(0, lb.A_MAX), edges(*lb.MU_RANGE)
    G = lb.bump(lb.X[:, None], mu[None, :])                       # (N, n)
    lin = a[:, None, None] + b[None, :, None] * lb.X[None, None, :]  # (n, n, N)
    acc = []
    for iA in range(n):                                           # one amplitude slice at a time
        m = lin[:, :, None, :] + A[iA] * G.T[None, None, :, :]    # (n_a, n_b, n_mu, N)
        c2 = np.sum((y - m) ** 2, axis=-1) / lb.SIGMA**2
        acc.append(np.log(np.mean(np.exp(-0.5 * (c2 - 30.0)))) - 15.0)
    return lb.ln_norm() + np.log(np.mean(np.exp(np.array(acc))))


ns = [8, 12, 16, 24, 32, 48, 64]
times, lnBs = [], []
for n in ns:
    t0 = time.perf_counter()
    lnZ = ln_Z_bump_bruteforce(y, n)
    times.append(time.perf_counter() - t0)
    lnBs.append(lnZ - lnZ0)
    print(n, times[-1], lnBs[-1])
times = np.array(times); lnBs = np.array(lnBs); ns = np.array(ns)
t_per_point = times[-1] / ns[-1] ** 4

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
axes[0].loglog(ns, times, "o", color=SERIES[0], label="measured")
theory_line(axes[0], ns, times[-1] * (ns / ns[-1]) ** 4, label=r"$\propto n^4$")
axes[0].set_xlabel("grid points per axis $n$"); axes[0].set_ylabel("wall time [s]"); axes[0].legend(fontsize=8)
axes[1].semilogx(ns, lnBs, "o-", color=SERIES[1], label="brute-force grid")
axes[1].axhline(lnB_exact, color="k", ls="--", lw=1.2, label="exact")
axes[1].set_xlabel("grid points per axis $n$"); axes[1].set_ylabel(r"$\ln B_{10}$"); axes[1].legend(fontsize=8)
for _ax in axes:
    _ax.set_xticks(ns); _ax.set_xticklabels([str(int(v)) for v in ns], fontsize=7); _ax.minorticks_off()
savefig(fig, "ch05", "grid_cost")

# ---- (2) the curse of dimension for a Gaussian likelihood in the unit cube
rng = rng_for("ch05", "27_grid_cost", 1)
s, ns_draw, reps = 0.05, 100_000, 20
Ds = np.arange(1, 11)
relerr, frac, posterr, medratio = [], [], [], []
for D in Ds:
    Z_true = (s * np.sqrt(2 * np.pi)) ** D                        # centre at 0.5: truncation negligible
    est, inside = [], []
    for r in range(reps):
        th = rng.random((ns_draw, D))
        r2 = np.sum((th - 0.5) ** 2, axis=1) / s**2
        est.append(np.mean(np.exp(-0.5 * r2)))
        inside.append(np.mean(r2 < D + 2 * np.sqrt(2 * D)))       # within the bulk of the posterior
    est = np.array(est)
    relerr.append(np.sqrt(np.mean((est / Z_true - 1) ** 2)))
    medratio.append(np.median(est / Z_true))
    frac.append(np.mean(inside))
    pm = [np.mean(0.5 + s * rng.standard_normal(ns_draw)) for _ in range(reps)]   # posterior draws of theta_1
    posterr.append(np.std(pm) / s)
relerr, frac, posterr = map(np.array, (relerr, frac, posterr))
# once almost no run hits the peak, the r.m.s. error saturates near 1 and means nothing: do not plot it
resolved = np.array(medratio) > 0.01

fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.semilogy(Ds[resolved], relerr[resolved], "o-", color=SERIES[1], label="evidence from prior draws: relative error")
ax.semilogy(Ds, posterr, "s-", color=SERIES[0], label=r"posterior mean from posterior draws: error$/\sigma$")
ax.semilogy(Ds, np.maximum(frac, 1e-6), "^-", color=SERIES[2], label="fraction of prior draws in the posterior bulk")
ax.set_xlabel("number of parameters $D$"); ax.set_ylim(1e-6, 30); ax.legend(fontsize=7, loc="lower left")
savefig(fig, "ch05", "dimension_curse")

yr = 3.15e7
save_numbers("ch05", "27_grid_cost", {
    "FiveCGcExact": lnB_exact, "FiveCGcNmax": int(ns[-1]), "FiveCGcTmax": times[-1],
    "FiveCGcLnBmax": lnBs[-1], "FiveCGcLnBsmall": lnBs[1], "FiveCGcNsmall": int(ns[1]),
    "FiveCGcLnBmid": lnBs[3], "FiveCGcNmid": int(ns[3]),
    "FiveCGcTpp": t_per_point,
    "FiveCGcSixD": 1000.0**6 * t_per_point / yr,                  # years for n = 1000, D = 6 at this speed
    "FiveCGcSixDcmb": 1000.0**6 * 1.0 / yr,                      # at one second per likelihood call
    "FiveCDcErrTwo": relerr[1], "FiveCDcErrFive": relerr[4], "FiveCDcErrEight": relerr[7],
    "FiveCDcErrTen": relerr[9],
    "FiveCDcFracTwo": frac[1], "FiveCDcFracFive": frac[4], "FiveCDcFracEight": frac[7],
    "FiveCDcPostTwo": posterr[1], "FiveCDcPostTen": posterr[9],
    "FiveCDcMedTen": medratio[9], "FiveCDcMedEight": medratio[7],
    "FiveCDcVolTen": (s * np.sqrt(2 * np.pi)) ** 10, "FiveCDcDraws": ns_draw,
})
