"""When the mean does not exist: running averages of Gaussian and Cauchy samples.

Question: the expectation E[X] is the long-run average of repeated draws.  For
X ~ N(0,1) the running average (X_1 + ... + X_n)/n settles down to 0 like 1/sqrt(n).
For the Cauchy (Breit-Wigner / Lorentzian) density 1/(pi(1+x^2)) the integral of |x| f(x)
diverges, so E[X] does not exist.  What does the running average do then?
We draw 4 independent runs of 10^4 values from each and plot the running averages.
Writes: figures/ch01/cauchy_running_mean.pdf, results/ch01/20_cauchy_running_mean.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "20_cauchy_running_mean")
setup(7.0, 2.8)

n_max, n_runs = 10_000, 4
n = np.arange(1, n_max + 1)
gauss = rng.normal(size=(n_runs, n_max))
cauchy = rng.standard_cauchy(size=(n_runs, n_max))
run_g = np.cumsum(gauss, axis=1) / n          # running averages, one row per run
run_c = np.cumsum(cauchy, axis=1) / n

fig, axes = plt.subplots(1, 2, sharex=True)
for j in range(n_runs):
    axes[0].plot(n, run_g[j], color=SERIES[j], lw=1.0)
    axes[1].plot(n, run_c[j], color=SERIES[j], lw=1.0)
axes[0].plot(n, 1 / np.sqrt(n), "k--", lw=0.9)
axes[0].plot(n, -1 / np.sqrt(n), "k--", lw=0.9, label="$\\pm1/\\sqrt{n}$")
axes[0].set_title("Gaussian $\\mathcal{N}(0,1)$", fontsize=9)
axes[1].set_title("Cauchy $1/[\\pi(1+x^2)]$", fontsize=9)
for ax in axes:
    ax.set_xscale("log")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("number of draws $n$")
axes[0].set_ylim(-1.5, 1.5)
axes[1].set_ylim(-8, 8)
axes[0].set_ylabel("running average")
axes[0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "ch01", "cauchy_running_mean")

save_numbers("ch01", "20_cauchy_running_mean", {
    "OneBRunGaussMax": f"{np.max(np.abs(run_g[:, -1])):.3f}",   # largest |average| at n = 10^4
    "OneBRunCauchyMin": f"{np.min(np.abs(run_c[:, -1])):.2f}",
    "OneBRunCauchyMax": f"{np.max(np.abs(run_c[:, -1])):.2f}",
    "OneBRunN": "10^4",
})
