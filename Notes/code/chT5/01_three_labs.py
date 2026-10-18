"""01_three_labs.py -- one quantity, three noise laws of the same variance: which average is best?

Question: three laboratories measure the same quantity mu, n readings each, x_i = mu + e_i.
The noise has variance 1 in all three, but a different law: Gaussian (lab A), Laplace (lab B),
and a Gaussian core with occasional wild readings, a 5% admixture ten times wider (lab C).
How large is the error of the sample mean, of the sample median, and of the maximum-likelihood
estimate of each lab, and how close does each come to the Cramer-Rao bound 1/(n I)?

Computes: M experiments of n readings per lab; rms error of mean, median and ML; the Fisher
          information I of each noise law (analytic for A and B, quadrature for C); the
          asymptotic median variance 1/(4 n f(0)^2); histograms of the three estimators in lab C.
Writes:   figures/chT5/three_labs.pdf, results/chT5/01_three_labs.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, stats
from lib_t5a import ml_location

rng = rng_for("chT5", "01_three_labs")
setup()

n, M = 25, 20_000
eps, wide = 0.05, 10.0                       # lab C: 5% of readings from a Gaussian 10x wider
s_core = 1.0 / np.sqrt(1 - eps + eps * wide**2)   # core width chosen so the total variance is 1
b_lap = 1.0 / np.sqrt(2.0)                   # Laplace scale: variance 2 b^2 = 1


def dens_C(e):
    """Density of lab C's noise: a two-component Gaussian mixture of total variance 1."""
    return (1 - eps) * stats.norm.pdf(e, 0, s_core) + eps * stats.norm.pdf(e, 0, wide * s_core)


def noise(lab, size):
    if lab == "A":
        return rng.standard_normal(size)
    if lab == "B":
        return rng.laplace(0.0, b_lap, size)
    bad = rng.random(size) < eps
    return np.where(bad, wide * s_core, s_core) * rng.standard_normal(size)


logf = {"A": lambda e: -0.5 * e**2,
        "B": lambda e: -np.abs(e) / b_lap,
        "C": lambda e: np.log(dens_C(e))}

# Fisher information I = E[(d ln f/de)^2] of each noise law
I_A = 1.0
I_B = 1.0 / b_lap**2
dC = lambda e: -((1 - eps) * stats.norm.pdf(e, 0, s_core) * e / s_core**2
                 + eps * stats.norm.pdf(e, 0, wide * s_core) * e / (wide * s_core)**2)
I_C = integrate.quad(lambda e: dC(e)**2 / dens_C(e), -60, 60, limit=400)[0]
f0 = {"A": stats.norm.pdf(0), "B": 1 / (2 * b_lap), "C": dens_C(0.0)}
info = {"A": I_A, "B": I_B, "C": I_C}

res, keep = {}, {}
for lab in "ABC":
    x = noise(lab, (M, n))                   # true mu = 0, so each estimate is its own error
    est = {"mean": x.mean(1), "median": np.median(x, 1), "ml": ml_location(x, logf[lab])}
    res[lab] = {k: n * np.mean(v**2) for k, v in est.items()}      # n * mean squared error
    res[lab]["medth"] = 1 / (4 * f0[lab]**2)
    res[lab]["cr"] = 1 / info[lab]
    keep[lab] = est
    print(lab, {k: round(v, 3) for k, v in res[lab].items()})

# sanity: the mean's error is the same in all labs (variance 1), Laplace ML is the median
assert all(abs(res[l]["mean"] - 1) < 0.05 for l in "ABC")
assert abs(res["B"]["cr"] - 0.5) < 1e-12 and res["C"]["cr"] < res["C"]["medth"]

# ---------- figure: noise laws (left) and sampling distributions in lab C (right) ----------
fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.0))
e = np.linspace(-4, 4, 801)
axs[0].plot(e, stats.norm.pdf(e), color=SERIES[0], label="A: Gaussian")
axs[0].plot(e, np.exp(-np.abs(e) / b_lap) / (2 * b_lap), color=SERIES[1], label="B: Laplace")
axs[0].plot(e, dens_C(e), color=SERIES[2], label="C: 5% wild readings")
axs[0].set_yscale("log"); axs[0].set_ylim(1e-4, 1.2)
axs[0].set_xlabel(r"noise $e$ (all three have variance 1)"); axs[0].set_ylabel("density")
axs[0].legend(fontsize=7, loc="lower center")
bins = np.linspace(-0.8, 0.8, 81)
for k, c, lab in [("mean", SERIES[0], "mean"), ("median", SERIES[1], "median"), ("ml", SERIES[2], "ML")]:
    axs[1].hist(keep["C"][k], bins=bins, density=True, histtype="step", color=c, lw=1.4,
                label=rf"{lab}: $n\,$MSE $={res['C'][k]:.2f}$")
axs[1].set_xlabel(r"estimate $-\mu$ in lab C ($n=%d$)" % n); axs[1].set_ylabel("density")
axs[1].legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "chT5", "three_labs")

nums = {"TFiveALabN": n, "TFiveALabM": M, "TFiveALabCore": f"{s_core:.3f}"}
for lab, L in zip("ABC", ["A", "B", "C"]):
    w = {"Mean": res[lab]["mean"], "Med": res[lab]["median"], "ML": res[lab]["ml"],
         "MedTh": res[lab]["medth"], "CR": res[lab]["cr"]}
    for k, v in w.items():
        nums[f"TFiveALab{L}{k}"] = f"{v:.3f}"
nums["TFiveALabCInfo"] = f"{I_C:.3f}"
nums["TFiveALabCfzero"] = f"{f0['C']:.4f}"
nums["TFiveALabCGain"] = f"{res['C']['mean'] / res['C']['median']:.2f}"
save_numbers("chT5", "01_three_labs", nums)
