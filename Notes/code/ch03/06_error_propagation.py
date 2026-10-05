"""06_error_propagation.py -- when does linear error propagation work, and when does it fail?

Question: (i) for the pendulum, g = 4 pi^2 L / T^2, does the first-order (Jacobian)
formula reproduce the Monte Carlo spread of g?  (ii) for a ratio y = x1/x2 of two
Gaussian quantities, how does the true distribution of y compare with the linear
prediction as the denominator's relative error grows (0.1, 0.33, 1)?

Writes: figures/ch03/ratio_failure.pdf, figures/ch03/pendulum_mc.pdf,
        results/ch03/06_error_propagation.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch03", "06_error_propagation")
setup()
M = 1_000_000

# ---------- (i) pendulum ----------
L0, sL = 1.000, 0.002           # m
T0, sT = 2.006, 0.004           # s
L = L0 + sL * rng.standard_normal(M)
T = T0 + sT * rng.standard_normal(M)
g = 4 * np.pi**2 * L / T**2
g0 = 4 * np.pi**2 * L0 / T0**2
sg_lin = g0 * np.sqrt((sL / L0) ** 2 + (2 * sT / T0) ** 2)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
bins = np.linspace(g0 - 5 * sg_lin, g0 + 5 * sg_lin, 121)
ax.hist(g, bins=bins, density=True, color=SERIES[0], alpha=0.6, label=r"Monte Carlo $g$")
gg = np.linspace(bins[0], bins[-1], 400)
theory_line(ax, gg, stats.norm.pdf(gg, g0, sg_lin), label="linear propagation")
ax.set_xlabel(r"$g$ [m s$^{-2}$]"); ax.set_ylabel("density"); ax.legend(fontsize=8)
savefig(fig, "ch03", "pendulum_mc")

# ---------- (ii) ratio of Gaussians ----------
mu1, s1, s2 = 10.0, 1.0, 1.0
fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.7))
out = {}
for ax, mu2, c, tag in zip(axs, [10.0, 3.0, 1.0], SERIES, ["A", "B", "C"]):
    x1 = mu1 + s1 * rng.standard_normal(M)
    x2 = mu2 + s2 * rng.standard_normal(M)
    y = x1 / x2
    y0 = mu1 / mu2
    sy = abs(y0) * np.sqrt((s1 / mu1) ** 2 + (s2 / mu2) ** 2)
    lo, med, hi = np.percentile(y, [15.865, 50, 84.135])
    out[tag] = dict(y0=y0, sy=sy, med=med, half=0.5 * (hi - lo),
                    neg=np.mean(y < 0), far=np.mean(np.abs(y - y0) > 5 * sy))
    w = 6 * sy
    bins = np.linspace(y0 - w, y0 + w, 121)
    ax.hist(y, bins=bins, density=True, color=c, alpha=0.6, label="Monte Carlo")
    yy = np.linspace(bins[0], bins[-1], 400)
    theory_line(ax, yy, stats.norm.pdf(yy, y0, sy), label="linear")
    ax.set_title(rf"$\sigma_2/\mu_2={s2/mu2:.2g}$")
    ax.set_xlabel(r"$y=x_1/x_2$")
axs[0].set_ylabel("density"); axs[0].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "ratio_failure")

vals = {
    "ThreeAPendG": g0, "ThreeAPendSgLin": sg_lin, "ThreeAPendSgMC": g.std(ddof=1),
    "ThreeAPendMeanMC": g.mean(),
}
for tag in "ABC":
    o = out[tag]
    vals.update({f"ThreeARatio{tag}Lin": o["sy"], f"ThreeARatio{tag}Half": o["half"],
                 f"ThreeARatio{tag}Med": o["med"], f"ThreeARatio{tag}Neg": o["neg"],
                 f"ThreeARatio{tag}Far": o["far"]})
save_numbers("ch03", "06_error_propagation", vals)
