"""02_blue_heavy_tails.py -- the best linear unbiased estimator, and where linear stops being enough.

Question (a): two masses are weighed singly and together, d = (m1, m2, m1+m2) + noise of spread
sigma. Does the best linear unbiased estimator m1_hat = (2 d1 - d2 + d3)/3 really have variance
2 sigma^2/3, below the sigma^2 of weighing m1 alone and the 2 sigma^2 of d3 - d2?
Question (b): for heavy-tailed noise (Laplace, Cauchy, a Gaussian with 5% wild readings), how do
the mean (the BLUE of a common level), the median and the maximum-likelihood estimate compare as
the number of readings n grows, and do they approach their asymptotic variances?

Computes: (a) M weighing experiments, variances and the correlation of (m1_hat, m2_hat);
          (b) n * MSE of mean, median and ML for n = 5 ... 101 under the three laws.
Writes:   figures/chT5/heavy_tails.pdf, results/chT5/02_blue_heavy_tails.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, stats
from lib_t5a import ml_location

rng = rng_for("chT5", "02_blue_heavy_tails")
setup()

# ---------- (a) the weighing design ----------
sigma, m_true, M = 0.1, np.array([3.0, 5.0]), 200_000
A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
d = A @ m_true + sigma * rng.standard_normal((M, 3))
W0 = np.linalg.solve(A.T @ A, A.T)              # (A^T A)^{-1} A^T  (C = sigma^2 1)
assert np.allclose(W0, np.array([[2, -1, 1], [-1, 2, 1]]) / 3)
blue = d @ W0.T
v_blue = blue[:, 0].var() / sigma**2
v_alone = d[:, 0].var() / sigma**2
v_diff = (d[:, 2] - d[:, 1]).var() / sigma**2
rho_blue = np.corrcoef(blue[:, 0], blue[:, 1])[0, 1]
bias_blue = np.abs(blue.mean(0) - m_true).max() / sigma

# ---------- (b) efficiency against n for three heavy-tailed laws ----------
eps, wide = 0.05, 10.0
s_core = 1.0 / np.sqrt(1 - eps + eps * wide**2)
b_lap = 1.0 / np.sqrt(2.0)


def dens_C(e):
    return (1 - eps) * stats.norm.pdf(e, 0, s_core) + eps * stats.norm.pdf(e, 0, wide * s_core)


laws = {
    "Laplace": dict(draw=lambda s: rng.laplace(0, b_lap, s), logf=lambda e: -np.abs(e) / b_lap,
                    f0=1 / (2 * b_lap), info=1 / b_lap**2),
    "Cauchy": dict(draw=lambda s: rng.standard_cauchy(s), logf=lambda e: -np.log1p(e**2),
                   f0=1 / np.pi, info=0.5),
    "contaminated": dict(draw=lambda s: np.where(rng.random(s) < eps, wide * s_core, s_core)
                         * rng.standard_normal(s),
                         logf=lambda e: np.log(dens_C(e)), f0=dens_C(0.0), info=None),
}
dC = lambda e: -((1 - eps) * stats.norm.pdf(e, 0, s_core) * e / s_core**2
                 + eps * stats.norm.pdf(e, 0, wide * s_core) * e / (wide * s_core)**2)
laws["contaminated"]["info"] = integrate.quad(lambda e: dC(e)**2 / dens_C(e), -60, 60, limit=400)[0]

ns = np.array([5, 11, 25, 51, 101])
Mb = 10_000
out = {}
for name, L in laws.items():
    rows = []
    for n in ns:
        x = L["draw"]((Mb, n))
        half = 6 / np.sqrt(n) + (2.0 if name == "Cauchy" else 0.3)
        ml = ml_location(x, L["logf"], half=half, ngrid=481)
        rows.append([n * np.mean(x.mean(1)**2), n * np.mean(np.median(x, 1)**2), n * np.mean(ml**2)])
    out[name] = np.array(rows)
    print(name, np.round(out[name], 3))

fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=False)
for ax, (name, L) in zip(axs, laws.items()):
    r = out[name]
    if name != "Cauchy":
        ax.plot(ns, r[:, 0], "o-", color=SERIES[0], label="mean")
        ax.axhline(1.0, color=SERIES[0], ls=":", lw=1)
    ax.plot(ns, r[:, 1], "s-", color=SERIES[1], label="median")
    ax.axhline(1 / (4 * L["f0"]**2), color=SERIES[1], ls=":", lw=1)
    ax.plot(ns, r[:, 2], "^-", color=SERIES[2], label="ML")
    ax.axhline(1 / L["info"], color="k", ls="--", lw=1.2, label="$1/I$ (Cramér–Rao)")
    ax.set_xscale("log"); ax.set_xlabel(r"readings $n$"); ax.set_title(name)
    ax.set_ylim(0, None)
axs[0].set_ylabel(r"$n\times$ mean squared error")
axs[0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "chT5", "heavy_tails")

big = -1
save_numbers("chT5", "02_blue_heavy_tails", {
    "TFiveAWeighM": M, "TFiveAWeighSigma": sigma,
    "TFiveAWeighVBlue": f"{v_blue:.4f}", "TFiveAWeighVAlone": f"{v_alone:.4f}",
    "TFiveAWeighVDiff": f"{v_diff:.4f}", "TFiveAWeighRho": f"{rho_blue:.4f}",
    "TFiveAWeighBias": f"{bias_blue:.4f}",
    "TFiveAHtM": Mb, "TFiveAHtNBig": int(ns[big]), "TFiveAHtNSmall": int(ns[0]),
    "TFiveAHtLapMean": f"{out['Laplace'][big, 0]:.3f}", "TFiveAHtLapMed": f"{out['Laplace'][big, 1]:.3f}",
    "TFiveAHtLapML": f"{out['Laplace'][big, 2]:.3f}",
    "TFiveAHtCauMed": f"{out['Cauchy'][big, 1]:.3f}", "TFiveAHtCauML": f"{out['Cauchy'][big, 2]:.3f}",
    "TFiveAHtCauMedSmall": f"{out['Cauchy'][0, 1]:.3f}", "TFiveAHtCauMLSmall": f"{out['Cauchy'][0, 2]:.3f}",
    "TFiveAHtConMean": f"{out['contaminated'][big, 0]:.3f}", "TFiveAHtConMed": f"{out['contaminated'][big, 1]:.3f}",
    "TFiveAHtConML": f"{out['contaminated'][big, 2]:.3f}",
    "TFiveAHtConCR": f"{1 / laws['contaminated']['info']:.3f}",
    "TFiveAHtCauMedTh": f"{np.pi**2 / 4:.3f}",
})
