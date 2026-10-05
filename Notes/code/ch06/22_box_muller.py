"""22_box_muller.py -- two uniform numbers in, two independent Gaussians out.

Question: the Gaussian CDF has no closed-form inverse, so how do we make
Gaussian numbers?  Box--Muller: R = sqrt(-2 ln U1), Theta = 2 pi U2,
X = R cos Theta, Y = R sin Theta are independent N(0, 1).  The polar (Marsaglia)
variant draws a point uniformly in the unit disc by rejection from the square
and needs no sine or cosine.

Checks: histograms against the N(0,1) density, the correlation of X and Y,
R^2 against the exponential of mean 2, a Kolmogorov--Smirnov test, and the
acceptance of the polar method against pi/4.

Writes: figures/ch06/box_muller.pdf, results/ch06/22_box_muller.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup(9.0, 3.0)
rng = rng_for("ch06", "22_box_muller")
N = 100_000


def box_muller(u1, u2):
    r = np.sqrt(-2.0 * np.log(u1))              # radius: R^2 = -2 ln U1 is exponential, mean 2
    th = 2.0 * np.pi * u2                       # angle: uniform on [0, 2 pi)
    return r * np.cos(th), r * np.sin(th)


u1 = 1.0 - rng.random(N)                        # in (0, 1], so log(u1) is finite
u2 = rng.random(N)
x, y = box_muller(u1, u2)

# polar method: points uniform in the square, keep those inside the unit disc
v = 2.0 * rng.random((N, 2)) - 1.0
s = np.sum(v**2, axis=1)
keep = (s < 1.0) & (s > 0.0)
vk, sk = v[keep], s[keep]
fac = np.sqrt(-2.0 * np.log(sk) / sk)
xp = vk[:, 0] * fac

out = {"SixBBmN": N,
       "SixBBmMeanX": x.mean(), "SixBBmVarX": x.var(ddof=1),
       "SixBBmMeanY": y.mean(), "SixBBmVarY": y.var(ddof=1),
       "SixBBmCorr": np.corrcoef(x, y)[0, 1],
       "SixBBmRsqMean": np.mean(x**2 + y**2),
       "SixBBmKS": stats.kstest(x, "norm").pvalue,
       "SixBBmKurt": np.mean(x**4),
       "SixBBmPolarAcc": keep.mean(), "SixBBmPolarVar": xp.var(ddof=1)}

fig, ax = plt.subplots(1, 3)
ax[0].scatter(u1[:2000], u2[:2000], s=1.5, color=SERIES[0])
ax[0].set(xlabel=r"$U_1$", ylabel=r"$U_2$", title="uniform input", aspect="equal")
ax[1].scatter(x[:2000], y[:2000], s=1.5, color=SERIES[0])
ax[1].set(xlabel=r"$X$", ylabel=r"$Y$", title="Gaussian output", aspect="equal",
          xlim=(-4.2, 4.2), ylim=(-4.2, 4.2))
bins = np.linspace(-4.5, 4.5, 61)
ax[2].hist(x, bins=bins, density=True, color=SERIES[0], alpha=0.5, label=r"$X$")
ax[2].hist(y, bins=bins, density=True, histtype="step", color=SERIES[1], lw=1.2, label=r"$Y$")
zz = np.linspace(-4.5, 4.5, 300)
theory_line(ax[2], zz, stats.norm.pdf(zz), label=r"$\mathcal{N}(0,1)$")
ax[2].set(xlabel="value", ylabel="density", title="histograms")
ax[2].legend()
fig.tight_layout()
savefig(fig, "ch06", "box_muller")
save_numbers("ch06", "22_box_muller", out)
print(out)
