"""The unbiased walker spreads forever; a walker pulled back to the centre settles down.

Question: a walker on the real line steps X_{t+1} = X_t + s Z_t with Z_t ~ N(0,1).  Does the
distribution of its position ever settle?  Compare with the AR(1) walker
X_{t+1} = rho X_t + sqrt(1 - rho^2) Z_t, which feels a pull towards 0, started far away.
Writes figures/ch09/random_walk_diffusion.pdf, figures/ch09/ar1_forgetting.pdf and
results/ch09/07_random_walk_diffusion.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

rng = rng_for("ch09", "07_random_walk_diffusion")
W, s, T = 20_000, 1.0, 1000
snaps = (10, 100, 1000)

# --- unbiased random walk ----------------------------------------------------------------------
x = np.zeros(W)
var_t, keep = [], {}
for t in range(1, T + 1):
    x += s * rng.standard_normal(W)
    var_t.append(x.var())
    if t in snaps:
        keep[t] = x.copy()
var_t = np.array(var_t)

setup(7.0, 3.0)
fig, (ax1, ax2) = plt.subplots(1, 2)
for c, t in zip(SERIES, snaps):
    sd = s * np.sqrt(t)
    ax1.hist(keep[t], bins=np.linspace(-110, 110, 89), density=True, color=c, alpha=0.5, label=f"$t={t}$")
    xx = np.linspace(-110, 110, 600)
    theory_line(ax1, xx, np.exp(-xx**2 / (2 * sd**2)) / np.sqrt(2 * np.pi * sd**2), label=None)
ax1.set_xlabel("position $x$")
ax1.set_ylabel("density")
ax1.set_title(r"(a) histograms keep widening")
ax1.legend(fontsize=8)
tt = np.arange(1, T + 1)
ax2.loglog(tt, var_t, color=SERIES[0], label=f"{W:,} walkers")
theory_line(ax2, tt, s**2 * tt, label=r"$s^2t$")
ax2.set_xlabel("step $t$")
ax2.set_ylabel(r"$\mathrm{Var}(X_t)$")
ax2.set_title("(b) the spread grows without limit")
ax2.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "ch09", "random_walk_diffusion")

# --- AR(1): the walker with a restoring pull -------------------------------------------------
rho, T2 = 0.9, 80
starts = (-20.0, 0.0, 20.0)
means, finals = {}, {}
for x0 in starts:
    y = np.full(W, x0)
    m = [x0]
    for t in range(T2):
        y = rho * y + np.sqrt(1 - rho**2) * rng.standard_normal(W)
        m.append(y.mean())
    means[x0], finals[x0] = np.array(m), y

setup(7.0, 3.0)
fig, (ax1, ax2) = plt.subplots(1, 2)
t2 = np.arange(T2 + 1)
for c, x0 in zip(SERIES, starts):
    ax1.plot(t2, means[x0], color=c, label=f"$x_0={x0:+.0f}$")
    theory_line(ax1, t2, x0 * rho**t2, label=None)
ax1.set_xlabel("step $t$")
ax1.set_ylabel(r"mean of $X_t$")
ax1.set_title(r"(a) the mean forgets $x_0$ as $\rho^t x_0$")
ax1.legend(fontsize=8)
for c, x0 in zip(SERIES, starts):
    ax2.hist(finals[x0], bins=np.linspace(-4.5, 4.5, 61), density=True, histtype="step", color=c, lw=1.4)
xx = np.linspace(-4.5, 4.5, 400)
theory_line(ax2, xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), label=r"$\mathcal{N}(0,1)$")
ax2.set_xlabel(f"$X_{{{T2}}}$")
ax2.set_ylabel("density")
ax2.set_title("(b) all three starts end in the same law")
ax2.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "ch09", "ar1_forgetting")

save_numbers("ch09", "07_random_walk_diffusion", {
    "NineADiffVarHundred": round(var_t[99], 1), "NineADiffVarThousand": round(var_t[-1], 0),
    "NineADiffWalkers": f"{W:,}".replace(",", r"\,"),
    "NineAARMeanForty": round(means[20.0][40], 3), "NineAARMeanFortyPred": round(20 * rho**40, 3),
    "NineAARVarEnd": round(finals[20.0].var(), 3), "NineAARVarEndPred": round(1 - rho ** (2 * T2), 4),
    "NineAARMeanEnd": round(finals[20.0].mean(), 3),
})
print("var t=100", var_t[99], "t=1000", var_t[-1], "AR mean40", means[20.0][40], 20 * rho**40,
      "var end", finals[20.0].var())
