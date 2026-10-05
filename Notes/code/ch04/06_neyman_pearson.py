"""06_neyman_pearson.py -- is the likelihood ratio really the best selection?

Question: signal and background events are described by two measured variables
x = (x1, x2), Gaussian under each hypothesis with different means and
covariances.  At a fixed signal efficiency (= 1 - alpha if H0 is "signal"), which
selection keeps the least background: a cut on x1 alone, Fisher's linear
discriminant, or the likelihood ratio (Neyman-Pearson)?  We also try many
random selections with the same signal efficiency; none should beat the ratio.

Writes: figures/ch04/np_scatter.pdf, figures/ch04/np_roc.pdf,
        results/ch04/06_neyman_pearson.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "06_neyman_pearson")
setup()
M = 400_000

mu_s, C_s = np.array([1.0, 1.0]), np.array([[0.6, 0.25], [0.25, 0.6]])
mu_b, C_b = np.array([-0.3, -0.2]), np.array([[1.4, -0.5], [-0.5, 1.2]])
S = stats.multivariate_normal(mu_s, C_s)
Bk = stats.multivariate_normal(mu_b, C_b)
xs = S.rvs(M, random_state=rng)
xb = Bk.rvs(M, random_state=rng)

def lnr(x):                      # log likelihood ratio  ln[ f(x|signal) / f(x|background) ]
    return S.logpdf(x) - Bk.logpdf(x)

# Fisher linear discriminant from training samples: a proportional to (W_s + W_b)^{-1} (mu_s - mu_b)
tr_s, tr_b = xs[:20000], xb[:20000]
Wsum = np.cov(tr_s.T) + np.cov(tr_b.T)
a_fish = np.linalg.solve(Wsum, tr_s.mean(0) - tr_b.mean(0))

stats_s = {"cut on $x_1$ alone": xs[:, 0], "Fisher linear": xs @ a_fish, "likelihood ratio": lnr(xs)}
stats_b = {"cut on $x_1$ alone": xb[:, 0], "Fisher linear": xb @ a_fish, "likelihood ratio": lnr(xb)}

eff_s = np.linspace(0.01, 0.99, 99)
roc = {}
for name in stats_s:
    thr = np.quantile(stats_s[name], 1 - eff_s)                 # keep events with statistic > thr
    sb = np.sort(stats_b[name])
    roc[name] = 1 - np.searchsorted(sb, thr) / M              # background efficiency
i90 = np.argmin(np.abs(eff_s - 0.90))
bkg90 = {k: v[i90] for k, v in roc.items()}

# random competitors: selections "keep events with g(x) > c" for random quadratic g, tuned to eff_s = 0.9
best_random = 1.0
for _ in range(300):
    A = rng.normal(size=(2, 2)); A = A + A.T; v = rng.normal(size=2)
    g = lambda x: np.einsum("ij,jk,ik->i", x, A, x) + x @ v
    gs, gb = g(xs[:100000]), g(xb[:100000])
    c = np.quantile(gs, 0.10)
    best_random = min(best_random, np.mean(gb > c))

# ---------- figure 1: the events and the boundaries at 90% signal efficiency ----------
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.plot(xb[:1500, 0], xb[:1500, 1], ".", ms=2, color=SERIES[0], alpha=0.6, label="background")
ax.plot(xs[:1500, 0], xs[:1500, 1], ".", ms=2, color=SERIES[1], alpha=0.6, label="signal")
g1, g2 = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
G = np.dstack([g1, g2]).reshape(-1, 2)
thr_lr = np.quantile(stats_s["likelihood ratio"], 0.10)
ax.contour(g1, g2, lnr(G).reshape(g1.shape), levels=[thr_lr], colors="k", linewidths=1.5, linestyles="solid")
thr_x1 = np.quantile(xs[:, 0], 0.10)
ax.axvline(thr_x1, color="k", ls=":", lw=1.2)
thr_f = np.quantile(xs @ a_fish, 0.10)
xx = np.linspace(-4, 4, 2)
ax.plot(xx, (thr_f - a_fish[0] * xx) / a_fish[1], color="k", ls="--", lw=1.0)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.legend(loc="upper right", framealpha=0.9, fontsize=8, markerscale=4)
savefig(fig, "ch04", "np_scatter")

# ---------- figure 2: ROC curves ----------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for j, name in enumerate(roc):
    ax.plot(eff_s, roc[name], color=SERIES[j], label=name)
ax.plot([0.9], [best_random], "kx", ms=6, label="best of 300 random selections")
ax.set_xlabel(r"signal efficiency $1-\alpha$"); ax.set_ylabel(r"background efficiency $\beta$")
ax.set_yscale("log"); ax.set_ylim(1e-3, 1.05); ax.legend(fontsize=8)
savefig(fig, "ch04", "np_roc")

save_numbers("ch04", "06_neyman_pearson", {
    "FourANPbkgCut": f"{bkg90['cut on $x_1$ alone']:.3f}",
    "FourANPbkgFisher": f"{bkg90['Fisher linear']:.3f}",
    "FourANPbkgLR": f"{bkg90['likelihood ratio']:.3f}",
    "FourANPbkgRandom": f"{best_random:.3f}",
})
print(bkg90, best_random, a_fish)
