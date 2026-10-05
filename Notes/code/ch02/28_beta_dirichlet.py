"""Probabilities of probabilities: three routes to the Beta distribution, and the Dirichlet.

Question answered
    (1) The k-th smallest of n uniform numbers is claimed to be Beta(k, n-k+1).
    (2) If X ~ Gamma(a) and Y ~ Gamma(b) are independent, X/(X+Y) is claimed to be
        Beta(a, b).
    (3) Preview of conjugacy, done by brute force: draw a coin bias theta from a
        Beta(2,2) prior, flip that coin 10 times, and keep only the runs that gave
        exactly 7 heads.  The kept thetas should follow Beta(2+7, 2+3) = Beta(9, 5).
    (4) Normalised independent gammas (G_1, G_2, G_3)/sum are Dirichlet(alpha);
        each component is Beta(alpha_j, alpha_0 - alpha_j).

What it computes
    Monte Carlo samples for (1)-(4) with the analytic densities overlaid, the
    acceptance fraction of (3) (which is the evidence P(7 heads)), and moments.

What it writes
    figures/ch02/beta_origins.pdf, figures/ch02/dirichlet.pdf,
    results/ch02/28_beta_dirichlet.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special

setup()
rng = rng_for("ch02", "28_beta_dirichlet")
NREP = 400_000

# ---------------------------------------------------------------- (1) order statistics
n, k = 10, 3
u = np.sort(rng.uniform(size=(NREP, n)), axis=1)
u_k = u[:, k - 1]                                   # k-th smallest

# ---------------------------------------------------------------- (2) gamma ratio
a, b = 2.5, 4.0
X = rng.gamma(a, size=NREP)
Y = rng.gamma(b, size=NREP)
ratio = X / (X + Y)

# ---------------------------------------------------------------- (3) rejection "posterior"
a0, b0, N, h = 2.0, 2.0, 10, 7
theta = rng.beta(a0, b0, size=2_000_000)
heads = rng.binomial(N, theta)
kept = theta[heads == h]
acc = kept.size / theta.size
evidence_th = special.comb(N, h) * special.beta(a0 + h, b0 + N - h) / special.beta(a0, b0)

fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
xx = np.linspace(0.001, 0.999, 500)
bins = np.linspace(0, 1, 81)
ax = axes[0]
ax.hist(u_k, bins=bins, density=True, color=SERIES[0], alpha=0.6, label=f"{k}rd smallest of {n}")
theory_line(ax, xx, stats.beta.pdf(xx, k, n - k + 1), label=f"Beta$({k},{n - k + 1})$")
ax.set_xlabel("$u_{(k)}$")
ax.set_ylabel("density")
ax.set_title("(a) order statistic of uniforms")
ax.legend(fontsize=7)
ax = axes[1]
ax.hist(ratio, bins=bins, density=True, color=SERIES[2], alpha=0.6, label="$X/(X+Y)$")
theory_line(ax, xx, stats.beta.pdf(xx, a, b), label=f"Beta$({a},{b:g})$")
ax.set_xlabel("fraction")
ax.set_title("(b) ratio of gammas")
ax.legend(fontsize=7)
ax = axes[2]
ax.hist(kept, bins=bins, density=True, color=SERIES[1], alpha=0.6, label="kept $\\theta$ (7 of 10 heads)")
ax.plot(xx, stats.beta.pdf(xx, a0, b0), color="0.4", ls=":", lw=1.4, label="prior Beta(2,2)")
theory_line(ax, xx, stats.beta.pdf(xx, a0 + h, b0 + N - h), label="Beta(9,5)")
ax.set_xlabel("coin bias $\\theta$")
ax.set_title("(c) keep the runs that match")
ax.legend(fontsize=7, loc="upper left")
savefig(fig, "ch02", "beta_origins")

# ---------------------------------------------------------------- (4) Dirichlet
alpha = np.array([4.0, 2.0, 2.0])
Gm = rng.gamma(alpha, size=(NREP, 3))
P = Gm / Gm.sum(axis=1, keepdims=True)
a_tot = alpha.sum()
mean_th = alpha / a_tot
var1_th = alpha[0] * (a_tot - alpha[0]) / (a_tot ** 2 * (a_tot + 1))
cov12_th = -alpha[0] * alpha[1] / (a_tot ** 2 * (a_tot + 1))

# ternary coordinates: vertex j at the corners of an equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
xy = P @ V
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
ax = axes[0]
ax.plot(xy[:5000, 0], xy[:5000, 1], ".", ms=1.4, color=SERIES[0], alpha=0.4)
tri = np.vstack([V, V[:1]])
ax.plot(tri[:, 0], tri[:, 1], color="k", lw=1.0)
ax.plot(*(mean_th @ V), "o", color=SERIES[1], ms=6, label="mean $\\alpha_j/\\alpha_0$")
for j, (dx, dy, ha) in enumerate([(-0.03, -0.05, "right"), (0.03, -0.05, "left"), (0.0, 0.04, "center")]):
    ax.text(V[j, 0] + dx, V[j, 1] + dy, f"$p_{j + 1}=1$", ha=ha, fontsize=9)
ax.set_aspect("equal")
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.15, 1.0)
ax.axis("off")
ax.set_title(r"(a) Dirichlet$(4,2,2)$ on the simplex")
ax.legend(fontsize=7, loc="upper left")
ax = axes[1]
for j in range(3):
    ax.hist(P[:, j], bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=SERIES[j],
            label=f"$p_{j + 1}$")
    theory_line(ax, xx, stats.beta.pdf(xx, alpha[j], a_tot - alpha[j]),
                label="Beta$(\\alpha_j,\\alpha_0-\\alpha_j)$" if j == 0 else None)
ax.set_xlabel("component value")
ax.set_ylabel("density")
ax.set_title("(b) each component is a Beta")
ax.legend(fontsize=7)
savefig(fig, "ch02", "dirichlet")

save_numbers("ch02", "28_beta_dirichlet", {
    "tbOSn": n, "tbOSk": k,
    "tbOSmean": f"{u_k.mean():.4f}", "tbOSmeanTh": f"{k / (n + 1):.4f}",
    "tbOSvar": f"{u_k.var(ddof=1):.5f}", "tbOSvarTh": f"{k * (n - k + 1) / ((n + 1) ** 2 * (n + 2)):.5f}",
    "tbGRmean": f"{ratio.mean():.4f}", "tbGRmeanTh": f"{a / (a + b):.4f}",
    "tbRJacc": f"{acc:.4f}", "tbRJaccTh": f"{evidence_th:.4f}", "tbRJkept": int(kept.size),
    "tbRJmean": f"{kept.mean():.4f}", "tbRJmeanTh": f"{(a0 + h) / (a0 + b0 + N):.4f}",
    "tbRJsd": f"{kept.std(ddof=1):.4f}",
    "tbRJsdTh": f"{np.sqrt(stats.beta.var(a0 + h, b0 + N - h)):.4f}",
    "tbDIRmeanOne": f"{P[:, 0].mean():.4f}", "tbDIRmeanOneTh": f"{mean_th[0]:.4f}",
    "tbDIRvarOne": f"{P[:, 0].var(ddof=1):.5f}", "tbDIRvarOneTh": f"{var1_th:.5f}",
    "tbDIRcov": f"{np.cov(P[:, 0], P[:, 1])[0, 1]:.5f}", "tbDIRcovTh": f"{cov12_th:.5f}",
})
