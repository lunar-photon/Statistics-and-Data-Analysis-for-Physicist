"""03_combine.py -- independent experiments add their Fisher matrices; a prior is one more experiment.

Question: two experiments measure the same straight line y = a + b x with the same noise sigma = 0.1,
experiment A at ten points spread over 0 <= x <= 1, experiment B at ten points over 2 <= x <= 3.
Each alone has a strongly correlated (a, b) ellipse, tilted in a different direction. How well does
the combination measure (a, b), and what does a Gaussian prior on a do?

Computes: F_A, F_B (F = sum_k (1, x_k)^T (1, x_k) / sigma^2), F_A + F_B, F_A + prior; marginalised and
conditional errors, correlations; the decorrelating "pivot" x_p = -F^-1_ab / F^-1_bb of each experiment
and the Jacobian transformation to (y_p, b).
Writes: figures/ch10/combine.pdf, results/ch10/03_combine.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES

setup()
SIG = 0.1
TRUE = np.array([1.0, 0.5])
xA = np.linspace(0, 1, 10)
xB = np.linspace(2, 3, 10)


def fisher_line(x):
    X = np.vstack([np.ones_like(x), x])
    return X @ X.T / SIG**2


FA, FB = fisher_line(xA), fisher_line(xB)
FAB = FA + FB
SIG_PRIOR_A = 0.02
P = np.diag([1 / SIG_PRIOR_A**2, 0.0])
FAP = FA + P


def summary(F):
    C = np.linalg.inv(F)
    s = np.sqrt(np.diag(C))
    return s, C[0, 1] / (s[0] * s[1]), 1 / np.sqrt(np.diag(F))


def ellipse(ax, F, k2=2.30, **kw):
    C = np.linalg.inv(F)
    t = np.linspace(0, 2 * np.pi, 400)
    L = np.linalg.cholesky(C)
    e = TRUE[:, None] + np.sqrt(k2) * L @ np.vstack([np.cos(t), np.sin(t)])
    ax.plot(e[0], e[1], **kw)


nums = {}
for tag, F in [("A", FA), ("B", FB), ("AB", FAB), ("AP", FAP)]:
    s, r, c = summary(F)
    key = {"A": "A", "B": "B", "AB": "Both", "AP": "Prior"}[tag]
    nums.update({f"TenCCo{key}Sa": f"{s[0]:.3f}", f"TenCCo{key}Sb": f"{s[1]:.3f}",
                 f"TenCCo{key}Rho": f"{r:+.2f}", f"TenCCo{key}Ca": f"{c[0]:.3f}", f"TenCCo{key}Cb": f"{c[1]:.3f}"})
# the pivot of experiment A: y_p = a + b x_p is uncorrelated with b when x_p = -Cov(a,b)/Var(b)
CA = np.linalg.inv(FA)
xp = -CA[0, 1] / CA[1, 1]
# Jacobian of the old parameters (a, b) with respect to the new ones (y_p, b): a = y_p - b x_p
J = np.array([[1.0, -xp], [0.0, 1.0]])
Fp = J.T @ FA @ J
sp, rp, _ = summary(Fp)
nums.update({"TenCCoPivot": f"{xp:.2f}", "TenCCoPivSy": f"{sp[0]:.3f}", "TenCCoPivSb": f"{sp[1]:.3f}",
             "TenCCoPivRho": f"{rp:+.3f}", "TenCCoSig": SIG, "TenCCoPriorA": SIG_PRIOR_A,
             "TenCCoGainA": f"{summary(FA)[0][0] / summary(FAB)[0][0]:.1f}",
             "TenCCoGainB": f"{summary(FA)[0][1] / summary(FAB)[0][1]:.1f}",
             "TenCCoGainPriorB": f"{summary(FA)[0][1] / summary(FAP)[0][1]:.1f}"})

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.5))
ax = axs[0]
ellipse(ax, FA, color=SERIES[0], lw=1.6, label=r"A: $0\leq x\leq1$")
ellipse(ax, FB, color=SERIES[1], lw=1.6, label=r"B: $2\leq x\leq3$")
ellipse(ax, FAB, color="k", lw=2.0, label=r"A+B: $\mathbf{F}_A+\mathbf{F}_B$")
ax.plot(*TRUE, "k+", ms=9)
ax.set_xlabel("intercept $a$"); ax.set_ylabel("slope $b$")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title(r"(a) two experiments, $\Delta\chi^2=2.30$", fontsize=9)
ax = axs[1]
ellipse(ax, FA, color=SERIES[0], lw=1.6, label="A alone")
ax.axvspan(TRUE[0] - SIG_PRIOR_A, TRUE[0] + SIG_PRIOR_A, color=SERIES[2], alpha=0.15,
           label=rf"prior $\sigma_a={SIG_PRIOR_A}$")
ellipse(ax, FAP, color=SERIES[2], lw=2.0, label=r"A + prior")
ax.plot(*TRUE, "k+", ms=9)
ax.set_xlabel("intercept $a$"); ax.set_ylabel("slope $b$")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("(b) a prior is one more Fisher matrix", fontsize=9)
axs[0].set_xlim(0.55, 1.45); axs[0].set_ylim(0.3, 0.7)
axs[1].set_xlim(0.85, 1.15); axs[1].set_ylim(0.3, 0.7)
savefig(fig, "ch10", "combine")
# naive combination of the two marginalised slope errors (the pitfall in the text)
nums["TenCCoNaive"] = f"{1/np.sqrt(1/float(nums['TenCCoASb'])**2+1/float(nums['TenCCoBSb'])**2):.3f}"
save_numbers("ch10", "03_combine", nums)
print(nums)
