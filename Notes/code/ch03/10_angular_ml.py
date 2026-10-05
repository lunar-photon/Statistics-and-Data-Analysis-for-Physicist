"""10_angular_ml.py -- a two-parameter maximum-likelihood fit done by hand with Newton-Raphson.

Question: an angular distribution f(x; alpha, beta) = (1 + alpha x + beta x^2)/norm on
-0.95 <= x <= 0.95 (x = cos theta, as for e+e- -> mu+mu-) has no closed-form MLE.  Can
Newton-Raphson, started from the method-of-moments estimate, find it?  Do the errors and the
correlation read off the Hessian of ln L agree with the scatter of the MLEs over many
simulated experiments, and is the MLE more precise than the method of moments?

Computes: one experiment of 2000 events (alpha = beta = 0.5), its MoM and ML estimates, the
          Newton iterations, the covariance from the inverse Hessian, the Delta lnL = 1/2
          contour; then 500 experiments to get the true spreads of the ML and MoM estimators.
Writes:   figures/ch03/angular_ml.pdf, results/ch03/10_angular_ml.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch03", "10_angular_ml")
setup()
a_true, b_true, xmin, xmax, n, M = 0.5, 0.5, -0.95, 0.95, 2000, 500
d = {k: (xmax**k - xmin**k) / k for k in range(1, 6)}      # d_k of Cowan (8.9)


def sample(size):
    """accept-reject from the flat envelope on [xmin, xmax]"""
    out = np.empty(0)
    gmax = 1 + abs(a_true) + abs(b_true)
    while out.size < size:
        x = rng.uniform(xmin, xmax, 2 * size)
        keep = rng.uniform(0, gmax, 2 * size) < 1 + a_true * x + b_true * x**2
        out = np.concatenate([out, x[keep]])
    return out[:size]


def lnL(a, b, x):
    D = d[1] + a * d[2] + b * d[3]
    return np.sum(np.log(1 + a * x + b * x**2)) - len(x) * np.log(D)


def grad_hess(th, x):
    a, b = th
    g = 1 + a * x + b * x**2
    D = d[1] + a * d[2] + b * d[3]
    N = len(x)
    grad = np.array([np.sum(x / g) - N * d[2] / D, np.sum(x**2 / g) - N * d[3] / D])
    H = np.array([[-np.sum(x**2 / g**2) + N * d[2] ** 2 / D**2, -np.sum(x**3 / g**2) + N * d[2] * d[3] / D**2],
                  [-np.sum(x**3 / g**2) + N * d[2] * d[3] / D**2, -np.sum(x**4 / g**2) + N * d[3] ** 2 / D**2]])
    return grad, H


def mom(x):
    """method of moments, Cowan (8.10)-(8.12): match <x> and <x^2>"""
    e1, e2 = x.mean(), (x**2).mean()
    A = np.array([[e1 * d[2] - d[3], e1 * d[3] - d[4]], [e2 * d[2] - d[4], e2 * d[3] - d[5]]])
    rhs = np.array([d[2] - e1 * d[1], d[3] - e2 * d[1]])
    return np.linalg.solve(A, rhs)


def newton(x, th0, tol=1e-10, maxit=50):
    th, path = np.array(th0, float), [np.array(th0, float)]
    for it in range(maxit):
        g, H = grad_hess(th, x)
        step = -np.linalg.solve(H, g)                # theta_{j+1} = theta_j - H^{-1} grad
        th = th + step
        path.append(th.copy())
        if np.max(np.abs(step)) < tol:
            break
    return th, np.array(path)


# ---------- one experiment ----------
x = sample(n)
th_mom = mom(x)
th_ml, path = newton(x, th_mom)
_, H = grad_hess(th_ml, x)
V = np.linalg.inv(-H)                                # covariance = inverse of minus the Hessian
sa, sb = np.sqrt(np.diag(V))
rho = V[0, 1] / (sa * sb)

# ---------- 500 experiments ----------
ml = np.empty((M, 2)); mm = np.empty((M, 2)); rho_hess = np.empty(M)
for k in range(M):
    xs = sample(n)
    mm[k] = mom(xs)
    ml[k], _ = newton(xs, mm[k])
    Vk = np.linalg.inv(-grad_hess(ml[k], xs)[1])      # this experiment's own inverse-Hessian covariance
    rho_hess[k] = Vk[0, 1] / np.sqrt(Vk[0, 0] * Vk[1, 1])
sd_ml = ml.std(0, ddof=1); sd_mm = mm.std(0, ddof=1)
rho_mc = np.corrcoef(ml.T)[0, 1]
rho_mc_err = (1 - rho_mc**2) / np.sqrt(M)            # statistical error of a correlation from M pairs
# moments versus ML: the same 500 data sets, so resample the pairs together (separate stream)
rb = rng_for("ch03", "10_angular_ml_pairs")
ratio_boot = np.empty((1000, 2))
for j in range(1000):
    ii = rb.integers(0, M, M)
    ratio_boot[j] = mm[ii].std(0, ddof=1) / ml[ii].std(0, ddof=1)
ratio_mm = sd_mm / sd_ml
ratio_err = ratio_boot.std(0, ddof=1)

# ---------- figure ----------
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
bins = np.linspace(xmin, xmax, 39)
axs[0].hist(x, bins=bins, density=True, histtype="step", color=SERIES[0], lw=1.3, label="2000 events")
xx = np.linspace(xmin, xmax, 300)
D = d[1] + th_ml[0] * d[2] + th_ml[1] * d[3]
axs[0].plot(xx, (1 + th_ml[0] * xx + th_ml[1] * xx**2) / D, color=SERIES[1], label="ML fit")
axs[0].set_xlabel(r"$x=\cos\theta$"); axs[0].set_ylabel(r"$f(x;\hat\alpha,\hat\beta)$"); axs[0].legend(fontsize=7)

axs[1].scatter(ml[:, 0], ml[:, 1], s=3, color=SERIES[2], alpha=0.6, label="MLEs, 500 experiments")
ag = np.linspace(th_ml[0] - 3 * sa, th_ml[0] + 3 * sa, 161)
bg = np.linspace(th_ml[1] - 3 * sb, th_ml[1] + 3 * sb, 161)
AA, BB = np.meshgrid(ag, bg)
Z = np.vectorize(lambda a, b: lnL(a, b, x))(AA, BB) - lnL(*th_ml, x)
axs[1].contour(AA, BB, Z, levels=[-0.5], colors=[SERIES[0]], linewidths=1.6, linestyles="solid", zorder=6)
for v in (th_ml[0] - sa, th_ml[0] + sa):
    axs[1].axvline(v, color=SERIES[0], lw=0.6, ls=":")
for v in (th_ml[1] - sb, th_ml[1] + sb):
    axs[1].axhline(v, color=SERIES[0], lw=0.6, ls=":")
axs[1].plot(*th_ml, "o", color=SERIES[0], ms=4, label="this experiment")
axs[1].plot(a_true, b_true, "*", color=SERIES[1], ms=8, label="true value")
axs[1].set_xlabel(r"$\hat\alpha$"); axs[1].set_ylabel(r"$\hat\beta$"); axs[1].legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
savefig(fig, "ch03", "angular_ml")

save_numbers("ch03", "10_angular_ml", {
    "ThreeBAngMomA": th_mom[0], "ThreeBAngMomB": th_mom[1],
    "ThreeBAngA": th_ml[0], "ThreeBAngB": th_ml[1],
    "ThreeBAngSa": sa, "ThreeBAngSb": sb, "ThreeBAngRho": rho,
    "ThreeBAngIter": len(path) - 1,
    "ThreeBAngStepOneA": path[1][0], "ThreeBAngStepOneB": path[1][1],
    "ThreeBAngSdA": sd_ml[0], "ThreeBAngSdB": sd_ml[1], "ThreeBAngRhoMC": rho_mc,
    "ThreeBAngMomSdA": sd_mm[0], "ThreeBAngMomSdB": sd_mm[1],
    "ThreeBAngMeanA": ml[:, 0].mean(), "ThreeBAngMeanB": ml[:, 1].mean(),
    "ThreeBAngRhoMCErr": rho_mc_err,
    "ThreeBAngRhoHessMean": rho_hess.mean(), "ThreeBAngRhoHessSd": rho_hess.std(ddof=1),
    "ThreeBAngMomRatioA": ratio_mm[0], "ThreeBAngMomRatioAErr": ratio_err[0],
    "ThreeBAngMomRatioB": ratio_mm[1], "ThreeBAngMomRatioBErr": ratio_err[1],
})
