"""01_cowan_two_param.py -- a two-parameter Fisher matrix computed before any data exist.

Question: Cowan (Statistical Data Analysis, sec. 6.8) fits the angular distribution
f(x; alpha, beta) = (1 + alpha x + beta x^2) / norm on -0.95 <= x <= 0.95 to n = 2000 events and
quotes alpha = 0.508 +- 0.052, beta = 0.47 +- 0.11, r = 0.46 (one data set, numerical Hessian) and,
from 500 simulated experiments, s_alpha = 0.051, s_beta = 0.111, r = 0.42.  Can the Fisher matrix,
an integral over the model alone, predict all of these numbers without simulating anything?

Computes: the Fisher matrix per event, I_ij = int d_i f d_j f / f dx (quadrature), its inverse for
n = 2000; 500 simulated experiments fitted by Newton-Raphson (the same procedure Cowan used);
the fraction of fitted points inside the Fisher ellipses Delta chi^2 = 2.30 and 6.18; and, for a range
of event numbers n, the ratio of the simulated variance of alpha-hat to the Cramer-Rao bound.
Writes: figures/ch10/cowan_scatter.pdf, figures/ch10/cowan_bound.pdf, results/ch10/01_cowan_two_param.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

setup()
rng = rng_for("ch10", "01_cowan_two_param")
XMIN, XMAX = -0.95, 0.95
TRUE = np.array([0.5, 0.5])          # (alpha, beta), Cowan's values
N_EVENTS, N_EXP = 2000, 500
XG = np.linspace(XMIN, XMAX, 201)    # a density must stay positive on the whole range, not only at the data


def norm(a, b):
    """Normalisation of 1 + a x + b x^2 on [XMIN, XMAX]."""
    return (XMAX - XMIN) + a * (XMAX**2 - XMIN**2) / 2 + b * (XMAX**3 - XMIN**3) / 3


def pdf(x, a, b):
    return (1 + a * x + b * x**2) / norm(a, b)


def fisher_per_event(a, b):
    """I_ij = E[d_i ln f d_j ln f] = int (d_i f)(d_j f)/f dx, with d_i f by hand."""
    n = norm(a, b)
    dn = np.array([(XMAX**2 - XMIN**2) / 2, (XMAX**3 - XMIN**3) / 3])   # d norm / d(a, b)

    def dlnf(x):
        g = 1 + a * x + b * x**2
        return np.array([x / g - dn[0] / n, x**2 / g - dn[1] / n])

    I = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            I[i, j] = integrate.quad(lambda x: dlnf(x)[i] * dlnf(x)[j] * pdf(x, a, b), XMIN, XMAX)[0]
    return I


def sample(n, size):
    """Accept-reject from 1 + a x + b x^2 (its maximum on the range is at x = XMAX for a, b > 0)."""
    fmax = 1 + TRUE[0] * XMAX + TRUE[1] * XMAX**2
    out = np.empty((size, n))
    for k in range(size):
        got = []
        while sum(len(g) for g in got) < n:
            x = rng.uniform(XMIN, XMAX, 2 * n)
            u = rng.uniform(0, fmax, 2 * n)
            got.append(x[u < 1 + TRUE[0] * x + TRUE[1] * x**2])
        out[k] = np.concatenate(got)[:n]
    return out


def fit(x, iters=30):
    """Newton-Raphson on ln L for every row of x at once. Returns MLEs and observed information."""
    th = np.tile(TRUE, (x.shape[0], 1)).astype(float)
    dn = np.array([(XMAX**2 - XMIN**2) / 2, (XMAX**3 - XMIN**3) / 3])
    for _ in range(iters):
        a, b = th[:, :1], th[:, 1:]
        g = 1 + a * x + b * x**2
        nn = norm(a, b)
        n_ev = x.shape[1]
        # score: sum_k d_i ln g_k - n d_i ln norm
        S = np.stack([np.sum(x / g, 1) - n_ev * dn[0] / nn[:, 0],
                      np.sum(x**2 / g, 1) - n_ev * dn[1] / nn[:, 0]], 1)
        # minus Hessian: sum_k x^(i+j)/g^2 - n dn_i dn_j / norm^2
        H = np.empty((x.shape[0], 2, 2))
        for i in range(2):
            for j in range(2):
                H[:, i, j] = np.sum(x**(i + j + 2) / g**2, 1) - n_ev * dn[i] * dn[j] / nn[:, 0]**2
        step = np.linalg.solve(H, S[..., None])[..., 0]
        # damping: halve the step of any experiment whose new density would go negative somewhere
        for _h in range(30):
            new = th + step
            bad = np.any(1 + new[:, :1] * XG + new[:, 1:] * XG**2 <= 0, 1)
            if not bad.any():
                break
            step[bad] *= 0.5
        th = th + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return th, H


# --- 1. the forecast: no data at all
I1 = fisher_per_event(*TRUE)
F = N_EVENTS * I1
cov = np.linalg.inv(F)
sa, sb = np.sqrt(np.diag(cov))
rho = cov[0, 1] / (sa * sb)
cond_a, cond_b = 1 / np.sqrt(F[0, 0]), 1 / np.sqrt(F[1, 1])

# --- 2. the check: 500 experiments of 2000 events, each fitted
x = sample(N_EVENTS, N_EXP)
th, H = fit(x)
mc_cov = np.cov(th.T)
msa, msb = np.sqrt(np.diag(mc_cov))
mrho = mc_cov[0, 1] / (msa * msb)
# observed-information errors of the first experiment (Cowan's single-data-set numbers)
obs_cov0 = np.linalg.inv(H[0])
d = th - TRUE
q = np.einsum("ki,ij,kj->k", d, F, d)            # Delta chi^2 of each fitted point under the Fisher ellipse
in68, in95 = np.mean(q <= 2.30), np.mean(q <= 6.18)
# MC errors of those fractions (binomial)
e68, e95 = np.sqrt(0.683 * 0.317 / N_EXP), np.sqrt(0.954 * 0.046 / N_EXP)

# --- figure: the fitted points and the ellipses
fig, ax = plt.subplots(figsize=(5.0, 4.4))
ax.scatter(th[:, 0], th[:, 1], s=5, color=SERIES[0], alpha=0.55, lw=0, label="500 fitted experiments")
t = np.linspace(0, 2 * np.pi, 400)
L = np.linalg.cholesky(cov)
for k2, ls, lab in [(2.30, "-", r"Fisher, $\Delta\chi^2=2.30$"), (6.18, "--", r"Fisher, $\Delta\chi^2=6.18$")]:
    e = TRUE[:, None] + np.sqrt(k2) * L @ np.vstack([np.cos(t), np.sin(t)])
    ax.plot(e[0], e[1], color="k", ls=ls, lw=1.3, label=lab)
ax.axvline(TRUE[0] - sa, color=SERIES[1], lw=0.9, ls=":")
ax.axvline(TRUE[0] + sa, color=SERIES[1], lw=0.9, ls=":", label=r"$\alpha\pm\sigma_\alpha$ (marginal)")
ax.plot([TRUE[0] - cond_a, TRUE[0] + cond_a], [TRUE[1]] * 2, color=SERIES[2], lw=2.4,
        label=r"$\alpha\pm1/\sqrt{F_{\alpha\alpha}}$ ($\beta$ fixed)")
ax.plot(*TRUE, "k+", ms=10, mew=1.5)
ax.set_xlabel(r"$\hat\alpha$")
ax.set_ylabel(r"$\hat\beta$")
ax.set_ylim(top=1.0)
ax.legend(loc="upper left", fontsize=7.5, ncol=2, columnspacing=1.0)
savefig(fig, "ch10", "cowan_scatter")

# --- 3. how fast is the bound reached?  n Var(alpha-hat) against the per-event bound (I1^-1)_aa
ns = np.array([50, 100, 200, 500, 1000, 2000])
ratio_a, ratio_b, err, edge, run = [], [], [], [], []
n_rep = 2000
for n in ns:
    th_n, _ = fit(sample(n, n_rep))
    ok = np.all(np.isfinite(th_n), 1) & (np.abs(th_n).max(1) < 20)   # runaway fits: MLE heading to infinity
    run.append(1 - ok.mean())
    edge.append(np.mean(np.min(1 + th_n[:, :1] * XG + th_n[:, 1:] * XG**2, 1) < 1e-3))
    v = np.var(th_n[ok], axis=0, ddof=1)
    bound = np.diag(np.linalg.inv(n * I1))
    ratio_a.append(v[0] / bound[0]); ratio_b.append(v[1] / bound[1])
    err.append(np.sqrt(2 / ok.sum()))
ratio_a, ratio_b, err = map(np.array, (ratio_a, ratio_b, err))

fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.errorbar(ns, ratio_a, yerr=err * ratio_a, fmt="o-", color=SERIES[0], capsize=2, label=r"$\hat\alpha$")
ax.errorbar(ns * 1.04, ratio_b, yerr=err * ratio_b, fmt="s-", color=SERIES[1], capsize=2, label=r"$\hat\beta$")
theory_line(ax, [40, 2500], [1, 1], label="Cramér–Rao bound")
ax.set_xscale("log")
ax.set_xlabel("events per experiment $n$")
ax.set_ylabel(r"$\mathrm{Var}(\hat\theta)\,/\,(\mathbf{F}^{-1})_{\theta\theta}$")
ax.legend()
savefig(fig, "ch10", "cowan_bound")

save_numbers("ch10", "01_cowan_two_param", {
    "TenACowFaa": f"{F[0, 0]:.0f}", "TenACowFab": f"{F[0, 1]:.0f}", "TenACowFbb": f"{F[1, 1]:.0f}",
    "TenACowIaa": f"{I1[0, 0]:.4f}", "TenACowIab": f"{I1[0, 1]:.4f}", "TenACowIbb": f"{I1[1, 1]:.4f}",
    "TenACowSa": f"{sa:.3f}", "TenACowSb": f"{sb:.3f}", "TenACowRho": f"{rho:.2f}",
    "TenACowCondA": f"{cond_a:.3f}", "TenACowCondB": f"{cond_b:.3f}",
    "TenACowMSa": f"{msa:.3f}", "TenACowMSb": f"{msb:.3f}", "TenACowMRho": f"{mrho:.2f}",
    "TenACowMeanA": f"{th[:, 0].mean():.3f}", "TenACowMeanB": f"{th[:, 1].mean():.3f}",
    "TenACowOneA": f"{th[0, 0]:.3f}", "TenACowOneB": f"{th[0, 1]:.3f}",
    "TenACowOneSa": f"{np.sqrt(obs_cov0[0, 0]):.3f}", "TenACowOneSb": f"{np.sqrt(obs_cov0[1, 1]):.3f}",
    "TenACowOneRho": f"{obs_cov0[0, 1] / np.sqrt(obs_cov0[0, 0] * obs_cov0[1, 1]):.2f}",
    "TenACowInSixEight": f"{in68:.3f}", "TenACowInNineFive": f"{in95:.3f}",
    "TenACowErrSixEight": f"{e68:.3f}", "TenACowErrNineFive": f"{e95:.3f}",
    "TenACowNexp": N_EXP, "TenACowNev": N_EVENTS, "TenACowNrep": n_rep,
    "TenACowRatioFifty": f"{ratio_a[0]:.2f}", "TenACowRatioFiftyB": f"{ratio_b[0]:.2f}",
    "TenACowRatioTwoK": f"{ratio_a[-1]:.2f}", "TenACowRatioTwoKB": f"{ratio_b[-1]:.2f}",
    "TenACowRatioErr": f"{err[0]:.2f}",
    "TenACowRatioHundred": f"{ratio_a[1]:.2f}", "TenACowRatioHundredB": f"{ratio_b[1]:.2f}",
    "TenACowRunFifty": f"{100 * run[0]:.1f}", "TenACowRunHundred": f"{100 * run[1]:.1f}",
    "TenACowEdgeFifty": f"{100 * edge[0]:.1f}", "TenACowEdgeHundred": f"{100 * edge[1]:.1f}",
})
print("F =", F, "\nsa sb rho", sa, sb, rho, "\nMC", msa, msb, mrho, "\nin68/95", in68, in95)
print("ratios a", ratio_a, "b", ratio_b, "edge", edge, "run", run)
