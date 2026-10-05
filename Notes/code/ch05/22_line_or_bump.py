"""22_line_or_bump.py -- Occam's razor at work: a line, or a line plus a bump?

Question: forty points scattered about a straight line.  Does a Gaussian bump
of known width sit on top of it?  The bump model always fits at least as well
(set A = 0 and it *is* the line), so the best-fit chi^2 can never decide.  The
evidence can: it averages the likelihood over each model's prior, and a model
that spreads its prior over (A, mu) values the data reject pays for it.

Computes, for a data set with a bump (A = 2.5 at mu = 6.2) and one without:
  * the best fits and chi^2_min of both models,
  * the exact log Bayes factor ln B_10 = ln Z(bump) - ln Z(line) (lib_bump),
  * its split into best-fit likelihood ratio and Occam factor, and the Laplace
    (Gaussian) approximation of the bump evidence,
  * the Bayes factor when the bump position is known in advance,
  * ln B_10 as a function of the prior width A_max of the amplitude.

Writes: figures/ch05/line_bump_fits.pdf, figures/ch05/line_bump_map.pdf,
        figures/ch05/line_bump_width.pdf, results/ch05/22_line_or_bump.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import lib_bump as lb

setup()


def fit_bump(y):
    """Best fit of the bump model: start from the (A, mu) grid minimum, polish all four."""
    A = np.linspace(0, lb.A_MAX, 401); mu = np.linspace(*lb.MU_RANGE, 401)
    dc = lb.dchi2_grid(y, A, mu)
    i, j = np.unravel_index(np.argmin(dc), dc.shape)
    ab = np.linalg.lstsq(lb.DESIGN, y - A[i] * lb.bump(lb.X, mu[j]), rcond=None)[0]
    res = optimize.minimize(lambda t: lb.chi2(t, y), [*ab, A[i], mu[j]], method="Nelder-Mead",
                            options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000))
    return res.x, res.fun


def hessian(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i = np.eye(n)[i] * h; e_j = np.eye(n)[j] * h
            H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
    return H


out = {}
data = {"bump": lb.make_data(rng_for("ch05", "22_line_or_bump", 0), True),
        "flat": lb.make_data(rng_for("ch05", "22_line_or_bump", 1), False)}
fits = {}
for key, y in data.items():
    ab = np.linalg.lstsq(lb.DESIGN, y, rcond=None)[0]
    c0 = lb.chi2min_line(y)
    th, c1 = fit_bump(y)
    lnB = lb.ln_bayes_factor(y)
    lnB_coarse = lb.ln_bayes_factor(y, nA=1001, nmu=1001)          # grid convergence check
    fits[key] = dict(ab=ab, th=th, c0=c0, c1=c1, lnB=lnB)
    tag = "Bump" if key == "bump" else "Flat"
    out.update({f"FiveC{tag}ChiLine": c0, f"FiveC{tag}ChiBump": c1, f"FiveC{tag}DChi": c0 - c1,
                f"FiveC{tag}lnLR": 0.5 * (c0 - c1), f"FiveC{tag}lnB": lnB,
                f"FiveC{tag}lnOccam": lnB - 0.5 * (c0 - c1), f"FiveC{tag}B": np.exp(lnB),
                f"FiveC{tag}Ahat": th[2], f"FiveC{tag}Muhat": th[3],
                f"FiveC{tag}lnZline": lb.ln_evidence_line(y),
                f"FiveC{tag}lnZbump": lb.ln_evidence_line(y) + lnB,
                f"FiveC{tag}GridErr": abs(lnB - lnB_coarse)})

# ---- Laplace approximation for the bump data: evidence = L_max x (posterior volume / prior volume)
y = data["bump"]; th = fits["bump"]["th"]
F1 = hessian(lambda t: 0.5 * lb.chi2(t, y), th)                   # Fisher matrix at the peak, 4x4
cov = np.linalg.inv(F1)
lnOcc1 = 0.5 * 4 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(F1)) - np.log(lb.V_AB * lb.A_MAX * lb.L_MU)
lnOcc0 = lb.ln_ab_integral()                                       # exact for the line
lnZ1_lap = lb.ln_norm() - 0.5 * fits["bump"]["c1"] + lnOcc1
out.update({"FiveCLapLnZbump": lnZ1_lap, "FiveCLapLnB": lnZ1_lap - lb.ln_evidence_line(y),
            "FiveCLapLnOccOne": lnOcc1, "FiveCLapLnOccZero": lnOcc0,
            "FiveCLapOccRatio": np.exp(lnOcc1 - lnOcc0),
            "FiveCSigA": np.sqrt(cov[2, 2]), "FiveCSigMu": np.sqrt(cov[3, 3]),
            "FiveCSigAMuVol": 2 * np.pi * np.sqrt(np.linalg.det(cov[2:, 2:])),
            "FiveCPriorAMuVol": lb.A_MAX * lb.L_MU})

# ---- bump position known in advance (a physicist told where to look)
lnB_fix = lb.ln_bayes_factor(y, mu_fixed=lb.TRUE["mu"])
# local significance of the amplitude at that fixed position (linear least squares)
Phi = np.column_stack([lb.DESIGN, lb.bump(lb.X, lb.TRUE["mu"])])
Cfix = np.linalg.inv(Phi.T @ Phi / lb.SIGMA**2)
Afix = (Cfix @ Phi.T @ y / lb.SIGMA**2)[2]
out.update({"FiveCFixLnB": lnB_fix, "FiveCFixB": np.exp(lnB_fix),
            "FiveCFreeOverFix": np.exp(lnB_fix - fits["bump"]["lnB"]),
            "FiveCFixAhat": Afix, "FiveCFixSigA": np.sqrt(Cfix[2, 2]),
            "FiveCFixZscore": Afix / np.sqrt(Cfix[2, 2])})

# ---- dependence on the prior width of the amplitude
amax = np.logspace(0, 3, 40)
lnB_w = {k: np.array([lb.ln_bayes_factor(v, a_max=m, nmu=801, nA=4001) for m in amax]) for k, v in data.items()}
out.update({"FiveCBumpLnBHundred": lb.ln_bayes_factor(data["bump"], a_max=100.0, nA=8001),
            "FiveCFlatLnBHundred": lb.ln_bayes_factor(data["flat"], a_max=100.0, nA=8001),
            "FiveCFlatLnBOne": lb.ln_bayes_factor(data["flat"], a_max=1.0)})

# ---- figures
xx = np.linspace(0, 10, 400)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
for ax, (key, y) in zip(axes, data.items()):
    f = fits[key]
    ax.errorbar(lb.X, y, yerr=lb.SIGMA, fmt="o", ms=3, color="0.35", elinewidth=0.6, capsize=0)
    ax.plot(xx, f["ab"][0] + f["ab"][1] * xx, color=SERIES[0], label=rf"line, $\chi^2_{{\min}}={f['c0']:.1f}$")
    ax.plot(xx, lb.model(f["th"], xx), color=SERIES[1], label=rf"line+bump, $\chi^2_{{\min}}={f['c1']:.1f}$")
    ax.set_title(("data with a bump" if key == "bump" else "data without a bump") + rf":  $\ln B_{{10}}={f['lnB']:.1f}$",
                 fontsize=9)
    ax.set_xlabel("$x$"); ax.legend(fontsize=7, loc="upper left")
axes[0].set_ylabel("$y$")
savefig(fig, "ch05", "line_bump_fits")

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for ax, (key, y) in zip(axes, data.items()):
    _, A, mu, dc = lb.ln_bayes_factor(y, nA=301, nmu=301, return_grid=True)
    lr = -0.5 * dc / np.log(10)
    im = ax.pcolormesh(mu, A, np.clip(lr, -4, None), cmap="Blues", vmin=-4, vmax=max(1, lr.max()), shading="auto")
    ax.contour(mu, A, lr, levels=[0], colors=[SERIES[1]], linewidths=1.0)
    ax.set_xlabel(r"bump position $\mu$")
    ax.set_title("with a bump" if key == "bump" else "without a bump", fontsize=9)
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ likelihood ratio to best line", shrink=0.9)
axes[0].set_ylabel(r"bump amplitude $A$")
savefig(fig, "ch05", "line_bump_map")

fig, ax = plt.subplots(figsize=(5.6, 3.0))
for c, (key, v) in zip(SERIES, lnB_w.items()):
    ax.semilogx(amax, v, color=c, label="data with a bump" if key == "bump" else "data without a bump")
ax.axhline(0, color="k", lw=0.7)
ax.set_xlabel(r"prior range of the amplitude $A_{\max}$"); ax.set_ylabel(r"$\ln B_{10}$ (bump vs line)")
ax.legend(fontsize=8)
savefig(fig, "ch05", "line_bump_width")

out.update({"FiveCAmax": lb.A_MAX, "FiveCNdata": lb.N_DATA, "FiveCWidth": lb.W,
            "FiveCTrueA": lb.TRUE["A"], "FiveCTrueMu": lb.TRUE["mu"]})
save_numbers("ch05", "22_line_or_bump", out)
for k, v in out.items():
    print(f"{k:24s} {v:.4g}")
