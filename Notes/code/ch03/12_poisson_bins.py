"""12_poisson_bins.py -- fitting a histogram of counts: Poisson likelihood versus chi^2.

Question: a histogram of decay times with only a few counts per bin is fitted for the
normalisation nu (expected total) and the lifetime tau.  Three recipes are common:
  Poisson maximum likelihood (the Cash statistic / Baker-Cousins chi^2_lambda),
  Neyman chi^2 (variance taken from the observed n_i),
  Pearson chi^2 (variance taken from the predicted mu_i).
Which of them gets the normalisation right?  Is the Baker-Cousins chi^2_lambda at its minimum
distributed as chi^2 with N_bins - 2 degrees of freedom, so that it can test goodness of fit?

Computes: one example fit, then 3000 simulated histograms fitted by all three methods;
          the mean fitted normalisation compared with the observed total and with Cowan's
          predictions nu_Pearson = n + chi^2/2, nu_Neyman = n - chi^2.
Writes:   figures/ch03/poisson_fit.pdf, figures/ch03/poisson_bias.pdf, results/ch03/12_poisson_bins.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize, special

rng = rng_for("ch03", "12_poisson_bins")
setup()
nu_true, tau_true, tmax, K, M = 120.0, 1.0, 5.0, 20, 3000
edges = np.linspace(0, tmax, K + 1)


def mu(par):
    """expected counts per bin for total nu and lifetime tau (pdf normalised on [0, tmax])"""
    nu, tau = par
    cdf = 1 - np.exp(-edges / tau)
    return nu * np.diff(cdf) / cdf[-1]


def cash(par, n):          # -2 ln L up to a constant: 2 sum (mu - n ln mu)
    m = mu(par)
    return 2 * np.sum(m - n * np.log(m))


def baker_cousins(par, n):  # 2 sum (mu - n + n ln(n/mu)), with n ln n -> 0 for n = 0
    m = mu(par)
    return 2 * np.sum(m - n + special.xlogy(n, n) - special.xlogy(n, m))


def neyman(par, n):
    m = mu(par)
    return np.sum((n - m) ** 2 / np.maximum(n, 1))


def pearson(par, n):
    m = mu(par)
    return np.sum((n - m) ** 2 / m)


def fit(fun, n):
    start = np.array([n.sum(), 1.0])
    r = optimize.minimize(fun, start, args=(n,), method="Nelder-Mead",
                          options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
    return r.x, r.fun


# ---------- one example ----------
n0 = rng.poisson(mu([nu_true, tau_true]))
res = {name: fit(f, n0) for name, f in [("P", cash), ("N", neyman), ("S", pearson)]}
bc0 = baker_cousins(res["P"][0], n0)

fig, ax = plt.subplots(figsize=(5.4, 3.2))
cent = 0.5 * (edges[1:] + edges[:-1])
ax.errorbar(cent, n0, np.sqrt(np.maximum(n0, 1)), fmt="o", ms=3, color="k", capsize=0, label="counts")
tt = np.linspace(0, tmax, 300)
for (name, lab), c in zip([("P", "Poisson ML"), ("N", r"Neyman $\chi^2$"), ("S", r"Pearson $\chi^2$")], SERIES):
    nu, tau = res[name][0]
    dens = nu * (tmax / K) * np.exp(-tt / tau) / tau / (1 - np.exp(-tmax / tau))
    ax.plot(tt, dens, color=c, label=rf"{lab}: $\hat\nu={nu:.1f}$")
ax.set_xlabel(r"decay time $t$"); ax.set_ylabel("counts per bin"); ax.legend(fontsize=7)
ax.set_title(rf"observed total $n={n0.sum()}$", fontsize=9)
fig.tight_layout()
savefig(fig, "ch03", "poisson_fit")

# ---------- many histograms ----------
out = {k: np.empty((M, 2)) for k in "PNS"}
fmin = {k: np.empty(M) for k in "PNS"}
ntot = np.empty(M); bc = np.empty(M)
for j in range(M):
    n = rng.poisson(mu([nu_true, tau_true]))
    ntot[j] = n.sum()
    for k, f in [("P", cash), ("N", neyman), ("S", pearson)]:
        out[k][j], fmin[k][j] = fit(f, n)
    bc[j] = baker_cousins(out["P"][j], n)

ratio = {k: out[k][:, 0] - ntot for k in "PNS"}
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0))
bins = np.linspace(-30, 20, 101)
for (k, lab), c in zip([("P", "Poisson ML"), ("N", r"Neyman $\chi^2$"), ("S", r"Pearson $\chi^2$")], SERIES):
    axs[0].hist(ratio[k], bins=bins, histtype="step", lw=1.4, color=c, density=True, label=lab)
axs[0].set_xlabel(r"fitted $\hat\nu$ $-$ observed total $n$"); axs[0].legend(fontsize=7)
cc = np.linspace(0.01, 50, 400)
axs[1].hist(bc, bins=np.linspace(0, 50, 76), density=True, color=SERIES[0], alpha=0.55,
            label=r"Baker--Cousins $\chi^2_\lambda$ at min")
theory_line(axs[1], cc, stats.chi2.pdf(cc, K - 2), label=rf"$\chi^2_{{{K-2}}}$")
axs[1].set_xlabel(r"$\chi^2_{\lambda,\min}$"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch03", "poisson_bias")

save_numbers("ch03", "12_poisson_bins", {
    "ThreeBPoisNobs": int(n0.sum()),
    "ThreeBPoisNuP": res["P"][0][0], "ThreeBPoisNuN": res["N"][0][0], "ThreeBPoisNuS": res["S"][0][0],
    "ThreeBPoisTauP": res["P"][0][1], "ThreeBPoisTauN": res["N"][0][1], "ThreeBPoisTauS": res["S"][0][1],
    "ThreeBPoisChiN": res["N"][1], "ThreeBPoisChiS": res["S"][1], "ThreeBPoisBC": bc0,
    "ThreeBPoisZeroBins": int((n0 == 0).sum()),
    "ThreeBPoisDiffP": np.abs(ratio["P"]).max(), "ThreeBPoisDiffN": ratio["N"].mean(), "ThreeBPoisDiffS": ratio["S"].mean(),
    "ThreeBPoisPredN": -fmin["N"].mean(), "ThreeBPoisPredS": fmin["S"].mean() / 2,
    "ThreeBPoisTauMeanP": out["P"][:, 1].mean(), "ThreeBPoisTauMeanN": out["N"][:, 1].mean(),
    "ThreeBPoisTauMeanS": out["S"][:, 1].mean(),
    "ThreeBPoisBCmean": bc.mean(), "ThreeBPoisDof": K - 2,
})
