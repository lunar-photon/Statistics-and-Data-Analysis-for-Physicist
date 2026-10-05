"""09_conjugate.py -- the three classic conjugate pairs, checked against a brute-force grid.

Question: when prior and likelihood have matching functional forms, the posterior is
in the prior's family with updated parameters.  Do the closed-form update rules
agree with the brute-force "multiply on a grid and normalise" posterior?

Computes:
  * Beta-Binomial: 17 heads in 20 tosses with three priors (Beta(100,100),
    Beta(18.25,6.75), Beta(1,1)): posterior modes and 95% HDIs;
  * Gamma-Poisson: muon counts in 5 one-minute windows, prior Gamma(3, 1 per min);
  * Normal-Normal: three measurements with different error bars and a Normal prior:
    posterior mean/sd, and the inverse-variance weighted mean it tends to;
  * the posterior predictive probability of two more heads after 10 heads in 14
    tosses with a flat prior, B(13,5)/B(11,5), against the plug-in (10/14)^2;
  * for every pair: the largest difference between the closed-form posterior and
    the grid posterior.

Writes: figures/ch05/conjugate_pairs.pdf, figures/ch05/conjugate_three_priors.pdf,
        results/ch05/09_conjugate.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special, optimize

rng = rng_for("ch05", "09_conjugate")
setup()


def grid_posterior(x, prior, like):
    """Brute force: multiply on the grid and normalise so that the density integrates to 1."""
    p = prior * like
    return p / (p.sum() * (x[1] - x[0]))


def hdi(dist, mass=0.95):
    """Shortest interval of a unimodal continuous distribution holding `mass`."""
    width = lambda lo: dist.ppf(lo + mass) - dist.ppf(lo)
    lo = optimize.minimize_scalar(width, bounds=(1e-9, 1 - mass - 1e-9), method="bounded").x
    return dist.ppf(lo), dist.ppf(lo + mass)


# ---------------------------------------------------------------- Beta-Binomial
z, N = 17, 20
th = np.linspace(0, 1, 2001)[1:-1]
priors = {"strong fair": (100, 100), "league": (18.25, 6.75), "flat": (1, 1)}
beta_res = {}
for name, (a, b) in priors.items():
    post = stats.beta(a + z, b + N - z)
    grid = grid_posterior(th, stats.beta(a, b).pdf(th), th**z * (1 - th) ** (N - z))
    mode = (a + z - 1) / (a + b + N - 2)
    beta_res[name] = dict(a=a, b=b, mode=mode, hdi=hdi(post),
                          err=np.abs(grid - post.pdf(th)).max() / post.pdf(th).max())

# ---------------------------------------------------------------- Gamma-Poisson
LAM_TRUE, n_win = 5.0, 5
counts = rng.poisson(LAM_TRUE, n_win)
al0, be0 = 3.0, 1.0                                  # prior mean 3 per minute
lam = np.linspace(1e-4, 14, 3000)
gpost = stats.gamma(al0 + counts.sum(), scale=1 / (be0 + n_win))
ggrid = grid_posterior(lam, stats.gamma(al0, scale=1 / be0).pdf(lam),
                       lam ** counts.sum() * np.exp(-n_win * lam))
gerr = np.abs(ggrid - gpost.pdf(lam)).max() / gpost.pdf(lam).max()

# ---------------------------------------------------------------- Normal-Normal
MU_TRUE = 10.0
sig_k = np.array([0.5, 1.0, 2.0])
x_k = MU_TRUE + sig_k * rng.standard_normal(3)
a_pr, b_pr = 8.0, 3.0
w_k = 1 / sig_k**2
prec = 1 / b_pr**2 + w_k.sum()
m_post = (a_pr / b_pr**2 + np.sum(w_k * x_k)) / prec
s_post = 1 / np.sqrt(prec)
m_ivw = np.sum(w_k * x_k) / w_k.sum()
s_ivw = 1 / np.sqrt(w_k.sum())
mu = np.linspace(4, 16, 3000)
ngrid = grid_posterior(mu, stats.norm(a_pr, b_pr).pdf(mu),
                       np.prod(stats.norm(x_k[None, :], sig_k[None, :]).pdf(mu[:, None]), axis=1))
nerr = np.abs(ngrid - stats.norm(m_post, s_post).pdf(mu)).max() / stats.norm(m_post, s_post).pdf(mu).max()

# ---------------------------------------------------------------- predictive (coin, 10 of 14)
pred = np.exp(special.betaln(13, 5) - special.betaln(11, 5))
plugin = (10 / 14) ** 2

# ---------------------------------------------------------------- figure: the three pairs
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.7))
a, b = 2, 2
pr = stats.beta(a, b).pdf(th); lk = th**z * (1 - th) ** (N - z); lk /= lk.sum() * (th[1] - th[0])
po = stats.beta(a + z, b + N - z).pdf(th); gr = grid_posterior(th, pr, lk)
ax[0].plot(th, pr, color=SERIES[1], ls="--", label="prior")
ax[0].plot(th, lk, color=SERIES[2], ls=":", label="likelihood (scaled)")
ax[0].plot(th, po, color=SERIES[0], label="posterior")
ax[0].plot(th[::100], gr[::100], "o", ms=2.5, color="k", label="grid")
ax[0].set_xlabel(r"$\theta$"); ax[0].set_title(r"Beta$(2,2)$ + 17 of 20 heads", fontsize=8)
ax[0].legend(fontsize=6, loc="upper left")
lk = lam ** counts.sum() * np.exp(-n_win * lam); lk /= lk.sum() * (lam[1] - lam[0])
ax[1].plot(lam, stats.gamma(al0, scale=1 / be0).pdf(lam), color=SERIES[1], ls="--")
ax[1].plot(lam, lk, color=SERIES[2], ls=":")
ax[1].plot(lam, gpost.pdf(lam), color=SERIES[0])
ax[1].plot(lam[::150], ggrid[::150], "o", ms=2.5, color="k")
ax[1].set_xlabel(r"rate $\lambda$ (per minute)")
ax[1].set_title(f"Gamma(3,1) + counts {', '.join(map(str, counts))}", fontsize=8)
lk = np.prod(stats.norm(x_k[None, :], sig_k[None, :]).pdf(mu[:, None]), axis=1); lk /= lk.sum() * (mu[1] - mu[0])
ax[2].plot(mu, stats.norm(a_pr, b_pr).pdf(mu), color=SERIES[1], ls="--")
ax[2].plot(mu, lk, color=SERIES[2], ls=":")
ax[2].plot(mu, stats.norm(m_post, s_post).pdf(mu), color=SERIES[0])
ax[2].plot(mu[::150], ngrid[::150], "o", ms=2.5, color="k")
ax[2].errorbar(x_k, [0.05, 0.1, 0.15], xerr=sig_k, fmt="s", ms=3, color=INK2, capsize=2, lw=0.8)
ax[2].set_xlabel(r"$\mu$"); ax[2].set_title("Normal(8,3) + three measurements", fontsize=8)
for x in ax:
    x.set_yticks([])
fig.tight_layout()
savefig(fig, "ch05", "conjugate_pairs")

# ---------------------------------------------------------------- figure: Kruschke's three priors
fig, ax = plt.subplots(2, 3, figsize=(7.6, 3.6), sharex=True)
for j, (name, r) in enumerate(beta_res.items()):
    a, b = r["a"], r["b"]
    ax[0, j].plot(th, stats.beta(a, b).pdf(th), color=SERIES[1])
    ax[0, j].set_title(f"prior Beta({a:g}, {b:g})", fontsize=8)
    post = stats.beta(a + z, b + N - z)
    ax[1, j].plot(th, post.pdf(th), color=SERIES[0])
    lo, hi = r["hdi"]
    m = th[(th > lo) & (th < hi)]
    ax[1, j].fill_between(m, post.pdf(m), color=SERIES[0], alpha=0.25, lw=0)
    ax[1, j].set_title(f"posterior: mode {r['mode']:.3f}, HDI [{lo:.2f}, {hi:.2f}]", fontsize=7)
    ax[1, j].set_xlabel(r"$\theta$")
    for x in ax[:, j]:
        x.set_yticks([])
fig.tight_layout()
savefig(fig, "ch05", "conjugate_three_priors")

bs, bl, bf = beta_res["strong fair"], beta_res["league"], beta_res["flat"]
save_numbers("ch05", "09_conjugate", {
    "FiveBBetaModeS": bs["mode"], "FiveBBetaModeL": bl["mode"], "FiveBBetaModeF": bf["mode"],
    "FiveBBetaHdiSlo": bs["hdi"][0], "FiveBBetaHdiShi": bs["hdi"][1],
    "FiveBBetaHdiLlo": bl["hdi"][0], "FiveBBetaHdiLhi": bl["hdi"][1],
    "FiveBBetaHdiFlo": bf["hdi"][0], "FiveBBetaHdiFhi": bf["hdi"][1],
    "FiveBBetaErr": max(r["err"] for r in beta_res.values()),
    "FiveBCountsSum": int(counts.sum()), "FiveBCountsList": ", ".join(map(str, counts)),
    "FiveBGamA": al0 + counts.sum(), "FiveBGamB": be0 + n_win,
    "FiveBGamMean": gpost.mean(), "FiveBGamSd": gpost.std(), "FiveBGamErr": gerr,
    "FiveBGamMLE": counts.mean(),
    "FiveBNxa": x_k[0], "FiveBNxb": x_k[1], "FiveBNxc": x_k[2],
    "FiveBNmPost": m_post, "FiveBNsPost": s_post, "FiveBNmIvw": m_ivw, "FiveBNsIvw": s_ivw,
    "FiveBNErr": nerr,
    "FiveBPred": pred, "FiveBPlugin": plugin,
})
