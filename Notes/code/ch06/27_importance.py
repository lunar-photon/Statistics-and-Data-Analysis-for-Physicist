"""27_importance.py -- importance sampling, and why g needs thicker tails than f.

Question: to estimate I = int h f dx we may draw from any density g and average
the weighted values h(X) f(X) / g(X).  The estimate is unbiased for every g that
is positive where h f is, but its variance contains int h^2 f^2 / g dx, which is
infinite when g falls off faster than f.

  1. Normalising constant of q(x) = exp(-x^2/2) (Z = sqrt(2 pi)) with g = N(0, s^2):
     relative variance per draw s^2 / sqrt(2 s^2 - 1) - 1 for s^2 > 1/2, infinite
     otherwise.  Running estimates (three runs each) for s = 0.5 (thin) and s = 1.5 (thick), and the
     scatter of the estimate over repeats against s.
  2. Wasserman Ex 24.6: P(Z > 3) with N = 100, plain Monte Carlo against g = N(4, 1).
  3. In the spirit of Wasserman Ex 24.7 (our own three data points, one an
     outlier): posterior mean of theta with a Student-t (nu = 3) likelihood and a
     flat prior, by self-normalised importance sampling with g = N(0, 1) and with
     g = Cauchy(0, 1); 2000 repeats of 1000 draws each.

Writes: figures/ch06/importance_tails.pdf, figures/ch06/importance_running.pdf,
        results/ch06/27_importance.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate

rng = rng_for("ch06", "27_importance")
out = {}
Z = np.sqrt(2 * np.pi)
q = lambda x: np.exp(-0.5 * x**2)


def z_hat(s, n):
    x = s * rng.standard_normal(n)
    w = q(x) / stats.norm.pdf(x, scale=s)       # importance weight q / g
    return w


# ---------------------------------------------------------------- 1a. running estimates
Nrun = 200_000
run = {}
for s, tag in [(0.5, "Thin"), (1.5, "Thick")]:
    for j in range(3):                            # three independent runs of each
        w = z_hat(s, Nrun)
        run[(s, j)] = np.cumsum(w) / np.arange(1, Nrun + 1) / Z
        if j == 0:
            out[f"SixBIs{tag}Final"] = run[(s, j)][-1]
            out[f"SixBIs{tag}Ess"] = w.sum() ** 2 / np.sum(w**2) / Nrun    # effective sample size fraction
            out[f"SixBIs{tag}MaxShare"] = w.max() / w.sum()
    out[f"SixBIs{tag}Spread"] = np.std([run[(s, j)][-1] for j in range(3)])
out["SixBIsThickRelVar"] = 1.5**2 / np.sqrt(2 * 1.5**2 - 1) - 1
out["SixBIsNrun"] = Nrun

# ---------------------------------------------------------------- 1b. scatter of the estimate against s
ss = np.linspace(0.45, 3.0, 35)
n_each, reps = 1000, 400
emp = np.array([np.std([z_hat(s, n_each).mean() / Z for _ in range(reps)]) for s in ss])
ssf = np.linspace(0.7072, 3.0, 300)
theo = np.sqrt((ssf**2 / np.sqrt(2 * ssf**2 - 1) - 1) / n_each)

setup(5.2, 3.3)
fig, ax = plt.subplots()
ax.plot(ss, emp, "o", ms=3.5, color=SERIES[0], label=f"scatter of $\\hat Z/Z$ over {reps} repeats")
theory_line(ax, ssf, theo, label=r"$\sqrt{[s^2/\sqrt{2s^2-1}-1]/N}$")
ax.axvline(1 / np.sqrt(2), color=SERIES[1], ls=":", lw=1.3)
ax.text(0.73, 0.2, r"$s=1/\sqrt{2}$", color=SERIES[1], fontsize=8)
ax.set(xlabel=r"width $s$ of the sampling density $g=\mathcal{N}(0,s^2)$",
       ylabel="standard deviation", yscale="log", ylim=(1e-3, 1))
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "ch06", "importance_tails")

setup(9.0, 3.0)
fig, ax = plt.subplots(1, 2)
nn = np.arange(1, Nrun + 1)
for s, c in [(0.5, SERIES[1]), (1.5, SERIES[0])]:
    for j in range(3):
        ax[0].plot(nn, run[(s, j)], color=c, lw=0.9, label=f"$s={s}$" if j == 0 else None)
ax[0].axhline(1.0, color="k", ls="--", lw=1.0)
ax[0].set(xscale="log", xlabel=r"number of draws $N$", ylabel=r"running $\hat Z/Z$", ylim=(0.6, 1.3))
ax[0].legend()
xx = np.linspace(-5, 5, 400)
ax[1].plot(xx, stats.norm.pdf(xx), color="k", lw=1.4, label=r"target $f=\mathcal{N}(0,1)$")
ax[1].plot(xx, stats.norm.pdf(xx, scale=0.5), color=SERIES[1], lw=1.2, label=r"$g$, $s=0.5$")
ax[1].plot(xx, stats.norm.pdf(xx, scale=1.5), color=SERIES[0], lw=1.2, label=r"$g$, $s=1.5$")
ax[1].plot(xx, stats.norm.pdf(xx) / stats.norm.pdf(xx, scale=0.5) / 8, color=SERIES[1], ls=":", lw=1.2,
           label=r"$f/g$ ($s=0.5$, $\div 8$)")
ax[1].set(xlabel=r"$x$", ylabel="density", ylim=(0, 0.8))
ax[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch06", "importance_running")

# ---------------------------------------------------------------- 2. tail probability P(Z > 3)
p_true = stats.norm.sf(3.0)
R, n = 20_000, 100
plain = np.array([np.mean(rng.standard_normal(n) > 3.0) for _ in range(R)])
isamp = []
for _ in range(R):
    x = 4.0 + rng.standard_normal(n)
    isamp.append(np.mean((x > 3.0) * stats.norm.pdf(x) / stats.norm.pdf(x, loc=4.0)))
isamp = np.array(isamp)
sd_plain_th = np.sqrt(p_true * (1 - p_true) / n)
second = np.exp(16.0) * stats.norm.sf(7.0)       # E_g[(h f / g)^2] = e^16 (1 - Phi(7))
sd_is_th = np.sqrt((second - p_true**2) / n)
out.update(SixBIsTailP=p_true, SixBIsTailN=n, SixBIsTailR=R,
           SixBIsPlainMean=plain.mean(), SixBIsPlainSd=plain.std(), SixBIsPlainSdTh=sd_plain_th,
           SixBIsPlainZero=np.mean(plain == 0),
           SixBIsIsMean=isamp.mean(), SixBIsIsSd=isamp.std(), SixBIsIsSdTh=sd_is_th,
           SixBIsGain=plain.std() / isamp.std(), SixBIsSecond=second)

# ---------------------------------------------------------------- 3. posterior mean with a t likelihood
xdat = np.array([-4.0, 0.5, 1.0])
like = lambda th: np.prod(stats.t.pdf(xdat[:, None] - th[None, :], df=3), axis=0)
num = integrate.quad(lambda t: t * like(np.array([t]))[0], -60, 60, limit=400)[0]
den = integrate.quad(lambda t: like(np.array([t]))[0], -60, 60, limit=400)[0]
post_mean = num / den
res = {}
for name, g in [("Norm", stats.norm(loc=0.0, scale=1.0)), ("Cauchy", stats.cauchy(loc=0.0, scale=1.0))]:
    ests = []
    for _ in range(2000):
        th = g.rvs(size=1000, random_state=rng)
        w = like(th) / g.pdf(th)
        ests.append(np.sum(w * th) / np.sum(w))     # self-normalised: the constant of f cancels
    ests = np.array(ests)
    res[name] = ests
    out[f"SixBIsPost{name}Mean"] = ests.mean()
    out[f"SixBIsPost{name}Sd"] = ests.std()
    out[f"SixBIsPost{name}Med"] = np.median(ests)
    out[f"SixBIsPost{name}Lo"], out[f"SixBIsPost{name}Hi"] = np.percentile(ests, [1, 99])
out.update(SixBIsPostTrue=post_mean, SixBIsPostXa=xdat[0], SixBIsPostXb=xdat[1], SixBIsPostXc=xdat[2])
save_numbers("ch06", "27_importance", out)
print(out)
