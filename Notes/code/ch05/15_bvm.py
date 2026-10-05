"""15_bvm.py -- forgetting the prior (through the likelihood) versus forgetting the start (of a chain).

Question: two analysts start from very different priors.  Do their posteriors agree once
enough data arrive?  When do they never agree?  And how is this different from a Markov
chain forgetting where it was started?

Computes:
  (a) a coin with theta = 0.3; priors Beta(1,1) and Beta(80,20) (confident and wrong);
      posteriors after N = 0, 10, 100, 1000, 100000 tosses; the gap between the two
      posterior means in units of the posterior sd, as a function of N;
  (b) two failures: (i) a detector that sees only the total count of two channels,
      so the likelihood depends on lambda1 + lambda2 alone: the posterior of the
      fraction f = lambda1/(lambda1+lambda2) stays equal to its prior; (ii) a prior that
      is zero above 0.25 when the truth is 0.3;
  (c) Metropolis chains on the posterior after 6 heads in 20: two priors x two starting
      points.  Chains with the same prior agree whatever the start; chains with
      different priors disagree however long they run.

Writes: figures/ch05/bvm_converge.pdf, figures/ch05/bvm_fail.pdf,
        figures/ch05/chain_vs_prior.pdf, results/ch05/15_bvm.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch05", "15_bvm")
setup()

# ---------------------------------------------------------------- (a) convergence
TH = 0.3
tosses = rng.uniform(size=100000) < TH
priors = [(1, 1), (80, 20)]
th = np.linspace(0, 1, 4001)
shown = [0, 10, 100, 1000, 100000]
fig, ax = plt.subplots(1, 6, figsize=(8.0, 2.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 1.5]))
for a, n in zip(ax[:5], shown):
    z = tosses[:n].sum()
    for k, (pa, pb) in enumerate(priors):
        a.plot(th, stats.beta(pa + z, pb + n - z).pdf(th), color=SERIES[k], ls="-" if k == 0 else "--")
    a.axvline(TH, color=INK2, ls=":", lw=0.7)
    a.set_title(f"N={n}", fontsize=8); a.set_yticks([]); a.set_xticks([0, 0.5, 1])
    if n >= 1000:
        a.set_xlim(0.2, 0.45) if n < 10**5 else a.set_xlim(0.28, 0.32)
        a.set_xticks([0.25, 0.4] if n < 10**5 else [0.29, 0.31])
Ns = np.unique(np.logspace(0, 5, 80).astype(int))
gap = []
for n in Ns:
    z = tosses[:n].sum()
    p1, p2 = (stats.beta(pa + z, pb + n - z) for pa, pb in priors)
    gap.append(abs(p1.mean() - p2.mean()) / p1.std())
gap = np.array(gap)
ax[5].loglog(Ns, gap, color=SERIES[0], label="gap / posterior sd")
ax[5].loglog(Ns, gap[-1] * np.sqrt(Ns[-1] / Ns), color="k", ls="--", lw=1, label=r"$\propto N^{-1/2}$")
ax[5].axhline(1, color=INK2, lw=0.6, ls=":")
ax[5].set_xlabel("N"); ax[5].legend(fontsize=6)
fig.tight_layout()
savefig(fig, "ch05", "bvm_converge")
ipk = np.argmax(gap)
n_one = Ns[ipk + np.argmax(gap[ipk:] < 1)]           # N after which the gap stays below 1 sd

# ---------------------------------------------------------------- (b) failures
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
f = np.linspace(0, 1, 1001)
T_obs = 1000.0                                          # observing time: total rate pinned
n_tot = rng.poisson((2.0 + 3.0) * T_obs)                 # lambda1 = 2, lambda2 = 3
for k, (fa, fb) in enumerate([(1, 1), (5, 2)]):
    # likelihood depends only on Lambda = lambda1 + lambda2, so p(f | n) = p(f): draw it
    Lam = stats.gamma(n_tot + 1, scale=1 / T_obs).rvs(size=50000, random_state=rng)
    fs = stats.beta(fa, fb).rvs(size=50000, random_state=rng)
    ax[0].hist(fs * Lam, bins=100, range=(0, 5), density=True, color=SERIES[k], alpha=0.5,
               label=f"prior on f: Beta({fa},{fb})")
ax[0].axvline(2.0, color=INK2, ls=":", lw=0.8)
ax[0].set_xlabel(r"$\lambda_1$ (per unit time)"); ax[0].set_yticks([])
ax[0].set_title(f"only the total is seen: {n_tot} counts", fontsize=8); ax[0].legend(fontsize=6)
thz = np.linspace(0, 0.5, 2001)
for k, n in enumerate([10, 100, 1000]):
    z = tosses[:n].sum()
    pz = np.where(thz <= 0.25, thz**z * (1 - thz) ** (n - z), 0.0)
    ax[1].plot(thz, pz / (pz.sum() * (thz[1] - thz[0])), color=SERIES[k], label=f"N={n}")
ax[1].axvline(TH, color=INK2, ls=":", lw=0.8)
ax[1].set_xlabel(r"$\theta$"); ax[1].set_yticks([])
ax[1].set_title(r"prior zero above 0.25, truth 0.3", fontsize=8); ax[1].legend(fontsize=6)
fig.tight_layout()
savefig(fig, "ch05", "bvm_fail")

# ---------------------------------------------------------------- (c) chains versus priors
Z, NN = 6, 20


def log_post(t, pa, pb):
    if t <= 0 or t >= 1:
        return -np.inf
    return (Z + pa - 1) * np.log(t) + (NN - Z + pb - 1) * np.log(1 - t)


def metropolis(t0, pa, pb, n=20000, step=0.08):
    chain = np.empty(n); t, lp = t0, log_post(t0, pa, pb)
    for i in range(n):
        tp = t + step * rng.standard_normal()
        lpp = log_post(tp, pa, pb)
        if np.log(rng.uniform()) < lpp - lp:           # only a ratio of posteriors enters
            t, lp = tp, lpp
        chain[i] = t
    return chain


fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
chain_means = {}
for k, (pa, pb) in enumerate(priors):
    for j, t0 in enumerate([0.02, 0.98]):
        c = metropolis(t0, pa, pb)
        chain_means[(k, j)] = c[5000:].mean()
        ax[0].plot(c[:400], color=SERIES[k], lw=0.7, ls="-" if j == 0 else "--")
        ax[1].hist(c[5000:], bins=60, range=(0, 1), density=True, histtype="step",
                   color=SERIES[k], ls="-" if j == 0 else "--")
ax[0].set_xlabel("step"); ax[0].set_ylabel(r"$\theta$"); ax[0].set_title("first 400 steps", fontsize=8)
for k, (pa, pb) in enumerate(priors):
    ax[1].plot([], [], color=SERIES[k], label=f"prior Beta({pa},{pb})")
ax[1].set_xlabel(r"$\theta$"); ax[1].set_yticks([]); ax[1].legend(fontsize=6)
ax[1].set_title("steps 5000-20000, both starts", fontsize=8)
fig.tight_layout()
savefig(fig, "ch05", "chain_vs_prior")

exact = [stats.beta(pa + Z, pb + NN - Z).mean() for pa, pb in priors]
save_numbers("ch05", "15_bvm", {
    "FiveBBvmGapTen": gap[np.searchsorted(Ns, 10)], "FiveBBvmGapThou": gap[np.searchsorted(Ns, 1000)],
    "FiveBBvmGapLast": gap[-1], "FiveBBvmGapPeak": gap[ipk], "FiveBBvmNpeak": int(Ns[ipk]), "FiveBBvmNone": int(n_one), "FiveBBvmNtot": int(n_tot),
    "FiveBChainFlatA": chain_means[(0, 0)], "FiveBChainFlatB": chain_means[(0, 1)],
    "FiveBChainStrA": chain_means[(1, 0)], "FiveBChainStrB": chain_means[(1, 1)],
    "FiveBChainExactFlat": exact[0], "FiveBChainExactStr": exact[1],
})
