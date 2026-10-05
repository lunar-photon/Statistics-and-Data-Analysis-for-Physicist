"""13_efficiency.py -- how good can an estimator be?  Consistency, efficiency and the Cramer-Rao bound.

Question: (i) why does the MLE converge to the truth?  The average log-likelihood ratio
M_n(theta) = (1/n) sum ln f(x_i;theta)/f(x_i;theta*) should approach -D(theta*, theta), minus a
Kullback-Leibler divergence, which peaks at the true value.  (ii) For Gaussian data the mean
(the MLE) and the median both estimate the centre: is n Var(mean) = sigma^2 (the Cramer-Rao
bound) and n Var(median) -> pi sigma^2/2?  (iii) For Uniform(0, theta) the 'bound' theta^2/n
is beaten by (n+1)/n max(x): the regularity condition fails.

Computes: M_n curves for exponential lifetimes at n = 10, 100, 1000 against -D; Monte Carlo
          variances of mean and median for several n; variance of (n+1)/n max(x) versus theta^2/n.
Writes:   figures/ch03/consistency_kl.pdf, figures/ch03/efficiency.pdf, results/ch03/13_efficiency.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch03", "13_efficiency")
setup()

# ---------- (i) consistency: M_n(tau) -> -D(tau*, tau) ----------
tau_s = 1.0
tau = np.linspace(0.3, 3.0, 400)
# for exponentials: ln f(t;tau) - ln f(t;tau*) = -ln(tau/tau*) - t/tau + t/tau*
Dkl = np.log(tau / tau_s) + tau_s / tau - 1          # D(tau*, tau) = E_tau*[ln f(.;tau*)/f(.;tau)]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
tmax_n = {}
for n, c in zip([10, 100, 1000], SERIES):
    t = rng.exponential(tau_s, n)
    Mn = -np.log(tau / tau_s) - t.mean() / tau + t.mean() / tau_s
    ax.plot(tau, Mn, color=c, lw=1.2, label=rf"$M_n(\tau)$, $n={n}$")
    tmax_n[n] = t.mean()
theory_line(ax, tau, -Dkl, label=r"$-D(\tau_\star,\tau)$")
ax.axvline(tau_s, color="k", lw=0.6, ls=":")
ax.set_ylim(-1.0, 0.25); ax.set_xlabel(r"$\tau$"); ax.set_ylabel("average log-likelihood ratio")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
savefig(fig, "ch03", "consistency_kl")

# ---------- (ii) mean versus median, (iii) uniform maximum ----------
ns = np.array([5, 11, 21, 51, 101, 201])
M = 60_000
vmean, vmed, vunif = [], [], []
for n in ns:
    X = rng.standard_normal((M, n))
    vmean.append(n * X.mean(1).var())
    vmed.append(n * np.median(X, 1).var())
    U = rng.uniform(0, 1, (M, n))
    vunif.append(n * ((n + 1) / n * U.max(1)).var())   # n Var / theta^2 with theta = 1
vmean, vmed, vunif = map(np.array, (vmean, vmed, vunif))

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0))
axs[0].plot(ns, vmean, "o-", color=SERIES[0], label="mean (MLE)")
axs[0].plot(ns, vmed, "s-", color=SERIES[1], label="median")
axs[0].axhline(1, color="k", ls="--", lw=1.2, label=r"Cram\'er--Rao: $\sigma^2$")
axs[0].axhline(np.pi / 2, color=SERIES[1], ls=":", lw=1, label=r"$\pi\sigma^2/2$")
axs[0].set_xscale("log"); axs[0].set_xlabel(r"$n$"); axs[0].set_ylabel(r"$n\,\mathrm{Var}/\sigma^2$")
axs[0].set_ylim(0.9, 2.25); axs[0].legend(fontsize=7, loc="upper center")
axs[1].plot(ns, vunif, "o-", color=SERIES[2], label=r"$\frac{n+1}{n}\max x_i$")
axs[1].plot(ns, np.ones_like(ns, dtype=float), "k--", lw=1.2, label=r"naive bound $\theta^2/n$")
axs[1].plot(ns, ns / (ns * (ns + 2.0)), ":", color=SERIES[2], label=r"exact $\theta^2/(n+2)$")
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel(r"$n$"); axs[1].set_ylabel(r"$n\,\mathrm{Var}/\theta^2$")
axs[1].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1, 0.85))
fig.tight_layout()
savefig(fig, "ch03", "efficiency")

save_numbers("ch03", "13_efficiency", {
    "ThreeBEffMeanBig": vmean[-1], "ThreeBEffMedBig": vmed[-1], "ThreeBEffNBig": int(ns[-1]),
    "ThreeBEffARE": vmean[-1] / vmed[-1], "ThreeBEffAREth": 2 / np.pi,
    "ThreeBEffUnifFifty": vunif[3], "ThreeBEffUnifFiftyTh": ns[3] / (ns[3] * (ns[3] + 2.0)) , "ThreeBEffNFifty": int(ns[3]),
    "ThreeBConsTen": tmax_n[10], "ThreeBConsHund": tmax_n[100], "ThreeBConsThou": tmax_n[1000],
})
