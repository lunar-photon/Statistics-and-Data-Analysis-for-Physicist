"""08_goodness_of_fit.py -- does the model fit the data at all?

Question: (i) Mendel's peas: Pearson chi^2 and its p-value.  (ii) Pearson's chi^2
for a Poisson histogram with small expected counts (as in Cowan's Fig. 4.4):
does chi^2_N still describe it, or must the null distribution be simulated?
(iii) fitting one parameter (an exponential decay length) to a multinomial
histogram: with the binned estimate, chi^2 ~ chi^2_{k-1-1}; with the unbinned MLE
the distribution lies between chi^2_{k-2} and chi^2_{k-1} (Chernoff-Lehmann).
(iv) the Kolmogorov-Smirnov test written from scratch, compared with scipy, and
the null distribution of sqrt(n) D_n compared with Kolmogorov's limit.

Writes: figures/ch04/gof_chi2.pdf, figures/ch04/ks_test.pdf,
        results/ch04/08_goodness_of_fit.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

rng = rng_for("ch04", "08_goodness_of_fit")
setup()

# ---------- (i) Mendel ----------
obs = np.array([315, 101, 108, 32]); p0 = np.array([9, 3, 3, 1]) / 16
E = obs.sum() * p0
chi_mendel = ((obs - E) ** 2 / E).sum()
p_mendel = stats.chi2.sf(chi_mendel, 3)

# ---------- (ii) small expected counts ----------
k_bins = 20
x = np.arange(k_bins)
nu_small = 0.8 + 3.0 * np.exp(-x / 8.0)          # expected counts between about 1 and 4
nu_large = 25 * nu_small                          # the same shape with 25 times the data
M = 200_000
def pearson(nu):
    n = rng.poisson(nu, size=(M, nu.size))
    return ((n - nu) ** 2 / nu).sum(1)
ch_small, ch_large = pearson(nu_small), pearson(nu_large)
q95_small, q95_large = np.quantile(ch_small, 0.95), np.quantile(ch_large, 0.95)
q95_th = stats.chi2.ppf(0.95, k_bins)
t_obs = 30.0
p_small_mc = np.mean(ch_small >= t_obs)
p_small_chi = stats.chi2.sf(t_obs, k_bins)

# ---------- (iii) one fitted parameter ----------
tau_true, Ntot, kb = 2.0, 400, 8
edges = np.append(np.arange(kb) * 1.0, np.inf)    # bins [0,1), ..., [6,7), [7, inf)
def probs(tau):
    c = 1 - np.exp(-edges / tau)
    return np.diff(c)
Mfit = 4000
q_binned, q_raw = np.empty(Mfit), np.empty(Mfit)
for j in range(Mfit):
    t = rng.exponential(tau_true, size=Ntot)
    N = np.histogram(t, bins=edges)[0]
    # binned (multinomial) maximum likelihood estimate
    nll = lambda tau: -(N * np.log(probs(tau))).sum()
    tau_b = optimize.minimize_scalar(nll, bounds=(0.3, 10), method="bounded").x
    tau_r = t.mean()                              # unbinned MLE of an exponential mean
    for tau, out in ((tau_b, q_binned), (tau_r, q_raw)):
        e = Ntot * probs(tau)
        out[j] = ((N - e) ** 2 / e).sum()
mean_binned, mean_raw = q_binned.mean(), q_raw.mean()

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
bins = np.linspace(0, 60, 61)
axs[0].hist(ch_small, bins=bins, density=True, histtype="step", lw=1.5, color=SERIES[1], label=r"toys, $\nu_i\approx1$--$4$")
axs[0].hist(ch_large, bins=bins, density=True, histtype="step", lw=1.5, color=SERIES[0], label=r"toys, $25\times$ more data")
g = np.linspace(0.1, 60, 300)
theory_line(axs[0], g, stats.chi2.pdf(g, k_bins), label=r"$\chi^2_{20}$")
axs[0].set_xlabel(r"Pearson $\chi^2$ (20 Poisson bins)"); axs[0].legend(fontsize=7)
bins = np.linspace(0, 25, 51)
axs[1].hist(q_binned, bins=bins, density=True, histtype="step", lw=1.5, color=SERIES[2], label=r"binned MLE $\tilde\tau$")
axs[1].hist(q_raw, bins=bins, density=True, histtype="step", lw=1.5, color=SERIES[3], label=r"unbinned MLE $\hat\tau$")
g = np.linspace(0.1, 25, 300)
theory_line(axs[1], g, stats.chi2.pdf(g, kb - 2), label=r"$\chi^2_{k-2}$")
axs[1].plot(g, stats.chi2.pdf(g, kb - 1), "k:", lw=1.4, label=r"$\chi^2_{k-1}$")
axs[1].set_xlabel(r"Pearson $\chi^2$ with fitted $\tau$ ($k=8$)"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch04", "gof_chi2")

# ---------- (iv) Kolmogorov-Smirnov from scratch ----------
def ks_stat(sample, cdf):
    """D_n = sup_x |F_n(x) - F(x)|; the sup is attained just before or at a data point."""
    xs = np.sort(sample); n = xs.size
    F = cdf(xs)
    d_plus = np.max(np.arange(1, n + 1) / n - F)   # F_n jumps up to i/n at x_(i)
    d_minus = np.max(F - np.arange(0, n) / n)      # just before the jump F_n = (i-1)/n
    return max(d_plus, d_minus)

n_ks = 50
tt = rng.exponential(1.3, size=n_ks)              # the truth: decay length 1.3, H0 says 1.0
cdf0 = lambda v: stats.expon.cdf(v, scale=1.0)
D_mine = ks_stat(tt, cdf0)
res = stats.kstest(tt, cdf0)
D_scipy, p_scipy = res.statistic, res.pvalue
p_kolm = stats.kstwobign.sf(np.sqrt(n_ks) * D_mine)
# null distribution by simulation
Dnull = np.array([ks_stat(rng.exponential(1.0, size=n_ks), cdf0) for _ in range(40_000)])
p_mc = np.mean(Dnull >= D_mine)

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
xs = np.sort(tt)
axs[0].step(np.concatenate([[0], xs]), np.arange(0, n_ks + 1) / n_ks, where="post", color=SERIES[0], label=r"empirical $F_n$")
gx = np.linspace(0, xs.max() * 1.05, 300)
theory_line(axs[0], gx, cdf0(gx), label=r"$H_0$: $F(x)=1-e^{-x}$")
Fn = np.arange(1, n_ks + 1) / n_ks
i = np.argmax(np.maximum(Fn - cdf0(xs), cdf0(xs) - (Fn - 1 / n_ks)))
lo, hi = sorted([cdf0(xs[i]), Fn[i] if Fn[i] - cdf0(xs[i]) >= cdf0(xs[i]) - (Fn[i] - 1 / n_ks) else Fn[i] - 1 / n_ks])
axs[0].plot([xs[i], xs[i]], [lo, hi], color=SERIES[1], lw=2.5, label=r"$D_n$")
axs[0].set_xlabel("$x$"); axs[0].legend(fontsize=7, loc="lower right")
axs[1].hist(np.sqrt(n_ks) * Dnull, bins=60, density=True, color=SERIES[0], alpha=0.6, label=r"toys, $n=50$")
gk = np.linspace(0.2, 2.2, 300)
theory_line(axs[1], gk, stats.kstwobign.pdf(gk), label="Kolmogorov limit")
axs[1].axvline(np.sqrt(n_ks) * D_mine, color=SERIES[1], lw=1.5, label="observed")
axs[1].set_xlabel(r"$\sqrt{n}\,D_n$"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch04", "ks_test")

save_numbers("ch04", "08_goodness_of_fit", {
    "FourAMendelChi": f"{chi_mendel:.3f}", "FourAMendelP": f"{p_mendel:.3f}",
    "FourAGofQsmall": f"{q95_small:.1f}", "FourAGofQlarge": f"{q95_large:.1f}", "FourAGofQth": f"{q95_th:.1f}",
    "FourAGofPsmallMC": f"{p_small_mc:.3f}", "FourAGofPsmallChi": f"{p_small_chi:.3f}",
    "FourAGofMeanBinned": f"{mean_binned:.2f}", "FourAGofMeanRaw": f"{mean_raw:.2f}",
    # both fits see the same histograms, so their difference is measured far better than either mean
    "FourAGofMeanDiff": f"{np.mean(q_raw - q_binned):.3f}",
    "FourAGofMeanDiffErr": f"{np.std(q_raw - q_binned, ddof=1) / np.sqrt(Mfit):.3f}",
    "FourAGofMeanErr": f"{np.std(q_raw, ddof=1) / np.sqrt(Mfit):.2f}",
    "FourAKsD": f"{D_mine:.4f}", "FourAKsDscipy": f"{D_scipy:.4f}", "FourAKsPscipy": f"{p_scipy:.4f}",
    "FourAKsPkolm": f"{p_kolm:.4f}", "FourAKsPmc": f"{p_mc:.4f}",
    "FourAKsSqrtnD": f"{np.sqrt(n_ks) * D_mine:.3f}",
})
print(chi_mendel, p_mendel, q95_small, q95_large, q95_th, p_small_mc, p_small_chi, mean_binned, mean_raw,
      D_mine, D_scipy, p_scipy, p_kolm, p_mc)
