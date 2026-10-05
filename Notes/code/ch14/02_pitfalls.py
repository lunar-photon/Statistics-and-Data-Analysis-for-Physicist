"""Seven recurring traps, each shown with numbers.

Question: for each of the traps that recur through the book, how large is the error
it causes in a small, fully controlled example?

Computes:
  (a) p-value against P(H0 | data): a population of two-sided Gaussian tests, half of them
      null, the other half with effects drawn from N(0, tau^2); the fraction of nulls among
      results with p near 0.05, against the analytic value and the Sellke bound;
  (b) confidence against credible: a Gaussian measurement of a quantity that cannot be
      negative; the posterior probability inside the 90% confidence interval, and the
      coverage of the 90% central credible interval as a function of the truth;
  (c) Fisher against posterior: exponential lifetimes with N = 5 and N = 50 decays;
  (d) chain start against prior: Metropolis chains from three starts, with a confident
      wrong prior, and the posterior mean as the number of data grows;
  (e) inverse of a simulated covariance: the mean chi^2 and the false-alarm rate with and
      without the Hartlap factor, p = 20 bins;
  (f) look-elsewhere: the global p-value of the largest of M = 20 independent windows;
  (g) a shared calibration: the error of an average of K measurements with a common offset.

Writes: figures/ch14/pitfalls_a.pdf, pitfalls_b.pdf, pitfalls_c.pdf,
        results/ch14/02_pitfalls.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = rng_for("ch14", "02_pitfalls")
out = {}

# ---------------------------------------------------------------- (a) p vs P(H0|d)
N = 4_000_000
tau = 2.0                                   # spread of true effects under H1
is_null = rng.random(N) < 0.5
mu = np.where(is_null, 0.0, rng.normal(0, tau, N))
z = mu + rng.normal(0, 1, N)
p = 2 * stats.norm.sf(np.abs(z))
sel = (p > 0.045) & (p < 0.055)
frac_null = is_null[sel].mean()
frac_err = np.sqrt(frac_null * (1 - frac_null) / sel.sum())
z05 = stats.norm.isf(0.025)
B01 = stats.norm.pdf(z05, 0, 1) / stats.norm.pdf(z05, 0, np.sqrt(1 + tau**2))
post_null = B01 / (1 + B01)
Bbar = -1 / (np.e * 0.05 * np.log(0.05))
sellke_post = 1 / (1 + Bbar)
out.update(ThPaTau="2", ThPaN=r"4\times10^{6}", ThPaNsel=f"{sel.sum():d}",
           ThPaFracNull=f"{frac_null:.3f}", ThPaFracErr=f"{frac_err:.3f}",
           ThPaBzeroone=f"{B01:.3f}", ThPaPostNull=f"{post_null:.3f}",
           ThPaBbar=f"{Bbar:.2f}", ThPaSellkePost=f"{sellke_post:.3f}")
# binned curve for the figure
edges = np.logspace(-4, 0, 25)
cent, fr = [], []
for lo_, hi_ in zip(edges[:-1], edges[1:]):
    m = (p > lo_) & (p < hi_)
    if m.sum() > 200:
        cent.append(np.sqrt(lo_ * hi_)); fr.append(is_null[m].mean())
pp = np.logspace(-4, -0.01, 200)
zz = stats.norm.isf(pp / 2)
b01 = stats.norm.pdf(zz) / stats.norm.pdf(zz, 0, np.sqrt(1 + tau**2))
th_curve = b01 / (1 + b01)
ppl = pp[pp < np.exp(-1)]
sel_curve = 1 / (1 + (-1 / (np.e * ppl * np.log(ppl))))

# ------------------------------------------------- (b) confidence vs credible interval
x_obs = -1.5
ci_lo, ci_hi = x_obs - 1.645, x_obs + 1.645
# posterior for mu >= 0 with a flat prior is N(x,1) truncated to [0, inf)
Zx = stats.norm.sf(-x_obs)                  # P(mu >= 0) under N(x_obs, 1)
post_in_ci = (stats.norm.cdf(ci_hi - x_obs) - stats.norm.cdf(0 - x_obs)) / Zx
def trunc_q(q, x):
    """quantile q of N(x,1) truncated to mu >= 0"""
    a = stats.norm.cdf(-x)
    return x + stats.norm.ppf(a + q * (1 - a))
up90 = trunc_q(0.90, x_obs)
out.update(ThPbX="-1.5", ThPbCIlo=f"{ci_lo:.3f}", ThPbCIhi=f"{ci_hi:.3f}",
           ThPbPostInCI=f"{post_in_ci:.2f}", ThPbUpNinety=f"{up90:.2f}")
mus = np.linspace(0, 3, 31)
cov_central, cov_upper = [], []
for m_ in mus:
    xs = m_ + rng.normal(0, 1, 200_000)
    lo_ = trunc_q(0.05, xs); hi_ = trunc_q(0.95, xs)
    cov_central.append(np.mean((lo_ <= m_) & (m_ <= hi_)))
    cov_upper.append(np.mean(m_ <= trunc_q(0.90, xs)))
cov_central = np.array(cov_central); cov_upper = np.array(cov_upper)
out.update(ThPbCovZero=f"{cov_central[0]:.3f}", ThPbCovHalf=f"{cov_central[5]:.3f}",
           ThPbCovThree=f"{cov_central[-1]:.3f}", ThPbCovUpZero=f"{cov_upper[0]:.3f}")

# ------------------------------------------------------- (c) Fisher vs posterior
res_c = {}
for Nd, tag in ((5, "Five"), (50, "Fifty")):
    S = float(Nd)                            # median experiment: tau_hat = S/N = 1
    shape = Nd - 1                           # flat prior in tau: inverse-gamma(N-1, S)
    ig = stats.invgamma(shape, scale=S)
    lo_, med_, hi_ = ig.ppf([0.16, 0.5, 0.84])
    sd_f = 1 / np.sqrt(Nd)
    res_c[Nd] = (ig, sd_f)
    out.update({f"ThPcLo{tag}": f"{lo_:.3f}", f"ThPcMed{tag}": f"{med_:.3f}",
                f"ThPcHi{tag}": f"{hi_:.3f}", f"ThPcMean{tag}": f"{ig.mean():.3f}",
                f"ThPcFisherLo{tag}": f"{1 - sd_f:.3f}", f"ThPcFisherHi{tag}": f"{1 + sd_f:.3f}",
                f"ThPcSd{tag}": f"{sd_f:.3f}"})

# ------------------------------------------------- (d) chain start vs prior
sigma, mu_true = 1.0, 1.0
mu_p, s_p = -2.0, 0.3                       # a confident, wrong prior
def post_mean(Nd, mu_p=mu_p, s_p=s_p, xbar=mu_true):
    w = 1 / s_p**2 + Nd / sigma**2
    return (mu_p / s_p**2 + Nd * xbar / sigma**2) / w, 1 / np.sqrt(w)
Ns = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
pm = np.array([post_mean(n_)[0] for n_ in Ns])
out.update(ThPdMuP="-2", ThPdSp="0.3", ThPdMuTrue="1",
           ThPdPmOne=f"{post_mean(1)[0]:.2f}", ThPdPmTen=f"{post_mean(10)[0]:.2f}",
           ThPdPmHundred=f"{post_mean(100)[0]:.2f}", ThPdPmThousand=f"{post_mean(1000)[0]:.3f}")
# Metropolis chains for N = 10 data points (xbar = 1 exactly: the median experiment)
Nd = 10
target_m, target_s = post_mean(Nd)
def lnpost(m):
    return -0.5 * Nd * (m - mu_true) ** 2 / sigma**2 - 0.5 * (m - mu_p) ** 2 / s_p**2
starts = [-20.0, 0.0, 20.0]
nstep = 20000
chains = []
for i, x0 in enumerate(starts):
    r = rng_for("ch14", "02_pitfalls", stream=10 + i)
    x, lp = x0, lnpost(x0)
    ch = np.empty(nstep)
    for t in range(nstep):
        y = x + 0.5 * r.normal()
        ly = lnpost(y)
        if np.log(r.random()) < ly - lp:
            x, lp = y, ly
        ch[t] = x
    chains.append(ch)
burn = 1000
cm = [c[burn:].mean() for c in chains]
out.update(ThPdTarget=f"{target_m:.3f}", ThPdChainA=f"{cm[0]:.3f}", ThPdChainB=f"{cm[1]:.3f}",
           ThPdChainC=f"{cm[2]:.3f}", ThPdNstep=f"{nstep:d}", ThPdBurn=f"{burn:d}")

# ------------------------------------------- (e) inverse of a simulated covariance
pdim = 20
ns = np.array([25, 30, 40, 60, 100, 200, 400, 1000])
ntrial = 4000
chi_mean, chi_mean_h, fa_raw, fa_h = [], [], [], []
crit = stats.chi2.isf(0.05, pdim)
r_e = rng_for("ch14", "02_pitfalls", stream=30)
for nsim in ns:
    c_raw = np.empty(ntrial)
    for t in range(ntrial):
        X = r_e.normal(size=(nsim, pdim))
        C = np.cov(X, rowvar=False)
        d = r_e.normal(size=pdim)
        c_raw[t] = d @ np.linalg.solve(C, d)
    h = (nsim - pdim - 2) / (nsim - 1)
    chi_mean.append(c_raw.mean()); chi_mean_h.append(h * c_raw.mean())
    fa_raw.append(np.mean(c_raw > crit)); fa_h.append(np.mean(h * c_raw > crit))
chi_mean = np.array(chi_mean); chi_mean_h = np.array(chi_mean_h)
i30 = list(ns).index(30); i100 = list(ns).index(100)
out.update(ThPeP=f"{pdim}", ThPeMeanThirty=f"{chi_mean[i30]:.1f}",
           ThPeExpThirty=f"{pdim * (30 - 1) / (30 - pdim - 2):.1f}",
           ThPeMeanHThirty=f"{chi_mean_h[i30]:.1f}",
           ThPeFaRawThirty=f"{fa_raw[i30]:.2f}", ThPeFaHThirty=f"{fa_h[i30]:.3f}",
           ThPeMeanHundred=f"{chi_mean[i100]:.1f}", ThPeFaRawHundred=f"{fa_raw[i100]:.3f}",
           ThPeFaHHundred=f"{fa_h[i100]:.3f}", ThPeNtrial=f"{ntrial}")

# ------------------------------------------------------------- (f) look-elsewhere
M = 20
p_loc = stats.norm.sf(3.0)
p_glob = 1 - (1 - p_loc) ** M
Z_glob = stats.norm.isf(p_glob)
r_f = rng_for("ch14", "02_pitfalls", stream=40)
Ns_f = 1_000_000
mx = np.zeros(Ns_f)
for j in range(M):
    mx = np.maximum(mx, r_f.normal(size=Ns_f)) if j else r_f.normal(size=Ns_f)
p_glob_sim = np.mean(mx > 3.0)
out.update(ThPfM=f"{M}", ThPfPloc=f"{p_loc:.5f}", ThPfPglob=f"{p_glob:.4f}",
           ThPfZglob=f"{Z_glob:.2f}", ThPfPglobSim=f"{p_glob_sim:.4f}",
           ThPfTrials=f"{p_glob / p_loc:.1f}")

# ------------------------------------------------------- (g) shared calibration
sig_stat, sig_cal = 2.0, 1.0                 # per cent
Ks = np.array([1, 4, 16, 100])
tot = np.sqrt(sig_stat**2 / Ks + sig_cal**2)
r_g = rng_for("ch14", "02_pitfalls", stream=50)
K = 100
offs = r_g.normal(0, sig_cal, 20000)
means = offs + r_g.normal(0, sig_stat, (20000, K)).mean(axis=1)
out.update(ThPgStat="2", ThPgCal="1", ThPgTotOne=f"{tot[0]:.2f}", ThPgTotHundred=f"{tot[-1]:.2f}",
           ThPgSimHundred=f"{means.std():.2f}")

save_numbers("ch14", "02_pitfalls", out)

# ================================================================== figures
setup(7.0, 3.0)
fig, ax = plt.subplots(1, 2)
a = ax[0]
a.plot(cent, fr, "o", color=SERIES[0], label="simulated experiments")
theory_line(a, pp, th_curve, label=r"Bayes, $\tau=2$")
a.plot(ppl, sel_curve, color=SERIES[1], lw=1.2, label="lowest possible (Sellke)")
a.axvline(0.05, color="k", lw=0.6, ls=":")
a.set_xscale("log"); a.set_ylim(0, 1)
a.set_xlabel("two-sided p-value"); a.set_ylabel(r"fraction with $H_0$ true")
a.set_title(r"(a) $p$ is not $P(H_0\mid$data$)$"); a.legend(fontsize=7.5, loc="upper left")
a = ax[1]
a.plot(mus, cov_central, "o-", color=SERIES[0], ms=3, label="90% central credible")
a.plot(mus, cov_upper, "s-", color=SERIES[2], ms=3, label="90% credible upper limit")
a.axhline(0.9, color="k", lw=0.8, ls="--")
a.set_ylim(-0.03, 1.03)
a.set_xlabel(r"true $\mu$ (in units of $\sigma$)"); a.set_ylabel("coverage")
a.set_title("(b) credible is not confidence"); a.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); savefig(fig, "ch14", "pitfalls_a")

setup(7.0, 3.0)
fig, ax = plt.subplots(1, 2)
a = ax[0]
tt = np.linspace(0.05, 4, 800)
for (Nd, (ig, sd_f)), c in zip(res_c.items(), (SERIES[0], SERIES[1])):
    a.plot(tt, ig.pdf(tt), color=c, label=f"posterior, $N={Nd}$")
    a.plot(tt, stats.norm.pdf(tt, 1, sd_f), color=c, ls="--", lw=1.1, label=f"Fisher, $N={Nd}$")
a.set_xlabel(r"lifetime $\tau$ (units of $\hat\tau$)"); a.set_ylabel("density")
a.set_title("(a) Fisher is not the posterior"); a.legend(fontsize=7.5)
a = ax[1]
a.plot(ns, chi_mean / pdim, "o-", color=SERIES[1], ms=3, label=r"$\hat{C}^{-1}$ as it comes")
a.plot(ns, chi_mean_h / pdim, "s-", color=SERIES[0], ms=3, label="with the Hartlap factor")
nn = np.linspace(24, 1000, 400)
theory_line(a, nn, (nn - 1) / (nn - pdim - 2), label=r"$(n-1)/(n-p-2)$")
a.axhline(1, color="k", lw=0.6)
a.set_xscale("log"); a.set_xlabel("number of simulations $n$")
a.set_ylabel(r"mean $\chi^2/p$")
a.set_title(r"(b) the inverse is biased ($p=20$)"); a.legend(fontsize=7.5)
fig.tight_layout(); savefig(fig, "ch14", "pitfalls_b")

setup(7.0, 2.9)
fig, ax = plt.subplots(1, 2)
a = ax[0]
for ch, c, x0 in zip(chains, SERIES, starts):
    a.plot(np.arange(400), ch[:400], color=c, lw=1.0, label=f"start {x0:+.0f}")
a.axhline(target_m, color="k", ls="--", lw=1.0, label="posterior mean")
a.set_xlabel("step"); a.set_ylabel(r"$\mu$")
a.set_title("(a) the start is forgotten"); a.legend(fontsize=7.5)
a = ax[1]
a.plot(Ns, pm, "o-", color=SERIES[1], ms=3, label=r"wrong prior $\mathcal{N}(-2,0.3^2)$")
a.axhline(mu_true, color="k", ls="--", lw=1.0, label="the truth")
a.set_xscale("log"); a.set_xlabel("number of data $N$"); a.set_ylabel("posterior mean")
a.set_title("(b) the prior is forgotten only by data"); a.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); savefig(fig, "ch14", "pitfalls_c")
