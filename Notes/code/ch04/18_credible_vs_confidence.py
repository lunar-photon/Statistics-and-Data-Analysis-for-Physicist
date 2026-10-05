"""18_credible_vs_confidence.py -- a confidence interval and a credible interval answer different questions.

Question: for the same data, how do a 90% confidence interval and a 90% credible interval
differ?  How often does each one cover a fixed true parameter in repeated experiments, and how
much posterior probability does each one hold?  (Casella-Berger Ex. 9.2.15-9.2.18.)  And for
Kruschke's coin (z = 7 heads in N = 24 flips), how does the 95% HDI compare with the confidence
intervals that different stopping intentions produce?

Computes: Poisson (n = 10 observations, sum = 6) confidence and gamma(1,1)-prior credible
intervals; coverage of both as a function of the true rate; posterior (credible) probability of
both as a function of the observed sum; Normal-prior credible interval coverage in closed form;
beta posteriors and 95% HDIs for the coin and for Kruschke's ROPE examples.

Writes: figures/ch04/credible_coverage.pdf, figures/ch04/hdi_vs_ci.pdf,
        results/ch04/18_credible_vs_confidence.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

setup()
CL = 0.90
a_t = (1 - CL) / 2

# ---------- Poisson: confidence (Garwood) and credible (gamma(1,1) prior) ----------
n_obs, a_pr, b_pr = 10, 1, 1.0          # prior gamma(shape a, scale b)

def conf_int(y, n=n_obs):
    lo = np.where(y == 0, 0.0, stats.chi2.ppf(a_t, 2 * np.maximum(y, 1)) / (2 * n))
    hi = stats.chi2.ppf(1 - a_t, 2 * (y + 1)) / (2 * n)
    return lo, hi

def post(y, n=n_obs):
    """posterior gamma(a + y, scale b/(n b + 1))"""
    return stats.gamma(a_pr + y, scale=b_pr / (n * b_pr + 1))

def cred_int(y, n=n_obs):
    p = post(y, n)
    return p.ppf(a_t), p.ppf(1 - a_t)

y0 = 6
c_lo, c_hi = conf_int(np.array(y0))
k_lo, k_hi = cred_int(y0)

lam = np.linspace(0.01, 3.0, 1500)
yy = np.arange(0, 120)
pm = stats.poisson.pmf(yy[None, :], n_obs * lam[:, None])
clo, chi = conf_int(yy)
klo, khi = cred_int(yy)
cov_conf = (pm * ((clo <= lam[:, None]) & (lam[:, None] <= chi))).sum(1)
cov_cred = (pm * ((klo <= lam[:, None]) & (lam[:, None] <= khi))).sum(1)
ys = np.arange(0, 51)
credprob_conf = np.array([post(y).cdf(conf_int(np.array(y))[1]) - post(y).cdf(conf_int(np.array(y))[0])
                          for y in ys])
# large-lambda coverage of the credible interval
lam_big = np.array([5.0, 20.0, 50.0])
def cov_at(l, lohi):
    yb = np.arange(0, int(n_obs * l + 20 * np.sqrt(n_obs * l) + 50))
    lo, hi = lohi(yb)
    return np.sum(stats.poisson.pmf(yb, n_obs * l) * ((lo <= l) & (l <= hi)))
cov_cred_big = [cov_at(l, cred_int) for l in lam_big]

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.1))
ax = axs[0]
ax.plot(lam, cov_conf, color=SERIES[0], lw=1.0, label="90% confidence interval")
ax.plot(lam, cov_cred, color=SERIES[1], lw=1.0, label="90% credible interval")
ax.axhline(CL, color="k", ls=":", lw=0.9)
ax.set_xlabel(r"true rate $\lambda$"); ax.set_ylabel("coverage (repeated experiments)")
ax.set_ylim(0.6, 1.01); ax.legend(fontsize=7.5, loc="lower left")
ax = axs[1]
ax.plot(ys, credprob_conf, "o-", color=SERIES[0], ms=2.5, lw=1.0, label="confidence interval")
ax.axhline(CL, color=SERIES[1], lw=1.2, label="credible interval (by construction)")
ax.set_xlabel(r"observed $\sum x_i$"); ax.set_ylabel("posterior probability of the interval")
ax.set_ylim(0.8, 1.0); ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
savefig(fig, "ch04", "credible_coverage")

# ---------- Normal prior: coverage of the credible interval in closed form ----------
z95 = stats.norm.isf(0.025)
def cov_normal_cred(delta, gam):
    """delta = (theta - mu_prior) sqrt(n)/sigma ; gamma = sigma^2/(n tau^2)."""
    w = np.sqrt(1 + gam) * z95
    return stats.norm.cdf(gam * delta + w) - stats.norm.cdf(gam * delta - w)
cov_g1_d0 = cov_normal_cred(0.0, 1.0)
cov_g1_d2 = cov_normal_cred(2.0, 1.0)
cov_g1_d4 = cov_normal_cred(4.0, 1.0)

# ---------- Kruschke's coin: HDIs ----------
def hdi(dist, mass=0.95):
    f = lambda p: dist.ppf(p + mass) - dist.ppf(p)
    p = optimize.minimize_scalar(f, bounds=(1e-9, 1 - mass - 1e-9), method="bounded").x
    return dist.ppf(p), dist.ppf(p + mass)

z, N = 7, 24
post_inf = stats.beta(11 + z, 11 + N - z)
post_flat = stats.beta(1 + z, 1 + N - z)
h_inf, h_flat = hdi(post_inf), hdi(post_flat)
p_half_inf = post_inf.cdf(0.5)
rope1 = hdi(stats.beta(1 + 325, 1 + 175))
rope2 = hdi(stats.beta(1 + 490, 1 + 510))
post_rope2_in = stats.beta(491, 511).cdf(0.55) - stats.beta(491, 511).cdf(0.45)
# Kruschke's CIs for the four stopping intentions (Sec. 11.3.1); the fixed-N one we recompute
cis = [("fixed N", (0.126, 0.511)), ("fixed z", (0.126, 0.484)),
       ("fixed duration", (0.135, 0.497)), ("two coins, fixed N", (0.110, 0.539))]
lo_fn = optimize.brentq(lambda t: stats.binom.sf(z - 1, N, t) - 0.025, 1e-6, z / N)
hi_fn = optimize.brentq(lambda t: stats.binom.cdf(z, N, t) - 0.025, z / N, 1 - 1e-6)
assert abs(lo_fn - 0.126) < 1e-3 and abs(hi_fn - 0.511) < 1e-3

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axs[0]
th = np.linspace(0, 1, 800)
for d, h, col, lab in ((post_flat, h_flat, SERIES[0], "uniform prior"),
                       (post_inf, h_inf, SERIES[2], r"beta(11,11) prior")):
    ax.plot(th, d.pdf(th), color=col, label=f"posterior, {lab}")
    sel = (th >= h[0]) & (th <= h[1])
    ax.fill_between(th[sel], d.pdf(th[sel]), color=col, alpha=0.15, lw=0)
ymax = ax.get_ylim()[1]
for i, (lab, (lo, hi)) in enumerate(cis):
    yb = -0.9 - 0.75 * i
    ax.plot([lo, hi], [yb, yb], color=SERIES[1], lw=2.2, solid_capstyle="butt")
    ax.text(hi + 0.02, yb, lab, fontsize=7, va="center")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylim(-0.9 - 0.75 * 3 - 0.6, ymax); ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= ymax])
ax.set_xlim(0, 1)
ax.set_xlabel(r"heads probability $\theta$"); ax.set_ylabel("posterior density")
ax.legend(fontsize=7, loc="upper right")
ax = axs[1]
dl = np.linspace(-6, 6, 600)
for gam, col in ((0.25, SERIES[0]), (1.0, SERIES[1]), (4.0, SERIES[2])):
    ax.plot(dl, cov_normal_cred(dl, gam), color=col, label=rf"$\gamma={gam:g}$")
ax.axhline(0.95, color="k", ls=":", lw=0.9)
ax.set_xlabel(r"$(\theta-\mu_{\rm prior})\sqrt{n}/\sigma$")
ax.set_ylabel("coverage of 95% credible interval")
ax.set_ylim(0, 1.02); ax.legend(fontsize=7.5, loc="center right")
fig.tight_layout()
savefig(fig, "ch04", "hdi_vs_ci")

save_numbers("ch04", "18_credible_vs_confidence", {
    "FourBCrConfLo": f"{float(c_lo):.3f}", "FourBCrConfHi": f"{float(c_hi):.3f}",
    "FourBCrCredLo": f"{k_lo:.3f}", "FourBCrCredHi": f"{k_hi:.3f}",
    "FourBCrCovConfMin": f"{cov_conf.min():.3f}", "FourBCrCovCredMin": f"{cov_cred.min():.3f}",
    "FourBCrCovCredFive": f"{cov_cred_big[0]:.3f}", "FourBCrCovCredTwenty": f"{cov_cred_big[1]:.3f}",
    "FourBCrCovCredFifty": f"{cov_cred_big[2]:.3f}",
    "FourBCrCredProbConfSix": f"{credprob_conf[6]:.3f}", "FourBCrCredProbConfFifty": f"{credprob_conf[50]:.3f}",
    "FourBCrCredProbConfZero": f"{credprob_conf[0]:.3f}",
    "FourBCrNormCovZero": f"{cov_g1_d0:.3f}", "FourBCrNormCovTwo": f"{cov_g1_d2:.3f}",
    "FourBCrNormCovFour": f"{cov_g1_d4:.3f}",
    "FourBHdiInfLo": f"{h_inf[0]:.3f}", "FourBHdiInfHi": f"{h_inf[1]:.3f}",
    "FourBHdiFlatLo": f"{h_flat[0]:.3f}", "FourBHdiFlatHi": f"{h_flat[1]:.3f}",
    "FourBHdiPhalf": f"{p_half_inf:.3f}",
    "FourBRopeOneLo": f"{rope1[0]:.3f}", "FourBRopeOneHi": f"{rope1[1]:.3f}",
    "FourBRopeTwoLo": f"{rope2[0]:.3f}", "FourBRopeTwoHi": f"{rope2[1]:.3f}",
    "FourBRopeTwoIn": f"{post_rope2_in:.3f}",
})
print(f"Poisson: conf [{float(c_lo):.3f},{float(c_hi):.3f}], cred [{k_lo:.3f},{k_hi:.3f}]")
print(f"coverage min: conf {cov_conf.min():.3f}, cred {cov_cred.min():.3f}; cred at lam 5,20,50: {np.round(cov_cred_big, 3)}")
print(f"credible prob of conf int at y=0,6,50: {credprob_conf[0]:.3f} {credprob_conf[6]:.3f} {credprob_conf[50]:.3f}")
print(f"normal cred coverage gamma=1: d=0 {cov_g1_d0:.3f} d=2 {cov_g1_d2:.3f} d=4 {cov_g1_d4:.3f}")
print(f"HDI beta(11,11) prior {h_inf}, flat {h_flat}, P(theta<0.5) {p_half_inf:.3f}; ROPE ex {rope1} {rope2} in {post_rope2_in:.3f}")
