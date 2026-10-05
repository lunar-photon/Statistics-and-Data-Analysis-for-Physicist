"""The handful of SciPy tools the book leans on, each answering one small physics question.

Question:  what do scipy.stats distributions (pdf, cdf, sf, ppf, rvs), integrate.quad,
           optimize.brentq, optimize.minimize, special.gammaln and interpolation return, and
           do they agree with answers we can check by hand?
Computes:  (a) the 95% two-sided point and the 5-sigma tail of a Gaussian, and 10^4 draws;
           (b) the same 5-sigma tail by numerical integration of the density;
           (c) the 95% upper limit on a Poisson mean after observing 0 and 3 events (a root);
           (d) the maximum-likelihood mean and width of Gaussian data (a minimisation);
           (e) ln(200!) from the log-gamma function, where 200! itself overflows a float;
           (f) linear versus cubic-spline interpolation of a coarse table.
Writes:    figures/chT0/03_scipy_tour.pdf, results/chT0/03_scipy_tour.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate, optimize, special, interpolate
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

rng = rng_for("chT0", "03_scipy_tour")
out = {}

# (a) one distribution object, five questions
z95 = stats.norm.ppf(0.975)                     # where does the cdf reach 0.975?
p5 = stats.norm.sf(5.0)                          # P(Z > 5) = 1 - cdf(5), computed without cancellation
x = stats.norm.rvs(loc=2.0, scale=0.5, size=10_000, random_state=rng)
out.update(TzZninetyfive=f"{z95:.4f}", TzPfive=p5, TzBackCdf=f"{stats.norm.cdf(z95):.4f}",
           TzRvsMean=f"{x.mean():.4f}", TzRvsStd=f"{x.std(ddof=1):.4f}")

# (b) the same tail by integrating the density from 5 to infinity
tail, tail_err = integrate.quad(stats.norm.pdf, 5.0, np.inf)
out.update(TzQuadTail=tail, TzQuadErr=tail_err, TzQuadRel=abs(tail - p5) / p5)

# (c) Poisson upper limit: the mean mu at which seeing n or fewer events has probability 0.05
def upper_limit(n, cl=0.95):
    return optimize.brentq(lambda mu: stats.poisson.cdf(n, mu) - (1 - cl), 1e-6, 50.0)
mu0, mu3 = upper_limit(0), upper_limit(3)
out.update(TzUpperZero=f"{mu0:.4f}", TzUpperZeroExact=f"{np.log(20):.4f}", TzUpperThree=f"{mu3:.3f}")

# (d) maximum likelihood by minimising minus the log-likelihood over (mu, ln sigma)
def neg_lnL(theta, data):
    mu, ln_sig = theta
    return -np.sum(stats.norm.logpdf(data, mu, np.exp(ln_sig)))
res = optimize.minimize(neg_lnL, x0=[0.0, 0.0], args=(x,), method="Nelder-Mead")
mu_hat, sig_hat = res.x[0], np.exp(res.x[1])
out.update(TzMinMu=f"{mu_hat:.4f}", TzMinSig=f"{sig_hat:.4f}", TzMeanCheck=f"{x.mean():.4f}",
           TzStdCheck=f"{x.std(ddof=0):.4f}", TzMinIters=res.nit)

# (e) logs of huge numbers: ln(200!) is fine, 200! as a float is not
ln_fact = special.gammaln(201)
try:
    float(math.factorial(200))
    overflow = "no"
except OverflowError:
    overflow = "yes"
out.update(TzLnFact=f"{ln_fact:.3f}", TzLnFactCheck=f"{math.lgamma(201):.3f}", TzFactOverflow=overflow)

# (f) interpolation of a coarse table of sin(x): straight lines versus a smooth spline
xt = np.linspace(0, np.pi, 6)
yt = np.sin(xt)
xf = np.linspace(0, np.pi, 400)
lin = np.interp(xf, xt, yt)
spl = interpolate.CubicSpline(xt, yt)(xf)
out.update(TzInterpLinErr=f"{np.max(np.abs(lin - np.sin(xf))):.4f}",
           TzInterpSplErr=f"{np.max(np.abs(spl - np.sin(xf))):.4f}")

# figure: left, pdf/cdf/ppf/sf of the standard Gaussian; right, the Poisson upper limits as roots
setup(8.4, 3.2)
fig, (ax0, ax1) = plt.subplots(1, 2)
z = np.linspace(-4, 4, 400)
ax0.plot(z, stats.norm.pdf(z), color=SERIES[0], label="pdf")
ax0.plot(z, stats.norm.cdf(z), color=SERIES[1], label="cdf")
ax0.fill_between(z[z > z95], stats.norm.pdf(z[z > z95]), color=SERIES[0], alpha=0.3, lw=0,
                 label="sf$(1.96)=0.025$")
ax0.plot([-4, z95, z95], [0.975, 0.975, 0], color=SERIES[2], ls=":", lw=1.2, label="ppf$(0.975)=1.96$")
ax0.set_xlabel("$z$")
ax0.legend(loc="center left", fontsize=8)
mu = np.linspace(0.01, 12, 400)
for n, root, c in [(0, mu0, SERIES[0]), (3, mu3, SERIES[1])]:
    ax1.plot(mu, stats.poisson.cdf(n, mu), color=c,
             label=f"$P(N\\leq {n}\\mid\\mu)$, brentq: $\\mu={root:.2f}$")
    ax1.plot(root, 0.05, "o", color=c)
    ax1.vlines(root, 0, 0.05, color=c, ls=":", lw=1)
ax1.axhline(0.05, color="0.4", lw=0.8, ls="--")
ax1.set_xlabel("Poisson mean $\\mu$")
ax1.set_ylabel("probability")
ax1.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "chT0", "03_scipy_tour")

save_numbers("chT0", "03_scipy_tour", out)
for k, v in out.items():
    print(f"{k:16s} {v}")
