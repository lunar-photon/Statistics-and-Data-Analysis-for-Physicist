"""40_line_fit.py -- a straight line fitted by the biased random walker, checked against the exact answer.

Question: data y_i = a + b x_i + noise.  What do a and b look like after the data, how does a
Metropolis chain represent that, and why is the histogram of one column of the chain the
marginal posterior of that parameter?  Can we reproduce the straight-line example of
Padilla et al. (2019, sec. 5.1): true line y = 3 + 2x, two data sets of 16 points with Gaussian
errors 0.3 and 0.2, flat priors a ~ U[0,5], b ~ U[0,3], five chains of 10 000 steps; their
result a = 2.982 +- 0.047, b = 1.994 +- 0.013?
Their design, as in the notebook of the repository the paper cites: x = 3 * N(0, 1) and, in the
fit, ONE unknown noise level sigma ~ U[0, 1.5] shared by all points (not the known errors).
Computes:
  * one simulated data set of that design;
  * model K (known errors 0.3 and 0.2, the paper's eq. 70): the exact posterior is a Gaussian
    with mean = weighted least squares and covariance (A^T W A)^-1; five MH chains, burn-in,
    R-hat, tau_int, ESS; marginals from single columns; emcee as the library cross-check;
  * model S (one unknown sigma, uniform prior): MH in (a, b, sigma); marginal of (a, b) by
    ignoring the sigma column, against the exact marginal, a Student-t with nu = N - 3;
  * the posterior widths of both models over 20 000 replications of the experiment, and where
    Padilla's published widths fall;
  * Kruschke's centring point: the same data with x shifted by +10, sampled with the same
    round proposal: correlation of (a, b), autocorrelation time.
Writes: figures/ch09/line_fit.pdf, figures/ch09/line_centring.pdf, results/ch09/40_line_fit.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
import lib_mcmc9c as mc

setup()
rng = rng_for("ch09", "40_line_fit")
A_TRUE, B_TRUE = 3.0, 2.0
N1, N2, S1, S2 = 16, 16, 0.3, 0.2
XSD = 3.0                                          # x = 3 N(0,1), as in the published notebook
PRIOR = dict(a=(0.0, 5.0), b=(0.0, 3.0), s=(0.0, 1.5))
PAD = dict(a=2.982, sa=0.047, b=1.994, sb=0.013)
nums = {}


def make_data(r):
    x = XSD * r.standard_normal(N1 + N2)
    sig = np.r_[np.full(N1, S1), np.full(N2, S2)]
    y = A_TRUE + B_TRUE * x + sig * r.standard_normal(x.size)
    return x, sig, y


x, sig, y = make_data(rng_for("ch09", "40_line_fit", stream=2))
A = np.c_[np.ones_like(x), x]                       # design matrix: columns 1 and x


def inside(a, b):
    return PRIOR["a"][0] < a < PRIOR["a"][1] and PRIOR["b"][0] < b < PRIOR["b"][1]


def logpost_K(th):
    """Known errors: ln p(a, b | d) = -chi^2/2 + const inside the flat prior box."""
    a, b = th
    if not inside(a, b):
        return -np.inf
    r = (y - a - b * x) / sig
    return -0.5 * r @ r


def logpost_S(th):
    """One unknown sigma, uniform prior: ln p(a, b, sigma | d) = -N ln sigma - RSS / (2 sigma^2)."""
    a, b, s = th
    if not (inside(a, b) and PRIOR["s"][0] < s < PRIOR["s"][1]):
        return -np.inf
    r = y - a - b * x
    return -x.size * np.log(s) - 0.5 * (r @ r) / s ** 2


def exact_K(x, sig, y=None):
    """Gaussian posterior for a flat prior: mean (A^T W A)^-1 A^T W y, covariance (A^T W A)^-1."""
    A = np.c_[np.ones_like(x), x]
    W = 1.0 / sig ** 2
    F = A.T @ (W[:, None] * A)
    cov = np.linalg.inv(F)
    mean = None if y is None else cov @ (A.T @ (W * y))
    return mean, cov


def exact_S(x, y):
    """Marginal of (a, b) after integrating sigma out (uniform prior): p ~ RSS(a,b)^{-(N-1)/2},
    a bivariate Student t with nu = N - 3, centre = ordinary least squares,
    covariance = RSS_min / (nu - 2) (A^T A)^-1."""
    A = np.c_[np.ones_like(x), x]
    G = np.linalg.inv(A.T @ A)
    m = G @ A.T @ y
    rss = np.sum((y - A @ m) ** 2)
    nu = x.size - 3
    return m, rss / (nu - 2) * G, nu, rss


m_K, C_K = exact_K(x, sig, y)
s_K = np.sqrt(np.diag(C_K))
m_S, C_S, nu_S, rss = exact_S(x, y)
s_S = np.sqrt(np.diag(C_S))

# ---------------------------------------------------------------- model K: five chains, like Padilla
NCH, NST, BURN = 5, 10000, 1000
starts = np.c_[rng.uniform(*PRIOR["a"], NCH), rng.uniform(*PRIOR["b"], NCH)]
prop = np.diag([0.05 ** 2, 0.015 ** 2])            # a rough guess of the widths, not the answer
ch, lp, acc = mc.run_chains(logpost_K, starts, prop, NST, rng)
post = ch[:, BURN:, :].reshape(-1, 2)
m_mc, s_mc, R_mc = mc.summary(post)
R = mc.rhat(ch[:, BURN:, :])
taus = [mc.tau_int(ch[0, BURN:, k]) for k in range(2)]
ess_tot = [sum(mc.ess(ch[c, BURN:, k]) for c in range(NCH)) for k in range(2)]
nums.update({
    "NineCLineAhat": f"{m_K[0]:.3f}", "NineCLineBhat": f"{m_K[1]:.4f}",
    "NineCLineSa": f"{s_K[0]:.4f}", "NineCLineSb": f"{s_K[1]:.4f}",
    "NineCLineRho": f"{C_K[0, 1] / (s_K[0] * s_K[1]):.2f}",
    "NineCLineAmc": f"{m_mc[0]:.3f}", "NineCLineBmc": f"{m_mc[1]:.4f}",
    "NineCLineSamc": f"{s_mc[0]:.4f}", "NineCLineSbmc": f"{s_mc[1]:.4f}", "NineCLineRhomc": f"{R_mc[0, 1]:.2f}",
    "NineCLineAcc": f"{acc.mean():.2f}", "NineCLineRa": f"{R[0]:.4f}", "NineCLineRb": f"{R[1]:.4f}",
    "NineCLineTaua": f"{taus[0]:.1f}", "NineCLineTaub": f"{taus[1]:.1f}",
    "NineCLineESSa": f"{ess_tot[0]:.0f}", "NineCLineESSb": f"{ess_tot[1]:.0f}",
    "NineCLineNkept": NCH * (NST - BURN), "NineCLineChisq": f"{-2 * logpost_K(m_K):.1f}",
    "NineCLineNdata": N1 + N2, "NineCLineNsteps": NST, "NineCLineBurn": BURN,
    "NineCLineXrms": f"{np.sqrt(np.mean((x - x.mean()) ** 2)):.2f}",
    "NineCLineSaFloor": f"{1 / np.sqrt(np.sum(1 / sig ** 2)):.4f}",
})
nums["NineCLineMCSEb"] = f"{s_mc[1] * np.sqrt(taus[1] / (NCH * (NST - BURN))):.5f}"

# library cross-check: emcee on the same posterior
import emcee
nw = 16
p0 = m_K + 1e-3 * rng.standard_normal((nw, 2))
sampler = emcee.EnsembleSampler(nw, 2, logpost_K)
sampler.random_state = np.random.RandomState(rng.integers(2 ** 31)).get_state()
sampler.run_mcmc(p0, 4000, progress=False)
em = sampler.get_chain(discard=1000, flat=True)
nums.update({"NineCLineAem": f"{em[:, 0].mean():.3f}", "NineCLineSaem": f"{em[:, 0].std():.4f}",
             "NineCLineBem": f"{em[:, 1].mean():.4f}", "NineCLineSbem": f"{em[:, 1].std():.4f}"})

# ---------------------------------------------------------------- model S: one unknown sigma
starts3 = np.c_[starts, rng.uniform(0.1, 1.0, NCH)]
prop3 = np.diag([0.05 ** 2, 0.015 ** 2, 0.03 ** 2])
ch3, _, acc3 = mc.run_chains(logpost_S, starts3, prop3, NST, rng)
post3 = ch3[:, BURN:, :].reshape(-1, 3)
m3, s3, R3 = mc.summary(post3)
nums.update({
    "NineCLineSAmc": f"{m3[0]:.3f}", "NineCLineSSamc": f"{s3[0]:.4f}",
    "NineCLineSBmc": f"{m3[1]:.4f}", "NineCLineSSbmc": f"{s3[1]:.4f}",
    "NineCLineSSigmc": f"{m3[2]:.3f}", "NineCLineSSSigmc": f"{s3[2]:.3f}",
    "NineCLineSAex": f"{m_S[0]:.3f}", "NineCLineSSaex": f"{s_S[0]:.4f}",
    "NineCLineSBex": f"{m_S[1]:.4f}", "NineCLineSSbex": f"{s_S[1]:.4f}",
    "NineCLineSnu": nu_S, "NineCLineSrms": f"{np.sqrt(rss / (x.size - 2)):.3f}",
    "NineCLineSAcc": f"{acc3.mean():.2f}",
    "NineCLineSRmax": f"{mc.rhat(ch3[:, BURN:, :]).max():.4f}",
    "NineCLinePooled": f"{np.sqrt((S1 ** 2 + S2 ** 2) / 2):.3f}",
})

# ---------------------------------------------------------------- Padilla's widths: how typical?
NREP = 20000
r2 = rng_for("ch09", "40_line_fit", stream=1)
wK, wS = np.empty((NREP, 2)), np.empty((NREP, 2))
for i in range(NREP):
    xr, sr, yr = make_data(r2)
    wK[i] = np.sqrt(np.diag(exact_K(xr, sr)[1]))
    wS[i] = np.sqrt(np.diag(exact_S(xr, yr)[1]))
for tag, w in (("K", wK), ("S", wS)):
    qa, qb = np.percentile(w[:, 0], [16, 50, 84]), np.percentile(w[:, 1], [16, 50, 84])
    nums.update({f"NineCLine{tag}SaMed": f"{qa[1]:.4f}", f"NineCLine{tag}SaLo": f"{qa[0]:.4f}",
                 f"NineCLine{tag}SaHi": f"{qa[2]:.4f}", f"NineCLine{tag}SbMed": f"{qb[1]:.4f}",
                 f"NineCLine{tag}SbLo": f"{qb[0]:.4f}", f"NineCLine{tag}SbHi": f"{qb[2]:.4f}",
                 f"NineCLine{tag}PadSaPct": f"{100 * np.mean(w[:, 0] < PAD['sa']):.0f}",
                 f"NineCLine{tag}PadSbPct": f"{100 * np.mean(w[:, 1] < PAD['sb']):.0f}"})
nums["NineCLineNrep"] = NREP

# ---------------------------------------------------------------- Kruschke: centre the predictor
SHIFT = 10.0
xs = x - x.mean() + SHIFT                           # same physics, the origin of x moved away
xc = x - x.mean()                                   # exactly centred
PRIOR_W = (-40.0, 40.0)


def make_lp(xx):
    def lp_(th):
        a, b = th
        if not (PRIOR_W[0] < a < PRIOR_W[1] and 0.0 < b < 3.0):
            return -np.inf
        r = (y - a - b * xx) / sig
        return -0.5 * r @ r
    return lp_


m_c, C_c = exact_K(xc, sig, y)
m_s, C_s = exact_K(xs, sig, y)
iso = np.diag([s_K[0] ** 2, s_K[0] ** 2])           # the same round proposal for both runs
c_c, _, acc_c = mc.mh(make_lp(xc), m_c, iso, 20000, rng, scale=1.0)
c_s, _, acc_s = mc.mh(make_lp(xs), m_s, iso, 20000, rng, scale=1.0)
rho_c = C_c[0, 1] / np.sqrt(C_c[0, 0] * C_c[1, 1])
rho_s = C_s[0, 1] / np.sqrt(C_s[0, 0] * C_s[1, 1])
tc, ts = mc.tau_int(c_c[:, 1]), mc.tau_int(c_s[:, 1])
nums.update({"NineCLineShift": f"{SHIFT:.0f}", "NineCLineRhoShift": f"{rho_s:.3f}",
             "NineCLineRhoCent": f"{rho_c:.3f}",
             "NineCLineTauCent": f"{tc:.1f}", "NineCLineTauShift": f"{ts:.0f}",
             "NineCLineAccCent": f"{acc_c:.2f}", "NineCLineAccShift": f"{acc_s:.2f}",
             "NineCLineSaShift": f"{np.sqrt(C_s[0, 0]):.3f}", "NineCLineSaCent": f"{np.sqrt(C_c[0, 0]):.4f}"})

# ---------------------------------------------------------------- figure 1: data, joint, marginal
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3))
ax = axs[0]
xx = np.linspace(-9, 9, 2)
pick = post[rng.integers(0, len(post), 40)]
for a, b in pick:
    ax.plot(xx, a + b * xx, color=SERIES[0], alpha=0.12, lw=0.8)
ax.errorbar(x[:N1], y[:N1], yerr=S1, fmt="o", ms=3, color=SERIES[1], lw=0.8, label=r"$D_1$, $\sigma=0.3$")
ax.errorbar(x[N1:], y[N1:], yerr=S2, fmt="s", ms=3, color=SERIES[2], lw=0.8, label=r"$D_2$, $\sigma=0.2$")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.set_title("(a) data and 40 lines drawn from the chain", fontsize=9)
ax.legend(fontsize=7.5, loc="upper left")
ax = axs[1]
ax.plot(post[::20, 0], post[::20, 1], ".", ms=1, color=SERIES[0], alpha=0.25)
mc.contour2d(ax, post[:, 0], post[:, 1], SERIES[0], filled=False, label="chain, known errors")
mc.gauss_ellipse(ax, m_K, C_K, color="k", label="exact Gaussian")
mc.contour2d(ax, post3[:, 0], post3[:, 1], SERIES[3], filled=False, ls="--", label=r"chain, unknown $\sigma$")
ax.plot(A_TRUE, B_TRUE, "x", color=SERIES[7], ms=7, mew=1.5, label="truth")
ax.errorbar(PAD["a"], PAD["b"], xerr=PAD["sa"], yerr=PAD["sb"], fmt="D", ms=4, color=SERIES[4],
            lw=1.0, label="Padilla et al.")
ax.set_xlabel("intercept $a$"); ax.set_ylabel("slope $b$")
ax.set_title("(b) joint posterior", fontsize=9)
ax.legend(fontsize=6.3, loc="lower left")
ax = axs[2]
h, e = np.histogram(post[:, 1], bins=60, density=True)
ax.stairs(h, e, color=SERIES[0], label="column $b$ of the chain")
bb = np.linspace(e[0], e[-1], 300)
theory_line(ax, bb, stats.norm.pdf(bb, m_K[1], s_K[1]), label="exact marginal")
ax.set_xlabel("slope $b$"); ax.set_ylabel("density")
ax.set_title("(c) marginal of $b$: ignore column $a$", fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "line_fit")

# ---------------------------------------------------------------- figure 2: centring
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.0))
ax = axs[0]
ax.plot(c_c[:3000, 1], lw=0.6, color=SERIES[0], label=f"$x$ centred, $\\tau={tc:.0f}$")
ax.plot(c_s[:3000, 1], lw=0.6, color=SERIES[1], label=f"$x$ shifted by {SHIFT:.0f}, $\\tau={ts:.0f}$")
ax.set_xlabel("step"); ax.set_ylabel("slope $b$"); ax.legend(fontsize=7)
ax.set_title("(a) trace, same round proposal", fontsize=9)
ax = axs[1]
lag = np.arange(301)
ax.plot(lag, mc.acf(c_c[:, 1], 300), color=SERIES[0], label="centred")
ax.plot(lag, mc.acf(c_s[:, 1], 300), color=SERIES[1], label="shifted")
ax.axhline(0, color="0.5", lw=0.6)
ax.set_xlabel("lag $k$"); ax.set_ylabel(r"$\rho(k)$"); ax.legend(fontsize=7)
ax.set_title("(b) autocorrelation of $b$", fontsize=9)
ax = axs[2]
ax.plot(c_s[::10, 0], c_s[::10, 1], ".", ms=1, color=SERIES[1], alpha=0.3)
mc.gauss_ellipse(ax, m_s, C_s, color="k")
t = np.linspace(0, 2 * np.pi, 100)
r0 = np.sqrt(iso[0, 0])
ax.plot(m_s[0] + r0 * np.cos(t), m_s[1] + r0 * np.sin(t), color=SERIES[2], lw=1.2, label="proposal (1$\\sigma$)")
ax.set_xlabel("intercept in shifted $x$"); ax.set_ylabel("slope $b$"); ax.legend(fontsize=7)
ax.set_title(f"(c) shifted: $\\rho(a,b)={rho_s:.2f}$", fontsize=9)
fig.tight_layout()
savefig(fig, "ch09", "line_centring")
save_numbers("ch09", "40_line_fit", nums)
print(nums)
