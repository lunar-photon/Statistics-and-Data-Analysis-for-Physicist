"""41_exponential.py -- a decaying exponential, an unknown noise level, and data sets that disagree.

Question 1: a source decays, y(t) = A exp(-t/tau) + noise.  The model is not linear in tau, so
the posterior of (A, tau) need not be Gaussian.  How far is it from the Gaussian (Laplace)
approximation at its peak, and does the chain see the difference?
Question 2: the noise level sigma is unknown.  Integrating it out analytically (Jeffreys prior
1/sigma) gives p(A, tau | d) ~ [chi0^2(A, tau)]^(-N/2) (Hobson et al. eq. 2.26).  Does a chain
in (A, tau, ln sigma), with the sigma column simply ignored, give the same marginal?  What
happens if sigma is fixed at a wrong value instead?
Question 3 (Padilla et al. 2019, sec. 5.2): two data sets drawn from different lines,
D1: y = 3 + 2x (sigma 0.3), D2: y = 3.5 + 1.5x (sigma 0.5), 16 points each, x = 3 N(0,1).
The plain fit (their H0) gives a = 3.528 +- 0.056, b = 1.795 +- 0.014.  With one weight per data
set, alpha_k, marginalised with the prior exp(-alpha) (their eq. 50, H1), the posterior has two
peaks.  Their Bayes factor K(H1 : H0) = 37 (case 2) and 3 (case 1, both sets from 3 + 2x,
sigma 0.3 and 0.2).  We compute the evidences by direct quadrature over the prior box.
Writes: figures/ch09/exp_fit.pdf, figures/ch09/exp_sigma.pdf, figures/ch09/hyper.pdf,
        results/ch09/41_exponential.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import gammaln, logsumexp
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
import lib_mcmc9c as mc

setup()
rng = rng_for("ch09", "41_exponential")
nums = {}

# ================================================================ 1. the decaying source
A_TRUE, TAU_TRUE, SIG_TRUE = 100.0, 4.0, 6.0     # counts/s at t = 0, hours, counts/s
N = 15
t = np.linspace(0.0, 14.0, N)                       # one reading per hour
y = A_TRUE * np.exp(-t / TAU_TRUE) + SIG_TRUE * rng.standard_normal(N)
PRIOR = dict(A=(0.0, 300.0), tau=(0.1, 30.0))


def model(A, tau, tt=t):
    return A * np.exp(-tt / tau)


def chi2_0(A, tau):
    """Sum of squared residuals (no division by sigma)."""
    r = y - model(A, tau)
    return r @ r


def lp_known(th, sig=SIG_TRUE):
    A, tau = th
    if not (PRIOR["A"][0] < A < PRIOR["A"][1] and PRIOR["tau"][0] < tau < PRIOR["tau"][1]):
        return -np.inf
    return -0.5 * chi2_0(A, tau) / sig ** 2


def lp_three(th):
    """(A, tau, ln sigma): flat in ln sigma is the Jeffreys prior 1/sigma."""
    A, tau, ls = th
    if not (-5 < ls < 5):
        return -np.inf
    base = lp_known((A, tau), 1.0)                # -chi0^2 / 2
    if not np.isfinite(base):
        return -np.inf
    return -N * ls + base * np.exp(-2 * ls)


def lp_marg(th):
    """sigma integrated out analytically: (N/2) ln chi0^2 with a minus sign."""
    A, tau = th
    if not (PRIOR["A"][0] < A < PRIOR["A"][1] and PRIOR["tau"][0] < tau < PRIOR["tau"][1]):
        return -np.inf
    return -0.5 * N * np.log(chi2_0(A, tau))


# maximum and Laplace (Gaussian) approximation: Hessian of -ln p at the peak
opt = optimize.minimize(lambda th: -lp_known(th), [90.0, 5.0], method="Nelder-Mead",
                        options=dict(xatol=1e-8, fatol=1e-10))
th_hat = opt.x


def hessian(f, x, h):
    x = np.asarray(x, float); d = x.size; H = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            ei, ej = np.eye(d)[i] * h[i], np.eye(d)[j] * h[j]
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


H = hessian(lambda th: -lp_known(th), th_hat, [0.01, 0.001])
C_lap = np.linalg.inv(H)

# exact posterior on a grid (known sigma): normalise, marginalise by summing a column
gA = np.linspace(70, 130, 400)
gT = np.linspace(2.5, 6.5, 400)
AA, TT = np.meshgrid(gA, gT, indexing="ij")
R = y[None, None, :] - AA[..., None] * np.exp(-t[None, None, :] / TT[..., None])
CHI0 = np.sum(R ** 2, axis=-1)
logp_grid = -0.5 * CHI0 / SIG_TRUE ** 2
P = np.exp(logp_grid - logp_grid.max()); P /= P.sum()
pT_exact = P.sum(axis=0) / (gT[1] - gT[0])
logm_grid = -0.5 * N * np.log(CHI0)
Pm = np.exp(logm_grid - logm_grid.max()); Pm /= Pm.sum()
pT_marg = Pm.sum(axis=0) / (gT[1] - gT[0])
SIG_WRONG = 3.0
lw = -0.5 * CHI0 / SIG_WRONG ** 2
Pw = np.exp(lw - lw.max()); Pw /= Pw.sum()
pT_wrong = Pw.sum(axis=0) / (gT[1] - gT[0])


def mom(p, g):
    w = p / p.sum(); m = np.sum(w * g); s = np.sqrt(np.sum(w * (g - m) ** 2))
    sk = np.sum(w * (g - m) ** 3) / s ** 3
    return m, s, sk


mT, sT, skT = mom(pT_exact, gT)
mTm, sTm, _ = mom(pT_marg, gT)
mTw, sTw, _ = mom(pT_wrong, gT)

# chains
NCH, NST, BURN = 4, 20000, 2000
st = np.c_[rng.uniform(60, 140, NCH), rng.uniform(2, 8, NCH)]
chK, _, accK = mc.run_chains(lp_known, st, C_lap, NST, rng)
pK = chK[:, BURN:].reshape(-1, 2)
st3 = np.c_[st, np.log(rng.uniform(2, 12, NCH))]
C3 = np.diag([C_lap[0, 0], C_lap[1, 1], 0.2 ** 2]); C3[0, 1] = C3[1, 0] = C_lap[0, 1]
ch3, _, acc3 = mc.run_chains(lp_three, st3, C3, NST, rng)
p3 = ch3[:, BURN:].reshape(-1, 3)
R3 = mc.rhat(ch3[:, BURN:])
mK, sK, rK = mc.summary(pK)
# the posterior of sigma itself: chi0^2 at fixed (A, tau) gives sigma^2 ~ chi0^2 / chi^2_N; marginal from chain
sig_chain = np.exp(p3[:, 2])
nums.update({
    "NineCExpN": N, "NineCExpAtrue": f"{A_TRUE:.0f}", "NineCExpTautrue": f"{TAU_TRUE:.0f}",
    "NineCExpSigtrue": f"{SIG_TRUE:.0f}", "NineCExpSigWrong": f"{SIG_WRONG:.0f}",
    "NineCExpAhat": f"{th_hat[0]:.1f}", "NineCExpTauhat": f"{th_hat[1]:.2f}",
    "NineCExpSaLap": f"{np.sqrt(C_lap[0, 0]):.1f}", "NineCExpStLap": f"{np.sqrt(C_lap[1, 1]):.2f}",
    "NineCExpRhoLap": f"{C_lap[0, 1] / np.sqrt(C_lap[0, 0] * C_lap[1, 1]):.2f}",
    "NineCExpTauMean": f"{mT:.2f}", "NineCExpTauSd": f"{sT:.2f}", "NineCExpTauSkew": f"{skT:.2f}",
    "NineCExpTauMeanMC": f"{mK[1]:.2f}", "NineCExpTauSdMC": f"{sK[1]:.2f}",
    "NineCExpAMeanMC": f"{mK[0]:.1f}", "NineCExpASdMC": f"{sK[0]:.1f}", "NineCExpRhoMC": f"{rK[0, 1]:.2f}",
    "NineCExpTauSkewMC": f"{stats.skew(pK[:, 1]):.2f}",
    "NineCExpAccK": f"{accK.mean():.2f}", "NineCExpAccThree": f"{acc3.mean():.2f}",
    "NineCExpRmax": f"{R3.max():.4f}",
    "NineCExpTauMeanMarg": f"{mTm:.2f}", "NineCExpTauSdMarg": f"{sTm:.2f}",
    "NineCExpTauMeanChain": f"{p3[:, 1].mean():.2f}", "NineCExpTauSdChain": f"{p3[:, 1].std():.2f}",
    "NineCExpTauSdWrong": f"{sTw:.2f}",
    "NineCExpSigMed": f"{np.median(sig_chain):.1f}", "NineCExpSigLo": f"{np.percentile(sig_chain, 16):.1f}",
    "NineCExpSigHi": f"{np.percentile(sig_chain, 84):.1f}",
    "NineCExpChiMin": f"{chi2_0(*th_hat) / SIG_TRUE ** 2:.1f}",
    "NineCExpNsteps": NST, "NineCExpBurn": BURN,
})

# ---- figure: data with posterior predictive band, joint, marginal of tau
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3))
ax = axs[0]
tt = np.linspace(0, 15, 200)
draws = pK[rng.integers(0, len(pK), 2000)]
curves = draws[:, :1] * np.exp(-tt[None, :] / draws[:, 1:])
lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
ax.fill_between(tt, lo, hi, color=SERIES[0], alpha=0.25, lw=0, label="95% band of the curve")
for A_, T_ in draws[:15]:
    ax.plot(tt, model(A_, T_, tt), color=SERIES[0], lw=0.5, alpha=0.4)
ax.errorbar(t, y, yerr=SIG_TRUE, fmt="o", ms=3.5, color=SERIES[1], lw=0.8, label="readings")
ax.plot(tt, model(A_TRUE, TAU_TRUE, tt), color="k", ls=":", lw=1, label="truth")
ax.set_xlabel("time $t$ (h)"); ax.set_ylabel("count rate $y$ (s$^{-1}$)")
ax.set_title("(a) data and curves drawn from the chain", fontsize=9); ax.legend(fontsize=7)
ax = axs[1]
ax.plot(pK[::40, 0], pK[::40, 1], ".", ms=1, color=SERIES[0], alpha=0.3)
ax.contour(gA, gT, P.T, levels=mc.hpd_levels(P, (0.95, 0.68)), colors=["k"], linewidths=1.0)
ax.plot([], [], color="k", lw=1, label="exact (grid)")
mc.gauss_ellipse(ax, th_hat, C_lap, color=SERIES[1], ls="--", label="Laplace (Gaussian)")
ax.plot(A_TRUE, TAU_TRUE, "x", color=SERIES[7], ms=7, mew=1.5, label="truth")
ax.set_xlabel("$A$ (s$^{-1}$)"); ax.set_ylabel(r"$\tau$ (h)")
ax.set_title("(b) joint posterior, $\\sigma$ known", fontsize=9); ax.legend(fontsize=6.5)
ax = axs[2]
h, e = np.histogram(pK[:, 1], bins=70, range=(2.5, 6.5), density=True)
ax.stairs(h, e, color=SERIES[0], label=r"column $\tau$ of the chain")
ax.plot(gT, pT_exact, color="k", lw=1.2, label="exact marginal (grid)")
ax.plot(gT, stats.norm.pdf(gT, th_hat[1], np.sqrt(C_lap[1, 1])), color=SERIES[1], ls="--", lw=1.2,
        label="Laplace")
ax.set_xlabel(r"$\tau$ (h)"); ax.set_ylabel("density")
ax.set_title(r"(c) marginal of $\tau$", fontsize=9); ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "exp_fit")

# ---- figure: the unknown sigma
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
ax = axs[0]
h, e = np.histogram(p3[:, 1], bins=70, range=(2.0, 7.0), density=True)
ax.stairs(h, e, color=SERIES[0], label=r"chain in $(A,\tau,\ln\sigma)$, column $\tau$")
ax.plot(gT, pT_marg, color="k", lw=1.2, label=r"$\sigma$ integrated out, eq. (2.26)")
ax.plot(gT, pT_exact, color=SERIES[2], lw=1.1, ls="-.", label=r"$\sigma$ fixed at the truth")
ax.plot(gT, pT_wrong, color=SERIES[1], lw=1.1, ls="--", label=fr"$\sigma$ fixed at {SIG_WRONG:.0f} (wrong)")
ax.set_xlabel(r"$\tau$ (h)"); ax.set_ylabel("density"); ax.legend(fontsize=6.8)
ax.set_title(r"(a) marginal of $\tau$, four treatments of $\sigma$", fontsize=9)
ax = axs[1]
h, e = np.histogram(sig_chain, bins=60, range=(2, 14), density=True)
ax.stairs(h, e, color=SERIES[0], label=r"column $\sigma$ of the chain")
ax.axvline(SIG_TRUE, color="k", ls=":", lw=1, label="truth")
ax.set_xlabel(r"$\sigma$ (s$^{-1}$)"); ax.set_ylabel("density"); ax.legend(fontsize=7)
ax.set_title(r"(b) what the data say about $\sigma$", fontsize=9)
fig.tight_layout()
savefig(fig, "ch09", "exp_sigma")

# ================================================================ 3. Padilla case 2: hyperparameters
PAD2 = dict(a=3.528, sa=0.056, b=1.795, sb=0.014, K=37.0, K1=3.0)
r2 = rng_for("ch09", "41_exponential", stream=1)
n = 16


def line_set(a, b, s, r):
    xx = 3.0 * r.standard_normal(n)
    return xx, a + b * xx + s * r.standard_normal(n), np.full(n, s)


case2 = [line_set(3.0, 2.0, 0.3, r2), line_set(3.5, 1.5, 0.5, r2)]
case1 = [line_set(3.0, 2.0, 0.3, r2), line_set(3.0, 2.0, 0.2, r2)]
BOX = dict(a=(-5.0, 8.0), b=(0.0, 4.0))


def chi2_k(a, b, ds):
    xx, yy, ss = ds
    r = (yy[..., :] - a[..., None] - b[..., None] * xx) / ss
    return np.sum(r ** 2, axis=-1)


def loglike_H0(a, b, sets):
    out = 0.0
    for ds in sets:
        out = out - 0.5 * chi2_k(a, b, ds) - np.sum(np.log(np.sqrt(2 * np.pi) * ds[2]))
    return out


def loglike_H1(a, b, sets):
    """Padilla eq. (50): each set's weight alpha_k ~ exp(-alpha) integrated out."""
    out = 0.0
    for ds in sets:
        nk = ds[0].size
        out = out + np.log(2.0) + gammaln(nk / 2 + 1) - nk / 2 * np.log(np.pi) - np.sum(np.log(ds[2])) \
            - (nk / 2 + 1) * np.log(chi2_k(a, b, ds) + 2.0)
    return out


ga = np.linspace(*BOX["a"], 1301)
gb = np.linspace(*BOX["b"], 801)
GA, GB = np.meshgrid(ga, gb, indexing="ij")
cell = (ga[1] - ga[0]) * (gb[1] - gb[0])
prior_dens = 1.0 / ((BOX["a"][1] - BOX["a"][0]) * (BOX["b"][1] - BOX["b"][0]))
res = {}
for name, sets in (("two", case2), ("one", case1)):
    L0 = loglike_H0(GA, GB, sets)
    L1 = loglike_H1(GA, GB, sets)
    lnZ0 = logsumexp(L0) + np.log(cell * prior_dens)
    lnZ1 = logsumexp(L1) + np.log(cell * prior_dens)
    res[name] = dict(L0=L0, L1=L1, lnK=lnZ1 - lnZ0)
# H0 posterior moments for case 2 (it is a Gaussian, exactly: the model is linear)
P0 = np.exp(res["two"]["L0"] - res["two"]["L0"].max()); P0 /= P0.sum()
ma, mb = np.sum(P0 * GA), np.sum(P0 * GB)
sa, sb = np.sqrt(np.sum(P0 * (GA - ma) ** 2)), np.sqrt(np.sum(P0 * (GB - mb) ** 2))
P1 = np.exp(res["two"]["L1"] - res["two"]["L1"].max()); P1 /= P1.sum()
# the two peaks of H1: split the plane by the line halfway between the two true slopes
upper = GB > 1.75
w_up = P1[upper].sum()
L1 = res["two"]["L1"]
dpeak = L1[upper].max() - L1[~upper].max()          # ln of the ratio of the two peak heights
i_lo = np.unravel_index(np.argmax(np.where(upper, -np.inf, L1)), L1.shape)
lo_peak = (GA[i_lo], GB[i_lo])
# chains on H1 from dispersed starts: do they find both peaks?
lpH1 = lambda th: (loglike_H1(np.array(th[0]), np.array(th[1]), case2)
                   if (BOX["a"][0] < th[0] < BOX["a"][1] and BOX["b"][0] < th[1] < BOX["b"][1]) else -np.inf)
starts = np.c_[r2.uniform(*BOX["a"], 6), r2.uniform(*BOX["b"], 6)]
chH, _, accH = mc.run_chains(lpH1, starts, np.diag([0.08 ** 2, 0.02 ** 2]), 20000, r2)
frac_up = [(c[2000:, 1] > 1.75).mean() for c in chH]
nums.update({
    "NineCHypAzero": f"{ma:.3f}", "NineCHypSAzero": f"{sa:.3f}", "NineCHypBzero": f"{mb:.3f}",
    "NineCHypSBzero": f"{sb:.3f}",
    "NineCHypKtwo": float(f"{np.exp(res['two']['lnK']):.2e}"), "NineCHypLnKtwo": f"{res['two']['lnK']:.1f}",
    "NineCHypKone": f"{np.exp(res['one']['lnK']):.2f}", "NineCHypLnKone": f"{res['one']['lnK']:.2f}",
    "NineCHypWup": f"{w_up:.2f}", "NineCHypWlow": float(f"{1 - w_up:.1e}"),
    "NineCHypDpeak": f"{dpeak:.1f}", "NineCHypAlow": f"{lo_peak[0]:.2f}", "NineCHypBlow": f"{lo_peak[1]:.2f}",
    "NineCHypChainsBoth": sum(1 for f in frac_up if 0.02 < f < 0.98),
    "NineCHypChainsN": len(frac_up),
    "NineCHypChiZero": f"{(-2 * (res['two']['L0'].max()) - 2 * sum(np.sum(np.log(np.sqrt(2*np.pi)*d[2])) for d in case2)):.0f}",
})
print("frac in upper peak per chain:", np.round(frac_up, 3))

fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.2))
ax = axs[0]
for (xx, yy, ss), c, lab in zip(case2, SERIES[1:3], [r"$D_1$: $3+2x$, $\sigma=0.3$", r"$D_2$: $3.5+1.5x$, $\sigma=0.5$"]):
    ax.errorbar(xx, yy, yerr=ss, fmt="o", ms=3, color=c, lw=0.8, label=lab)
xl = np.linspace(-9, 9, 2)
ax.plot(xl, ma + mb * xl, color="k", lw=1, label="one line through both ($H_0$)")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.legend(fontsize=6.5)
ax.set_title("(a) two data sets that disagree", fontsize=9)
for k, (key, title) in enumerate((("L0", "(b) plain fit $H_0$"), ("L1", r"(c) weights $\alpha_k$ integrated out, $H_1$"))):
    ax = axs[k + 1]
    Pk = np.exp(res["two"][key] - res["two"][key].max()); Pk /= Pk.sum()
    ax.contour(ga, gb, Pk.T, levels=mc.hpd_levels(Pk, (0.95, 0.68)), colors=["k"], linewidths=1.0)
    if key == "L1":                                  # much lower contours reveal the second peak
        d = res["two"][key] - res["two"][key].max()
        ax.contour(ga, gb, d.T, levels=[-dpeak - 3.0, -dpeak - 1.15], colors=[SERIES[6]],
                   linewidths=0.9, linestyles="--")
        ax.plot([], [], color=SERIES[6], ls="--", lw=0.9, label="contours far below the top")
    ax.plot([3.0, 3.5], [2.0, 1.5], "x", color=SERIES[7], ms=7, mew=1.5, label="the two true lines")
    if key == "L0":
        ax.errorbar(PAD2["a"], PAD2["b"], xerr=PAD2["sa"], yerr=PAD2["sb"], fmt="D", ms=4,
                    color=SERIES[4], lw=1, label="Padilla et al.")
    else:
        for c in chH[:, 2000::50]:
            ax.plot(c[:, 0], c[:, 1], ".", ms=1.2, alpha=0.5)
    ax.set_xlim(2.6, 4.2); ax.set_ylim(1.35, 2.5)
    ax.set_xlabel("intercept $a$"); ax.set_ylabel("slope $b$"); ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title(title, fontsize=9)
fig.tight_layout()
savefig(fig, "ch09", "hyper")
save_numbers("ch09", "41_exponential", nums)
print(nums)
