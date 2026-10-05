"""c05_fisher_post.py -- how well does one map measure eps?  Fisher forecast and posterior.

Question: for the local model of lib_fnl (same power spectrum for every eps), what error on eps
does each summary deliver from one 256 x 256 map, according to the Fisher matrix with
Monte Carlo derivatives, and does a Metropolis-Hastings posterior built on a simulated
Gaussian likelihood agree with it and cover the true value as often as it claims?
Computes, from the cache of c02_mc_fnl.py:
  1. derivatives dmu/deps by central differences with common random numbers (the same g at
     +eps and -eps), for steps 0.025, 0.05, 0.1, and without common random numbers (the two
     sides from different seeds), with the noise bias of the Fisher number tr(C^-1 Sigma_d);
  2. sigma(eps) = F^-1/2 per summary and for combinations;
  3. a likelihood for the Betti-curve vector: mean mu(eps) = quadratic fit through the grid
     means (seeds 0-249), covariance from the 1000 null maps with the Hartlap factor;
  4. for one mock map (seed 250, eps = 0.05): a Metropolis-Hastings chain (lib_mcmc), an emcee
     cross-check, and the posterior on a grid;
  5. coverage: the 68% central credible interval for each of the 250 mock maps
     (seeds 250-499) at eps = 0.05 and at eps = 0.
Writes: figures/ch12/c_fisher.pdf, figures/ch12/c_post.pdf, results/ch12/c05_fisher_post.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch09"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, DATA
from lib_fnl import NU
from lib_mcmc import metropolis, tau_int

rng = rng_for("ch12", "c05_fisher_post")
z = np.load(DATA / "ch12" / "c_fnl_mc.npz")
eg = list(z["eps_grid"])
EPS = np.array(eg)
i0 = eg.index(0.0)

SUMMARIES = {
    "pk": (["pk"], "power spectrum"),
    "skew": (["skew"], "skewness"),
    "area": (["area"], "area fraction"),
    "chi": (["chi"], r"$\chi(\nu)$"),
    "betti": (["b0", "b1"], r"Betti curves"),
    "pd": (["pd0", "pd1"], "persistence hist."),
    "bettiA": (["b0A", "b1A"], "Betti, area thr."),
    "pdA": (["pd0A", "pd1A"], "pers., area thr."),
    "betti+skew": (["b0", "b1", "skew"], "Betti + skewness"),
    "betti+pk": (["b0", "b1", "pk"], "Betti + power sp."),
}


def vec(prefix, keys):
    return np.concatenate([z[prefix + k] for k in keys], axis=-1)


def null_model(keys):
    N = vec("null_", keys)
    ok = N.std(0) > 1e-9
    N = N[:, ok]
    n, p = N.shape
    Ci = np.linalg.inv(np.atleast_2d(np.cov(N.T))) * (n - p - 2) / (n - 1)
    return N.mean(0), Ci, ok, n, p


def fisher(keys, step=0.05, seeds=slice(None), crn=True):
    """(F, noise bias of F, derivative, its standard error) for one summary."""
    mu, Ci, ok, n, p = null_model(keys)
    G = vec("grid_", keys)[seeds][..., ok]
    jp, jm = eg.index(step), eg.index(-step)
    if crn:
        D = (G[:, jp] - G[:, jm]) / (2 * step)             # same g on both sides
    else:
        h = len(G) // 2                                    # +step from one half, -step from the other
        D = (G[:h, jp] - G[h:2 * h, jm]) / (2 * step)
    d = D.mean(0)
    Sd = np.atleast_2d(np.cov(D.T)) / len(D)               # covariance of the averaged derivative
    F = d @ Ci @ d
    return F, np.trace(Ci @ Sd), d, np.sqrt(np.diag(Sd))


# ------------------------------------------------------------ 1-2. Fisher numbers
sig = {}
for name, (keys, _) in SUMMARIES.items():
    F, b, _, _ = fisher(keys)
    sig[name] = (1 / np.sqrt(F), b / F)
steps = {s: 1 / np.sqrt(fisher(SUMMARIES["betti"][0], step=s)[0]) for s in (0.025, 0.05, 0.1)}
F_crn, b_crn, d_crn, se_crn = fisher(SUMMARIES["betti"][0])
F_ind, b_ind, d_ind, se_ind = fisher(SUMMARIES["betti"][0], crn=False)
# variance reduction of one derivative component (beta0 at nu = 1) by common random numbers
k1 = list(NU).index(1.0)
G = z["grid_b0"][:, :, k1]
jp, jm = eg.index(0.05), eg.index(-0.05)
var_crn = np.var(G[:, jp] - G[:, jm])
var_ind = np.var(G[:250, jp] - G[250:, jm])
rho = np.corrcoef(G[:, jp], G[:, jm])[0, 1]
# convergence of sigma with the number of derivative pairs
npairs = [25, 50, 100, 250, 500]
conv = [1 / np.sqrt(fisher(SUMMARIES["betti"][0], seeds=slice(0, m))[0]) for m in npairs]

setup(9.0, 3.2)
fig, ax = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.2, 1]))
mu, Ci, ok, n, p = null_model(SUMMARIES["betti"][0])
full = np.concatenate([NU, NU])[ok]
nb0 = int(ok[:13].sum())
p_betti = int(ok.sum())                                    # length of the Betti vector
for sl, c, lab in ((slice(0, nb0), SERIES[0], r"$\partial\beta_0/\partial\epsilon$"),
                   (slice(nb0, None), SERIES[1], r"$\partial\beta_1/\partial\epsilon$")):
    ax[0].errorbar(full[sl] - 0.04, d_crn[sl], 2 * se_crn[sl], fmt="o-", color=c, ms=3,
                   label=lab + ", common random numbers")
    ax[0].errorbar(full[sl] + 0.04, d_ind[sl], 2 * se_ind[sl], fmt="s", color=c, ms=3, mfc="none",
                   alpha=0.6, label=lab + ", independent seeds")
ax[0].axhline(0, color=INK2, lw=0.6)
ax[0].set_xlabel(r"threshold $\nu$")
ax[0].set_ylabel(r"derivative (counts per unit $\epsilon$)")
ax[0].legend(fontsize=6.5)
order = ["pk", "skew", "area", "bettiA", "pdA", "chi", "betti", "pd", "betti+pk", "betti+skew"]
vals = [sig[k][0] for k in order]
cols = [SERIES[0] if k == "pk" else (SERIES[1] if k in ("skew", "area") else
        (SERIES[3] if k.endswith("A") else SERIES[2])) for k in order]
y = np.arange(len(order))
ax[1].barh(y, vals, color=cols)
ax[1].set_yticks(y)
ax[1].set_yticklabels([SUMMARIES[k][1] for k in order], fontsize=7)
ax[1].set_xscale("log")
ax[1].set_xlabel(r"Fisher $\sigma(\epsilon)$ from one map")
for yi, v in zip(y, vals):
    ax[1].text(v * 1.08, yi, f"{v:.3f}" if v < 1 else r"$\approx$" + f"{v:.0f} (noise)", va="center", fontsize=6.5)
ax[1].set_xlim(0.005, 60)
ax[1].invert_yaxis()
fig.tight_layout()
savefig(fig, "ch12", "c_fisher")

# ------------------------------------------------------------ 3. the likelihood
keys = SUMMARIES["betti"][0]
dseeds, mseeds = np.arange(250), np.arange(250, 500)
Gm = vec("grid_", keys)[dseeds][..., ok]                   # (250, 7, p)
means = Gm.mean(0)                                         # (7, p) mean vector at each grid eps
coef = np.polyfit(EPS, means, 2)                           # quadratic in eps, per component
fit_res = np.abs(np.polyval(coef, EPS[:, None]) - means).max() / (Gm.std(0).max())


def mu_of(e):
    return coef[0] * e ** 2 + coef[1] * e + coef[2]


PRIOR = 0.15


def loglike(e, S):
    if abs(e) > PRIOR:
        return -np.inf
    r = S - mu_of(e)
    return -0.5 * r @ Ci @ r


egrid = np.linspace(-PRIOR, PRIOR, 1201)
MU = coef[0][None, :] * egrid[:, None] ** 2 + coef[1][None, :] * egrid[:, None] + coef[2][None, :]


def grid_post(S):
    R = S[None, :] - MU
    lp = -0.5 * np.einsum("gi,ij,gj->g", R, Ci, R)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    return w


def summary_post(w):
    m = (w * egrid).sum()
    sd = np.sqrt((w * (egrid - m) ** 2).sum())
    cdf = np.cumsum(w)
    lo, hi = np.interp([0.16, 0.84], cdf, egrid)
    return m, sd, lo, hi


# ------------------------------------------------------------ 4. one mock map
TRUE = 0.05
# the illustration uses the first mock map (seeds 250, 251, ...) whose posterior mean lies within
# one posterior sd of the truth, i.e. a typical map; section 5 below uses all 250 of them
for MOCK in mseeds:
    S_obs = vec("grid_", keys)[MOCK, eg.index(TRUE)][ok]
    _m, _sd, _, _ = summary_post(grid_post(S_obs))
    if abs(_m - TRUE) < _sd:
        break
w_obs = grid_post(S_obs)
m_g, sd_g, lo_g, hi_g = summary_post(w_obs)
chain, lp, acc = metropolis(lambda x: loglike(x[0], S_obs), [0.0], 20000, 1.5 * sd_g, rng)
burn = 1000
xs = chain[burn:, 0]
m_mh, sd_mh = xs.mean(), xs.std()
tau = tau_int(xs)[0]
import emcee                                               # cross-check: affine-invariant ensemble
np.random.seed(1)                                          # emcee draws from numpy's global generator
sampler = emcee.EnsembleSampler(16, 1, lambda x: loglike(x[0], S_obs))
sampler.run_mcmc(m_g + 0.1 * sd_g * rng.standard_normal((16, 1)), 2000, progress=False)
ex = sampler.get_chain(discard=300, flat=True)[:, 0]
m_em, sd_em = ex.mean(), ex.std()

# ------------------------------------------------------------ 5. coverage over 250 mock maps
cover, zsc, sds = {}, {}, {}
for tv in (0.0, 0.05):
    hits, zz, ss = [], [], []
    for s in mseeds:
        S = vec("grid_", keys)[s, eg.index(tv)][ok]
        m, sd, lo, hi = summary_post(grid_post(S))
        hits.append(lo <= tv <= hi)
        zz.append((m - tv) / sd)
        ss.append(sd)
    cover[tv], zsc[tv], sds[tv] = np.mean(hits), np.array(zz), np.array(ss)

# variance of the compressed statistic t = d^T C^-1 S at eps = 0.05 relative to eps = 0
wv = Ci @ d_crn
t_of = {tv: vec("grid_", keys)[:, eg.index(tv)][:, ok] @ wv for tv in (0.0, 0.05)}
var_t = {tv: np.var(t) for tv, t in t_of.items()}
# its uncertainty: bootstrap over the 500 seeds (both eps of a seed resampled together)
rb = rng_for("ch12", "c05_boot")
boot = [np.var(t_of[0.05][ix]) / np.var(t_of[0.0][ix]) for ix in rb.integers(0, len(t_of[0.0]), (1000, len(t_of[0.0])))]
var_t_se = np.std(boot)
k_w = np.sqrt(var_t[0.05] / var_t[0.0])                   # width factor and the coverage it implies
from scipy.stats import norm
cov_k = 2 * norm.cdf(1 / k_w) - 1

setup(9.0, 3.0)
fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
ax[0].plot(np.arange(len(chain)), chain[:, 0], color=SERIES[0], lw=0.4)
ax[0].axhline(TRUE, color="k", ls="--", lw=1)
ax[0].set_xlabel("step")
ax[0].set_ylabel(r"$\epsilon$")
ax[0].set_title(f"Metropolis chain, acceptance {acc:.2f}", fontsize=9)
ax[1].hist(xs, bins=50, density=True, color=SERIES[0], alpha=0.5, label="Metropolis")
ax[1].hist(ex, bins=50, density=True, histtype="step", color=SERIES[1], lw=1.2, label="emcee")
de = egrid[1] - egrid[0]
ax[1].plot(egrid, w_obs / de, color="k", lw=1, label="grid")
sF = 1 / np.sqrt(F_crn)
ax[1].plot(egrid, np.exp(-0.5 * ((egrid - m_g) / sF) ** 2) / (np.sqrt(2 * np.pi) * sF), ":",
           color=INK2, lw=1.2, label="Fisher width")
ax[1].axvline(TRUE, color="k", ls="--", lw=1)
ax[1].set_xlim(m_g - 5 * sd_g, m_g + 5 * sd_g)
ax[1].set_xlabel(r"$\epsilon$")
ax[1].legend(fontsize=6.5)
ax[1].set_title("posterior, one map", fontsize=9)
xx = np.linspace(-4, 4, 200)
for tv, c in ((0.05, SERIES[0]), (0.0, SERIES[2])):
    ax[2].hist(zsc[tv], bins=20, range=(-4, 4), density=True, histtype="step", color=c, lw=1.3,
               label=rf"$\epsilon_{{\rm true}}={tv}$, 68% covers {100 * cover[tv]:.0f}%")
ax[2].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "--", color="k", lw=1)
ax[2].set_xlabel("(posterior mean $-\\epsilon_{\\rm true}$)\n/ posterior sd")
ax[2].legend(fontsize=6.5, loc="upper left")
ax[2].set_title("250 mock maps", fontsize=9)
ax[2].set_ylim(0, 0.62)
fig.tight_layout()
savefig(fig, "ch12", "c_post")

f3 = lambda x: f"{x:.4f}"  # noqa: E731
save_numbers("ch12", "c05_fisher_post", {
    "CfSigPk": f"{sig['pk'][0]:.1f}", "CfBiasPk": f"{sig['pk'][1]:.2f}",
    "CfSigSkew": f3(sig["skew"][0]), "CfSigArea": f3(sig["area"][0]), "CfSigChi": f3(sig["chi"][0]),
    "CfSigBetti": f3(sig["betti"][0]), "CfSigPd": f3(sig["pd"][0]),
    "CfSigBettiA": f"{sig['bettiA'][0]:.3f}", "CfSigPdA": f"{sig['pdA'][0]:.3f}",
    "CfPdAOverBettiA": f"{sig['pdA'][0] / sig['bettiA'][0]:.2f}",
    "CfSigBettiSkew": f3(sig["betti+skew"][0]), "CfSigBettiPk": f3(sig["betti+pk"][0]),
    "CfBiasBetti": f"{100 * sig['betti'][1]:.1f}", "CfBiasPd": f"{100 * sig['pd'][1]:.1f}",
    "CfBiasPdA": f"{100 * sig['pdA'][1]:.1f}",
    "CfStepA": f3(steps[0.025]), "CfStepB": f3(steps[0.05]), "CfStepC": f3(steps[0.1]),
    "CfSigInd": f3(1 / np.sqrt(F_ind)), "CfBiasInd": f"{100 * b_ind / F_ind:.0f}",
    "CfBiasCrn": f"{100 * b_crn / F_crn:.1f}",
    "CfP": p_betti, "CfBiasEst": f"{200 * p_betti / 250:.0f}",   # the bias estimate of section 5 in the text
    "CfFbetti": f"{F_crn:.0f}", "CfBiasEstPct": f"{100 * (200 * p_betti / 250) / F_crn:.1f}",
    "CfVarRatio": f"{var_ind / var_crn:.0f}", "CfRho": f"{rho:.3f}", "CfRhoFactor": f"{1 / (1 - rho):.0f}",
    "CfConvA": f3(conv[0]), "CfConvB": f3(conv[1]), "CfConvC": f3(conv[2]), "CfConvD": f3(conv[3]),
    "CfConvE": f3(conv[4]),
    "CfFitRes": f"{fit_res:.2f}",
    "CfTrue": TRUE, "CfGridMean": f3(m_g), "CfGridSd": f3(sd_g), "CfGridLo": f3(lo_g), "CfGridHi": f3(hi_g),
    "CfMhMean": f3(m_mh), "CfMhSd": f3(sd_mh), "CfMhAcc": f"{acc:.2f}", "CfMhTau": f"{tau:.1f}", "CfMhNeff": f"{100 * round(len(xs) / tau / 100):.0f}",
    "CfEmMean": f3(m_em), "CfEmSd": f3(sd_em),
    "CfMock": int(MOCK), "CfVarT": f"{var_t[0.05] / var_t[0.0]:.2f}",
    "CfCoverFive": f"{100 * cover[0.05]:.0f}", "CfCoverZero": f"{100 * cover[0.0]:.0f}",
    "CfZsdFive": f"{zsc[0.05].std():.2f}", "CfZmeanFive": f"{zsc[0.05].mean():.2f}",
    "CfZsdSe": f"{zsc[0.05].std() / np.sqrt(2 * len(mseeds)):.2f}",
    "CfZmeanSe": f"{zsc[0.05].std() / np.sqrt(len(mseeds)):.2f}",
    "CfVarTSe": f"{var_t_se:.2f}", "CfK": f"{k_w:.3f}", "CfKinv": f"{1 / k_w:.3f}",
    "CfPhiK": f"{norm.cdf(1 / k_w):.3f}", "CfCovK": f"{cov_k:.3f}", "CfCovKPct": f"{100 * cov_k:.0f}",
    "CfKPct": f"{100 * (k_w - 1):.0f}",
    "CfSdMedFive": f3(np.median(sds[0.05])), "CfSdMedZero": f3(np.median(sds[0.0])),
})
print({k: v for k, v in sig.items()})
print("steps", steps, "crn", 1 / np.sqrt(F_crn), b_crn / F_crn, "ind", 1 / np.sqrt(F_ind), b_ind / F_ind,
      "var ratio", var_ind / var_crn, rho)
print("conv", conv, "fit res", fit_res)
print("grid", m_g, sd_g, lo_g, hi_g, "mh", m_mh, sd_mh, acc, tau, "emcee", m_em, sd_em)
print("coverage", cover, {k: (v.mean(), v.std()) for k, v in zsc.items()})
