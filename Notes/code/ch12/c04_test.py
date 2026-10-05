"""c04_test.py -- is a map Gaussian?  A chi-square test of topological summaries with simulated p-values.

Question: the local model u = W * R_eps * [g + eps (g^2 - 1)] of lib_fnl has the same power
spectrum for every eps.  Given ONE map, can we reject "eps = 0" (a Gaussian field)?
Computes, from the Monte Carlo cache of c02_mc_fnl.py (256 x 256 maps):
  1. mean and covariance of the Betti-curve vector from the 1000 null maps, its correlation
     matrix, and the Hartlap factor;
  2. the distribution of chi^2 = (S - mean)^T C^-1 (S - mean) for 500 independent null maps,
     against chi^2_p and against the exact (Hotelling) law, for the Betti vector (p = its components that vary across maps,
     n = 1000) and for the persistence-diagram histogram (p ~ 98) with n = 1000 and n = 150;
  3. simulated p-values (rank of chi^2_obs among the 500 'check' chi^2 values) for the 500
     grid maps at eps = 0 (calibration; since all share one reference set, the check is a
     two-sample KS test of their chi^2 against the reference chi^2) and at eps != 0 (power);
     the tests drop rare-feature bins (fewer than 5% of null maps away from the median);
  4. the power of the omnibus chi^2 test and of the score (directional) test
     t = dmu^T C^-1 (S - mean) for each summary: power spectrum, skewness, area fraction,
     Euler characteristic, Betti curves, persistence histograms, and their area-fraction
     ('gaussianised') versions.  The derivative dmu/deps for the score test comes from grid
     seeds 0-249; the power is measured on seeds 250-499.
Writes: figures/ch12/c_cov.pdf, figures/ch12/c_chi2.pdf, figures/ch12/c_power.pdf,
        results/ch12/c04_test.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, DATA
from lib_fnl import NU

rng = rng_for("ch12", "c04_test")
z = np.load(DATA / "ch12" / "c_fnl_mc.npz")
eg = list(z["eps_grid"])
i0 = eg.index(0.0)

SUMMARIES = {                       # name -> (keys in the cache, label)
    "pk": (["pk"], "power spectrum"),
    "skew": (["skew"], "skewness"),
    "area": (["area"], "area fraction"),
    "chi": (["chi"], r"Euler char. $\chi$"),
    "betti": (["b0", "b1"], r"Betti curves"),
    "pd": (["pd0", "pd1"], "persistence hist."),
    "bettiA": (["b0A", "b1A"], "Betti, area thresholds"),
    "pdA": (["pd0A", "pd1A"], "pers., area thresholds"),
}


def vec(prefix, keys):
    return np.concatenate([z[prefix + k] for k in keys], axis=-1)


def chi2(X, mu, Cinv):
    D = X - mu
    return np.einsum("...i,ij,...j->...", D, Cinv, D)


def fit(N, hartlap=True, rare=0.0):
    """Mean, inverse covariance (Hartlap-corrected) and kept columns of a null sample N (n, p);
    rare > 0 also drops bins in which fewer than that fraction of the maps differ from the median."""
    ok = (N.std(0) > 1e-9) & ((N != np.median(N, 0)).mean(0) >= rare)   # drop bins that (almost) never change
    N = N[:, ok]
    n, p = N.shape
    C = np.atleast_2d(np.cov(N.T))
    Ci = np.linalg.inv(C)
    if hartlap:
        Ci = Ci * (n - p - 2) / (n - 1)
    return N.mean(0), Ci, ok, n, p


# ------------------------------------------------------------ 1. the Betti vector: mean, covariance
Nb = vec("null_", SUMMARIES["betti"][0])
mu_b, Ci_b, ok_b, n_b, p_b = fit(Nb)
C_b = np.cov(Nb[:, ok_b].T)
sd_b = np.sqrt(np.diag(C_b))
R_b = C_b / np.outer(sd_b, sd_b)
k1 = list(NU).index(1.0)
km1 = list(NU).index(-1.0)
# correlations quoted in the text: beta0(1) with beta0(1.5), beta0(1) with beta1(-1)
idx = np.flatnonzero(ok_b)                                # positions in the full 26-vector
pos = {j: i for i, j in enumerate(idx)}
r_adj = R_b[pos[k1], pos[k1 + 1]]
r_cross = R_b[pos[k1], pos[13 + km1]]
hart_b = (n_b - p_b - 2) / (n_b - 1)

setup(8.6, 3.3)
fig, ax = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.25, 1]))
for key, c, lab in (("b0", SERIES[0], r"$\beta_0$"), ("b1", SERIES[1], r"$\beta_1$")):
    m0, s0 = z["null_" + key].mean(0), z["null_" + key].std(0)
    m1 = z["grid_" + key][:, eg.index(0.1)].mean(0)
    ax[0].fill_between(NU, m0 - s0, m0 + s0, color=c, alpha=0.25, lw=0)
    ax[0].plot(NU, m0, "-", color=c, label=lab + r", $\epsilon=0$ (mean $\pm$ one map)")
    ax[0].plot(NU, m1, "--", color=c, label=lab + r", $\epsilon=0.1$ (mean)")
ax[0].set_xlabel(r"threshold $\nu$ (units of the map's $\sigma$)")
ax[0].set_ylabel("count on a $256^2$ map")
ax[0].legend(fontsize=6.5)
im = ax[1].imshow(R_b, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
nb0 = int(ok_b[:13].sum())
ax[1].axhline(nb0 - 0.5, color="k", lw=0.6)
ax[1].axvline(nb0 - 0.5, color="k", lw=0.6)
ax[1].set_xticks([nb0 / 2, nb0 + (p_b - nb0) / 2])
ax[1].set_xticklabels([r"$\beta_0(\nu)$", r"$\beta_1(\nu)$"])
ax[1].set_yticks([nb0 / 2, nb0 + (p_b - nb0) / 2])
ax[1].set_yticklabels([r"$\beta_0(\nu)$", r"$\beta_1(\nu)$"])
ax[1].grid(False)
ax[1].set_title("correlation matrix, 1000 null maps")
fig.colorbar(im, ax=ax[1], fraction=0.046)
fig.tight_layout()
savefig(fig, "ch12", "c_cov")

# ------------------------------------------------------------ 2. distribution of chi^2 for null maps
def hotelling_pdf(x, p, n):
    """Density of chi^2 = (x - mean)^T Chat^-1 (x - mean) (no Hartlap), x independent of the n sims:
    chi^2 = (1 + 1/n) (n - 1) p / (n - p) F(p, n - p)."""
    a = (1 + 1 / n) * (n - 1) * p / (n - p)
    return stats.f.pdf(x / a, p, n - p) / a


Kb = vec("check_", SUMMARIES["betti"][0])[:, ok_b]
X2b = chi2(Kb, mu_b, Ci_b / hart_b)                       # raw inverse (no Hartlap)
Np = vec("null_", SUMMARIES["pd"][0])
mu_p, Ci_p, ok_p, n_p, p_p = fit(Np, hartlap=False)
Kp = vec("check_", SUMMARIES["pd"][0])[:, ok_p]
X2p = chi2(Kp, mu_p, Ci_p)
sub = rng.choice(len(Np), 150, replace=False)              # only 150 simulations for the covariance
Nps = Np[sub][:, ok_p]
okk = Nps.std(0) > 1e-9
mu_s, Ci_s, _, n_s, p_s = fit(Nps[:, okk], hartlap=False)
X2s = chi2(Kp[:, okk], mu_s, Ci_s)
hart_s = (n_s - p_s - 2) / (n_s - 1)

setup(9.0, 3.0)
fig, ax = plt.subplots(1, 3)
for a, X2, p, n, t in ((ax[0], X2b, p_b, n_b, f"Betti curves, $p={p_b}$, $n={n_b}$"),
                       (ax[1], X2p, p_p, n_p, f"persistence hist., $p={p_p}$, $n={n_p}$"),
                       (ax[2], X2s, p_s, n_s, f"persistence hist., $p={p_s}$, $n={n_s}$")):
    hi = np.quantile(X2, 0.99)
    lo = min(stats.chi2.ppf(0.001, p), X2.min())
    a.hist(X2, bins=40, range=(lo, hi), density=True, color=SERIES[0], alpha=0.6,
           label="500 independent null maps")
    xx = np.linspace(lo, hi, 300)
    a.plot(xx, stats.chi2.pdf(xx, p), ":", color=INK2, label=r"$\chi^2_p$ (known covariance)")
    a.plot(xx, hotelling_pdf(xx, p, n), "--", color="k", label="estimated mean and covariance")
    a.set_title(t, fontsize=8)
    a.set_xlabel(r"$\chi^2$")
ax[0].legend(fontsize=6.5)
fig.tight_layout()
savefig(fig, "ch12", "c_chi2")


def mean_pred(p, n):
    return p * (1 + 1 / n) * (n - 1) / (n - p - 2)


# simulated p-value versus the chi^2_p shortcut, for the 150-simulation case: false-alarm rates
grid0_p = vec("grid_", SUMMARIES["pd"][0])[:, i0][:, ok_p][:, okk]
X2g = chi2(grid0_p, mu_s, Ci_s)
p_short = stats.chi2.sf(X2g, p_s)                          # pretend the covariance were exact
p_hart = stats.chi2.sf(X2g * hart_s, p_s)                  # Hartlap-rescaled, still chi^2_p
p_sim = (1 + (X2s[None, :] >= X2g[:, None]).sum(1)) / (1 + len(X2s))
fa_short, fa_hart, fa_sim = [(pv < 0.05).mean() for pv in (p_short, p_hart, p_sim)]


# ------------------------------------------------------------ 3-4. calibration and power
# The tests follow the rare-feature pitfall: a bin in which fewer than RARE of the 1000 null maps
# differ from the median count is dropped, so that one rare feature cannot dominate chi^2.
RARE = 0.05


def pvals_omnibus(name, eps_idx, seeds, values=False):
    keys = SUMMARIES[name][0]
    mu, Ci, ok, n, p = fit(vec("null_", keys), rare=RARE)
    ref = chi2(vec("check_", keys)[:, ok], mu, Ci)
    obs = chi2(vec("grid_", keys)[seeds, eps_idx][:, ok], mu, Ci)
    pv = (1 + (ref[None, :] >= obs[:, None]).sum(1)) / (1 + len(ref))
    return (pv, obs, ref) if values else pv


def pvals_score(name, eps_idx, seeds, dseeds):
    keys = SUMMARIES[name][0]
    mu, Ci, ok, n, p = fit(vec("null_", keys), rare=RARE)
    G = vec("grid_", keys)[dseeds][..., ok]
    d = ((G[:, eg.index(0.05)] - G[:, eg.index(-0.05)]) / 0.1).mean(0)
    w = Ci @ d                                             # the compression weights
    ref = (vec("check_", keys)[:, ok] - mu) @ w
    obs = (vec("grid_", keys)[seeds, eps_idx][:, ok] - mu) @ w
    # two-sided: how often is |t| under the null at least as large as observed
    return (1 + (np.abs(ref)[None, :] >= np.abs(obs)[:, None]).sum(1)) / (1 + len(ref))


dseeds = np.arange(250)
pseeds = np.arange(250, 500)
eps_list = [e for e in eg]
power = {}
for name in SUMMARIES:
    power[name] = {
        "omni": np.array([(pvals_omnibus(name, eg.index(e), pseeds) < 0.05).mean() for e in eps_list]),
        "score": np.array([(pvals_score(name, eg.index(e), pseeds, dseeds) < 0.05).mean() for e in eps_list]),
    }
calib, calib_obs, calib_ref = pvals_omnibus("betti", i0, np.arange(500), values=True)
# the 500 p-values share ONE reference set, so they are not independent draws: a one-sample KS test
# against the uniform would ignore the noise of that set.  Compare the two samples of chi^2 instead.
ks_calib = stats.ks_2samp(calib_obs, calib_ref).pvalue
# error of the 5% rate: binomial for the 500 maps plus the Beta scatter of the reference's 95th percentile
sd_five = np.sqrt(0.05 * 0.95 / 500 + 0.05 * 0.95 / 501)
z_five = ((calib < 0.05).mean() - 0.05) / sd_five
calib_score = pvals_score("betti", i0, pseeds, dseeds)

# the trimmed Betti vector of the tests: its length, Fisher error, and the 50% points of section 4's estimate
mu_t, Ci_t, ok_t, n_t, p_t = fit(Nb, rare=RARE)
Gt = vec("grid_", SUMMARIES["betti"][0])[dseeds][..., ok_t]
d_t = ((Gt[:, eg.index(0.05)] - Gt[:, eg.index(-0.05)]) / 0.1).mean(0)
sig_t = 1 / np.sqrt(d_t @ Ci_t @ d_t)
crit_t = stats.chi2.ppf(0.95, p_t)                         # 95th percentile of chi^2_p
omni_t = np.sqrt(crit_t - p_t)                             # eps sqrt(F) the omnibus test needs

setup(9.0, 3.2)
fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
ax[0].hist(calib, bins=10, range=(0, 1), density=True, color=SERIES[0], alpha=0.7,
           label=r"Betti $\chi^2$, 500 maps")
ax[0].hist(p_hart, bins=10, range=(0, 1), density=True, histtype="step", color=SERIES[1], lw=1.4,
           label=r"pers., $n=150$, Hartlap $+\ \chi^2_p$ table")
ax[0].hist(p_sim, bins=10, range=(0, 1), density=True, histtype="step", color=SERIES[2], lw=1.4,
           label=r"pers., $n=150$, simulated")
ax[0].axhline(1, color="k", ls="--", lw=1)
ax[0].set_xlabel(r"$p$-value of a Gaussian map")
ax[0].set_ylabel("density")
ax[0].legend(fontsize=6.5, loc="upper right")
ax[0].set_ylim(0, 4.5)
order = ["pk", "skew", "area", "chi", "betti", "pd", "bettiA", "pdA"]
cols = dict(zip(order, SERIES))
ee = np.array(eps_list)
for a, kind, t in ((ax[1], "omni", r"omnibus $\chi^2$ test"), (ax[2], "score", "score test (aimed at $\\epsilon$)")):
    for name in order:
        ls = "--" if name.endswith("A") else "-"
        a.plot(ee, power[name][kind], ls, marker="o", ms=2.5, color=cols[name], label=SUMMARIES[name][1])
    a.axhline(0.05, color=INK2, lw=0.6, ls=":")
    a.set_xlabel(r"$\epsilon$ of the observed map")
    a.set_title(t, fontsize=9)
    a.set_ylim(-0.02, 1.02)
ax[1].set_ylabel(r"power (fraction with $p<0.05$)")
ax[1].legend(fontsize=6, loc="upper center", ncol=1, framealpha=0.9, bbox_to_anchor=(0.5,0.93))
fig.tight_layout()
savefig(fig, "ch12", "c_power")


# ------------------------------------------------------------ 5. why the counts beat the one-point summaries
# response of single components to eps (in units of their own one-map scatter), and how much the
# components overlap: area fractions at neighbouring thresholds versus peak and lake counts in the tails
k25, k3, km25 = list(NU).index(2.5), list(NU).index(3.0), list(NU).index(-2.5)


def response(key, k):
    G = z["grid_" + key][..., k]
    return ((G[:, eg.index(0.05)] - G[:, eg.index(-0.05)]) / 0.1).mean() / z["null_" + key][:, k].std()


A, B0, B1, SK = z["null_area"], z["null_b0"], z["null_b1"], z["null_skew"][:, 0]
corr = lambda a, b: np.corrcoef(a, b)[0, 1]  # noqa: E731
# the null map with the largest Betti chi^2, and the component that made it large
iout = int(np.argmax(X2b))
zsc = (Kb[iout] - mu_b) / sd_b
jout = int(np.argmax(np.abs(zsc)))
labels_full = [rf"\beta_0({nu:g})" for nu in NU] + [rf"\beta_1({nu:g})" for nu in NU]
lab_out = labels_full[np.flatnonzero(ok_b)[jout]]

j05, j025, j10 = eg.index(0.05), eg.index(0.025), eg.index(0.1)
pw = lambda name, kind, j: f"{100 * power[name][kind][j]:.0f}"  # noqa: E731
save_numbers("ch12", "c04_test", {
    "CcNnull": n_b, "CcNcheck": len(Kb), "CcPb": p_b, "CcTailb": f"{100 * (X2b > stats.chi2.ppf(0.999, p_b)).mean():.1f}",
    "CcMaxb": f"{X2b.max():.0f}", "CcHartb": f"{hart_b:.3f}",
    "CcRadj": f"{r_adj:.2f}", "CcRcross": f"{r_cross:.2f}",
    "CcMeanb": f"{X2b.mean():.1f}", "CcMeanbTh": f"{mean_pred(p_b, n_b):.1f}",
    "CcPp": p_p, "CcMeanp": f"{X2p.mean():.1f}", "CcMeanpTh": f"{mean_pred(p_p, n_p):.1f}",
    "CcHartp": f"{(n_p - p_p - 2) / (n_p - 1):.3f}",
    "CcPs": p_s, "CcNs": n_s, "CcMeans": f"{X2s.mean():.0f}", "CcMeansTh": f"{mean_pred(p_s, n_s):.0f}",
    "CcHarts": f"{hart_s:.2f}",
    "CcNmPb": n_b - p_b, "CcNmPs": n_s - p_s,                  # the width estimate (12c:eq:width)
    "CcWidA": f"{2 / (n_s - p_s):.3f}", "CcWidB": f"{2 / p_s:.3f}", "CcWidSum": f"{2 / (n_s - p_s) + 2 / p_s:.3f}",
    "CcWidRatio": f"{np.sqrt((2 / (n_s - p_s) + 2 / p_s) / (2 / p_s)):.1f}",
    "CcFaShort": f"{100 * fa_short:.0f}", "CcFaHart": f"{100 * fa_hart:.0f}", "CcFaSim": f"{100 * fa_sim:.0f}",
    "CcFaHartX": f"{fa_hart / 0.05:.0f}",
    "CcCalibKS": f"{ks_calib:.2f}", "CcCalibFive": f"{100 * (calib < 0.05).mean():.1f}",
    "CcCalibFiveSd": f"{100 * sd_five:.1f}", "CcCalibZ": f"{abs(z_five):.1f}",
    "CcPt": p_t, "CcRare": f"{100 * RARE:.0f}", "CcSqrtTwoP": f"{np.sqrt(2 * p_t):.0f}",
    "CcCritT": f"{crit_t:.1f}", "CcShiftT": f"{crit_t - p_t:.1f}", "CcOmniT": f"{omni_t:.1f}",
    "CcOmniRatio": f"{omni_t / 1.96:.1f}", "CcSigT": f"{sig_t:.4f}",
    "CcEpsScore": f"{1.96 * sig_t:.3f}", "CcEpsOmni": f"{omni_t * sig_t:.3f}",
    "CcCalibScoreFive": f"{100 * (calib_score < 0.05).mean():.1f}",
    "CcPowPkOmni": pw("pk", "omni", j10), "CcPowPkScore": pw("pk", "score", j10),
    "CcPowSkewFive": pw("skew", "score", j05), "CcPowAreaFive": pw("area", "score", j05),
    "CcPowChiFive": pw("chi", "score", j05), "CcPowBettiFive": pw("betti", "score", j05),
    "CcPowPdFive": pw("pd", "score", j05),
    "CcPowBettiOmniFive": pw("betti", "omni", j05), "CcPowPdOmniFive": pw("pd", "omni", j05),
    "CcPowBettiOmniTen": pw("betti", "omni", j10), "CcPowBettiTwo": pw("betti", "score", j025),
    "CcPowBettiOmniTwo": pw("betti", "omni", j025),
    "CcPowBettiAFive": pw("bettiA", "score", j05), "CcPowBettiATen": pw("bettiA", "score", j10),
    "CcPowPdATen": pw("pdA", "score", j10), "CcPowBettiAOmniTen": pw("bettiA", "omni", j10),
    "CcRespSkew": f"{response('skew', 0):.0f}", "CcRespArea": f"{response('area', k25):.0f}",
    "CcRespBz": f"{response('b0', k25):.0f}", "CcRespBo": f"{response('b1', km25):.0f}",
    "CcCorrAreaTail": f"{corr(A[:, k25], A[:, k3]):.2f}", "CcCorrAreaTwo": f"{corr(A[:, k25], A[:, km25]):.2f}",
    "CcCorrAreaSkew": f"{corr(A[:, k25], SK):.2f}",
    "CcCorrBzTail": f"{corr(B0[:, k25], B0[:, k3]):.2f}", "CcCorrBzBo": f"{corr(B0[:, k25], B1[:, km25]):.2f}",
    "CcCorrBzSkew": f"{corr(B0[:, k25], SK):.2f}",
    "CcOutBin": lab_out, "CcOutVal": f"{Kb[iout][jout]:.0f}", "CcOutMean": f"{mu_b[jout]:.2f}",
    "CcOutSd": f"{sd_b[jout]:.2f}", "CcOutZ": f"{zsc[jout]:.0f}",
})
for name in order:
    print(f"{name:7s} omni", np.round(power[name]["omni"], 2), " score", np.round(power[name]["score"], 2))
print("chi2 means", X2b.mean(), mean_pred(p_b, n_b), X2p.mean(), mean_pred(p_p, n_p), X2s.mean(), mean_pred(p_s, n_s))
print("false alarms short/hart/sim", fa_short, fa_hart, fa_sim, "KS calib", ks_calib)
