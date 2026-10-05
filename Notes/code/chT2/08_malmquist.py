"""What does a flux limit do to standard candles, and how do we undo it?

Question: a survey only sees objects brighter than a limiting magnitude. (1) For candles with
a Gaussian spread of absolute magnitudes spread uniformly through Euclidean space, what is
the mean absolute magnitude of the objects we detect? (Malmquist: M0 - 1.382 sigma^2.)
(2) For a supernova survey with a magnitude limit, how biased are the Hubble residuals near
the limit, how biased is the fitted Omega_m, and does a likelihood that includes the
selection (a truncated Gaussian) remove the bias?
Computes:
  * Monte Carlo of the classical Malmquist bias for several sigma, against -0.6 ln10 sigma^2,
  * mock magnitude-limited SN surveys: binned residuals against the truncated-normal mean,
  * Omega_m from a naive chi^2 and from the selection-aware likelihood over many surveys.
Writes: figures/chT2/malmquist.pdf, results/chT2/08_malmquist.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

C_KMS = 299792.458
ZG = np.linspace(0, 1.3, 1301)


def mu_model(z, om, h0=70.0):
    E = np.sqrt(om * (1 + ZG) ** 3 + 1 - om)
    dc = np.concatenate([[0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(ZG))])
    return 5 * np.log10((1 + z) * C_KMS / h0 * np.interp(z, ZG, dc)) + 25


rng = rng_for("chT2", "08_malmquist")

# ---------- (1) classical Malmquist bias in a static Euclidean space
M0, MLIM = -19.25, 19.0
sigmas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
mal_mc, mal_err = [], []
for s in sigmas:
    n = 400_000
    r = 10 ** (np.log10(1500.0) + np.log10(rng.uniform(0, 1, n)) / 3)   # uniform in a ball, Mpc
    M = rng.normal(M0, s, n)
    m = M + 5 * np.log10(r) + 25
    sel = m < MLIM
    mal_mc.append(M[sel].mean() - M0); mal_err.append(M[sel].std() / np.sqrt(sel.sum()))
mal_mc = np.array(mal_mc)
mal_th = -0.6 * np.log(10) * sigmas**2

# ---------- (2) a magnitude-limited supernova survey
OM, MB, SIG, LIM = 0.3, -19.25, 0.15, 24.0
# comoving volume element per dz ~ D_M^2 / E(z), rate per unit volume constant, time dilation 1/(1+z)
E = np.sqrt(OM * (1 + ZG) ** 3 + 1 - OM)
DM = np.concatenate([[0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(ZG))])
pz = DM**2 / E / (1 + ZG)
cdf = np.cumsum(pz); cdf /= cdf[-1]


def survey(rng, nlow=150, nhigh=1200, zmax=1.2):
    zl = rng.uniform(0.02, 0.1, nlow)                                  # complete low-z sample
    zh = np.interp(rng.uniform(np.interp(0.1, ZG, cdf), np.interp(zmax, ZG, cdf), nhigh), cdf, ZG)
    z = np.concatenate([zl, zh])
    m = MB + mu_model(z, OM) + rng.normal(0, SIG, z.size)
    lim = np.where(z < 0.1, np.inf, LIM)
    keep = m < lim
    return z[keep], m[keep], lim[keep]


def nll(p, z, m, lim, corrected):
    om, off = p
    if not 0.02 < om < 0.98:
        return 1e10
    mod = mu_model(z, om) + off
    t = (m - mod) / SIG
    out = 0.5 * np.sum(t**2)
    if corrected:
        fin = np.isfinite(lim)
        out += np.sum(norm.logcdf((lim[fin] - mod[fin]) / SIG))
    return out


def fit(z, m, lim, corrected):
    r = minimize(nll, x0=[0.3, MB], args=(z, m, lim, corrected), method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-6, maxiter=2000))
    return r.x


nmock = 200
res = np.array([[fit(*survey(rng), c)[0] for c in (False, True)] for _ in range(nmock)])

# one large survey for the residual plot
zb, mb, lb = survey(rng, nlow=600, nhigh=8000)
resid = mb - MB - mu_model(zb, OM)
edges = np.linspace(0.1, 1.0, 10)
cen = 0.5 * (edges[1:] + edges[:-1])
nbin = np.array([((zb >= a) & (zb < b)).sum() for a, b in zip(edges[:-1], edges[1:])])
ok = nbin >= 20                                        # keep bins that still contain supernovae
bmean = np.array([resid[(zb >= a) & (zb < b)].mean() if n >= 20 else np.nan
                  for a, b, n in zip(edges[:-1], edges[1:], nbin)])
berr = np.array([resid[(zb >= a) & (zb < b)].std() / np.sqrt(n) if n >= 20 else np.nan
                 for a, b, n in zip(edges[:-1], edges[1:], nbin)])
cen, bmean, berr = cen[ok], bmean[ok], berr[ok]
zz = np.linspace(0.1, 1.2, 300)
tt = (LIM - MB - mu_model(zz, OM)) / SIG
trunc_mean = -SIG * norm.pdf(tt) / norm.cdf(tt)
z_half = zz[np.argmin(np.abs(tt))]                     # where half of the SNe are lost

# ---------- figure
setup(7.2, 2.6)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax[0].errorbar(sigmas, mal_mc, yerr=mal_err, fmt="o", color=SERIES[0], ms=4, label="simulation")
ss = np.linspace(0, 0.55, 100)
theory_line(ax[0], ss, -0.6 * np.log(10) * ss**2, label=r"$-1.382\,\sigma_M^2$")
ax[0].set(xlabel=r"spread of absolute magnitudes $\sigma_M$", ylabel=r"$\langle M\rangle_{\rm det}-M_0$")
ax[0].legend(fontsize=7, frameon=False)
ax[1].scatter(zb[::8], resid[::8], s=1.5, color="0.6", lw=0)
ax[1].errorbar(cen, bmean, yerr=berr, fmt="o", ms=3.5, color=SERIES[1], label="binned mean")
theory_line(ax[1], zz, trunc_mean, label="truncated-normal mean")
ax[1].axhline(0, color="0.3", lw=0.6)
ax[1].set(xlabel="redshift $z$", ylabel="Hubble residual [mag]", ylim=(-0.6, 0.45))
ax[1].legend(fontsize=7, frameon=False, loc="lower left")
bins = np.linspace(0.1, 0.5, 41)
ax[2].hist(res[:, 0], bins=bins, color=SERIES[3], alpha=0.7, label="naive $\\chi^2$")
ax[2].hist(res[:, 1], bins=bins, color=SERIES[0], alpha=0.7, label="with selection")
ax[2].axvline(OM, color="k", ls="--", lw=1)
ax[2].set(xlabel=r"$\hat\Omega_{m0}$", ylabel="mock surveys"); ax[2].set_ylim(0, ax[2].get_ylim()[1] * 1.3)
ax[2].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
savefig(fig, "chT2", "malmquist")

save_numbers("chT2", "08_malmquist", {
    "TwmSigThree": f"{mal_mc[2]:.4f}", "TwmSigThreeTh": f"{mal_th[2]:.4f}",
    "TwmSigOne": f"{mal_mc[0]:.4f}", "TwmSigOneTh": f"{mal_th[0]:.4f}",
    "TwmSigFive": f"{mal_mc[4]:.3f}", "TwmSigFiveTh": f"{mal_th[4]:.3f}",
    "TwmLim": LIM, "TwmSig": SIG, "TwmNmock": nmock, "TwmOm": OM,
    "TwmNaiveMean": f"{res[:, 0].mean():.3f}", "TwmNaiveSd": f"{res[:, 0].std(ddof=1):.3f}",
    "TwmCorrMean": f"{res[:, 1].mean():.3f}", "TwmCorrSd": f"{res[:, 1].std(ddof=1):.3f}",
    "TwmZhalf": f"{z_half:.2f}", "TwmResLast": f"{bmean[-1]:.3f}", "TwmResLastTh": f"{np.interp(cen[-1], zz, trunc_mean):.3f}",
    "TwmNkept": f"{zb.size}", "TwmZlast": f"{cen[-1]:.2f}",
})
print("Malmquist MC", np.round(mal_mc, 4), "theory", np.round(mal_th, 4))
print(f"Om naive {res[:,0].mean():.3f}+-{res[:,0].std():.3f}, corrected {res[:,1].mean():.3f}+-{res[:,1].std():.3f}; z_half {z_half:.2f}")
