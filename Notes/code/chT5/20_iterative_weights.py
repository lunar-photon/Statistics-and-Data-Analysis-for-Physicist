"""20_iterative_weights.py -- weights that need the parameter they are used to estimate.

Question: Poisson counts scatter by the square root of their own mean, so the best
weights for combining them depend on the unknown signal.  What happens if we guess
the signal, weight, re-estimate and repeat?  Where does the repetition settle, how
fast, and how does that answer compare with the usual shortcuts?

  A. Two detectors see the same source of strength s on backgrounds b = (30, 2):
     n = (44, 10).  Fixed-point iteration s_{k+1} = g(s_k) with inverse-variance
     weights 1/(s_k + b_i); the fixed point against the root of the Poisson score;
     the contraction factor g'(s_hat) from the formula against the ratio of errors.
  B. A binned fit: a Gaussian peak of known shape and unknown total theta on a known
     flat background, 20 bins, at several exposures.  Over many toy data sets:
       Neyman chi^2   (weights 1/y_i, fixed by the data),
       Pearson chi^2  (weights 1/lambda_i(theta), minimised directly),
       iterated reweighting (weights 1/lambda_i(theta_k), re-solved) = maximum likelihood.
     Bias in units of the statistical error, and the contraction factor of the iteration.

Writes: figures/chT5/iter_weights.pdf, results/chT5/20_iterative_weights.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

rng = rng_for("chT5", "20_iterative_weights")
out = {}
setup()

# ---------------------------------------------------------------- A. two detectors
b = np.array([30.0, 2.0])                 # known backgrounds
n = np.array([44.0, 10.0])                # observed counts
x = n - b                                 # background-subtracted readings (14, 8)


def g(s):                                 # one reweighting step: weights 1/(s + b_i)
    w = 1.0 / (s + b)
    return np.sum(w * x) / np.sum(w)


s_hist = [x.mean()]                       # start from the equal-weight average
for _ in range(14):
    s_hist.append(g(s_hist[-1]))
s_hist = np.array(s_hist)

score = lambda s: np.sum(n / (s + b) - 1.0)          # Poisson score d lnL / ds
s_ml = brentq(score, 0.1, 40.0)
assert abs(s_hist[-1] - s_ml) < 1e-10, "fixed point is not the ML root"

lam = s_ml + b
gprime = -np.sum((n - lam) / lam**2) / np.sum(1.0 / lam)   # derived in the text
err = np.abs(s_hist - s_ml)
meas_rate = err[2] / err[1]
assert abs(meas_rate - abs(gprime)) < 0.01 * abs(gprime) + 1e-3, (meas_rate, gprime)

w_ney = 1.0 / n                           # weights from the observed counts
s_ney = np.sum(w_ney * x) / np.sum(w_ney)
var_ml = 1.0 / np.sum(1.0 / lam)          # variance of the inverse-variance mean
var_eq = np.sum(lam) / 4.0                # variance of the plain average (x1 + x2)/2

out.update(TFiveBTwoSzero=s_hist[0], TFiveBTwoSone=s_hist[1], TFiveBTwoStwo=s_hist[2],
           TFiveBTwoSthree=s_hist[3], TFiveBTwoShat=s_ml, TFiveBTwoGprime=gprime,
           TFiveBTwoRate=meas_rate, TFiveBTwoSney=s_ney,
           TFiveBTwoSigML=np.sqrt(var_ml), TFiveBTwoSigEq=np.sqrt(var_eq),
           TFiveBTwoVarRatio=var_ml / var_eq)
print("two detectors: iterates", np.round(s_hist[:5], 5), "ML", s_ml, "g'", gprime)

# ---------------------------------------------------------------- B. binned fit, many toys
nb = 20
edges = np.linspace(0.0, 1.0, nb + 1)
cen = 0.5 * (edges[1:] + edges[:-1])
shape = np.exp(-0.5 * ((cen - 0.5) / 0.1) ** 2)
shape /= shape.sum()                      # signal shape s_i, sums to one
exposures = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32])
S0, B0 = 20.0, 100.0                      # signal and background totals at exposure 1
M = 20000                                 # toys per exposure


def solve_decreasing(h, lo, hi, iters=80):
    """vectorised bisection for the root of a function decreasing in theta"""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = h(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


rows = []
for E in exposures:
    theta0, beta = S0 * E, np.full(nb, B0 * E / nb)
    y = rng.poisson(theta0 * shape + beta, size=(M, nb)).astype(float)
    lo = np.full(M, -0.999 * np.min(beta / shape))     # keeps every lambda_i > 0
    hi = np.full(M, 50.0 * theta0 + 50.0)
    # Neyman: weights fixed by the data (empty bins get variance 1)
    v = np.maximum(y, 1.0)
    th_ney = np.sum(shape * (y - beta) / v, 1) / np.sum(shape**2 / v, 1)
    # Pearson: minimise sum (y - lam)^2 / lam directly; stationarity sum s (y^2/lam^2 - 1) = 0
    th_pea = solve_decreasing(lambda t: np.sum(shape * (y**2 / (t[:, None] * shape + beta) ** 2 - 1), 1), lo, hi)
    # iterated reweighting: weights 1/lambda(theta_k), solve the weighted fit, repeat
    th = th_ney.copy()
    for k in range(60):
        lam_k = np.maximum(th[:, None] * shape + beta, 1e-9)
        th = np.sum(shape * (y - beta) / lam_k, 1) / np.sum(shape**2 / lam_k, 1)
    # maximum likelihood directly: Poisson score sum s (y/lam - 1) = 0
    th_ml = solve_decreasing(lambda t: np.sum(shape * (y / (t[:, None] * shape + beta) - 1), 1), lo, hi)
    # contraction factor at the fixed point, g' = -sum s^2 (y - lam)/lam^2 / sum s^2/lam
    lam = th_ml[:, None] * shape + beta
    gp = -np.sum(shape**2 * (y - lam) / lam**2, 1) / np.sum(shape**2 / lam, 1)
    conv = np.abs(th - th_ml) < 1e-6 * (1 + theta0)
    print(f"   E={E}: {np.sum(~conv)} of {M} toys not converged after 60 steps; "
          f"their |g'| min {np.min(np.abs(gp[~conv])) if np.any(~conv) else np.nan:.3f}; "
          f"max |g'| of converged {np.max(np.abs(gp[conv])):.3f}")
    # wherever the contraction factor is safely below one, the iteration must land on the ML root
    assert np.all(conv[np.abs(gp) < 0.7]), "IRLS fixed point != ML although |g'| < 0.7"
    sig = th_ml.std()
    rows.append(dict(E=E, counts=B0 * E / nb, sig=sig, nonconv=np.mean(~conv),
                     ney=(th_ney.mean() - theta0) / sig, pea=(th_pea.mean() - theta0) / sig,
                     ml=(th_ml.mean() - theta0) / sig, mcerr=1 / np.sqrt(M),
                     gp_med=np.median(np.abs(gp)), gp_90=np.quantile(np.abs(gp), 0.9)))
    print(f"E={E:5}: bg/bin {B0*E/nb:6.2f}  bias/sigma: Neyman {rows[-1]['ney']:+.3f} "
          f"Pearson {rows[-1]['pea']:+.3f}  ML {rows[-1]['ml']:+.3f}  |g'| median {rows[-1]['gp_med']:.3f}")

r1 = [r for r in rows if r["E"] == 1][0]
r16 = [r for r in rows if r["E"] == 16][0]
out.update(TFiveBBinNbins=nb, TFiveBBinNtoys=M, TFiveBBinSig=S0, TFiveBBinBkg=B0,
           TFiveBBinBkgPerBin=B0 / nb,
           TFiveBBinNeyOne=r1["ney"], TFiveBBinPeaOne=r1["pea"], TFiveBBinMLOne=r1["ml"],
           TFiveBBinNeySixteen=r16["ney"], TFiveBBinPeaSixteen=r16["pea"], TFiveBBinMLSixteen=r16["ml"],
           TFiveBBinGpOne=r1["gp_med"], TFiveBBinGpNineOne=r1["gp_90"], TFiveBBinGpSixteen=r16["gp_med"],
           TFiveBBinMCerr=1 / np.sqrt(M),
           TFiveBBinCountsLow=rows[0]["counts"], TFiveBBinCountsHigh=rows[-1]["counts"],
           TFiveBBinNeyLow=rows[0]["ney"], TFiveBBinPeaLow=rows[0]["pea"], TFiveBBinMLLow=rows[0]["ml"],
           TFiveBBinNonconvLow=100 * rows[0]["nonconv"], TFiveBBinNonconvOne=100 * r1["nonconv"])
# ML must be the least biased at low counts, the two chi^2 shortcuts biased in opposite directions
assert r1["ney"] < 0 < r1["pea"] and abs(r1["ml"]) < abs(r1["ney"]) and abs(r1["ml"]) < abs(r1["pea"])

# one binned toy at exposure 1: the path of the iteration
E = 1.0
beta = np.full(nb, B0 * E / nb)
y1 = rng.poisson(S0 * E * shape + beta).astype(float)
v = np.maximum(y1, 1.0)
t = [np.sum(shape * (y1 - beta) / v) / np.sum(shape**2 / v)]
for _ in range(10):
    lam_k = np.maximum(t[-1] * shape + beta, 1e-9)
    t.append(np.sum(shape * (y1 - beta) / lam_k) / np.sum(shape**2 / lam_k))
t = np.array(t)
t_ml = brentq(lambda q: np.sum(shape * (y1 / (q * shape + beta) - 1)), -0.99 * np.min(beta / shape), 500)
lam1 = t_ml * shape + beta
gp1 = -np.sum(shape**2 * (y1 - lam1) / lam1**2) / np.sum(shape**2 / lam1)
out.update(TFiveBToyGprime=abs(gp1))

save_numbers("chT5", "20_iterative_weights", out)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
c = [r["counts"] for r in rows]
ax[0].axhline(0, color="0.5", lw=0.8)
ax[0].plot(c, [r["ney"] for r in rows], "o-", color=SERIES[0], label=r"Neyman $\chi^2$ (weights $1/y_i$)")
ax[0].plot(c, [r["pea"] for r in rows], "s-", color=SERIES[1], label=r"Pearson $\chi^2$ (weights $1/\lambda_i(\theta)$)")
ax[0].plot(c, [r["ml"] for r in rows], "^-", color=SERIES[2], label="re-solved weights = ML")
ax[0].set_xscale("log")
ax[0].set_xlabel("mean background counts per bin")
ax[0].set_ylabel(r"bias of $\hat\theta$ / statistical error")
ax[0].legend(loc="lower right")
ax[0].set_title("binned fit, %d toys per point" % M)

k = np.arange(len(s_hist))
ax[1].semilogy(k[:7], err[:7], "o-", color=SERIES[0], label="two detectors")
ax[1].semilogy(k[:7], abs(gprime) ** k[:7] * err[0], "--", color=SERIES[0], lw=1)
e1 = np.abs(t - t_ml)
ax[1].semilogy(np.arange(len(e1))[:7], e1[:7], "s-", color=SERIES[1], label="one binned toy")
ax[1].semilogy(np.arange(7), abs(gp1) ** np.arange(7) * e1[0], "--", color=SERIES[1], lw=1)
ax[1].set_ylim(1e-12, 10)
ax[1].set_xlabel("iteration $k$")
ax[1].set_ylabel(r"$|\theta_k-\hat\theta|$")
ax[1].legend(loc="upper right")
ax[1].set_title(r"dashed: $|g'(\hat\theta)|^k$ from the formula")
fig.tight_layout()
savefig(fig, "chT5", "iter_weights")
