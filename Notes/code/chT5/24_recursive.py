"""24_recursive.py -- estimators that update with each new reading.

Question: when readings arrive one at a time, can we update the estimate without keeping
the data, and get exactly the batch answer?  And when the quantity itself drifts, which
update rule tracks it best?

  A. Running mean and Welford's running variance on readings of a 1 GHz frequency
     counter with 1 Hz jitter, against the textbook one-pass formula
     (sum x^2 - (sum x)^2/N)/(N-1), which cancels catastrophically, and the two-pass result.
  B. Recursive least squares for a straight-line calibration, one point at a time,
     against the batch normal equations.
  C. A detector gain drifting as a random walk, read through noisy calibration pulses.
     The Kalman filter against the running mean (all data so far) and trailing windows of
     every width; its steady state against the formula; the filter with no drift against
     the running mean; the filter with a wrongly assumed drift rate.

Writes: figures/chT5/kalman_track.pdf, figures/chT5/kalman_rms.pdf, results/chT5/24_recursive.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import math
import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("chT5", "24_recursive")
out = {}
setup()

# ---------------------------------------------------------------- A. running mean and variance
N = 100_000
x = 1.0e9 + rng.standard_normal(N)           # Hz
mean_run, M2 = 0.0, 0.0
for k, xk in enumerate(x, start=1):          # Welford: one pass, one reading at a time
    d = xk - mean_run
    mean_run += d / k
    M2 += d * (xk - mean_run)
var_welford = M2 / (N - 1)
s1, s2 = 0.0, 0.0
for xk in x:                                 # the textbook one-pass formula, also one reading at a time
    s1 += xk
    s2 += xk * xk
var_naive = (s2 - s1 * s1 / N) / (N - 1)
xbar = math.fsum(x) / N
var_two = math.fsum((x - xbar) ** 2) / (N - 1)   # two passes, accurately summed: the reference
err_w, err_n = abs(var_welford / var_two - 1), abs(var_naive / var_two - 1)
print(f"A diagnostics: |mean_run - xbar| = {abs(mean_run - xbar):.3e} Hz, "
      f"relative variance error: Welford {err_w:.3e}, naive {err_n:.3e}")
assert err_w < 1e-3 * err_n, "Welford must beat the naive formula by orders of magnitude"
out.update(TFiveBWelN=N, TFiveBWelVar=f"{var_welford:.7f}", TFiveBWelTwo=f"{var_two:.7f}", TFiveBWelNaive=int(round(var_naive)),
           TFiveBWelMeanDiff=abs(mean_run - xbar), TFiveBWelErr=err_w, TFiveBWelErrNaive=int(round(err_n)))
print(f"A: Welford {var_welford:.9f}, two-pass {var_two:.9f}, naive one-pass {var_naive:.6g}")

# ---------------------------------------------------------------- B. recursive least squares
Nl = 200
xs = np.sort(rng.uniform(0, 10, Nl))
ys = 0.5 + 2.0 * xs + 0.3 * rng.standard_normal(Nl)
A = np.column_stack([np.ones(Nl), xs])
theta = np.linalg.solve(A[:2].T @ A[:2], A[:2].T @ ys[:2])     # start: exact fit to the first two points
P = np.linalg.inv(A[:2].T @ A[:2])
maxdiff = 0.0
for i in range(2, Nl):
    a = A[i]
    Pa = P @ a
    k = Pa / (1.0 + a @ Pa)                  # gain vector
    theta = theta + k * (ys[i] - a @ theta)  # correct by the prediction error
    P = P - np.outer(k, Pa)                  # Sherman-Morrison update of (A^T A)^{-1}
    batch = np.linalg.solve(A[:i + 1].T @ A[:i + 1], A[:i + 1].T @ ys[:i + 1])
    maxdiff = max(maxdiff, np.max(np.abs(theta - batch)))
assert maxdiff < 1e-10
out.update(TFiveBRlsN=Nl, TFiveBRlsMaxDiff=maxdiff, TFiveBRlsA=theta[0], TFiveBRlsB=theta[1])
print("B: RLS vs batch, max |difference| over the stream:", maxdiff)

# ---------------------------------------------------------------- C. drifting gain
T, R = 5000, 200
sq, sr = 0.002, 0.05                         # drift per reading, noise per reading
q, r = sq**2, sr**2
m0, P0 = 1.0, 0.1**2                         # nominal gain and a vague prior
burn = 500


def kalman(y, q, r, m0, P0):
    m, Pk = m0, P0
    est, var = np.empty(len(y)), np.empty(len(y))
    for t, yt in enumerate(y):
        Pk = Pk + q                          # predict: the gain may have wandered
        K = Pk / (Pk + r)                    # update: weigh prediction against reading
        m = m + K * (yt - m)
        Pk = Pk * r / (Pk + r)          # = (1 - K) P, written without cancellation
        est[t], var[t] = m, Pk
    return est, var


def kalman_vec(Y, q, r, m0, P0):             # same recursion, all streams at once
    m = np.full(Y.shape[0], m0)
    Pk = P0
    est = np.empty_like(Y)
    for t in range(Y.shape[1]):
        Pk = Pk + q
        K = Pk / (Pk + r)
        m = m + K * (Y[:, t] - m)
        Pk = Pk * r / (Pk + r)          # = (1 - K) P, written without cancellation
        est[:, t] = m
    return est, Pk


G = 1.0 + np.cumsum(sq * rng.standard_normal((R, T)), 1)        # true gains
Y = G + sr * rng.standard_normal((R, T))                         # calibration readings

# no drift + flat prior = running mean, exactly
est0, _ = kalman(Y[0], 0.0, r, 0.0, 1e12)
runmean = np.cumsum(Y[0]) / np.arange(1, T + 1)
assert np.max(np.abs(est0 - runmean)) < 1e-8, np.max(np.abs(est0 - runmean))
out.update(TFiveBKfRunDiff=np.max(np.abs(est0 - runmean)))

estK, Pend = kalman_vec(Y, q, r, m0, P0)
rms = lambda E: np.sqrt(np.mean((E[:, burn:] - G[:, burn:]) ** 2))
rmsK = rms(estK)
Mss = (q + np.sqrt(q**2 + 4 * q * r)) / 2                         # steady prior variance
Kss = Mss / (Mss + r)
Pss = Mss * r / (Mss + r)
assert abs(Pend - Pss) < 1e-6 * Pss
cs = np.cumsum(Y, 1)
run_all = cs / np.arange(1, T + 1)
rmsRun = rms(run_all)

Ws = np.unique(np.round(np.logspace(np.log10(2), np.log10(2000), 60)).astype(int))
rmsW = []
cs0 = np.concatenate([np.zeros((R, 1)), cs], 1)
for W in Ws:
    idx = np.arange(T)
    lo = np.maximum(idx + 1 - W, 0)
    win = (cs0[:, idx + 1] - cs0[:, lo]) / (idx + 1 - lo)
    rmsW.append(rms(win))
rmsW = np.array(rmsW)
Wbest = Ws[np.argmin(rmsW)]
idx = np.arange(T)
lo = np.maximum(idx + 1 - Wbest, 0)
winbest = (cs0[:, idx + 1] - cs0[:, lo]) / (idx + 1 - lo)
per_stream_K = np.sqrt(np.mean((estK[:, burn:] - G[:, burn:]) ** 2, 1))
per_stream_W = np.sqrt(np.mean((winbest[:, burn:] - G[:, burn:]) ** 2, 1))
win_frac = np.mean(per_stream_K < per_stream_W)
Wtheory = np.sqrt(3 * r / q)
rmsWtheory = np.sqrt(2 * np.sqrt(r * q / 3))
assert rmsK < rmsW.min() < rmsRun
out.update(TFiveBKfT=T, TFiveBKfR=R, TFiveBKfSq=sq, TFiveBKfSr=sr, TFiveBKfBurn=burn,
           TFiveBKfRms=rmsK, TFiveBKfPss=np.sqrt(Pss), TFiveBKfKss=Kss, TFiveBKfMem=1 / Kss,
           TFiveBKfRmsRun=rmsRun, TFiveBKfWbest=int(Wbest), TFiveBKfRmsWbest=rmsW.min(),
           TFiveBKfWtheory=Wtheory, TFiveBKfRmsWtheory=rmsWtheory, TFiveBKfWinFrac=100 * win_frac,
           TFiveBKfGain=100 * (1 - rmsK / rmsW.min()))
print(f"C: Kalman rms {rmsK:.5f} (steady sqrt P {np.sqrt(Pss):.5f}), best window W={Wbest} rms {rmsW.min():.5f} "
      f"(theory W={Wtheory:.1f}, {rmsWtheory:.5f}), running mean {rmsRun:.4f}; Kalman wins on {100*win_frac:.1f}% of streams")

ratios = np.logspace(-2, 2, 17)
rmsQ = np.array([rms(kalman_vec(Y, q * f, r, m0, P0)[0]) for f in ratios])
for f, v in zip(ratios[::4], rmsQ[::4]):
    print(f"   assumed q / true q = {f:7.2f}: rms {v:.5f}")
out.update(TFiveBKfRmsQtenth=rmsQ[np.argmin(abs(ratios - 0.1))], TFiveBKfRmsQten=rmsQ[np.argmin(abs(ratios - 10))],
           TFiveBKfRmsQhundredth=rmsQ[0], TFiveBKfRmsQhundred=rmsQ[-1])

save_numbers("chT5", "24_recursive", out)

# ---------------------------------------------------------------- figures
s = 0
tt = np.arange(T)
estS, varS = kalman(Y[s], q, r, m0, P0)
fig, ax = plt.subplots(figsize=(9.4, 3.8))
sl = slice(0, 2000)
ax.plot(tt[sl], Y[s, sl], ".", color="0.75", ms=1.5, label="calibration readings")
ax.plot(tt[sl], G[s, sl], color="k", lw=1.4, label="true gain")
ax.plot(tt[sl], run_all[s, sl], color=SERIES[2], lw=1.2, label="running mean of all readings")
ax.plot(tt[sl], winbest[s, sl], color=SERIES[1], lw=1.0, label="trailing window, $W=%d$" % Wbest)
ax.plot(tt[sl], estS[sl], color=SERIES[0], lw=1.4, label="Kalman filter")
ax.fill_between(tt[sl], estS[sl] - np.sqrt(varS[sl]), estS[sl] + np.sqrt(varS[sl]), color=SERIES[0], alpha=0.25, lw=0)
lo_, hi_ = G[s, sl].min(), G[s, sl].max()
ax.set_ylim(lo_ - 0.12, hi_ + 0.12)
ax.set_xlabel("reading number $t$")
ax.set_ylabel("gain")
ax.legend(fontsize=8, ncol=3, loc="upper center")
fig.tight_layout()
savefig(fig, "chT5", "kalman_track")

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
ax[0].loglog(Ws, rmsW, color=SERIES[1], label="trailing window of width $W$")
ax[0].axhline(rmsK, color=SERIES[0], label="Kalman filter")
ax[0].axhline(rmsRun, color=SERIES[2], label="running mean of all readings")
ax[0].axvline(1 / Kss, color="0.5", ls=":", lw=1)
ax[0].set_xlabel("window width $W$ (readings)")
ax[0].set_ylabel("RMS error of the gain")
ax[0].legend(fontsize=8)
ax[1].semilogx(ratios, rmsQ, "o-", color=SERIES[0], ms=4)
ax[1].axhline(rmsW.min(), color=SERIES[1], ls="--", lw=1, label="best trailing window")
ax[1].axvline(1.0, color="0.5", ls=":", lw=1)
ax[1].set_xlabel("assumed drift variance / true drift variance")
ax[1].set_ylabel("RMS error of the Kalman filter")
ax[1].legend(fontsize=8)
fig.tight_layout()
savefig(fig, "chT5", "kalman_rms")
