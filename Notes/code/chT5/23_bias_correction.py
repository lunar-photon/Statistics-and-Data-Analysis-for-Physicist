"""23_bias_correction.py -- removing the 1/n bias of an estimator, and what it costs.

Question: the decay constant lambda_hat = 1/mean(t) from n lifetimes is biased by
lambda/(n-1).  The jackknife, the bootstrap and the iterated bootstrap each estimate that
bias from the data and subtract it.  How much bias does each leave, what happens to the
variance, and when does the bias matter at all?

  A. Exponential lifetimes, lambda = 1, n = 5 ... 40, many toy experiments:
     plain 1/tbar; jackknife n T - (n-1) mean(T_(i)); bootstrap 2T - E*T*;
     double bootstrap 3T - 3E*T* + E**T** (n = 5, 10).  Exact results to check against:
     E[1/tbar] = lambda n/(n-1), jackknife bias -lambda/((n-1)(n-2)).
  B. The price: the sample standard deviation of n Gaussian readings, plain against
     jackknife-corrected and against the exactly unbiased s/c4; bias, variance, MSE.
  C. Why the bias matters: the average of K independent estimates (n = 10 each);
     its RMS error against K for each estimator.

Writes: figures/chT5/bias_ladder.pdf, results/chT5/23_bias_correction.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

rng = rng_for("chT5", "23_bias_correction")
out = {}
setup()
words = {5: "Five", 10: "Ten", 20: "Twenty", 40: "Forty"}

# ---------------------------------------------------------------- A. decay constant
M, B1, B2 = 40000, 100, 25
chunk = 200
res = {}
for n in [5, 10, 20, 40]:
    plain = np.empty(M); jk = np.empty(M); boot = np.empty(M); dboot = np.full(M, np.nan)
    for c0 in range(0, M, chunk):
        t = rng.exponential(1.0, size=(chunk, n))
        S = t.sum(1)
        T = n / S
        loo = (n - 1) / (S[:, None] - t)                      # leave-one-out estimates T_(i)
        plain[c0:c0 + chunk] = T
        jk[c0:c0 + chunk] = n * T - (n - 1) * loo.mean(1)
        i1 = rng.integers(0, n, size=(chunk, B1, n))
        x1 = np.take_along_axis(t[:, None, :].repeat(B1, 1), i1, 2)   # first-level resamples
        T1 = 1.0 / x1.mean(2)
        m1 = T1.mean(1)                                        # estimate of E* T*
        boot[c0:c0 + chunk] = 2 * T - m1
        if n <= 20:
            i2 = rng.integers(0, n, size=(chunk, B1, B2, n))
            x2 = np.take_along_axis(x1[:, :, None, :].repeat(B2, 2), i2, 3)   # resamples of resamples
            m2 = (1.0 / x2.mean(3)).mean((1, 2))               # estimate of E* E** T**
            dboot[c0:c0 + chunk] = 3 * T - 3 * m1 + m2
    c = n / (n - 1.0)                     # parametric bootstrap: E* T* = c T exactly
    par1, par2, parinf = (2 - c) * plain, (3 - 3 * c + c * c) * plain, plain / c
    res[n] = dict(plain=plain, jk=jk, boot=boot, dboot=dboot, par1=par1, par2=par2, parinf=parinf)
    exact = dict(plain=c - 1, jk=-1.0 / ((n - 1) * (n - 2)), par1=-1.0 / (n - 1) ** 2,
                 par2=1.0 / (n - 1) ** 3, parinf=0.0)
    for k_, v in exact.items():
        se_ = res[n][k_].std() / np.sqrt(M)
        assert abs(res[n][k_].mean() - 1 - v) < 5 * se_, (n, k_, res[n][k_].mean() - 1, v)
    assert abs(boot.mean() - 1) < abs(plain.mean() - 1)
    iqr = lambda v: np.subtract(*np.nanquantile(v, [0.75, 0.25]))
    print(f"n={n:2d}: bias plain {plain.mean()-1:+.4f} (exact {c-1:+.4f}), jackknife {jk.mean()-1:+.4f} "
          f"(exact {exact['jk']:+.4f}), boot {boot.mean()-1:+.4f}, double boot {np.nanmean(dboot)-1:+.4f} "
          f"(median {np.nanmedian(dboot)-1:+.4f}, sd {np.nanstd(dboot):.3f}); plain sd {plain.std():.3f}; "
          f"IQR plain {iqr(plain):.3f} boot {iqr(boot):.3f} dboot {iqr(dboot):.3f}")
    w = words[n]
    for k_, tag in [("plain", "Plain"), ("jk", "JK"), ("boot", "Boot"), ("par1", "ParOne"),
                    ("par2", "ParTwo"), ("parinf", "ParInf")]:
        out["TFiveBLamBias" + tag + w] = res[n][k_].mean() - 1
        out["TFiveBLamSd" + tag + w] = res[n][k_].std()
        out["TFiveBLamMse" + tag + w] = np.mean((res[n][k_] - 1) ** 2)
    out["TFiveBLamBiasErr" + w] = boot.std() / np.sqrt(M)
    out["TFiveBLamBiasErrJK" + w] = jk.std() / np.sqrt(M)
    if n <= 20:
        out["TFiveBLamBiasDboot" + w] = np.nanmean(dboot) - 1
        out["TFiveBLamMedDboot" + w] = np.nanmedian(dboot) - 1
        out["TFiveBLamSdDboot" + w] = np.nanstd(dboot)
        out["TFiveBLamBiasErrD" + w] = np.nanstd(dboot) / np.sqrt(M)
        out["TFiveBLamIqrDboot" + w] = iqr(dboot)
        out["TFiveBLamIqrBoot" + w] = iqr(boot)
        out["TFiveBLamIqrPlain" + w] = iqr(plain)
        out["TFiveBLamFracBigDboot" + w] = 100 * np.mean(dboot > 3) - 0
        out["TFiveBLamFracBigPlain" + w] = 100 * np.mean(plain > 3)
out.update(TFiveBLamNtoys=M, TFiveBLamBone=B1, TFiveBLamBtwo=B2)

# ---------------------------------------------------------------- B. the price: sample standard deviation
def c4(n):
    return np.sqrt(2.0 / (n - 1)) * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2))


Ms = 200000
price = {}
for n in [5, 10, 20]:
    xg = rng.standard_normal((Ms, n))
    s = xg.std(1, ddof=1)
    tot, tot2 = xg.sum(1), (xg**2).sum(1)
    s_loo = np.sqrt(((tot2[:, None] - xg**2) - (tot[:, None] - xg) ** 2 / (n - 1)) / (n - 2))
    s_jk = n * s - (n - 1) * s_loo.mean(1)
    s_c4 = s / c4(n)
    st = {}
    for name, v in [("plain", s), ("jk", s_jk), ("cfour", s_c4)]:
        st[name] = dict(bias=v.mean() - 1, var=v.var(), mse=np.mean((v - 1) ** 2))
    price[n] = st
    print(f"s, n={n}: " + "; ".join(f"{k}: bias {d['bias']:+.4f} var {d['var']:.4f} mse {d['mse']:.4f}" for k, d in st.items()))
    w = words[n]
    for k_, d in st.items():
        tag = {"plain": "Plain", "jk": "JK", "cfour": "Cfour"}[k_]
        out["TFiveBSd" + tag + "Bias" + w] = d["bias"]
        out["TFiveBSd" + tag + "Var" + w] = d["var"]
        out["TFiveBSd" + tag + "Mse" + w] = d["mse"]
    # exact values for the plain s and s/c4
    assert abs(st["cfour"]["mse"] - (1 / c4(n) ** 2 - 1)) < 0.02 * (1 / c4(n) ** 2 - 1)
    assert abs(st["plain"]["mse"] - 2 * (1 - c4(n))) < 0.02 * 2 * (1 - c4(n))
out.update(TFiveBSdNtoys=Ms, TFiveBCfourFive=c4(5), TFiveBCfourTen=c4(10))
assert price[5]["jk"]["var"] > price[5]["plain"]["var"] and price[5]["cfour"]["mse"] > price[5]["plain"]["mse"]

# ---------------------------------------------------------------- C. averaging K estimates (n = 10)
n = 10
r = res[n]
Ks = np.unique(np.round(np.logspace(0, 3, 40)).astype(int))
curves = {}
for name in ["plain", "jk", "boot", "dboot"]:
    v = r[name]
    b, var = v.mean() - 1, v.var()
    curves[name] = np.sqrt(var / Ks + b**2)
# direct check at K = 100: average groups of 100 toys
K = 100
chk = {name: np.sqrt(np.mean((r[name].reshape(-1, K).mean(1) - 1) ** 2)) for name in ["plain", "jk", "boot", "dboot"]}
b_plain, v_plain = r["plain"].mean() - 1, r["plain"].var()
out.update(TFiveBAvgKcross=v_plain / b_plain**2, TFiveBAvgRmsPlainHundred=chk["plain"],
           TFiveBAvgRmsJKHundred=chk["jk"], TFiveBAvgRmsBootHundred=chk["boot"],
           TFiveBAvgRmsDbootHundred=chk["dboot"], TFiveBAvgK=K)
print("K=100 RMS of the average:", {k: round(v, 4) for k, v in chk.items()}, "crossover K", v_plain / b_plain**2)

save_numbers("chT5", "23_bias_correction", out)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9))
nn = np.linspace(4.5, 45, 200)
ax[0].axhline(0, color="0.5", lw=0.8)
ax[0].plot(nn, 1 / (nn - 1), color=SERIES[0], lw=1.2, label=r"exact: $\lambda/(n-1)$")
ax[0].plot(nn, -1 / ((nn - 1) * (nn - 2)), color=SERIES[1], lw=1.2, label=r"jackknife exact: $-\lambda/((n-1)(n-2))$")
ax[0].plot(nn, -1 / (nn - 1) ** 2, ":", color=SERIES[2], lw=1.4, label=r"parametric, one step: $-\lambda/(n-1)^2$")
ax[0].plot(nn, 1 / (nn - 1) ** 3, ":", color=SERIES[3], lw=1.4, label=r"parametric, two steps: $\lambda/(n-1)^3$")
ns = np.array([5, 10, 20, 40])
mk = dict(plain=("o", 0, "plain, simulated"), jk=("s", 1, "jackknife, simulated"), boot=("^", 2, "bootstrap, simulated"),
          dboot=("D", 3, "double bootstrap, simulated"))
for name, (m_, ci, lab) in mk.items():
    vals = [(n_, np.nanmean(res[n_][name]) - 1, np.nanstd(res[n_][name]) / np.sqrt(M)) for n_ in ns
            if not np.all(np.isnan(res[n_][name])) and not (name == "dboot" and n_ == 5)]
    vals = np.array(vals)
    ax[0].errorbar(vals[:, 0], vals[:, 1], yerr=vals[:, 2], fmt=m_, color=SERIES[ci], ms=5, capsize=2, label=lab)
ax[0].set_xscale("log")
ax[0].set_xticks([5, 10, 20, 40])
ax[0].set_xticklabels(["5", "10", "20", "40"])
ax[0].xaxis.set_minor_formatter(plt.NullFormatter())
ax[0].set_xlabel("number of lifetimes $n$")
ax[0].set_ylabel(r"bias of the estimate of $\lambda$ (units of $\lambda$)")
ax[0].legend(fontsize=6.5, loc="upper right")
ax[0].set_title("lines: exact or parametric results")
for name, (m_, ci, lab) in mk.items():
    ax[1].loglog(Ks, curves[name], color=SERIES[ci], label=lab)
    ax[1].plot([K], [chk[name]], m_, color=SERIES[ci], ms=5)
ax[1].set_xlabel("number $K$ of independent experiments averaged ($n=10$ each)")
ax[1].set_ylabel("RMS error of the average")
ax[1].legend(fontsize=8, loc="lower left")
fig.tight_layout()
savefig(fig, "chT5", "bias_ladder")
