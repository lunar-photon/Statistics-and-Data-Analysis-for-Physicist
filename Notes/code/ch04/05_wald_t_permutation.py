"""05_wald_t_permutation.py -- three workhorse tests: Wald, Student t, permutation.

Question: (i) the Wald test divides by an *estimated* standard error.  For small
n, is its type I error rate still alpha = 0.05?  Does the t-test fix it?
(ii) two calibration runs of a calorimeter measure the same reference peak.
Did the energy scale shift between runs?  Compare the Wald test, Welch's t-test
and a permutation test (no distributional assumption).  (iii) Wasserman's toy
permutation example (1, 9 | 3): enumerate all 3! orderings.

Writes: figures/ch04/wald_size.pdf, figures/ch04/permutation_dist.pdf,
        results/ch04/05_wald_t_permutation.tex
"""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "05_wald_t_permutation")
setup()
alpha = 0.05
M = 200_000

# ---------- (i) size of Wald vs t for small n ----------
ns = np.array([3, 4, 5, 7, 10, 15, 20, 30, 50, 100])
size_wald, size_t = [], []
for n in ns:
    x = rng.standard_normal((M, n))                      # H0 true: mu = 0
    T = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
    size_wald.append(np.mean(np.abs(T) > stats.norm.isf(alpha / 2)))
    size_t.append(np.mean(np.abs(T) > stats.t.isf(alpha / 2, n - 1)))
size_wald, size_t = np.array(size_wald), np.array(size_t)
# exact size of the Wald test: P(|t_{n-1}| > z_{alpha/2})
size_wald_exact = 2 * stats.t.sf(stats.norm.isf(alpha / 2), ns - 1)

fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.plot(ns, size_wald, "o", color=SERIES[1], label="Wald: reject if $|T|>z_{0.025}$")
ax.plot(ns, size_wald_exact, "-", color=SERIES[1], lw=1)
ax.plot(ns, size_t, "s", color=SERIES[0], label=r"t-test: reject if $|T|>t_{n-1,0.025}$")
ax.axhline(alpha, color="k", ls="--", lw=1)
ax.set_xscale("log"); ax.set_xlabel("sample size $n$"); ax.set_ylabel("type I error rate")
ax.legend(fontsize=8)
savefig(fig, "ch04", "wald_size")

# ---------- (ii) two calibration runs ----------
m, n = 12, 15
run1 = rng.normal(661.7, 4.0, size=m)                    # Cs-137 line, keV, resolution 4 keV
run2 = rng.normal(665.0, 4.0, size=n)                    # a 3.3 keV scale shift in run 2
d_hat = run2.mean() - run1.mean()
se_hat = np.sqrt(run1.var(ddof=1) / m + run2.var(ddof=1) / n)
W = d_hat / se_hat
p_wald = 2 * stats.norm.sf(abs(W))
p_welch = stats.ttest_ind(run2, run1, equal_var=False).pvalue
pooled = np.concatenate([run1, run2])
B = 100_000
Tperm = np.empty(B)
for j in range(B):
    perm = rng.permutation(pooled)
    Tperm[j] = perm[m:].mean() - perm[:m].mean()
p_perm = np.mean(np.abs(Tperm) >= abs(d_hat))
p_perm_scipy = stats.permutation_test((run2, run1), lambda a, b: a.mean() - b.mean(),
                                      n_resamples=20_000, alternative="two-sided",
                                      random_state=rng).pvalue

fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.hist(Tperm, bins=80, density=True, color=SERIES[0], alpha=0.6, label="permutation distribution")
ax.axvline(d_hat, color=SERIES[1], lw=1.5, label="observed difference")
ax.axvline(-d_hat, color=SERIES[1], lw=1.0, ls=":")
ax.set_xlabel(r"$\bar x_2-\bar x_1$ after shuffling the run labels [keV]"); ax.set_ylabel("density")
ax.legend(fontsize=8)
savefig(fig, "ch04", "permutation_dist")

# ---------- (iii) Wasserman's toy example ----------
data = (1, 9, 3)                                         # X1, X2 | Y1
T = lambda v: abs((v[0] + v[1]) / 2 - v[2])
tobs = T(data)
Ts = [T(p) for p in itertools.permutations(data)]
p_toy = np.mean([t > tobs for t in Ts])

# ---------- AoS cholesterol numbers ----------
p_chol = 2 * stats.norm.sf(3.78)

save_numbers("ch04", "05_wald_t_permutation", {
    "FourAWaldSizeThree": f"{size_wald[0]:.3f}", "FourAWaldSizeFive": f"{size_wald[2]:.3f}",
    "FourAWaldSizeTen": f"{size_wald[4]:.3f}", "FourAWaldSizeThirty": f"{size_wald[7]:.3f}",
    "FourAWaldSizeThreeEx": f"{size_wald_exact[0]:.3f}",
    "FourATSizeThree": f"{size_t[0]:.3f}", "FourATSizeTen": f"{size_t[4]:.3f}",
    "FourACalMeanOne": f"{run1.mean():.2f}", "FourACalMeanTwo": f"{run2.mean():.2f}",
    "FourACalSdOne": f"{run1.std(ddof=1):.2f}", "FourACalSdTwo": f"{run2.std(ddof=1):.2f}",
    "FourACalD": f"{d_hat:.2f}", "FourACalSe": f"{se_hat:.2f}", "FourACalW": f"{W:.2f}",
    "FourACalPwald": f"{p_wald:.4f}", "FourACalPwelch": f"{p_welch:.4f}",
    "FourACalPperm": f"{p_perm:.4f}", "FourACalPpermScipy": f"{p_perm_scipy:.4f}",
    # difference of the two Monte Carlo p-values in units of their combined binomial error
    "FourACalPermDiffSig": f"{abs(p_perm - p_perm_scipy) / np.sqrt(p_perm * (1 - p_perm) / B + p_perm_scipy * (1 - p_perm_scipy) / 20_000):.1f}",
    "FourAToyP": f"{p_toy:.3f}", "FourACholP": f"{p_chol:.5f}",
})
print(size_wald, size_t, size_wald_exact)
print(run1.mean(), run2.mean(), d_hat, se_hat, W, p_wald, p_welch, p_perm, p_perm_scipy, p_toy, Ts, p_chol)
