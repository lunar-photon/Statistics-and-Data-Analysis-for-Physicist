"""Empirical tests of randomness: four null-hypothesis tests on four generators.

Question: the hypothesis H0 is "these numbers are independent draws from
Uniform(0,1)".  For each test we know the distribution of its statistic under
H0, so a p-value says how surprising the observed statistic would be if H0 were
true.  Which generators do the tests catch, and which test catches which flaw?

Computes, for PCG64, MT19937, RANDU and the LCG (1229 x + 1) mod 2048, on
N = 999999 numbers each:
 - equidistribution: chi^2 of a 64-bin histogram (63 degrees of freedom);
 - serial correlation at lag 1: z = r sqrt(N);
 - runs up and down: number of runs R, z = (R - (2N-1)/3) / sqrt((16N-29)/90);
 - triples: chi^2 of non-overlapping triples in a 20^3 grid (7999 d.o.f.).
Also 400 repetitions of the histogram test on 10^4 PCG64 numbers, to show the
p-value is itself uniform under H0.
Writes: figures/ch06/tests_pvalues.pdf, figures/ch06/tests_triples.pdf,
        results/ch06/08_randomness_tests.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from lib_prng import randu, lcg_ints_vec

N = 999_999


def chi2_hist(u, k=64):
    obs = np.bincount(np.minimum((u * k).astype(int), k - 1), minlength=k)
    e = len(u) / k
    X2 = np.sum((obs - e) ** 2 / e)
    return X2, stats.chi2.sf(X2, k - 1)


def serial(u):
    r = np.corrcoef(u[:-1], u[1:])[0, 1]
    z = r * np.sqrt(len(u))
    return z, 2 * stats.norm.sf(abs(z))


def runs_updown(u):
    s = np.sign(np.diff(u))
    s = s[s != 0]
    R = 1 + np.count_nonzero(s[1:] != s[:-1])
    n = len(u)
    z = (R - (2 * n - 1) / 3) / np.sqrt((16 * n - 29) / 90)
    return z, 2 * stats.norm.sf(abs(z))


def chi2_triples(u, k=20):
    T = u[: len(u) // 3 * 3].reshape(-1, 3)
    idx = np.minimum((T * k).astype(int), k - 1)
    cell = (idx[:, 0] * k + idx[:, 1]) * k + idx[:, 2]
    obs = np.bincount(cell, minlength=k**3)
    e = len(T) / k**3
    X2 = np.sum((obs - e) ** 2 / e)
    return X2, stats.chi2.sf(X2, k**3 - 1)


rng = rng_for("ch06", "08_randomness_tests")
mt = np.random.Generator(np.random.MT19937(rng.integers(2**63)))
gens = {
    "PCG64": rng.random(N),
    "MT19937": mt.random(N),
    "RANDU": randu(1, N),
    "LCG2048": lcg_ints_vec(1229, 1, 2048, 1, N) / 2048,
}
table = {}
for name, u in gens.items():
    table[name] = dict(hist=chi2_hist(u), ser=serial(u), runs=runs_updown(u), tri=chi2_triples(u))
    print(name, {k: (round(v[0], 3), f"{v[1]:.3g}") for k, v in table[name].items()})

# p-values of many small histogram tests under H0
reps = 400
pv = np.array([chi2_hist(rng.random(10_000))[1] for _ in range(reps)])
ks = stats.kstest(pv, "uniform")
print("uniformity of p-values: KS p =", ks.pvalue)

# ---------------- figures ----------------
setup(6.2, 2.7)
fig, axes = plt.subplots(1, 2, gridspec_kw={"wspace": 0.45})
axes[0].hist(pv, bins=20, range=(0, 1), color=SERIES[0], alpha=0.85, density=True)
theory_line(axes[0], [0, 1], [1, 1], label="uniform")
axes[0].set_xlabel("p-value of the histogram test")
axes[0].set_ylabel("density")
axes[0].set_title(f"PCG64, {reps} batches of $10^4$", fontsize=8)
x = np.linspace(7400, 8600, 400)
axes[1].plot(x, stats.chi2.pdf(x, 7999), color="k", ls="--", lw=1.2, label=r"$\chi^2_{7999}$ under $H_0$")
for j, name in enumerate(["PCG64", "MT19937"]):
    axes[1].axvline(table[name]["tri"][0], color=SERIES[j], lw=1.5, label=name)
axes[1].annotate(f"RANDU: {table['RANDU']['tri'][0]:.0f}\n(far off scale)", xy=(8600, 0.0006), xytext=(8230, 0.0019),
                 fontsize=8, color=SERIES[2], arrowprops=dict(arrowstyle="->", color=SERIES[2]))
axes[1].set_xlabel(r"$X^2$ of triples in $20^3$ cells")
axes[1].set_title("the triple test", fontsize=8)
axes[1].set_ylim(top=0.0049)
axes[1].legend(fontsize=7, loc="upper left", framealpha=0.9)
savefig(fig, "ch06", "tests_pvalues")

nums = {}
key = {"PCG64": "Pcg", "MT19937": "Mt", "RANDU": "Randu", "LCG2048": "Lam"}
tkey = {"hist": "Hist", "ser": "Ser", "runs": "Runs", "tri": "Tri"}


def fmtp(p):
    if p < 1e-300:
        return r"<10^{-300}"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return rf"{m}\times10^{{{int(e)}}}"
    return f"{p:.3f}"


for g, d in table.items():
    for t, (s, p) in d.items():
        nums[f"SixAT{key[g]}{tkey[t]}Stat"] = float(f"{s:.4g}") if t in ("hist", "tri") else f"{s:.2f}"
        nums[f"SixAT{key[g]}{tkey[t]}P"] = fmtp(p)
nums["SixATLamHistLowP"] = fmtp(stats.chi2.cdf(table["LCG2048"]["hist"][0], 63))
nums["SixATreps"] = reps
nums["SixATksP"] = f"{ks.pvalue:.2f}"
save_numbers("ch06", "08_randomness_tests", nums)
