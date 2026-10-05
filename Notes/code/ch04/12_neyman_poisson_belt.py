"""12_neyman_poisson_belt.py -- the Neyman construction done by brute force for a Poisson mean.

Question: we count n events from a Poisson process with unknown mean nu.  For every
hypothetical nu, which counts are "acceptable" (central 90%: at most 5% probability
in each tail)?  Drawing those acceptance sets for all nu gives the confidence belt;
reading it vertically at the observed count gives the confidence interval.
Does the brute-force belt reproduce Garwood's chi-square formula (Cowan eq. 9.18,
Casella-Berger eq. 9.2.17) and the textbook tables?

Computes: acceptance sets [n1(nu), n2(nu)] on a fine nu grid, the interval for
n_obs = 0..10 by inversion, the chi-square formula for comparison, Cowan's Table 9.3
entries, and the Casella-Berger example (n = 10 observations, sum = 6, 90%).

Writes: figures/ch04/poisson_belt.pdf, results/ch04/12_neyman_poisson_belt.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup()
CL = 0.90
a_tail = (1 - CL) / 2                      # probability allowed in each tail

# ---------- the construction: one horizontal acceptance set per hypothetical nu ----------
nu_grid = np.arange(0.0005, 20.0, 0.0005)
n_max = 60
n = np.arange(n_max + 1)
cdf = stats.poisson.cdf(n[None, :], nu_grid[:, None])        # P(N <= n | nu)
sf = 1 - cdf + stats.poisson.pmf(n[None, :], nu_grid[:, None])  # P(N >= n | nu)
# n1(nu): the smallest n with P(N < n) <= a_tail AND P(N < n+1) > a_tail, i.e. the lowest
# count we keep.  Keep n if P(N <= n) > a_tail  and  P(N >= n) > a_tail  (both tails "not too small").
keep = (cdf > a_tail) & (sf > a_tail)
n1 = np.array([n[k].min() for k in keep])
n2 = np.array([n[k].max() for k in keep])
acc_prob = np.array([stats.poisson.pmf(n[k], nu).sum() for k, nu in zip(keep, nu_grid)])

# ---------- the inversion: the interval for n_obs collects every nu whose set contains n_obs ----------
def interval_from_belt(n_obs):
    inside = (n1 <= n_obs) & (n_obs <= n2)
    nus = nu_grid[inside]
    lo = 0.0 if n_obs == 0 else nus.min()
    return lo, nus.max()

def garwood(n_obs, cl_lo_tail, cl_hi_tail):
    lo = 0.0 if n_obs == 0 else 0.5 * stats.chi2.ppf(cl_lo_tail, 2 * n_obs)
    hi = 0.5 * stats.chi2.ppf(1 - cl_hi_tail, 2 * (n_obs + 1))
    return lo, hi

rows = []
for k in range(11):
    b_lo, b_hi = interval_from_belt(k)
    g_lo, g_hi = garwood(k, a_tail, a_tail)
    rows.append((k, b_lo, b_hi, g_lo, g_hi))
    assert abs(b_hi - g_hi) < 2e-3 and abs(b_lo - g_lo) < 2e-3, (k, b_lo, g_lo, b_hi, g_hi)

# ---------- figure: the belt ----------
fig, ax = plt.subplots(figsize=(6.0, 4.0))
nu_show = np.arange(0.5, 15.01, 0.5)
for nu in nu_show:
    i = np.argmin(np.abs(nu_grid - nu))
    ax.plot([n1[i] - 0.3, n2[i] + 0.3], [nu, nu], color=SERIES[0], lw=2.2, alpha=0.55,
            solid_capstyle="butt")
ax.step(n1 - 0.5, nu_grid, where="post", color=SERIES[0], lw=0.9)
ax.step(n2 + 0.5, nu_grid, where="post", color=SERIES[0], lw=0.9)
n_obs = 6
lo6, hi6 = garwood(n_obs, a_tail, a_tail)
ax.axvline(n_obs, color=SERIES[1], lw=1.4, ls="--")
ax.plot([n_obs, n_obs], [lo6, hi6], color=SERIES[1], lw=4, solid_capstyle="butt",
        label=rf"90% interval for $n_{{\rm obs}}={n_obs}$: [{lo6:.2f}, {hi6:.2f}]")
ax.set_xlabel(r"observed count $n$")
ax.set_ylabel(r"hypothetical Poisson mean $\nu$")
ax.set_xlim(-0.8, 22); ax.set_ylim(0, 15.5)
ax.set_xticks(range(0, 23, 2))
ax.legend(loc="lower right", fontsize=8)
savefig(fig, "ch04", "poisson_belt")

# ---------- textbook checks ----------
up95_n0 = 0.5 * stats.chi2.ppf(0.95, 2)                 # Cowan Table 9.3: 3.00 (= -ln 0.05)
up90_n0 = 0.5 * stats.chi2.ppf(0.90, 2)                 # 2.30
lo95_n3 = 0.5 * stats.chi2.ppf(0.05, 6)                 # Cowan Table 9.3: 0.818
up95_n3 = 0.5 * stats.chi2.ppf(0.95, 8)                 # 7.75
# Casella-Berger Ex. 9.2.15: n = 10 observations, sum = 6, 90% interval for lambda
cb_lo = stats.chi2.ppf(0.05, 12) / 20
cb_hi = stats.chi2.ppf(0.95, 14) / 20

vals = {"FourBBeltCL": "90", "FourBBeltNobs": n_obs,
        "FourBBeltLoSix": f"{lo6:.2f}", "FourBBeltHiSix": f"{hi6:.2f}",
        "FourBBeltMinAcc": f"{acc_prob.min():.4f}", "FourBBeltMaxAcc": f"{acc_prob.max():.4f}",
        "FourBUpNinetyFiveZero": f"{up95_n0:.3f}", "FourBUpNinetyZero": f"{up90_n0:.3f}",
        "FourBLoNinetyFiveThree": f"{lo95_n3:.3f}", "FourBUpNinetyFiveThree": f"{up95_n3:.2f}",
        "FourBCBlo": f"{cb_lo:.3f}", "FourBCBhi": f"{cb_hi:.3f}"}
names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
for (k, b_lo, b_hi, g_lo, g_hi) in rows:
    vals[f"FourBBeltLo{names[k]}"] = f"{g_lo:.2f}"
    vals[f"FourBBeltHi{names[k]}"] = f"{g_hi:.2f}"
save_numbers("ch04", "12_neyman_poisson_belt", vals)
for r in rows:
    print("n=%2d belt [%.3f, %.3f]  Garwood [%.3f, %.3f]" % r)
print(f"acceptance probability ranges {acc_prob.min():.4f} .. {acc_prob.max():.4f}")
print(f"n=0 upper 95% {up95_n0:.3f}, 90% {up90_n0:.3f}; n=3 95%: [{lo95_n3:.3f}, {up95_n3:.3f}]; "
      f"CB [{cb_lo:.3f}, {cb_hi:.3f}]")
