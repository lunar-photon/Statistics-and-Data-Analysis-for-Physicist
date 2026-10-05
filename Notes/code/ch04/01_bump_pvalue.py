"""01_bump_pvalue.py -- a bump in an invariant-mass histogram: new particle or luck?

Question: a histogram of diphoton invariant masses shows an excess in a two-bin
window.  If only background were present (H0: the Poisson mean in the window is
the known b), how often would we see a count like this, or even more in the
upward direction?  That tail probability is the p-value.

Computes: a toy spectrum (falling exponential background + a small Gaussian
signal), the observed window count, the exact Poisson p-value, a Monte Carlo
p-value from 10^6 background-only experiments, the naive Gaussian estimate, and
Cowan's textbook numbers (11 observed on 3.2 expected; 5 on 0.5 or 0.8).

Writes: figures/ch04/bump_spectrum.pdf, figures/ch04/bump_null.pdf,
        results/ch04/01_bump_pvalue.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "01_bump_pvalue")
setup()

# ---------- the toy spectrum ----------
edges = np.arange(100.0, 162.0, 2.0)            # 30 bins of 2 GeV, 100-160 GeV
centres = 0.5 * (edges[1:] + edges[:-1])
B_TOT, SLOPE = 60.0, 25.0                       # expected background events, decay length (GeV)
# expected background per bin: integral of B_TOT * exponential shape over each bin
norm = 1 - np.exp(-(edges[-1] - edges[0]) / SLOPE)
cdf = lambda m: (1 - np.exp(-(m - edges[0]) / SLOPE)) / norm
b_bin = B_TOT * np.diff(cdf(edges))
M_SIG, W_SIG, S_TOT = 126.0, 1.2, 9.0            # signal position, width, expected events
s_bin = S_TOT * np.diff(stats.norm.cdf(edges, M_SIG, W_SIG))
n_bin = rng.poisson(b_bin + s_bin)               # the one data set we "observed"

win = (edges[:-1] >= 124.0) & (edges[1:] <= 128.0)   # two-bin window 124-128 GeV
b_win = b_bin[win].sum()
n_win = int(n_bin[win].sum())

# exact Poisson p-value:  P(N >= n_obs | b) = 1 - F(n_obs - 1)
p_exact = stats.poisson.sf(n_win - 1, b_win)
# Monte Carlo: background-only experiments in the window
M = 1_000_000
n_toys = rng.poisson(b_win, size=M)
p_mc = np.mean(n_toys >= n_win)
p_mc_err = np.sqrt(p_mc * (1 - p_mc) / M)
z_exact = stats.norm.isf(p_exact)
z_naive = (n_win - b_win) / np.sqrt(b_win)       # Gaussian approximation with sigma = sqrt(b)
z_wrong = (n_win - b_win) / np.sqrt(n_win)       # the "n +- sqrt(n)" error bar

# ---------- Cowan's numbers (Statistical Data Analysis, Sec. 4.6) ----------
pC1 = stats.poisson.sf(11 - 1, 3.2)
pC2 = stats.poisson.sf(5 - 1, 0.5)
pC3 = stats.poisson.sf(5 - 1, 0.8)

# ---------- figure 1: the spectrum ----------
fig, ax = plt.subplots(figsize=(6.0, 3.3))
ax.axvspan(124, 128, color=SERIES[3], alpha=0.18, lw=0, label="search window")
ax.stairs(b_bin, edges, color="k", ls="--", lw=1.3, label="expected background")
ax.errorbar(centres, n_bin, yerr=np.sqrt(np.maximum(n_bin, 1)), fmt="o", color=SERIES[0],
            ms=3.5, lw=1, capsize=0, label="data")
ax.set_xlabel(r"invariant mass $m_{\gamma\gamma}$ [GeV]")
ax.set_ylabel("events / 2 GeV")
ax.set_xlim(100, 160); ax.set_ylim(0, None)
ax.legend(loc="upper right")
savefig(fig, "ch04", "bump_spectrum")

# ---------- figure 2: the null distribution of the window count ----------
k = np.arange(0, 26)
pmf = stats.poisson.pmf(k, b_win)
counts = np.bincount(n_toys, minlength=k.size)[: k.size] / M
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, logy in zip(axs, (False, True)):
    tail = k >= n_win
    ax.bar(k[~tail], pmf[~tail], width=0.8, color=SERIES[0], alpha=0.55, label=r"$\mathrm{Pois}(n;b)$")
    ax.bar(k[tail], pmf[tail], width=0.8, color=SERIES[1], alpha=0.9, label=r"tail $n\geq n_{\rm obs}$")
    ax.plot(k, np.where(counts > 0, counts, np.nan), "k.", ms=4, label=r"$10^6$ toys")
    ax.axvline(n_win, color=SERIES[1], lw=1.0, ls=":")
    ax.set_xlabel(r"count $n$ in the window")
    if logy:
        ax.set_yscale("log"); ax.set_ylim(1e-7, 1)
    else:
        ax.set_ylabel("probability"); ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch04", "bump_null")

save_numbers("ch04", "01_bump_pvalue", {
    "FourABumpBtot": B_TOT, "FourABumpStot": S_TOT,
    "FourABumpB": f"{b_win:.2f}", "FourABumpN": n_win,
    "FourABumpP": p_exact, "FourABumpPmc": p_mc, "FourABumpPmcErr": p_mc_err,
    "FourABumpZ": f"{z_exact:.2f}", "FourABumpZnaive": f"{z_naive:.2f}", "FourABumpZwrong": f"{z_wrong:.2f}",
    "FourACowanPone": pC1, "FourACowanPtwo": pC2, "FourACowanPthree": pC3,
    "FourACowanZone": f"{stats.norm.isf(pC1):.2f}",
    "FourACowanZnaive": f"{(11 - 3.2) / np.sqrt(3.2):.2f}",
})
print(f"window b={b_win:.3f}, n={n_win}, p={p_exact:.3e}, pmc={p_mc:.3e}, Z={z_exact:.2f}, "
      f"Znaive={z_naive:.2f}, Zwrong={z_wrong:.2f}")
print(f"Cowan: {pC1:.3e} {pC2:.3e} {pC3:.3e}")
