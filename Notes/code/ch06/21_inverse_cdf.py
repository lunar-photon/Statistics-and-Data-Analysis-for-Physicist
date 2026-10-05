"""21_inverse_cdf.py -- turning uniform numbers into decay times, cosmic-ray energies and counts.

Question: given uniform numbers U in [0, 1), how do we produce numbers with a
prescribed distribution F?  Answer: X = F^{-1}(U), because P(F^{-1}(U) <= x) =
P(U <= F(x)) = F(x).

Three cases, each checked against the exact density:
  1. muon decay times, exponential with tau = 2.197 us:  t = -tau ln(1 - u);
  2. a cosmic-ray-like power law dN/dE ~ E^-gamma (gamma = 2.7) between
     E_min = 10 GeV and E_max = 1e6 GeV:  E = [E_min^(1-g) - u (E_min^(1-g) - E_max^(1-g))]^(1/(1-g));
  3. discrete outcomes by cumulative sums: Binomial(4, 5/8) (the Casella--Berger
     table) and Poisson(3.5), found with np.searchsorted on the cumulative table.

Writes: figures/ch06/inverse_cdf.pdf, results/ch06/21_inverse_cdf.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup(9.0, 3.0)
rng = rng_for("ch06", "21_inverse_cdf")
N = 100_000
out = {"SixBInvN": N}

# ---------------------------------------------------------------- 1. exponential decay times
tau = 2.197                                     # muon lifetime in microseconds
u = rng.random(N)
t = -tau * np.log(1.0 - u)                      # inverse CDF of 1 - exp(-t/tau)
out.update(SixBInvTmean=t.mean(), SixBInvTstd=t.std(ddof=1),
           SixBInvTse=t.std(ddof=1) / np.sqrt(N))

# ---------------------------------------------------------------- 2. power-law energies
g, Emin, Emax = 2.7, 10.0, 1.0e6
a, b = Emin ** (1 - g), Emax ** (1 - g)
E = (a - rng.random(N * 10) * (a - b)) ** (1.0 / (1 - g))   # 10^6 energies
frac_TeV = np.mean(E > 1000.0)
exact_TeV = (1000.0 ** (1 - g) - b) / (a - b)
out.update(SixBInvNE=E.size, SixBInvFracTeV=frac_TeV, SixBInvExactTeV=exact_TeV,
           SixBInvCountTeV=int(np.sum(E > 1000.0)), SixBInvEmax=E.max())

# ---------------------------------------------------------------- 3. discrete via cumulative sums
k4 = np.arange(5)
p4 = stats.binom.pmf(k4, 4, 5 / 8)
cum4 = np.cumsum(p4)                              # 0.020, 0.152, 0.481, 0.847, 1
y4 = np.searchsorted(cum4, rng.random(N), side="right")     # first k with U < F(k)
freq4 = np.bincount(y4, minlength=5) / N

lam = 3.5
kp = np.arange(0, 30)
cump = np.cumsum(stats.poisson.pmf(kp, lam))
yp = np.searchsorted(cump, rng.random(N), side="right")
freqp = np.bincount(yp, minlength=kp.size)[: kp.size] / N
out.update({f"SixBInvCum{w}": c for w, c in zip(["Zero", "One", "Two", "Three"], cum4[:4])})
out.update(SixBInvPoisMean=yp.mean(), SixBInvPoisVar=yp.var(ddof=1),
           SixBInvBinMax=np.max(np.abs(freq4 - p4)))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3)
bins = np.linspace(0, 15, 61)
ax[0].hist(t, bins=bins, density=True, color=SERIES[0], alpha=0.55, label="samples")
xx = np.linspace(0, 15, 300)
theory_line(ax[0], xx, np.exp(-xx / tau) / tau, label=r"$e^{-t/\tau}/\tau$")
ax[0].set(xlabel=r"decay time $t$ [$\mu$s]", ylabel="density", title="exponential")
ax[0].legend()

lb = np.logspace(1, 6, 41)
h, _ = np.histogram(E, bins=lb)
dens = h / (E.size * np.diff(lb))
cen = np.sqrt(lb[1:] * lb[:-1])
ok = h >= 10                                  # bins with enough counts for a sqrt(n) error bar
ax[1].errorbar(cen[ok], dens[ok], yerr=np.sqrt(h[ok]) / (E.size * np.diff(lb)[ok]),
               fmt="o", ms=3, color=SERIES[0], label="samples")
EE = np.logspace(1, 6, 200)
theory_line(ax[1], EE, (g - 1) * EE ** (-g) / (a - b), label=r"$\propto E^{-2.7}$")
ax[1].set(xscale="log", yscale="log", xlabel=r"energy $E$ [GeV]", title="power law")
ax[1].legend()

ax[2].bar(kp[:12] - 0.2, freqp[:12], width=0.4, color=SERIES[0], alpha=0.7, label="Poisson(3.5) samples")
ax[2].bar(k4 + 0.2, freq4, width=0.4, color=SERIES[1], alpha=0.7, label="Binomial(4,5/8) samples")
ax[2].plot(kp[:12], stats.poisson.pmf(kp[:12], lam), "k_", ms=12, mew=1.6, label="exact")
ax[2].plot(k4, p4, "k_", ms=12, mew=1.6)
ax[2].set(xlabel=r"count $k$", ylabel="probability", title="discrete")
ax[2].set_ylim(0, 0.52)
ax[2].legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "ch06", "inverse_cdf")
save_numbers("ch06", "21_inverse_cdf", out)
print(out)
