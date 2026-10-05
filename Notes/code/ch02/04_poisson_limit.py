"""Rare events in many trials: the Poisson distribution as a limit of the binomial.

Question:  n nuclei each decay in the counting window with a tiny probability
           p = lam/n.  As n grows with lam fixed, does Binomial(n, lam/n) turn
           into Poisson(lam) = e^-lam lam^k / k!, and how fast?
Computes:  (a) explicit simulation: n = 2000 nuclei, each a Bernoulli(lam/n)
           trial, 20000 counting windows, histogram vs Poisson(3);
           (b) total-variation distance between Binomial(n, lam/n) and
           Poisson(lam) for several n, with the Le Cam bound lam^2/n;
           (c) the typesetter example (1500 words, p = 1/500): exact vs Poisson;
           (d) Poisson shapes for lam = 2, 5, 10.
Writes:    figures/ch02/04_poisson_limit.pdf, results/ch02/04_poisson_limit.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from common import setup, savefig, save_numbers, rng_for, SERIES

setup(6.4, 2.9)
rng = rng_for("ch02", "04_poisson_limit")
lam = 3.0

# (a) the generating process, literally: many nuclei, each with a tiny chance
n_nuclei, windows = 2000, 20_000
counts = np.empty(windows, dtype=int)
for w in range(windows):
    counts[w] = np.count_nonzero(rng.uniform(size=n_nuclei) <= lam / n_nuclei)

# (b) distance between the two pmfs:  TV = (1/2) sum_k |Bin(k) - Pois(k)|
def tv(n):
    k = np.arange(0, n + 1)
    tail = poisson.sf(n, lam)                 # Poisson mass above n (binomial has none there)
    return 0.5 * (np.abs(binom.pmf(k, n, lam / n) - poisson.pmf(k, lam)).sum() + tail)
ns = [5, 20, 100, 1000]
tvs = {n: tv(n) for n in ns}

# (c) typesetter: X ~ Binomial(1500, 1/500);  P(X <= 2)
typ_exact = binom.cdf(2, 1500, 1 / 500)
typ_pois = poisson.cdf(2, 3.0)

fig, (ax1, ax2) = plt.subplots(1, 2)
ks = np.arange(0, 13)
ax1.bar(ks, np.bincount(counts, minlength=13)[:13] / windows, color=SERIES[0], alpha=0.4,
        label=f"{n_nuclei} nuclei, sim.")
for c, n in zip(SERIES[1:], [5, 20]):
    ax1.plot(ks, binom.pmf(ks, n, lam / n), "o-", color=c, ms=3, lw=0.8, label=f"Bin($n={n}$, $3/n$)")
ax1.plot(ks, poisson.pmf(ks, lam), "k_", ms=10, mew=1.8, label="Poisson(3)")
ax1.set_xlabel("counts $k$"); ax1.set_ylabel("$P(k)$")
ax1.legend(fontsize=7.5)
for c, L in zip(SERIES, [2, 5, 10]):
    k = np.arange(0, 21)
    ax2.plot(k, poisson.pmf(k, L), "o-", color=c, ms=3, lw=0.9, label=f"$\\lambda={L}$")
ax2.set_xlabel("counts $k$"); ax2.set_ylabel("$P(k)$")
ax2.legend()
fig.tight_layout()
savefig(fig, "ch02", "04_poisson_limit")

save_numbers("ch02", "04_poisson_limit", {
    "twoaPLMean": float(counts.mean()), "twoaPLVar": float(counts.var(ddof=1)),
    "twoaPLZeroSim": float(np.mean(counts == 0)), "twoaPLZeroTh": poisson.pmf(0, lam),
    "twoaTVfive": tvs[5], "twoaTVtwenty": tvs[20], "twoaTVhundred": tvs[100], "twoaTVthousand": tvs[1000],
    "twoaTypExact": typ_exact, "twoaTypPois": typ_pois,
})
print(tvs, typ_exact, typ_pois)
