"""Same oscillator, different sampling rule: choose the POSITION uniformly, and ask how the time is distributed.

Question: the oscillator x(t) = A cos(omega t) has a deterministic motion.  A probability
distribution needs, in addition, a rule saying what is chosen at random.  If the instant t is
uniform over a period, x follows the arcsine law.  If instead the position x is chosen uniformly,
p(x) = 1/(2A), with the direction of motion (sign of the momentum) decided by a fair coin, the
instant at which the oscillator is at that point is no longer uniform:
        P(t) = (omega / 4) |sin(omega t)|,   0 <= t < T.
We sample x uniformly, map each x back to its time t on the orbit, histogram t, and compare.
Writes: figures/ch01/oscillator_sampling_rule.pdf, results/ch01/14b_sampling_rule.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "14b_sampling_rule")
setup(7.0, 2.8)

A, T = 1.0, 1.0
omega = 2 * np.pi / T
N = 200_000

# rule 1 (for comparison): uniform time -> arcsine in x
t_u = rng.uniform(0.0, T, size=N)
x_from_t = A * np.cos(omega * t_u)

# rule 2: uniform position, and a fair coin for the branch (moving left, p<0, or right, p>0)
x = rng.uniform(-A, A, size=N)
moving_left = rng.random(N) < 0.5
t0 = np.arccos(x / A) / omega              # the instant in [0, T/2] with x(t) = x and p < 0
t = np.where(moving_left, t0, T - t0)      # the other branch: the instant in [T/2, T], p > 0


def p_t(tt):
    return omega / 4 * np.abs(np.sin(omega * tt))


# chi-square of the t-histogram against the bin-integrated prediction (CDF by the trapezoid rule)
edges = np.linspace(0, T, 51)
counts, _ = np.histogram(t, bins=edges)
fine = np.linspace(0, T, 200_001)
cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p_t(fine[1:]) + p_t(fine[:-1])) * np.diff(fine))])
expected = N * np.diff(np.interp(edges, fine, cdf))
chi2 = np.sum((counts - expected) ** 2 / expected)
frac_first_tenth = np.mean(t < T / 10)     # fraction of instants in the first tenth of the period
frac_exact = (1 - np.cos(omega * T / 10)) / 4

fig, (ax_x, ax_t) = plt.subplots(1, 2, figsize=(7.0, 2.8))
xx = np.linspace(-0.999, 0.999, 600)
ax_x.hist(x_from_t, bins=50, range=(-1, 1), density=True, color=SERIES[0], alpha=0.45,
          label="uniform $t$")
ax_x.hist(x, bins=50, range=(-1, 1), density=True, color=SERIES[1], alpha=0.45,
          label="uniform $x$")
theory_line(ax_x, xx, 1 / (np.pi * np.sqrt(A**2 - xx**2)), label="arcsine")
ax_x.set_ylim(0, 3.2)
ax_x.set_xlabel("$x/A$"); ax_x.set_ylabel("density")
ax_x.set_title("(a) the position under the two rules", fontsize=9)
ax_x.legend(fontsize=7, loc="upper center")

tt = np.linspace(0, T, 600)
ax_t.hist(t_u, bins=50, range=(0, T), density=True, color=SERIES[0], alpha=0.45,
          label="uniform $t$")
ax_t.hist(t, bins=50, range=(0, T), density=True, color=SERIES[1], alpha=0.45,
          label="uniform $x$")
theory_line(ax_t, tt, p_t(tt) * T, label=r"$(\omega/4)|\sin\omega t|$")
ax_t.set_ylim(0, 2.25)
ax_t.set_xlabel("$t/T$"); ax_t.set_ylabel(r"density $\times\,T$")
ax_t.set_title("(b) the instant under the two rules", fontsize=9)
ax_t.legend(fontsize=7, loc="upper center", ncol=3)
fig.tight_layout()
savefig(fig, "ch01", "oscillator_sampling_rule")

save_numbers("ch01", "14b_sampling_rule", {
    "OneBRuleChi": f"{chi2:.1f}",
    "OneBRuleNbins": 50,
    "OneBRuleFracSim": f"{frac_first_tenth:.4f}",
    "OneBRuleFracExact": f"{frac_exact:.4f}",
})
