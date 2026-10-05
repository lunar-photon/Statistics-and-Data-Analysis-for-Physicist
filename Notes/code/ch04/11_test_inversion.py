"""11_test_inversion.py -- a confidence interval is the set of null values a test does not reject.

Question: we flipped a coin N = 24 times and saw z = 7 heads (Kruschke's example).
For which values theta0 of the heads probability would a two-sided level-0.05 test
NOT reject H0: theta = theta0?  That set of survivors is the 95% confidence interval.
How does it compare with the quick Normal (Wald) interval and the Hoeffding interval?
And a warm-up on "coverage is not a statement about this data set": the Berger-Wolpert
two-point example, whose 75% interval is sometimes certainly right.

Computes: the p-value function p(theta0) (exact binomial, and the Wald approximation),
the exact (Clopper-Pearson) interval by root finding, the Wald and Hoeffding intervals,
and a Monte Carlo of the Berger-Wolpert interval (overall and conditional coverage).

Writes: figures/ch04/pvalue_function.pdf, results/ch04/11_test_inversion.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

rng = rng_for("ch04", "11_test_inversion")
setup()

N, z, alpha = 24, 7, 0.05

# ---------- exact p-value function: two tails, each compared with alpha/2 ----------
def p_lower_tail(t):          # P(Z <= z | theta = t): small when t is too large
    return stats.binom.cdf(z, N, t)

def p_upper_tail(t):          # P(Z >= z | theta = t): small when t is too small
    return stats.binom.sf(z - 1, N, t)

def p_two_sided(t):
    return np.minimum(1.0, 2 * np.minimum(p_lower_tail(t), p_upper_tail(t)))

# the interval ends are where each one-sided tail equals alpha/2
lo = optimize.brentq(lambda t: p_upper_tail(t) - alpha / 2, 1e-6, z / N)
hi = optimize.brentq(lambda t: p_lower_tail(t) - alpha / 2, z / N, 1 - 1e-6)
# the same numbers from the binomial-beta identity (a check)
lo_beta = stats.beta.ppf(alpha / 2, z, N - z + 1)
hi_beta = stats.beta.ppf(1 - alpha / 2, z + 1, N - z)
assert abs(lo - lo_beta) < 1e-6 and abs(hi - hi_beta) < 1e-6

# ---------- Wald (Normal) and Hoeffding intervals for the same data ----------
phat = z / N
se_hat = np.sqrt(phat * (1 - phat) / N)
zq = stats.norm.isf(alpha / 2)
wald = (phat - zq * se_hat, phat + zq * se_hat)
eps = np.sqrt(np.log(2 / alpha) / (2 * N))
hoef = (max(0.0, phat - eps), min(1.0, phat + eps))

def p_wald(t):                # the Wald test uses the estimated se, centred on phat
    return 2 * stats.norm.sf(np.abs(phat - t) / se_hat)

# ---------- figure: the p-value function ----------
t = np.linspace(0.001, 0.85, 2000)
fig, ax = plt.subplots(figsize=(6.0, 3.3))
ax.plot(t, p_two_sided(t), color=SERIES[0], label="exact binomial test")
ax.plot(t, p_wald(t), color=SERIES[1], ls="-.", lw=1.3, label="Wald (Normal) test")
ax.axhline(alpha, color="k", lw=0.9, ls=":")
ax.text(0.83, alpha + 0.02, r"$\alpha=0.05$", ha="right", fontsize=8)
ax.axvspan(lo, hi, color=SERIES[0], alpha=0.12, lw=0)
ax.axvline(phat, color="0.4", lw=0.8)
ax.text(phat + 0.008, 0.12, r"$\hat\theta=7/24$", fontsize=8, color="0.3")
ax.text(0.37, 0.6, "not rejected:\n95% confidence\ninterval", ha="left", fontsize=8,
        color=SERIES[0])
ax.set_xlabel(r"null value $\theta_0$ of the heads probability")
ax.set_ylabel(r"two-sided $p$-value of $H_0:\theta=\theta_0$")
ax.set_xlim(0, 0.85); ax.set_ylim(0, 1.02)
ax.legend(loc="upper right", fontsize=8)
savefig(fig, "ch04", "pvalue_function")

# ---------- Berger-Wolpert: a 75% interval that is sometimes certainly right ----------
theta = 16.0
M = 400_000
X = rng.choice([-1, 1], size=(M, 2))
Y = theta + X
same = Y[:, 0] == Y[:, 1]
C = np.where(same, Y[:, 0] - 1, 0.5 * (Y[:, 0] + Y[:, 1]))
hit = C == theta
cov_all = hit.mean()
cov_same = hit[same].mean()
cov_diff = hit[~same].mean()

save_numbers("ch04", "11_test_inversion", {
    "FourBKrN": N, "FourBKrZ": z,
    "FourBKrThetaHat": f"{phat:.3f}",
    "FourBKrLo": f"{lo:.3f}", "FourBKrHi": f"{hi:.3f}",
    "FourBKrWaldLo": f"{wald[0]:.3f}", "FourBKrWaldHi": f"{wald[1]:.3f}",
    "FourBKrSe": f"{se_hat:.4f}",
    "FourBKrHoefEps": f"{eps:.3f}",
    "FourBKrHoefLo": f"{hoef[0]:.3f}", "FourBKrHoefHi": f"{hoef[1]:.3f}",
    "FourBKrPhalf": f"{p_two_sided(0.5):.3f}",
    "FourBBWcovAll": f"{cov_all:.4f}", "FourBBWcovSame": f"{cov_same:.4f}",
    "FourBBWcovDiff": f"{cov_diff:.4f}", "FourBBWfracSame": f"{same.mean():.4f}",
})
print(f"exact CI [{lo:.4f}, {hi:.4f}], Wald [{wald[0]:.4f}, {wald[1]:.4f}], "
      f"Hoeffding [{hoef[0]:.4f}, {hoef[1]:.4f}], p(0.5) = {p_two_sided(0.5):.4f}")
print(f"Berger-Wolpert coverage all={cov_all:.4f} same={cov_same:.4f} diff={cov_diff:.4f}")
