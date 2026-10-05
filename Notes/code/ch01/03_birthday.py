"""The birthday problem: exact counting versus simulation.

Question: among k people (or k particle hits in a detector with N cells), what
is the probability that at least two share a birthday (land in the same cell)?

Computes: the exact probability 1 - N!/((N-k)! N^k) for N = 365, its
exponential approximation 1 - exp(-k(k-1)/(2N)), and a Monte Carlo estimate from
repeated random draws; the smallest k with probability > 1/2; and the same
question for k hits spread over a pixel detector with N = 10^4 pixels.
Writes: figures/ch01/1a_birthday.pdf, results/ch01/1a_birthday.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt


def p_shared_exact(k, N):
    """1 - prod_{j=0}^{k-1} (N - j)/N, computed as a product of factors < 1."""
    j = np.arange(k)
    return 1.0 - np.prod((N - j) / N)


def p_shared_approx(k, N):
    return 1.0 - np.exp(-k * (k - 1) / (2.0 * N))


def p_shared_mc(k, N, trials, rng):
    """Draw k labels uniformly from N, many times; count draws with a repeat."""
    draws = rng.integers(0, N, size=(trials, k))
    draws.sort(axis=1)
    has_repeat = np.any(draws[:, 1:] == draws[:, :-1], axis=1)
    return has_repeat.mean()


rng = rng_for("ch01", "1a_birthday")
N = 365
ks = np.arange(1, 71)
exact = np.array([p_shared_exact(k, N) for k in ks])
approx = p_shared_approx(ks, N)
k_mc = np.arange(5, 71, 5)
trials = 20_000
mc = np.array([p_shared_mc(k, N, trials, rng) for k in k_mc])

k_half = int(ks[np.argmax(exact > 0.5)])
mc23 = p_shared_mc(23, N, 200_000, rng)

# pixel detector: k hits in N_pix pixels
N_pix = 10_000
k_pix = 100
pix_exact = p_shared_exact(k_pix, N_pix)
pix_mc = p_shared_mc(k_pix, N_pix, 100_000, rng)

setup(6.0, 3.6)
fig, ax = plt.subplots()
ax.plot(ks, exact, color=SERIES[0], label="exact count")
theory_line(ax, ks, approx, label=r"$1-e^{-k(k-1)/2N}$")
ax.plot(k_mc, mc, "o", color=SERIES[1], label=f"simulation ({trials:,} trials)")
ax.axhline(0.5, color="0.5", lw=0.8)
ax.axvline(k_half, color="0.5", lw=0.8)
ax.set_xlabel("number of people $k$")
ax.set_ylabel("P(at least one shared birthday)")
ax.set_title("birthday problem, $N=365$")
ax.legend(loc="lower right")
fig.tight_layout()
savefig(fig, "ch01", "1a_birthday")

save_numbers("ch01", "1a_birthday", {
    "OneABdayExact": f"{p_shared_exact(23, N):.4f}",
    "OneABdayApprox": f"{p_shared_approx(23, N):.4f}",
    "OneABdayMC": f"{mc23:.4f}",
    "OneABdayKhalf": k_half,
    "OneABdayExactFifty": f"{p_shared_exact(50, N):.3f}",
    "OneABdayPixExact": f"{pix_exact:.4f}",
    "OneABdayPixMC": f"{pix_mc:.4f}",
    "OneABdayPixApprox": f"{p_shared_approx(k_pix, N_pix):.4f}",
    # Monte Carlo error sqrt(q(1-q)/trials) of each frequency, and the distance from the
    # exact value in units of that error (a random number: about 1 on average)
    "OneABdayMCErr": f"{np.sqrt(p_shared_exact(23, N) * (1 - p_shared_exact(23, N)) / 200_000):.4f}",
    "OneABdayPixMCErr": f"{np.sqrt(pix_exact * (1 - pix_exact) / 100_000):.4f}",
    "OneABdayPull": f"{abs(mc23 - p_shared_exact(23, N)) / np.sqrt(p_shared_exact(23, N) * (1 - p_shared_exact(23, N)) / 200_000):.1f}",
    "OneABdayPixPull": f"{abs(pix_mc - pix_exact) / np.sqrt(pix_exact * (1 - pix_exact) / 100_000):.1f}",
})
print(k_half, p_shared_exact(23, N), mc23, pix_exact, pix_mc)
