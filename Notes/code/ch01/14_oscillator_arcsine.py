"""Where is a classical harmonic oscillator? Histogram x(t) at random times; compare with arcsine.

Question: an oscillator x(t) = A cos(omega t), p(t) = -m omega A sin(omega t) is observed at a
time t chosen uniformly over a period.  The phase-space points (x, p) on the energy ellipse are
the sample space; a random instant picks a random point of it.  The change of variables t -> x predicts the density
        p(x) = 1 / (pi sqrt(A^2 - x^2)),   |x| < A      (the arcsine density),
which piles up at the turning points.  We check it by brute force (histogram of x at random t),
check the CDF-based prediction P(|x| > 0.9A) = 1 - (2/pi) arcsin(0.9), and compare with the
quantum probability density |psi_n|^2 for n = 30 at the same energy (correspondence principle).
Writes: figures/ch01/oscillator_arcsine.pdf, results/ch01/14_oscillator_arcsine.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "14_oscillator_arcsine")
setup(7.0, 3.0)

A, T, m = 1.0, 1.0, 1.0
omega = 2 * np.pi / T
N = 200_000
t = rng.uniform(0.0, T, size=N)      # the random ingredient: WHEN we look
x = A * np.cos(omega * t)            # the deterministic map t -> (x, p): a point on the orbit
p = -m * omega * A * np.sin(omega * t)

frac_sim = np.mean(np.abs(x) > 0.9 * A)
frac_exact = 1 - (2 / np.pi) * np.arcsin(0.9)


def arcsine_pdf(xx):
    return 1.0 / (np.pi * np.sqrt(A**2 - xx**2))


# quantum n = 30 in units where hbar = m = omega = 1: classical amplitude sqrt(2n+1)
def hermite_function(n, xi):
    psi_prev = np.pi**-0.25 * np.exp(-xi**2 / 2)
    if n == 0:
        return psi_prev
    psi = np.sqrt(2.0) * xi * psi_prev
    for k in range(1, n):
        psi_prev, psi = psi, np.sqrt(2.0 / (k + 1)) * xi * psi - np.sqrt(k / (k + 1)) * psi_prev
    return psi


n_q = 30
A_q = np.sqrt(2 * n_q + 1)

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
(ax_t, ax_ps), (ax_h, ax_q) = axes

# (a) random instants on x(t)
tt = np.linspace(0, T, 400)
ax_t.plot(tt, A * np.cos(omega * tt), color=SERIES[0])
ts = t[:25]
ax_t.plot(ts, A * np.cos(omega * ts), "o", ms=3, color=SERIES[1])
ax_t.set_xlabel("$t/T$"); ax_t.set_ylabel("$x/A$")
ax_t.set_title("(a) look at random instants", fontsize=9)

# (b) the same instants as points in phase space, and a thin energy shell
pm = m * omega * A
th = np.linspace(0, 2 * np.pi, 400)
dE = 0.40                                  # exaggerated shell thickness, for the picture only
s_out = np.sqrt(1 + dE)
ax_ps.fill(s_out * np.cos(th), s_out * np.sin(th), color=SERIES[0], alpha=0.12, lw=0)
ax_ps.fill(np.cos(th), np.sin(th), color="white", lw=0)
ax_ps.plot(np.cos(th), np.sin(th), color=SERIES[0], lw=1.0)
ax_ps.plot(x[:300] / A, p[:300] / pm, ".", ms=1.8, color=SERIES[1], zorder=3)
x0, dx = 0.55, 0.08                        # strip between x0 and x0+dx meets the shell twice
ax_ps.axvspan(x0, x0 + dx, color=SERIES[3], alpha=0.25, lw=0)
xs = np.linspace(x0, x0 + dx, 50)          # the two patches where the strip meets the shell (p_+ and p_-)
for sgn in (+1, -1):
    ax_ps.fill_between(xs, sgn * np.sqrt(1 - xs**2), sgn * np.sqrt(s_out**2 - xs**2), color=SERIES[3], lw=0)
ax_ps.set_xlabel("$x/A$"); ax_ps.set_ylabel(r"$p/(m\omega A)$")
ax_ps.set_title("(b) phase space: orbit, samples, shell", fontsize=9)
ax_ps.set_aspect("equal")
ax_ps.set_xlim(-1.35, 1.35); ax_ps.set_ylim(-1.35, 1.35)

# (c) histogram of x
xx = np.linspace(-0.999, 0.999, 800)
ax_h.hist(x, bins=60, range=(-1, 1), density=True, color=SERIES[0], alpha=0.6,
          label="histogram of $x(t)$")
theory_line(ax_h, xx, arcsine_pdf(xx), label="$1/\\pi\\sqrt{A^2-x^2}$")
ax_h.set_ylim(0, 3.5)
ax_h.set_xlabel("$x/A$"); ax_h.set_ylabel("density")
ax_h.set_title("(c) the arcsine density", fontsize=9)
ax_h.legend(fontsize=7, loc="upper center")

# (d) quantum n = 30 at the same energy
xi = np.linspace(-1.25 * A_q, 1.25 * A_q, 2000)
ax_q.plot(xi / A_q, A_q * hermite_function(n_q, xi)**2, color=SERIES[2], lw=1.0,
          label=f"quantum $|\\psi_{{{n_q}}}|^2$")
theory_line(ax_q, xx, arcsine_pdf(xx), label="classical")
ax_q.set_ylim(0, 3.5)
ax_q.set_xlabel("$x/A$")
ax_q.set_title("(d) correspondence principle", fontsize=9)
ax_q.legend(fontsize=7, loc="upper center")
fig.tight_layout()
savefig(fig, "ch01", "oscillator_arcsine")

# chi-square of histogram vs bin-integrated arcsine density (CDF differences)
edges = np.linspace(-1, 1, 61)
counts, _ = np.histogram(x, bins=edges)
F = 0.5 + np.arcsin(edges / A) / np.pi
expected = N * np.diff(F)
chi2 = np.sum((counts - expected)**2 / expected)

save_numbers("ch01", "14_oscillator_arcsine", {
    "OneBOscN": "2\\times10^{5}",
    "OneBOscFracSim": f"{frac_sim:.4f}",
    "OneBOscFracExact": f"{frac_exact:.4f}",
    "OneBOscChi": f"{chi2:.1f}",
    "OneBOscNbins": 60,
    "OneBOscMeanSq": f"{np.mean(x**2):.4f}",
})
