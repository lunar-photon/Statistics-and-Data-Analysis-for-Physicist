"""When the dynamics samples for us: the oscillator's phase-space orbit is the sample space.

Question: if we look at a harmonic oscillator of fixed energy at a random instant, where is it?
A uniform random instant t (equivalently a uniform phase theta = omega t) picks a uniformly
random point of the orbit (x, p); histogramming x gives the arcsine law
P(x) = 1 / (pi sqrt(A^2 - x^2)).  No Markov chain is needed: drawing samples is drawing times.

House-style version of the author's original script (monte_carlo_harmonic_oscillator.py).
Writes figures/ch09/oscillator_phase_space.pdf and results/ch09/01_oscillator_phase_space.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

A, N, NBINS = 1.0, 50_000, 80          # amplitude (units m = omega = 1), samples, histogram bins
rng = rng_for("ch09", "01_oscillator_phase_space")


def arcsine_pdf(x, A=1.0):
    """Exact density of the position at a uniformly random instant."""
    return 1.0 / (np.pi * np.sqrt(A**2 - x**2))


def arcsine_cdf(x, A=1.0):
    return 0.5 + np.arcsin(np.clip(x / A, -1, 1)) / np.pi


# --- the whole sampler: a random instant, then the deterministic motion -------------------
theta = rng.uniform(0.0, 2.0 * np.pi, size=N)   # random phase = omega * (random instant)
x = A * np.cos(theta)                            # where the motion has carried the oscillator
p = -A * np.sin(theta)                           # its momentum at that instant

# --- compare with the exact law, bin by bin (bin probabilities from CDF differences) -------
edges = np.linspace(-A, A, NBINS + 1)
counts, _ = np.histogram(x, bins=edges)
expected = N * np.diff(arcsine_cdf(edges, A))
chi2 = float(((counts - expected) ** 2 / expected).sum())

# time average of x^2 (over random instants) versus the ensemble value A^2/2 of the energy shell
mean_x2 = float(np.mean(x**2))

setup(7.0, 3.2)
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.5]})
show = slice(0, 400)
ax1.plot(np.cos(np.linspace(0, 2 * np.pi, 300)), np.sin(np.linspace(0, 2 * np.pi, 300)),
         color="0.75", lw=1.0, zorder=1)
ax1.scatter(x[show], p[show], s=6, color=SERIES[0], zorder=2, label="random instants")
ax1.set_aspect("equal")
ax1.set_xlabel(r"position $x/A$")
ax1.set_ylabel(r"momentum $p/(m\omega A)$")
ax1.set_title("(a) samples are points of the orbit")
xx = np.linspace(-0.998 * A, 0.998 * A, 800)
ax2.hist(x, bins=edges, density=True, color=SERIES[0], alpha=0.55, label=f"histogram of $x$, $N={N:,}$")
theory_line(ax2, xx, arcsine_pdf(xx, A), label=r"$1/\pi\sqrt{A^2-x^2}$")
ax2.set_ylim(0, 3.2)
ax2.set_xlabel(r"position $x/A$")
ax2.set_ylabel(r"density $P(x)$")
ax2.set_title("(b) the arcsine law")
ax2.legend(loc="upper center")
fig.tight_layout()
savefig(fig, "ch09", "oscillator_phase_space")

save_numbers("ch09", "01_oscillator_phase_space", {
    "NineAOscN": f"{N:,}".replace(",", r"\,"),
    "NineAOscChi": round(chi2, 1),
    "NineAOscBins": NBINS,
    "NineAOscMeanSq": round(mean_x2, 4),
})
print(f"chi2 = {chi2:.1f} over {NBINS} bins, <x^2> = {mean_x2:.4f} (exact 0.5)")
