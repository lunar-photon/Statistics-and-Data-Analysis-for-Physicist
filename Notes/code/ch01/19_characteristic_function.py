"""Moment generating and characteristic functions: moments by differentiation, sums by products.

Questions:
  * Exp(1) has mgf psi(t) = 1/(1-t).  Do finite-difference derivatives of the EMPIRICAL mgf
    (average of e^{tX} over samples) at t = 0 give E[X] = 1 and E[X^2] = 2?
  * U ~ Uniform(-1,1) has cf phi(t) = sin(t)/t.  For S = U1 + U2 + U3 (independent) the cf is
    (sin t / t)^3.  Does the empirical cf (average of e^{itS}) agree, and does the numerical
    inverse Fourier transform of (sin t/t)^n return the density of the sum (triangle for n=2)?
  * Cauchy: the mgf does not exist, but the cf exp(-|t|) does.
Writes: figures/ch01/characteristic_function.pdf, results/ch01/19_characteristic_function.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "19_characteristic_function")
setup(7.0, 3.0)

# ---- moments of Exp(1) from the empirical mgf ---------------------------------------
X = rng.exponential(1.0, size=1_000_000)
h = 0.01
psi = lambda t: np.mean(np.exp(t * X))
m1 = (psi(h) - psi(-h)) / (2 * h)
m2 = (psi(h) - 2 * psi(0.0) + psi(-h)) / h**2

# ---- empirical cf of sums of uniforms ---------------------------------------------
t = np.linspace(-15, 15, 601)
U = rng.uniform(-1, 1, size=(3, 100_000))
sinc = np.sinc(t / np.pi)                       # numpy sinc(x) = sin(pi x)/(pi x)
emp = {}
for n in (1, 2, 3):
    S = U[:n].sum(axis=0)
    emp[n] = np.array([np.mean(np.cos(tt * S)) for tt in t])   # imaginary part is ~0 (symmetric)
maxdev = max(np.max(np.abs(emp[n] - sinc**n)) for n in (1, 2, 3))

# Cauchy empirical cf
Cy = rng.standard_cauchy(size=100_000)
emp_c = np.array([np.mean(np.cos(tt * Cy)) for tt in t])

# ---- invert (sin t / t)^2 numerically: p(s) = (1/2pi) int phi(t) e^{-its} dt ------------
tg = np.linspace(-400, 400, 160_001)
dt = tg[1] - tg[0]
phi2 = np.sinc(tg / np.pi)**2
s = np.linspace(-2.5, 2.5, 201)
p2 = np.array([np.sum(phi2 * np.cos(tg * ss)) * dt / (2 * np.pi) for ss in s])
tri = np.clip(1 - np.abs(s) / 2, 0, None) / 2
maxdev_inv = np.max(np.abs(p2 - tri))

fig, axes = plt.subplots(1, 2)
ax = axes[0]
for n, c in zip((1, 2, 3), SERIES):
    ax.plot(t, emp[n], color=c, lw=1.4, label=f"empirical, $n={n}$")
    ax.plot(t, sinc**n, color="k", ls="--", lw=0.9)
ax.plot(t, emp_c, color=SERIES[3], lw=1.2, label="empirical, Cauchy")
ax.plot(t, np.exp(-np.abs(t)), color="k", ls="--", lw=0.9, label="exact")
ax.set_xlabel("$t$"); ax.set_ylabel("$\\phi(t)$")
ax.legend(fontsize=7, loc="upper right")
ax.set_xlim(-15, 15)

ax = axes[1]
S2 = U[:2].sum(axis=0)
ax.hist(S2, bins=60, range=(-2.5, 2.5), density=True, color=SERIES[1], alpha=0.5,
        label="histogram of $U_1+U_2$")
ax.plot(s, p2, color=SERIES[0], lw=1.6, label="inverse FT of $(\\sin t/t)^2$")
theory_line(ax, s, tri, label="triangle $(1-|s|/2)/2$")
ax.set_xlabel("$s$"); ax.set_ylabel("density")
ax.legend(fontsize=7, loc="upper right")
ax.set_ylim(0, 0.75)
fig.tight_layout()
savefig(fig, "ch01", "characteristic_function")

save_numbers("ch01", "19_characteristic_function", {
    "OneBMgfMone": f"{m1:.4f}", "OneBMgfMtwo": f"{m2:.4f}",
    "OneBCfMaxDev": f"{maxdev:.4f}",
    "OneBCfInvDev": f"{maxdev_inv:.4f}",
    "OneBCfN": "10^5",
})
