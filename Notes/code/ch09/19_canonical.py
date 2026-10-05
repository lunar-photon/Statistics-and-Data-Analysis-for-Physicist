"""Metropolis for what it was invented for: the canonical ensemble rho(x, p) ∝ exp(-H / kT).

Question 1 (harmonic oscillator, m = k = 1): sample (x, p) with Metropolis at temperature kT = 1.
Do x and p come out Gaussian with variance kT, does equipartition <x^2/2> = <p^2/2> = kT/2 hold,
and is the energy exponentially distributed?  Compare with the microcanonical arcsine law at the
single energy E = kT, and check that Boltzmann-weighted energy shells rebuild the Gaussian.
Question 2 (double well V(x) = dV (x^2 - 1)^2, dV = 1): sample exp(-V/kT) for several kT, check
against the exact density by quadrature, and count hops between the wells.  Does the hop rate
follow Arrhenius, rate ∝ exp(-dV / kT)?
Writes figures/ch09/canonical_oscillator.pdf, figures/ch09/double_well.pdf,
results/ch09/19_canonical.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_mcmc import metropolis

rng = rng_for("ch09", "19_canonical")
nums = {}

# ---- 1. harmonic oscillator -------------------------------------------------------------------------
kT = 1.0
H = lambda x, p: 0.5 * p**2 + 0.5 * x**2                     # m = k = 1
log_rho = lambda z: -H(z[0], z[1]) / kT                       # the partition function never appears
n, burn = 200_000, 1_000
ch, _, acc = metropolis(log_rho, [3.0, -3.0], n, 2.4 / np.sqrt(2), rng)
x, p = ch[burn:, 0], ch[burn:, 1]
E = H(x, p)
nums.update({"NineBCaN": f"{n:,}".replace(",", r"\,"), "NineBCaAcc": acc,
             "NineBCaVarX": x.var(), "NineBCaVarP": p.var(),
             "NineBCaKin": np.mean(0.5 * p**2), "NineBCaPot": np.mean(0.5 * x**2),
             "NineBCaE": E.mean(), "NineBCaFracHot": np.mean(E > 2 * kT),
             "NineBCaFracHotExact": np.exp(-2.0), "NineBCaCorrXP": np.corrcoef(x, p)[0, 1]})
# Boltzmann-weighted shells: E ~ Exp(kT) (density of states constant in 1-D), phase uniform
Es = rng.exponential(kT, 200_000)
xs = np.sqrt(2 * Es) * np.cos(2 * np.pi * rng.random(Es.size))
nums["NineBCaShellVar"] = xs.var()

setup(7.2, 2.7)
fig, (a1, a2, a3) = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1.15, 1]})
a1.plot(x[:3000], p[:3000], ".", ms=1.2, color=SERIES[0], alpha=0.6)
th = np.linspace(0, 2 * np.pi, 200)
for e in (0.5, 1.0, 2.0):
    a1.plot(np.sqrt(2 * e) * np.cos(th), np.sqrt(2 * e) * np.sin(th), color="0.4", lw=0.7)
a1.set_aspect("equal"); a1.set_xlim(-4, 4); a1.set_ylim(-4, 4)
a1.set_xlabel("$x$"); a1.set_ylabel("$p$")
a1.set_title("(a) samples and orbits", fontsize=9, loc="left")
xx = np.linspace(-3.5, 3.5, 600)
a2.hist(x, bins=80, range=(-3.5, 3.5), density=True, color=SERIES[0], alpha=0.5,
        label="Metropolis, canonical")
theory_line(a2, xx, np.exp(-xx**2 / (2 * kT)) / np.sqrt(2 * np.pi * kT), label=r"$N(0,kT)$")
A = np.sqrt(2 * kT)
xa = np.linspace(-A + 1e-3, A - 1e-3, 400)
a2.plot(xa, 1 / (np.pi * np.sqrt(A**2 - xa**2)), color=SERIES[1], lw=1.2,
        label=r"fixed $E=kT$ (arcsine)")
a2.set_ylim(0, 1.0); a2.set_xlabel("$x$")
a2.set_title("(b) position", fontsize=9, loc="left")
a2.legend(fontsize=6.5, loc="upper left")
ee = np.linspace(0, 6, 300)
a3.hist(E, bins=60, range=(0, 6), density=True, color=SERIES[2], alpha=0.5, label="Metropolis")
theory_line(a3, ee, np.exp(-ee / kT) / kT, label=r"$e^{-E/kT}/kT$")
a3.set_xlabel("energy $E$")
a3.set_title("(c) energy", fontsize=9, loc="left")
a3.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "canonical_oscillator")

# ---- 2. double well ------------------------------------------------------------------------------------
dV = 1.0
V = lambda x: dV * (x**2 - 1.0) ** 2
temps = [1.0, 0.7, 0.5, 0.4, 0.33, 0.25, 0.2]
n2 = 400_000
grid = np.linspace(-2.2, 2.2, 4001)


def hop_rate(xT):
    """Hops per 10^5 steps between the wells x < -0.5 and x > 0.5 (the gap gives hysteresis)."""
    side = np.where(xT > 0.5, 1, np.where(xT < -0.5, -1, 0))
    s = side[side != 0]
    return np.count_nonzero(np.diff(s)) / xT.size * 1e5


hops, hops_big, wells = [], [], {}
for T in temps:
    logV = lambda z: -V(z[0]) / T
    xT = metropolis(logV, [-1.0], n2, 0.15, rng)[0][1:, 0]      # local steps, like diffusion
    hops.append(hop_rate(xT))
    hops_big.append(hop_rate(metropolis(logV, [-1.0], n2, 0.5, rng)[0][1:, 0]))  # big steps
    wells[T] = xT
    pe = np.exp(-V(grid) / T); pe /= np.trapezoid(pe, grid)
    if T in (1.0, 0.25):
        tag = {1.0: "Hot", 0.25: "Cold"}[T]
        nums[f"NineBDw{tag}XsqMC"] = np.mean(xT**2)
        nums[f"NineBDw{tag}XsqExact"] = np.trapezoid(grid**2 * pe, grid)
        nums[f"NineBDw{tag}Right"] = np.mean(xT > 0)
        nums[f"NineBDw{tag}Hops"] = hops[-1]
hops, hops_big = np.array(hops), np.array(hops_big)
invT = 1 / np.array(temps)
cold = invT >= 2                                   # Kramers regime: barrier at least 2 kT
# overdamped Kramers: rate ∝ (1/kT) exp(-dV/kT), so ln(rate * kT) is linear in 1/kT with slope -dV
slope, icpt = np.polyfit(invT[cold], np.log(hops[cold] / invT[cold]), 1)
slope_big = np.polyfit(invT[cold], np.log(hops_big[cold] / invT[cold]), 1)[0]
nums.update({"NineBDwSlope": slope, "NineBDwSlopeBig": slope_big, "NineBDwBarrier": dV,
             "NineBDwN": f"{n2:,}".replace(",", r"\,"),
             "NineBDwVBig": V(0.6), "NineBDwHopsColdBig": hops_big[temps.index(0.25)]})
save_numbers("ch09", "19_canonical", nums)

setup(7.2, 2.7)
fig, (b1, b2, b3) = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 1.3, 1]})
for T, col in zip((1.0, 0.4, 0.25), SERIES):
    pe = np.exp(-V(grid) / T); pe /= np.trapezoid(pe, grid)
    b1.hist(wells[T], bins=90, range=(-2.2, 2.2), density=True, histtype="step", color=col,
            lw=1.1, label=f"$kT={T:g}$")
    theory_line(b1, grid, pe, label="exact" if T == 0.25 else None)
b1.set_xlabel("$x$"); b1.set_title(r"(a) $e^{-V/kT}$, three temperatures", fontsize=9, loc="left")
b1.set_ylim(0, 1.5)
b1.legend(fontsize=6.5, loc="upper center", ncol=2, handlelength=1.5, columnspacing=1.0)
b2.plot(np.arange(100_000), wells[0.25][:100_000], color=SERIES[2], lw=0.4)
b2.set_xlabel("step"); b2.set_ylabel("$x_t$")
b2.set_xticks([0, 50000, 100000]); b2.set_xticklabels(["0", "50k", "100k"])
b2.set_title(r"(b) $kT=0.25$: rare hops", fontsize=9, loc="left")
b3.semilogy(invT, hops / invT, "o", color=SERIES[0], ms=4, label="steps 0.15")
b3.semilogy(invT, hops_big / invT, "s", color=SERIES[1], ms=3.5, label="steps 0.5")
ii = np.linspace(1.8, 5.2, 50)
theory_line(b3, ii, np.exp(icpt + slope * ii), label=f"slope {slope:.2f}")
b3.set_xlabel(r"$\Delta V/kT$"); b3.set_ylabel(r"hops per $10^5$ steps $\times\, kT$")
b3.set_title("(c) Kramers–Arrhenius", fontsize=9, loc="left")
b3.legend(fontsize=6.5, loc="lower left")
fig.tight_layout()
savefig(fig, "ch09", "double_well")
