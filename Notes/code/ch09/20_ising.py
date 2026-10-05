"""The 2-D Ising model by single-spin-flip Metropolis: a distribution nobody can sample directly.

Question: on an L x L periodic lattice (L = 32, 2^1024 configurations) with energy
E(s) = -J sum_<ij> s_i s_j, sample the canonical distribution P(s) ∝ exp(-E(s)/kT) at a range of
temperatures.  Does the mean |magnetisation| per spin follow Onsager--Yang's exact infinite-lattice
result M(T) = [1 - sinh(2J/kT)^-4]^(1/8) below T_c = 2J / ln(1 + sqrt 2), and does the energy per
spin follow Onsager's exact formula?  What do typical configurations look like?
Writes figures/ch09/ising.pdf and results/ch09/20_ising.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.special import ellipk
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line


@njit(cache=True)
def sweep(s, beta, u):
    """One sweep: visit every site once (in order) and propose to flip its spin.
    dE = 2 J s_i (sum of the 4 neighbours), J = 1.  Accept with min(1, exp(-beta dE)).
    Each single-site update satisfies detailed balance, so each preserves the Boltzmann law."""
    L = s.shape[0]
    acc = 0
    k = 0
    for i in range(L):
        for j in range(L):
            nb = s[(i + 1) % L, j] + s[(i - 1) % L, j] + s[i, (j + 1) % L] + s[i, (j - 1) % L]
            dE = 2.0 * s[i, j] * nb
            if dE <= 0.0 or u[k] < np.exp(-beta * dE):
                s[i, j] = -s[i, j]
                acc += 1
            k += 1
    return acc


def energy_per_spin(s):
    return -np.mean(s * (np.roll(s, 1, 0) + np.roll(s, 1, 1)))      # each bond counted once


def onsager_m(T):
    x = 1.0 - np.sinh(2.0 / T) ** -4.0
    return np.where(x > 0, np.abs(x) ** 0.125, 0.0)


def onsager_u(T):
    """Exact energy per spin of the infinite lattice (J = 1)."""
    K = 1.0 / T
    kap = 2.0 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
    return -1.0 / np.tanh(2 * K) * (1 + 2 / np.pi * (2 * np.tanh(2 * K) ** 2 - 1) * ellipk(kap**2))


rng = rng_for("ch09", "20_ising")
L = 32
Tc = 2.0 / np.log(1 + np.sqrt(2))
temps = np.array([1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.3, 2.4, 2.5, 2.7, 3.0, 3.5])
n_burn, n_meas = 2000, 6000
s = np.ones((L, L), dtype=np.float64)              # start fully ordered, then heat up
m_abs, u_mean, acc_rate, snaps = [], [], [], {}
for T in temps:
    beta = 1.0 / T
    for _ in range(n_burn):
        sweep(s, beta, rng.random(L * L))
    ms, us, acc = [], [], 0
    for t in range(n_meas):
        acc += sweep(s, beta, rng.random(L * L))
        ms.append(abs(s.mean())); us.append(energy_per_spin(s))
    m_abs.append(np.mean(ms)); u_mean.append(np.mean(us)); acc_rate.append(acc / (n_meas * L * L))
    if T in (1.8, 2.3, 3.0):
        snaps[T] = s.copy()
m_abs, u_mean = np.array(m_abs), np.array(u_mean)

i18, i30 = list(temps).index(1.8), list(temps).index(3.0)
nums = {"NineBIsL": L, "NineBIsSites": L * L, "NineBIsTc": Tc,
        "NineBIsBurn": n_burn, "NineBIsMeas": n_meas,
        "NineBIsMLow": m_abs[i18], "NineBIsMLowExact": float(onsager_m(1.8)),
        "NineBIsULow": u_mean[i18], "NineBIsULowExact": float(onsager_u(1.8)),
        "NineBIsMHigh": m_abs[i30], "NineBIsUHigh": u_mean[i30],
        "NineBIsUHighExact": float(onsager_u(3.0)),
        "NineBIsAccLow": acc_rate[i18], "NineBIsAccHigh": acc_rate[i30],
        "NineBIsLogConfigs": L * L * np.log10(2)}
save_numbers("ch09", "20_ising", nums)

setup(7.2, 4.6)
fig = plt.figure()
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
a1 = fig.add_subplot(gs[0, :2]); a2 = fig.add_subplot(gs[0, 2])
TT = np.linspace(1.2, 3.7, 400)
a1.plot(temps, m_abs, "o", color=SERIES[0], ms=4, label=rf"$\langle|m|\rangle$, Metropolis, $L={L}$")
theory_line(a1, TT, onsager_m(TT), label="Onsager--Yang, infinite lattice")
a1.plot(temps, -u_mean / 2, "s", color=SERIES[1], ms=3.5, label=r"$-u/2$, Metropolis")
theory_line(a1, TT, -onsager_u(TT) / 2, label=None)
a1.axvline(Tc, color="0.5", ls=":", lw=0.9)
a1.text(Tc + 0.03, 0.93, r"$T_c$", fontsize=8)
a1.set_xlabel(r"temperature $kT/J$")
a1.set_title("(a) magnetisation and energy per spin", fontsize=9, loc="left")
a1.legend(fontsize=7, loc="lower left")
a2.plot(temps, acc_rate, "o-", color=SERIES[2], ms=3.5, lw=0.8)
a2.set_xlabel(r"$kT/J$"); a2.set_ylabel("flip acceptance")
a2.set_title("(b) acceptance", fontsize=9, loc="left")
for k, T in enumerate((1.8, 2.3, 3.0)):
    ax = fig.add_subplot(gs[1, k])
    ax.imshow(snaps[T], cmap="Greys", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"({'cde'[k]}) $kT/J={T:g}$", fontsize=9)
fig.tight_layout()
savefig(fig, "ch09", "ising")
