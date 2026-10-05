"""Time averages along one chain: they converge, but correlated steps carry less information.

Question 1: for a two-state chain with a = P(1->2), b = P(2->1), second eigenvalue
lambda = 1 - a - b, the fraction of time spent in state 1 over N steps estimates pi_1.  Is its
variance pi_1 pi_2 tau / N with tau = (1 + lambda)/(1 - lambda), as the correlation sum predicts?
Question 2 (the tilted die): a six-faced die whose next face is, with probability eps, a
neighbour (+-1 around the ring) of the current one and otherwise a fresh fair throw.  The
time average of the face value still converges to 3.5; how much slower than for a fair die?
Writes figures/ch09/ergodic_average.pdf and results/ch09/06_ergodic_average.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_markov import stationary, step_many

rng = rng_for("ch09", "06_ergodic_average")
R, N = 4000, 1000                                   # independent chains, steps per chain


def run_many(P, n, start_from_pi=True):
    """R chains of n steps, started in the stationary distribution; returns array (R, n)."""
    pi = stationary(P)
    s = rng.choice(P.shape[0], size=R, p=pi)
    out = np.empty((R, n), dtype=np.int8)
    for t in range(n):
        out[:, t] = s
        s = step_many(s, P, rng)
    return out


# --- two-state chains: anticorrelated (Wasserman's weather) and sticky -----------------------
two = {"weather": (0.6, 0.8), "sticky": (0.1, 0.2)}
res = {}
for name, (a, b) in two.items():
    P = np.array([[1 - a, a], [b, 1 - b]])
    lam = 1 - a - b
    pi1 = b / (a + b)
    tau = (1 + lam) / (1 - lam)
    frac = (run_many(P, N) == 0).mean(axis=1)
    res[name] = dict(lam=lam, tau=tau, pred=pi1 * (1 - pi1) * tau / N, iid=pi1 * (1 - pi1) / N,
                     meas=frac.var(ddof=1), mean=frac.mean(), pi1=pi1)

# --- the tilted die ----------------------------------------------------------------------------
faces = np.arange(1, 7)


def die_matrix(eps):
    ring = np.zeros((6, 6))
    for i in range(6):
        ring[i, (i + 1) % 6] = ring[i, (i - 1) % 6] = 0.5
    return (1 - eps) * np.full((6, 6), 1 / 6) + eps * ring


def tau_of(P, g, kmax=4000):
    """Integrated autocorrelation time 1 + 2 sum_k rho_k of g(X) for the stationary chain."""
    pi = stationary(P)
    gc = g - pi @ g
    var = pi @ gc**2
    v, s = gc.copy(), 0.0
    for _ in range(kmax):
        v = P @ v                                   # v = E[g(X_k) - mean | X_0]
        s += (pi * gc) @ v
    return 1 + 2 * s / var


Ns = np.unique(np.logspace(1, 3, 15).astype(int))
die = {}
for eps in (0.0, 0.9):
    P = die_matrix(eps)
    x = faces[run_many(P, Ns[-1])]
    err = [np.sqrt(np.mean((x[:, :n].mean(axis=1) - 3.5) ** 2)) for n in Ns]
    die[eps] = dict(err=np.array(err), tau=tau_of(P, faces.astype(float)),
                    lam=np.sort(np.abs(np.linalg.eigvals(P)))[-2])
sig = np.sqrt(35 / 12)                               # sd of a fair die

setup(7.0, 3.1)
fig, (ax1, ax2) = plt.subplots(1, 2)
names = list(two)
xpos = np.arange(len(names))
ax1.bar(xpos - 0.2, [res[k]["meas"] * N for k in names], 0.35, color=SERIES[0], alpha=0.7, label="measured, 4000 chains")
ax1.plot(xpos + 0.2, [res[k]["pred"] * N for k in names], "k_", ms=22, mew=2, label=r"$\pi_1\pi_2\,\tau$")
ax1.plot(xpos + 0.2, [res[k]["iid"] * N for k in names], "_", color=SERIES[1], ms=22, mew=2, label=r"$\pi_1\pi_2$ (if independent)")
ax1.set_xticks(xpos, [f"{k}\n$\\lambda={res[k]['lam']:+.1f}$" for k in names])
ax1.set_ylabel(r"$N\times\mathrm{Var}(\mathrm{time\ fraction})$")
ax1.set_title("(a) two-state chains, $N=1000$")
ax1.legend(fontsize=7, loc="upper left")
for c, eps in zip(SERIES, die):
    ax2.loglog(Ns, die[eps]["err"], "o", ms=3, color=c, label=fr"tilted die $\epsilon={eps}$")
    theory_line(ax2, Ns, sig * np.sqrt(die[eps]["tau"] / Ns), label=None)
ax2.set_xlabel("number of throws $N$")
ax2.set_ylabel("rms error of the mean face")
ax2.set_title(r"(b) error $\sigma\sqrt{\tau/N}$ (dashed)")
ax2.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "ergodic_average")

# --- one long weather history: running fraction of rainy days, and Kac's mean return time -----
from lib_markov import walk
P3 = np.array([[0.6, 0.2, 0.2], [0.3, 0.7, 0.0], [0.6, 0.0, 0.4]])
pi3 = stationary(P3)
Nlong = 1_000_000
hist = walk(P3, 2, Nlong - 1, rng)
rain = (hist == 2).astype(float)
running = np.cumsum(rain) / np.arange(1, Nlong + 1)
tau_rain = tau_of(P3, (np.arange(3) == 2).astype(float))
gaps = np.diff(np.flatnonzero(hist == 2))
setup(6.0, 3.0)
fig, ax = plt.subplots()
nn = np.unique(np.logspace(0, np.log10(Nlong), 3000).astype(int))   # thin to ~3000 log-spaced points
running_plot = running[nn - 1]
ax.semilogx(nn, running_plot, color=SERIES[0], lw=1.0, label="fraction of rainy days so far")
band = 2 * np.sqrt(pi3[2] * (1 - pi3[2]) * tau_rain / nn)
ax.fill_between(nn, pi3[2] - band, pi3[2] + band, color=SERIES[0], alpha=0.15, lw=0,
                label=r"$\pi_R\pm2\sqrt{\pi_R(1-\pi_R)\tau/N}$")
ax.axhline(pi3[2], color="k", ls="--", lw=1.2)
ax.set_ylim(0, 1.05)
ax.set_xlabel("days $N$")
ax.set_ylabel("running fraction")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
savefig(fig, "ch09", "ergodic_running")

save_numbers("ch09", "06_ergodic_average", {
    "NineAErgRainTau": round(tau_rain, 3), "NineAErgRainFrac": round(running[-1], 4),
    "NineAErgRainGap": round(float(gaps.mean()), 3), "NineAErgRainErr": f"{np.sqrt(pi3[2]*(1-pi3[2])*tau_rain/Nlong):.1e}".replace("e-0", r"\times10^{-") + "}",
    "NineAErgLong": r"10^6",
    "NineAErgWeatherTau": round(res["weather"]["tau"], 3), "NineAErgStickyTau": round(res["sticky"]["tau"], 3),
    "NineAErgWeatherMeas": round(res["weather"]["meas"] * N, 4), "NineAErgWeatherPred": round(res["weather"]["pred"] * N, 4),
    "NineAErgStickyMeas": round(res["sticky"]["meas"] * N, 3), "NineAErgStickyPred": round(res["sticky"]["pred"] * N, 3),
    "NineAErgStickyMean": round(res["sticky"]["mean"], 4),
    "NineAErgDieTau": round(die[0.9]["tau"], 2), "NineAErgDieLam": round(die[0.9]["lam"], 2),
    "NineAErgDieErr": round(die[0.9]["err"][-1], 4), "NineAErgFairErr": round(die[0.0]["err"][-1], 4),
    "NineAErgChains": R,
})
for k, v in res.items():
    print(k, {a: round(b, 5) for a, b in v.items()})
print("rain tau", tau_rain, running[-1], gaps.mean())
print("die tau", die[0.9]["tau"], "lam", die[0.9]["lam"], "err1000", die[0.9]["err"][-1], die[0.0]["err"][-1])
