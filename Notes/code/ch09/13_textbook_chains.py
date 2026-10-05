"""Two textbook Metropolis runs, reproduced: Wasserman's Cauchy chains and Kruschke's coin.

Question 1 (Wasserman Ex. 24.10, Fig. 24.2): random-walk Metropolis for the Cauchy density
f(x) = 1/(pi (1+x^2)) with N(x, b^2) proposals, b = 0.1, 1, 10, 1000 steps each.  What do the
traces look like, and how good is the histogram?
Question 2 (Kruschke Sec. 7.3.1, Fig. 7.4): posterior of a coin bias theta with a flat prior and
z = 14 heads in N = 20 flips, Metropolis from theta = 0.01 with proposal sd 0.02, 0.2, 2 and
50 000 steps.  Do we get Kruschke's acceptance rates and effective sizes?
Writes figures/ch09/cauchy_chains.pdf, figures/ch09/coin_chains.pdf and
results/ch09/13_textbook_chains.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_mcmc import metropolis, ess, tau_int

rng = rng_for("ch09", "13_textbook_chains")
nums = {}

# ---------------- Wasserman: Cauchy target ----------------------------------------------------
logp_cauchy = lambda x: -np.log1p(x[0] ** 2)        # log f up to the constant -log(pi)
bs = [0.1, 1.0, 10.0]
short, longrun = {}, {}
for b in bs:
    ch, _, a = metropolis(logp_cauchy, [0.0], 1000, b, rng)
    short[b] = (ch[:, 0], a)
    chl, _, al = metropolis(logp_cauchy, [0.0], 200_000, b, rng)
    longrun[b] = (chl[1000:, 0], al)
key = {0.1: "Small", 1.0: "One", 10.0: "Big"}

for b in bs:
    x, a = short[b]
    xl, al = longrun[b]
    nums[f"NineBCauAcc{key[b]}"] = al
    nums[f"NineBCauHalf{key[b]}"] = np.mean(np.abs(x) < 1)      # should be 0.5
    nums[f"NineBCauHalfLong{key[b]}"] = np.mean(np.abs(xl) < 1)
    nums[f"NineBCauTau{key[b]}"] = tau_int(np.abs(xl) < 1)[0]   # tau of the indicator |x|<1

setup(7.0, 4.6)
fig = plt.figure()
gs = fig.add_gridspec(3, 2, width_ratios=[2.4, 1])
for k, b in enumerate(bs):
    ax = fig.add_subplot(gs[k, 0])
    x, a = short[b]
    ax.plot(x, color=SERIES[k], lw=0.6)
    ax.set_ylim(-20, 20)
    ax.set_ylabel(r"$x_t$")
    ax.set_title(rf"$b={b:g}$: acceptance {a:.2f}", fontsize=9, loc="left")
    if k == 2:
        ax.set_xlabel("step $t$")
    axh = fig.add_subplot(gs[k, 1])
    axh.hist(x, bins=60, range=(-8, 8), density=True, color=SERIES[k], alpha=0.55)
    xx = np.linspace(-8, 8, 400)
    theory_line(axh, xx, 1 / (np.pi * (1 + xx**2)), label="Cauchy")
    axh.set_ylim(0, 0.75)
    axh.set_yticks([0, 0.3, 0.6])
    if k == 0:
        axh.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "cauchy_chains")

# ---------------- Kruschke: coin bias -----------------------------------------------------------
z, Nf = 14, 20


def logp_coin(x):
    th = x[0]
    if not 0.0 < th < 1.0:
        return -np.inf                                   # prior and likelihood vanish outside
    return z * np.log(th) + (Nf - z) * np.log1p(-th)     # flat Beta(1,1) prior


sds = [0.02, 0.2, 2.0]
kr = {0.02: (0.936, 468.9), 0.2: (0.495, 11723.9), 2.0: (0.0624, 2113.4)}   # Kruschke Fig. 7.4
names = {0.02: "Small", 0.2: "Mid", 2.0: "Big"}
coin = {}
for sd in sds:
    ch, _, a = metropolis(logp_coin, [0.01], 50_000, sd, rng)
    th = ch[:, 0]
    coin[sd] = (th, a)
    nums[f"NineBCoinAcc{names[sd]}"] = a
    nums[f"NineBCoinEss{names[sd]}"] = f"{ess(th[1:]):.0f}"
    nums[f"NineBCoinKrAcc{names[sd]}"] = kr[sd][0]
    nums[f"NineBCoinKrEss{names[sd]}"] = f"{kr[sd][1]:.0f}"
    nums[f"NineBCoinMean{names[sd]}"] = th[500:].mean()
post = stats.beta(z + 1, Nf - z + 1)
nums["NineBCoinMeanExact"] = post.mean()
lo, hi = np.percentile(coin[0.2][0][500:], [2.5, 97.5])
nums["NineBCoinLo"], nums["NineBCoinHi"] = lo, hi
nums["NineBCoinLoExact"], nums["NineBCoinHiExact"] = post.ppf(0.025), post.ppf(0.975)
save_numbers("ch09", "13_textbook_chains", nums)

setup(7.0, 4.4)
fig, axes = plt.subplots(2, 3, sharey="row")
tt = np.linspace(0, 1, 400)
for j, sd in enumerate(sds):
    th, a = coin[sd]
    axes[0, j].plot(th[:500], np.arange(500), color=SERIES[j], lw=0.6, marker=".", ms=1.5)
    axes[0, j].set_xlim(0, 1)
    axes[0, j].set_title(rf"proposal sd {sd:g}: acc. {a:.2f}", fontsize=9)
    axes[0, j].set_xlabel(r"$\theta$")
    axes[1, j].hist(th[500:], bins=60, range=(0, 1), density=True, color=SERIES[j], alpha=0.55)
    theory_line(axes[1, j], tt, post.pdf(tt), label="Beta(15, 7)")
    axes[1, j].set_xlabel(r"$\theta$")
    axes[1, j].text(0.03, 0.9, f"ESS {ess(th[1:]):.0f}", transform=axes[1, j].transAxes,
                    fontsize=8)
axes[0, 0].set_ylabel("step (first 500)")
axes[1, 0].set_ylabel("density")
axes[1, 0].legend(fontsize=7, loc="center left")
fig.tight_layout()
savefig(fig, "ch09", "coin_chains")
