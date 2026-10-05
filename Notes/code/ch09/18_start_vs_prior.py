"""What a chain forgets and what it does not: the starting point versus the prior.

Question: counts n_1..n_N from a Poisson process with true rate lambda = 4 (events per unit
exposure).  Two priors for lambda:
  A  flat on lambda > 0                          (posterior Gamma(S + 1, N), S = sum n_i)
  B  confident and wrong: ln lambda ~ N(ln 10, 0.2^2)
For N = 3 and N = 300 counts we run Metropolis--Hastings (multiplicative steps, Hastings factor
y/x) from four starts spread over four decades, lambda_0 = 0.05, 1, 40, 200.
(i) Do the four chains agree with each other (start forgotten)?  (ii) Do the answers under A and
B agree with each other (prior forgotten)?  (iii) Does a ten times longer chain under B with
N = 3 move towards the truth?
Writes figures/ch09/start_vs_prior.pdf and results/ch09/18_start_vs_prior.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_mcmc import mh, gelman_rubin

rng = rng_for("ch09", "18_start_vs_prior")
LAM_TRUE, MU_B, SIG_B = 4.0, np.log(10.0), 0.2
starts = [0.05, 1.0, 40.0, 200.0]
counts = {3: rng.poisson(LAM_TRUE, 3), 300: rng.poisson(LAM_TRUE, 300)}


def log_prior(lam, prior):
    if prior == "A":
        return 0.0
    return -np.log(lam) - 0.5 * ((np.log(lam) - MU_B) / SIG_B) ** 2    # log-normal density


def make_logpost(N, prior):
    S = counts[N].sum()
    def logp(x):
        lam = x[0]
        if lam <= 0:
            return -np.inf
        return S * np.log(lam) - N * lam + log_prior(lam, prior)       # Poisson likelihood x prior
    return logp


def make_proposal(s):
    """y = x e^{s Z}: a random walk in ln(lambda).  q(y|x) = 1/(y s sqrt(2 pi)) e^{-(ln y - ln x)^2/2s^2},
    so q(x|y)/q(y|x) = y/x and log q(a, b) = -ln b + (terms symmetric in a, b)."""
    propose = lambda x, rng: x * np.exp(s * rng.standard_normal(1))
    log_q = lambda a, b: -np.log(b[0])
    return propose, log_q


def exact(N, prior, grid):
    lp = np.array([make_logpost(N, prior)([g]) for g in grid])
    p = np.exp(lp - lp.max())
    return p / np.trapezoid(p, grid)


steps = {3: 0.6, 300: 0.12}
n, burn = 20_000, 2_000
res, nums = {}, {}
for N in (3, 300):
    nums[f"NineBSp{'Few' if N == 3 else 'Many'}Sum"] = int(counts[N].sum())
    for prior in "AB":
        logp = make_logpost(N, prior)
        prop, lq = make_proposal(steps[N])
        chains = np.array([mh(logp, [x0], n, prop, lq, rng)[0][:, 0] for x0 in starts])
        res[N, prior] = chains
        post = chains[:, burn:]
        tag = f"{'Few' if N == 3 else 'Many'}{prior}"
        nums[f"NineBSp{tag}Mean"] = post.mean()
        nums[f"NineBSp{tag}Sd"] = post.std()
        nums[f"NineBSp{tag}Rhat"] = f"{gelman_rubin(post):.3f}"
        nums[f"NineBSp{tag}Spread"] = np.ptp(post.mean(axis=1))      # chain-to-chain spread of means
        grid = np.linspace(1e-3, 40, 40_001)
        pe = exact(N, prior, grid)
        nums[f"NineBSp{tag}MeanExact"] = np.trapezoid(grid * pe, grid)
for N, tag in ((3, "Few"), (300, "Many")):
    dA, dB = nums[f"NineBSp{tag}AMean"], nums[f"NineBSp{tag}BMean"]
    nums[f"NineBSp{tag}Shift"] = (dB - dA) / nums[f"NineBSp{tag}ASd"]   # in posterior sd's
# (iii) ten times longer under the wrong prior with three counts
logp = make_logpost(3, "B")
prop, lq = make_proposal(steps[3])
long = mh(logp, [1.0], 10 * n, prop, lq, rng)[0][burn:, 0]
nums["NineBSpLongMean"] = long.mean()
nums["NineBSpLongN"] = f"{10 * n:,}".replace(",", r"\,")
nums["NineBSpTrue"] = LAM_TRUE
nums["NineBSpN"] = f"{n:,}".replace(",", r"\,")
nums["NineBSpCountsFew"] = ", ".join(str(c) for c in counts[3])
save_numbers("ch09", "18_start_vs_prior", nums)

# ---- figure ---------------------------------------------------------------------------------------------
setup(7.2, 4.6)
fig, axes = plt.subplots(2, 2, gridspec_kw={"width_ratios": [1.3, 1]})
for row, N in enumerate((3, 300)):
    ax = axes[row, 0]
    for c, x0, col in zip(res[N, "B"], starts, SERIES):
        ax.semilogy(np.arange(301), c[:301], color=col, lw=0.8, label=rf"$\lambda_0={x0:g}$")
    ax.axhline(LAM_TRUE, color="k", ls=":", lw=0.8)
    ax.set_ylabel(r"$\lambda_t$")
    ax.set_title(rf"({'ab'[row]}) $N={N}$ counts, prior B: first 300 steps", fontsize=9, loc="left")
    if row == 1:
        ax.set_xlabel("step $t$")
    else:
        ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    axh = axes[row, 1]
    lo, hi = (0, 16) if N == 3 else (3.0, 5.0)
    grid = np.linspace(max(lo, 1e-3), hi, 600)
    for prior, col in (("A", SERIES[0]), ("B", SERIES[1])):
        axh.hist(res[N, prior][:, burn:].ravel(), bins=70, range=(lo, hi), density=True,
                 color=col, alpha=0.45, label=f"prior {prior}")
        theory_line(axh, grid, exact(N, prior, grid), label="exact" if prior == "A" else None)
    axh.axvline(LAM_TRUE, color="k", ls=":", lw=0.9)
    axh.set_title(rf"({'cd'[row]}) posteriors, $N={N}$", fontsize=9, loc="left")
    if row == 1:
        axh.set_xlabel(r"rate $\lambda$")
    axh.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "start_vs_prior")
