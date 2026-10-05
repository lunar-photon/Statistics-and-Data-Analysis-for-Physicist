"""A curved (banana-shaped) target: what a proposal covariance can and cannot fix, and a
cross-check of our sampler against emcee.

Question: p(t1, t2) ∝ exp[-t1^2/2 - (t2 - t1^2)^2 / (2 * 0.3^2)].  How well do (a) round
Metropolis steps, (b) steps shaped by the covariance of a pilot run, (c) round steps in the
straightened variables (t1, u = t2 - t1^2) sample it, and does the affine-invariant ensemble
sampler emcee agree with our chains?  Exact answers: t1 ~ N(0,1), t2 = t1^2 + 0.3 z, so
E t2 = 1 and Var t2 = 2 + 0.09.
Writes figures/ch09/banana_paths.pdf, figures/ch09/banana_corner.pdf,
results/ch09/15_banana.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
import emcee
from common import setup, savefig, save_numbers, rng_for, SERIES
from lib_mcmc import metropolis, tau_int, corner

SIG = 0.3
rng = rng_for("ch09", "15_banana")


def logp(t):
    return -0.5 * t[0] ** 2 - 0.5 * ((t[1] - t[0] ** 2) / SIG) ** 2


def logp_straight(v):
    """The same density in (t1, u = t2 - t1^2); the Jacobian of the map is 1."""
    return -0.5 * v[0] ** 2 - 0.5 * (v[1] / SIG) ** 2


n = 200_000
burn = 2000
x0 = np.array([0.0, 1.0])

# (a) round steps: best of a short scan
best = None
for s in np.geomspace(0.1, 2.0, 10):
    ch, _, a = metropolis(logp, x0, 20_000, s, rng)
    t = max(tau_int(ch[burn:, i])[0] for i in range(2))
    best = min(best, (t, s)) if best else (t, s)
ch_a, _, acc_a = metropolis(logp, x0, n, best[1], rng)
# (b) pilot run -> covariance -> shaped proposal with the 2.38^2/D scaling
pilot, _, _ = metropolis(logp, x0, 20_000, best[1], rng)
Cpil = np.cov(pilot[burn:].T)
ch_b, _, acc_b = metropolis(logp, x0, n, 2.38 / np.sqrt(2), rng, cov=Cpil)
# (c) straightened variables, mapped back
ch_c_u, _, acc_c = metropolis(logp_straight, [0.0, 0.0], n, 2.38 / np.sqrt(2), rng,
                              cov=np.diag([1.0, SIG**2]))
ch_c = np.column_stack([ch_c_u[:, 0], ch_c_u[:, 1] + ch_c_u[:, 0] ** 2])

# emcee: 32 walkers started in a small ball
nw, nsteps = 32, 20_000
p0 = x0 + 0.1 * rng.standard_normal((nw, 2))
sampler = emcee.EnsembleSampler(nw, 2, logp)
sampler.random_state = np.random.RandomState(int(rng.integers(2**31))).get_state()  # reproducible
sampler.run_mcmc(p0, nsteps, progress=False)
tau_em = sampler.get_autocorr_time(tol=0)
ch_em = sampler.get_chain(discard=2000, flat=True)

# repeat each sampler on fresh random numbers (same proposals) to see how much Var t2 scatters
# from run to run; one run of a slowly mixing chain says little about a long-tailed variance
R = 12
r_rep = rng_for("ch09", "15_banana", stream=1)
rep = {k: np.empty(R) for k in ("Round", "Shaped", "Straight", "Em")}
for j in range(R):
    c, _, _ = metropolis(logp, x0, n, best[1], r_rep)
    rep["Round"][j] = c[burn:, 1].var()
    c, _, _ = metropolis(logp, x0, n, 2.38 / np.sqrt(2), r_rep, cov=Cpil)
    rep["Shaped"][j] = c[burn:, 1].var()
    c, _, _ = metropolis(logp_straight, [0.0, 0.0], n, 2.38 / np.sqrt(2), r_rep,
                         cov=np.diag([1.0, SIG**2]))
    rep["Straight"][j] = (c[burn:, 1] + c[burn:, 0] ** 2).var()
    s_rep = emcee.EnsembleSampler(nw, 2, logp)
    s_rep.random_state = np.random.RandomState(int(r_rep.integers(2**31))).get_state()
    s_rep.run_mcmc(x0 + 0.1 * r_rep.standard_normal((nw, 2)), nsteps, progress=False)
    rep["Em"][j] = s_rep.get_chain(discard=2000, flat=True)[:, 1].var()

nums = {"NineBBaSig": SIG, "NineBBaN": f"{n:,}".replace(",", r"\,")}
nums["NineBBaRepN"] = R
for tag, v in rep.items():
    nums[f"NineBBa{tag}VarRepMean"] = f"{v.mean():.2f}"           # average over the R runs
    nums[f"NineBBa{tag}VarRepSd"] = f"{v.std(ddof=1):.2f}"        # scatter of one run
    nums[f"NineBBa{tag}VarRepErr"] = f"{v.std(ddof=1) / np.sqrt(R):.2f}"  # error of the average
for tag, ch, a in (("Round", ch_a, acc_a), ("Shaped", ch_b, acc_b), ("Straight", ch_c, acc_c)):
    post = ch[burn:]
    nums[f"NineBBa{tag}Acc"] = a
    nums[f"NineBBa{tag}Tau"] = max(tau_int(post[:, i])[0] for i in range(2))
    nums[f"NineBBa{tag}MeanTwo"] = post[:, 1].mean()
    nums[f"NineBBa{tag}VarTwo"] = post[:, 1].var()
nums["NineBBaRoundStep"] = best[1]
nums["NineBBaPilotCorr"] = Cpil[0, 1] / np.sqrt(Cpil[0, 0] * Cpil[1, 1])
nums["NineBBaEmAcc"] = np.mean(sampler.acceptance_fraction)
nums["NineBBaEmTau"] = tau_em.max()
nums["NineBBaEmWalkers"] = nw
nums["NineBBaEmSteps"] = nsteps
nums["NineBBaEmMeanTwo"] = ch_em[:, 1].mean()
nums["NineBBaEmVarTwo"] = ch_em[:, 1].var()
nums["NineBBaVarTwoExact"] = 2 + SIG**2
save_numbers("ch09", "15_banana", nums)

# --- figure 1: paths -----------------------------------------------------------------------------
setup(7.2, 2.7)
fig, axes = plt.subplots(1, 3, sharey=True)
g1 = np.linspace(-3.2, 3.2, 300)
g2 = np.linspace(-1.5, 9, 300)
G1, G2 = np.meshgrid(g1, g2)
Z = np.exp(-0.5 * G1**2 - 0.5 * ((G2 - G1**2) / SIG) ** 2)
lev = [np.exp(-0.5 * 2.45**2), np.exp(-0.5 * 1.52**2)]       # ~95% and ~68% for a 2-D Gaussian
for ax, ch, c, title in zip(axes, (ch_a, ch_b, ch_c), SERIES,
                            ("(a) round steps", "(b) pilot-covariance steps",
                             "(c) round steps, straightened")):
    ax.contour(G1, G2, Z, levels=lev, colors="0.55", linewidths=0.8)
    ax.plot(ch[burn:burn + 600, 0], ch[burn:burn + 600, 1], "-o", color=c, lw=0.4, ms=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_xlim(-3.2, 3.2)
axes[0].set_ylabel(r"$\theta_2$")
axes[0].set_ylim(-1.5, 9)
fig.tight_layout()
savefig(fig, "ch09", "banana_paths")

# --- figure 2: corner plot, our sampler against emcee ---------------------------------------------
setup()
ranges = [(-3.5, 3.5), (-1.2, 9)]
fig = corner(ch_c[burn:], [r"$\theta_1$", r"$\theta_2$"], color=SERIES[0], ranges=ranges,
             label="our Metropolis (straightened)")
fig = corner(ch_em, [r"$\theta_1$", r"$\theta_2$"], color=SERIES[1], ranges=ranges, fig=fig,
             label="emcee")
fig.set_size_inches(4.6, 4.2)
fig.tight_layout()
h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", fontsize=8, frameon=False)
savefig(fig, "ch09", "banana_corner")
