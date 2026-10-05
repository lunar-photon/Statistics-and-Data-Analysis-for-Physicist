"""28_sampling_evidence.py -- sampling without the denominator, and sampling for it.

Question 1: can we explore the bump posterior of 22_line_or_bump.py knowing
only likelihood x prior, never its normalisation?  A Metropolis random walk
that accepts a step with probability min(1, ratio of posteriors) does it; the
ratio is all it ever uses (a preview of the Markov chains of Chapter 9).

Question 2: the evidence itself.  Two routes from samples:
  * Savage--Dickey: for nested models, B_01 = p(A = 0 | data, M1) / pi(A = 0),
    read off the chain for the no-bump data;
  * a cross-check of the chain with the ensemble sampler emcee;
  * nested sampling (Skilling): live points climb nested likelihood contours,
    each step shrinking the enclosed prior mass X by about exp(-1/N_live), and
    Z = integral of L dX.  Run from scratch on the (A, mu) problem with the
    line integrated out exactly, so that Z is the Bayes factor B_10.

Writes: figures/ch05/mh_bump.pdf, figures/ch05/nested_sampling.pdf,
        results/ch05/28_sampling_evidence.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import lib_bump as lb

setup()
data = {"bump": lb.make_data(rng_for("ch05", "22_line_or_bump", 0), True),
        "flat": lb.make_data(rng_for("ch05", "22_line_or_bump", 1), False)}
out = {}


# ------------------------------------------------------------------ 1. Metropolis on ratios only
def metropolis(y, rng, n_steps=1_000_000, step=(0.25, 0.04, 0.35, 0.12)):
    ab = np.linalg.lstsq(lb.DESIGN, y, rcond=None)[0]
    theta = np.array([ab[0], ab[1], 1.0, 5.0])                   # start: best line, small bump mid-range
    lp = lb.ln_post_unnorm(theta, y)
    chain = np.empty((n_steps, 4)); acc = 0
    for k in range(n_steps):
        prop = theta + np.array(step) * rng.standard_normal(4)
        lp_new = lb.ln_post_unnorm(prop, y)
        if np.log(rng.random()) < lp_new - lp:                   # only the RATIO of posteriors enters
            theta, lp = prop, lp_new; acc += 1
        chain[k] = theta
    return chain, acc / n_steps


t0 = time.perf_counter()
chains = {k: metropolis(y, rng_for("ch05", "28_sampling_evidence", i)) for i, (k, y) in enumerate(data.items())}
t_mh = time.perf_counter() - t0
burn = 50_000
cb = chains["bump"][0][burn:]

# grid marginals of mu and A for the bump data (reference)
_, A, mu, dc = lb.ln_bayes_factor(data["bump"], nA=801, nmu=801, return_grid=True)
w = np.exp(-0.5 * (dc - dc.min()))
p_mu = np.trapezoid(w, A, axis=0); p_mu /= np.trapezoid(p_mu, mu)
p_A = np.trapezoid(w, mu, axis=1); p_A /= np.trapezoid(p_A, A)
A_b = A.copy()
mu_mean_grid = np.trapezoid(mu * p_mu, mu)
mu_sd_grid = np.sqrt(np.trapezoid((mu - mu_mean_grid) ** 2 * p_mu, mu))

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
axes[0].hist(cb[:, 3], bins=np.linspace(5.5, 7.2, 70), density=True, color=SERIES[0], alpha=0.75, label="Metropolis chain")
axes[0].plot(mu, p_mu, color="k", ls="--", lw=1.2, label="grid")
axes[0].set_xlim(5.5, 7.2); axes[0].set_xlabel(r"bump position $\mu$"); axes[0].set_ylabel("posterior density")
axes[0].legend(fontsize=7)
axes[1].hist(cb[:, 2], bins=np.linspace(0, 5, 70), density=True, color=SERIES[0], alpha=0.75)
axes[1].plot(A, p_A, color="k", ls="--", lw=1.2)
axes[1].set_xlim(0, 5); axes[1].set_xlabel(r"bump amplitude $A$")
savefig(fig, "ch05", "mh_bump")

# cross-check of the from-scratch chain with the standard ensemble sampler emcee
import emcee
rng_e = rng_for("ch05", "28_sampling_evidence", 50)
p0 = cb[rng_e.integers(0, len(cb), 32)] + 1e-4 * rng_e.standard_normal((32, 4))
sampler = emcee.EnsembleSampler(32, 4, lb.ln_post_unnorm, args=(data["bump"],))
np.random.seed(int(rng_e.integers(2**31)))
sampler.run_mcmc(p0, 20_000, progress=False)
ce = sampler.get_chain(discard=2_000, flat=True)
out.update({"FiveCEmMuMean": ce[:, 3].mean(), "FiveCEmAMean": ce[:, 2].mean(),
            "FiveCEmFloor": np.mean(ce[:, 2] < 0.5), "FiveCEmSteps": 32 * 20_000})

# Savage--Dickey on the no-bump data: density of A at 0 from the chain
cf = chains["flat"][0][burn:]
h = 0.1
pA0_chain = np.mean(cf[:, 2] < h) / h
batches = np.array_split(cf[:, 2] < h, 20)                          # batch means -> Monte Carlo error
pA0_err = np.std([b.mean() for b in batches]) / h / np.sqrt(len(batches))
pA0_extrap = 2 * pA0_chain - np.mean((cf[:, 2] >= h) & (cf[:, 2] < 2 * h)) / h   # linear extrapolation to A = 0
_, A, mu, dcf = lb.ln_bayes_factor(data["flat"], nA=2001, nmu=801, return_grid=True)
wf = np.exp(-0.5 * dcf)
pAf = np.trapezoid(wf, mu, axis=1); pAf /= np.trapezoid(pAf, A)
B01_exact = np.exp(-lb.ln_bayes_factor(data["flat"]))
out.update({"FiveCMhSteps": len(chains["bump"][0]), "FiveCMhBurn": burn, "FiveCMhAcc": chains["bump"][1],
            "FiveCMhTime": t_mh / 2, "FiveCMhMuMean": cb[:, 3].mean(), "FiveCMhMuSd": cb[:, 3].std(),
            "FiveCGridMuMean": mu_mean_grid, "FiveCGridMuSd": mu_sd_grid,
            "FiveCMhAMean": cb[:, 2].mean(), "FiveCGridAMean": np.trapezoid(A_b * p_A, A_b),
            "FiveCSdBin": h, "FiveCSdChain": pA0_chain * lb.A_MAX, "FiveCSdExtrap": pA0_extrap * lb.A_MAX,
            "FiveCSdChainErr": pA0_err * lb.A_MAX,
            "FiveCMhFloorChain": np.mean(cb[:, 2] < 0.5), "FiveCMhFloorGrid": np.trapezoid(p_A[A_b < 0.5], A_b[A_b < 0.5]), "FiveCSdAvg": np.trapezoid(pAf[A < h], A[A < h]) / h * lb.A_MAX,
            "FiveCSdGridZero": pAf[0] * lb.A_MAX, "FiveCSdExact": B01_exact})

# is the missed pedestal a property of the sampler, or the luck of one chain?  More chains, new seeds.
N_EXTRA = 5
floors, musds = [np.mean(cb[:, 2] < 0.5)], [cb[:, 3].std()]
for i in range(N_EXTRA):
    ce_i = metropolis(data["bump"], rng_for("ch05", "28_sampling_evidence_chains", i))[0][burn:]
    floors.append(np.mean(ce_i[:, 2] < 0.5)); musds.append(ce_i[:, 3].std())
out.update({"FiveCMhNchains": N_EXTRA + 1,
            "FiveCMhFloorMin": min(floors), "FiveCMhFloorMax": max(floors),
            "FiveCMhMuSdMin": min(musds), "FiveCMhMuSdMax": max(musds)})


# ------------------------------------------------------------------ 2. nested sampling from scratch
MU_FINE = np.linspace(*lb.MU_RANGE, 20001)
NLIVE = 100


def make_lnL(y):
    """ln L(A, mu) = -(chi2_min(A, mu) - chi2_min(line))/2, line integrated out; prior U(box)."""
    ry = lb.PPERP @ y
    U = lb.PPERP @ lb.bump(lb.X[:, None], MU_FINE[None, :])
    uy = ry @ U / lb.SIGMA**2; uu = np.sum(U * U, axis=0) / lb.SIGMA**2
    def lnL(A, m):
        return A * np.interp(m, MU_FINE, uy) - 0.5 * A**2 * np.interp(m, MU_FINE, uu)
    return lnL


def draw_prior(rng, k):
    return rng.random(k) * lb.A_MAX, lb.MU_RANGE[0] + rng.random(k) * lb.L_MU


def nested_sampling(lnL, rng, n_live=100, tol=1e-3):
    A, m = draw_prior(rng, n_live); L = lnL(A, m)
    dead_lnL, dead_pts, n_eval, i = [], [], n_live, 0
    lnZ = -np.inf
    while True:
        i += 1
        j = np.argmin(L); Lstar = L[j]
        dead_lnL.append(Lstar); dead_pts.append((A[j], m[j]))
        lnX = -i / n_live                                          # expected shrinkage per step
        lnZ = np.logaddexp(lnZ, Lstar + np.log(np.exp(-(i - 1) / n_live) - np.exp(lnX)))
        if L.max() + lnX < lnZ + np.log(tol):                      # what is left is negligible
            break
        while True:                                                # replace: prior draw with L > L*
            k = int(min(2e6, 5 * np.exp(-lnX) + 50))
            An, mn = draw_prior(rng, k); Ln = lnL(An, mn); n_eval += k
            ok = np.flatnonzero(Ln > Lstar)
            if ok.size:
                A[j], m[j], L[j] = An[ok[0]], mn[ok[0]], Ln[ok[0]]
                break
    lnZ = np.logaddexp(lnZ, logsumexp(L) - np.log(n_live) + lnX)  # the remaining live points
    return lnZ, np.array(dead_lnL), np.array(dead_pts), i, n_eval, (A, m, L)


def lnZ_uncertainty(dead_lnL, live_L, n_live, rng, n_rep=100):
    """Skilling: redraw the unknown shrinkage factors t ~ Beta(N,1) and recompute ln Z."""
    n = len(dead_lnL); res = []
    for _ in range(n_rep):
        lnX = np.cumsum(np.log(rng.random(n)) / n_live)            # ln t with t ~ N t^(N-1)
        lnXprev = np.concatenate([[0.0], lnX[:-1]])
        lw = dead_lnL + np.log(np.exp(lnXprev) - np.exp(lnX))
        res.append(np.logaddexp(logsumexp(lw), logsumexp(live_L) - np.log(n_live) + lnX[-1]))
    return np.std(res)


lnL = make_lnL(data["bump"])
rng = rng_for("ch05", "28_sampling_evidence", 7)
t0 = time.perf_counter()
lnZ, dL, dP, n_it, n_eval, live = nested_sampling(lnL, rng)
t_ns = time.perf_counter() - t0
err = lnZ_uncertainty(dL, live[2], NLIVE, rng)
runs = [nested_sampling(lnL, rng_for("ch05", "28_sampling_evidence", 100 + r))[0] for r in range(8)]
lnB_ref = lb.ln_bayes_factor(data["bump"])
lnX = -np.arange(1, n_it + 1) / NLIVE
wts = dL + np.log(np.exp(lnX + 1 / NLIVE) - np.exp(lnX))
H = np.sum(np.exp(wts - lnZ) * (dL - lnZ))                         # information, prior -> posterior

fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"wspace": 0.85})
sc = axes[0].scatter(dP[:, 1], dP[:, 0], c=np.arange(n_it), s=3, cmap="Blues")
axes[0].scatter(live[1], live[0], s=4, color=SERIES[1], label="final live points")
axes[0].set_xlabel(r"bump position $\mu$"); axes[0].set_ylabel(r"bump amplitude $A$")
axes[0].set_xlim(*lb.MU_RANGE); axes[0].set_ylim(0, lb.A_MAX); axes[0].legend(fontsize=7, loc="upper left")
fig.colorbar(sc, ax=axes[0], label="iteration", shrink=0.9)
axes[1].plot(lnX, np.exp(dL), color=SERIES[0], label=r"likelihood $L(X)$")
ax2 = axes[1].twinx()
ax2.plot(lnX, np.exp(wts - lnZ), color=SERIES[1], label=r"weight $L\,\Delta X/Z$")
ax2.set_ylabel(r"$L\,\Delta X/Z$", color=SERIES[1]); ax2.grid(False)
axes[1].set_xlabel(r"$\ln X$ (enclosed prior mass)")
axes[1].set_ylabel(r"$L$ (relative to best line)", color=SERIES[0])   # axis colours replace a legend
savefig(fig, "ch05", "nested_sampling")

out.update({"FiveCNsLnZ": lnZ, "FiveCNsErr": err, "FiveCNsRef": lnB_ref, "FiveCNsIter": n_it,
            "FiveCNsEval": n_eval, "FiveCNsEvalM": n_eval / 1e6, "FiveCNsTime": t_ns, "FiveCNsLive": NLIVE, "FiveCNsRuns": len(runs), "FiveCNsH": H,
            "FiveCNsSkilling": np.sqrt(H / NLIVE), "FiveCNsRunSd": np.std(runs), "FiveCNsRunMean": np.mean(runs),
            "FiveCNsLnXend": lnX[-1]})
save_numbers("ch05", "28_sampling_evidence", out)
for k, v in out.items():
    print(f"{k:22s} {v:.4g}")
