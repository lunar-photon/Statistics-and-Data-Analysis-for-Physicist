"""c07_abc.py -- inference without writing down a likelihood: approximate Bayesian computation.

Question: if we refuse to assume that the Betti-curve vector is Gaussian, can we still get a
posterior for eps from one map, using only the simulator?  And why does it matter to compress
the 25 numbers into one before comparing?
Computes:
  * the score compression t(S) = dmu^T C^-1 (S - mu0) / F (an estimate of eps), with dmu, C, mu0
    from the cache of c02_mc_fnl.py (as in c05_fisher_post.py);
  * 20000 new maps with eps drawn from the prior U(-0.15, 0.15), their Betti curves and t;
  * rejection ABC for the mock map of c05 (seed 252, eps = 0.05): keep the 2% of simulations whose
    t is closest to t_obs; and, for comparison, keep the 2% closest in the full 25-dimensional
    Mahalanobis distance;
  * the Gaussian-likelihood posterior of c05 for the same map.
Writes: figures/ch12/c_abc.pdf, results/ch12/c07_abc.tex, data/ch12/c_abc.npz (cache)
"""
import sys
import time
import pathlib
import re
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, DATA
from lib_fnl import Model, NU, standardise
from lib_topo import betti_plane

N_ABC, PRIOR, KEEP = 20000, 0.15, 0.02
TRUE = 0.05
# the mock map: the one c05_fisher_post.py picked for its illustration (\CfMock), so both use the same map
MOCK = int(re.search(r"\\CfMock\}\{(\d+)\}", (pathlib.Path(__file__).resolve().parents[2] / "results" / "ch12" / "c05_fisher_post.tex").read_text()).group(1))
OUT = DATA / "ch12" / "c_abc.npz"
model = Model(256, 2.0)


def betti_vector(u):
    x = standardise(u)
    b = np.array([betti_plane(x >= nu) for nu in NU], float)
    return np.concatenate([b[:, 0], b[:, 1]])


def one(i):
    r = rng_for("ch12", "c07_abc", i)
    eps = r.uniform(-PRIOR, PRIOR)                       # 1. draw a parameter from the prior
    return eps, betti_vector(model.draw(eps, r))         # 2. simulate a map, 3. summarise it


def simulate():
    if OUT.exists():
        z = np.load(OUT)
        return z["eps"], z["S"]
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(one, range(N_ABC), chunksize=20)
    print(f"{N_ABC} maps in {time.time() - t0:.0f} s")
    eps = np.array([o[0] for o in out])
    S = np.array([o[1] for o in out])
    np.savez_compressed(OUT, eps=eps, S=S)
    return eps, S


if __name__ == "__main__":
    z = np.load(DATA / "ch12" / "c_fnl_mc.npz")
    eg = list(z["eps_grid"])
    N = np.concatenate([z["null_b0"], z["null_b1"]], 1)
    ok = N.std(0) > 1e-9
    N = N[:, ok]
    n, p = N.shape
    mu0 = N.mean(0)
    Ci = np.linalg.inv(np.cov(N.T)) * (n - p - 2) / (n - 1)
    G = np.concatenate([z["grid_b0"], z["grid_b1"]], 2)[:250][..., ok]
    d = ((G[:, eg.index(0.05)] - G[:, eg.index(-0.05)]) / 0.1).mean(0)
    F = d @ Ci @ d
    w = Ci @ d / F                                        # t = w . (S - mu0) estimates eps

    S_obs = np.concatenate([z["grid_b0"][MOCK, eg.index(TRUE)], z["grid_b1"][MOCK, eg.index(TRUE)]])[ok]
    t_obs = (S_obs - mu0) @ w

    eps, S = simulate()
    S = S[:, ok]
    t = (S - mu0) @ w
    nkeep = int(KEEP * len(eps))
    acc1 = np.argsort(np.abs(t - t_obs))[:nkeep]          # 4. keep the closest: compressed
    tol1 = np.abs(t - t_obs)[acc1].max()
    D = S - S_obs
    dist = np.sqrt(np.einsum("ij,jk,ik->i", D, Ci, D))
    acc2 = np.argsort(dist)[:nkeep]                       # 4'. keep the closest: all p numbers
    e1, e2 = eps[acc1], eps[acc2]

    # the Gaussian-likelihood posterior of c05 for the same map (quadratic mean, fixed C)
    EPS = np.array(eg)
    coef = np.polyfit(EPS, G.mean(0), 2)
    egrid = np.linspace(-PRIOR, PRIOR, 1201)
    MU = coef[0][None] * egrid[:, None] ** 2 + coef[1][None] * egrid[:, None] + coef[2][None]
    R = S_obs[None] - MU
    lp = -0.5 * np.einsum("gi,ij,gj->g", R, Ci, R)
    wl = np.exp(lp - lp.max())
    wl /= wl.sum()
    m_l = (wl * egrid).sum()
    sd_l = np.sqrt((wl * (egrid - m_l) ** 2).sum())

    # linearity of t in eps and its scatter at fixed eps (the 'likelihood' ABC is sampling)
    slope, icpt = np.polyfit(eps, t, 1)
    scatter = np.std(t - (slope * eps + icpt))
    resid = t - (slope * eps + icpt)
    sc0 = resid[np.abs(eps) < 0.02].std()                 # scatter of t near eps = 0
    sc5 = resid[np.abs(eps - 0.05) < 0.02].std()          # ... and near eps = 0.05

    # the same simulations serve every observed map: ABC on t for all 250 mock maps of c05
    hits, sds = [], []
    for m in range(250, 500):
        So = np.concatenate([z["grid_b0"][m, eg.index(TRUE)], z["grid_b1"][m, eg.index(TRUE)]])[ok]
        keep = eps[np.argsort(np.abs(t - (So - mu0) @ w))[:nkeep]]
        lo, hi = np.percentile(keep, [16, 84])            # the same 68% interval as c05
        hits.append(lo <= TRUE <= hi)
        sds.append(keep.std())

    setup(9.0, 3.1)
    fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ax[0].plot(eps, t, ".", ms=1.2, color=INK2, alpha=0.5, label=f"{len(eps)} simulations")
    ax[0].plot(eps[acc1], t[acc1], ".", ms=2, color=SERIES[1], label=f"kept (closest {int(100 * KEEP)}%)")
    ax[0].axhline(t_obs, color=SERIES[1], lw=1)
    ax[0].set_xlabel(r"$\epsilon$ drawn from the prior")
    ax[0].set_ylabel(r"compressed summary $t$")
    ax[0].legend(fontsize=6.5, markerscale=4, loc="upper left")
    bins = np.linspace(-0.05, 0.15, 41)
    de = egrid[1] - egrid[0]
    for a, e, t_ in ((ax[1], e1, r"ABC on $t$ (one number)"), (ax[2], e2, f"ABC on all {p} numbers")):
        a.hist(e, bins=bins, density=True, color=SERIES[0], alpha=0.6, label="ABC")
        a.plot(egrid, wl / de, "k-", lw=1, label="Gaussian likelihood")
        a.axvline(TRUE, color="k", ls="--", lw=0.8)
        a.set_xlim(-0.05, 0.15)
        a.set_xlabel(r"$\epsilon$")
        a.set_title(t_, fontsize=9)
    ax[1].legend(fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "ch12", "c_abc")

    save_numbers("ch12", "c07_abc", {
        "CbN": len(eps), "CbKeep": nkeep, "CbKeepPct": int(100 * KEEP),
        "CbTobs": f"{t_obs:.4f}", "CbTol": f"{tol1:.4f}",
        "CbMeanT": f"{e1.mean():.4f}", "CbSdT": f"{e1.std():.4f}",
        "CbMeanAll": f"{e2.mean():.4f}", "CbSdAll": f"{e2.std():.4f}",
        "CbMeanL": f"{m_l:.4f}", "CbSdL": f"{sd_l:.4f}",
        "CbSlope": f"{slope:.2f}", "CbScatter": f"{scatter:.4f}", "CbSigF": f"{1 / np.sqrt(F):.4f}",
        "CbScZero": f"{sc0:.4f}", "CbScFive": f"{sc5:.4f}",
        "CbDistTol": f"{dist[acc2].max():.1f}", "CbP": p,
        "CbNmaps": len(hits), "CbCover": f"{100 * np.mean(hits):.0f}",
        "CbCoverErr": f"{100 * np.sqrt(np.mean(hits) * (1 - np.mean(hits)) / len(hits)):.0f}",
        "CbSdMed": f"{np.median(sds):.4f}",
    })
    print("t_obs", t_obs, "tol", tol1, "ABC t", e1.mean(), e1.std(), "ABC all", e2.mean(), e2.std(),
          "lik", m_l, sd_l, "slope", slope, "scatter", scatter, 1 / np.sqrt(F), "dist tol", dist[acc2].max(), "sc0", sc0, "sc5", sc5,
          "coverage", np.mean(hits), "median sd", np.median(sds))
