"""b04_betti_persistence.py -- Betti curves and persistence diagrams of the real sky against Gaussian skies.

Question: the Euler characteristic is beta0 - beta1.  Do the two parts separately -- the number of
hot regions and the number of holes in them -- behave as in Gaussian skies at the four resolutions?
And does the whole H0 persistence diagram of the hot regions (every birth and merger of a hot
spot as the threshold is lowered) sit at a typical distance from the diagrams of Gaussian skies?

Computes: the chi^2 test (Hartlap-corrected, rank p-value) of the 24 numbers (beta0, beta1) per
          steradian at Planck's 12 thresholds, dropping thresholds where no simulation varies;
          at N_side 64 and 128 the 1-Wasserstein distance between diagrams, computed as an optimal
          assignment (scipy.optimize.linear_sum_assignment) after dropping the short-lived points
          below a fixed lifetime cut: for each map, its mean distance to a fixed set of reference
          skies; the rank of the data's mean distance among the simulations'.
Writes:   figures/ch13/b_betti.pdf, figures/ch13/b_persistence.pdf, results/ch13/b04_betti_persistence.tex,
          data/ch13/b_wdist.npz
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment
from common import setup, savefig, save_numbers, SERIES, DATA
from lib_skytopo import NU12

setup()
SCALES = [(64, 160), (128, 80), (256, 40), (512, 20)]
WORD = {64: "SixFour", 128: "OneTwoEight", 256: "TwoFiveSix", 512: "FiveOneTwo"}
NREF, NTEST = {64: 10, 128: 5}, {64: 400, 128: 200}
CUT = {64: 0.1, 128: 0.5}          # lifetime cut in units of sigma_0: shorter blips are dropped
D = np.load(DATA / "ch13" / "b_data_stats.npz", allow_pickle=True)["stats"].item()


def w1(A, B):
    """1-Wasserstein distance between diagrams A, B (rows: birth, death), L-infinity ground cost.

    Each point is matched either to a point of the other diagram, at cost max(|db|, |dd|), or to the
    diagonal, at cost (birth - death)/2.  Giving every point a private diagonal copy turns this into
    a square assignment problem: rows are A's points then B's diagonal copies, columns are B's points
    then A's diagonal copies; diagonal copy to diagonal copy costs nothing.
    """
    n, m = len(A), len(B)
    C = np.zeros((n + m, n + m))
    C[:n, :m] = np.maximum(np.abs(A[:, None, 0] - B[None, :, 0]), np.abs(A[:, None, 1] - B[None, :, 1]))
    C[:n, m:] = ((A[:, 0] - A[:, 1]) / 2)[:, None]          # A's point i sent to the diagonal
    C[n:, :m] = ((B[:, 0] - B[:, 1]) / 2)[None, :]          # B's point j sent to the diagonal
    r, c = linear_sum_assignment(C)
    return C[r, c].sum()


def w1_check():
    """The worked example of the text: D = {(2, 0.5)}, D' = {(2.2, 0.4), (0.3, 0.2)} gives 0.25."""
    return w1(np.array([[2.0, 0.5]]), np.array([[2.2, 0.4], [0.3, 0.2]]))


def load(nside, keys):
    zs = [np.load(f) for f in sorted((DATA / "ch13").glob(f"b_sims_{nside}_*.npz"))]
    return {k: np.concatenate([z[k] for z in zs]) for k in keys}, zs


def rank_chi2(y, Y):
    keep = Y.std(0) > 0                                     # thresholds where nothing varies carry no test
    y, Y = y[keep], Y[:, keep]
    n, p = Y.shape
    C = np.cov(Y, rowvar=False)
    d = y - Y.mean(0)
    cd = (n - p - 2) / (n - 1) * d @ np.linalg.solve(C, d)
    cs = np.zeros(n)
    for i in range(n):
        rest = np.delete(Y, i, axis=0)
        di = Y[i] - rest.mean(0)
        cs[i] = (n - 1 - p - 2) / (n - 2) * di @ np.linalg.solve(np.cov(rest, rowvar=False), di)
    return cd, cs, p


nums, curves = {}, {}
for nside, fwhm in SCALES:
    S, _ = load(nside, ("b0", "b1", "b0c"))
    area = D[nside]["area"]
    Y = np.concatenate([S["b0"], S["b1"]], axis=1) / area
    y = np.concatenate([D[nside]["b0"], D[nside]["b1"]]) / area
    cd, cs, p = rank_chi2(y, Y)
    w = WORD[nside]
    pv = float(np.mean(cs >= cd))
    nums.update({f"BeP{w}": round(100 * pv, 1), f"BeChiN{w}": round(cd / p, 2), f"BeDof{w}": p,
                 f"BePerr{w}": round(100 * np.sqrt(max(pv * (1 - pv), 1 / len(cs)) / len(cs)), 1)})
    curves[nside] = (S, D[nside])

# ---------------------------------------------------------------- persistence: Wasserstein distances
out = DATA / "ch13" / "b_wdist.npz"
if out.exists():
    W = dict(np.load(out))
else:
    W = {}
    for nside in (64, 128):
        _, zs = load(nside, ("b0",))
        z = zs[0]
        off, pb, pdd = z["pair_off"], z["pair_b"], z["pair_d"]

        def diagram(i):
            b, d = pb[off[i]:off[i + 1]], pdd[off[i]:off[i + 1]]
            f = np.isfinite(d) & (b - d > CUT[nside])
            return np.c_[b[f], d[f]]

        refs = [diagram(i) for i in range(NREF[nside])]
        bd, dd = D[nside]["pairs"]
        f = np.isfinite(dd) & (bd - dd > CUT[nside])
        data_dg = np.c_[bd[f], dd[f]]
        t0 = time.time()
        mean_dist = lambda dg: np.mean([w1(dg, r) for r in refs])
        W[f"data_{nside}"] = np.array(mean_dist(data_dg))
        W[f"sims_{nside}"] = np.array([mean_dist(diagram(i))
                                       for i in range(NREF[nside], NREF[nside] + NTEST[nside])])
        W[f"npts_{nside}"] = np.array([len(data_dg), np.mean([len(r) for r in refs])])
        print(nside, "done", f"{time.time() - t0:.0f}s", flush=True)
    np.savez(out, **W)
for nside in (64, 128):
    w = WORD[nside]
    s, d = W[f"sims_{nside}"], float(W[f"data_{nside}"])
    pv = float(np.mean(s >= d))
    nums.update({f"WdData{w}": round(d, 1), f"WdSim{w}": round(float(s.mean()), 1),
                 f"WdSd{w}": round(float(s.std(ddof=1)), 1), f"WdP{w}": round(100 * pv, 1),
                 f"WdNtest{w}": len(s), f"WdNref{w}": NREF[nside],
                 f"WdPtsData{w}": int(W[f"npts_{nside}"][0]), f"WdPtsSim{w}": int(round(W[f"npts_{nside}"][1])),
                 f"WdCut{w}": CUT[nside]})
nums["WdExample"] = round(float(w1_check()), 3)
save_numbers("ch13", "b04_betti_persistence", nums)
for k, v in nums.items():
    print(k, v)

# ---------------------------------------------------------------- figure: Betti curves at two resolutions
fig, axs = plt.subplots(2, 3, figsize=(10, 5.2))
for r, nside in enumerate((128, 512)):
    S, d = curves[nside]
    for j, (k, lab) in enumerate((("b0", r"$\beta_0$ hot regions"), ("b1", r"$\beta_1$ holes in them"),
                                  ("b0c", r"cold regions"))):
        ax = axs[r, j]
        lo, hi = np.percentile(S[k], [0.5, 99.5], axis=0)
        ax.fill_between(NU12, lo, hi, color="0.82", lw=0, label="central 99% of Gaussian skies")
        ax.plot(NU12, S[k].mean(0), "k--", lw=1, label="mean")
        ax.plot(NU12, d[k], "o-", ms=3, color=SERIES[0], lw=1.1, label="SMICA map")
        ax.set_title(fr"{lab}, $N_{{\rm side}}={nside}$", fontsize=9)
        if r == 1:
            ax.set_xlabel(r"threshold $\nu$")
axs[0, 0].legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch13", "b_betti")

# ---------------------------------------------------------------- figure: diagrams and distances
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
_, zs = load(128, ("b0",))
z = zs[0]
off = z["pair_off"]
bs, ds = z["pair_b"][off[0]:off[1]], z["pair_d"][off[0]:off[1]]
bd, dd = D[128]["pairs"]
for ax, (b, d, lab, c) in zip(axs[:2], ((bd, dd, "SMICA map", SERIES[0]), (bs, ds, "one Gaussian sky", SERIES[2]))):
    f = np.isfinite(d)
    ax.plot(d[f], b[f], ".", ms=2.5, color=c, alpha=0.7)
    ax.plot([-4, 4.5], [-4, 4.5], color="k", lw=0.7)
    ax.set_xlim(-4, 3)
    ax.set_ylim(-3, 4.5)
    ax.set_xlabel(r"merger level (death) $\nu_d$")
    ax.set_ylabel(r"birth level $\nu_b$")
    ax.set_title(fr"{lab}, $N_{{\rm side}}=128$: {f.sum()} points", fontsize=9)
ax = axs[2]
s = W["sims_128"]
ax.hist(s, bins=25, color=SERIES[0], alpha=0.5, label="Gaussian skies")
ax.axvline(float(W["data_128"]), color=SERIES[1], lw=2, label="SMICA map")
ax.set_xlabel(r"mean $W_1$ distance to reference skies")
ax.set_title(r"$N_{\rm side}=128$", fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch13", "b_persistence")
