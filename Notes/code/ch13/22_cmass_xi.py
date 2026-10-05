"""22_cmass_xi.py -- the redshift-space correlation function of CMASS South, with jackknife errors.

Question: what is the monopole xi_0(s) of the CMASS galaxies of the southern Galactic cap, and how
uncertain is it, estimated from the data alone?
Computes: weighted pair counts DD, DR, RR in bins of 8 Mpc/h out to 184 Mpc/h with
scipy.spatial.cKDTree.count_neighbors (galaxy weight w_tot w_FKP, random weight w_FKP), region by
region, so that every leave-one-region-out count follows by subtraction; the Landy-Szalay
estimator for the full sample and for each of the NJK jackknife samples; the jackknife
covariance; and xi_0 again with no weights and with FKP weights only, to show what the weights do.
Writes: data/ch13/cmass_xi.npz, results/ch13/22_cmass_xi.tex
"""
import os
import pathlib
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial import cKDTree
from common import save_numbers, DATA

EDGES = np.arange(0.0, 184.1, 8.0)          # bin edges [Mpc/h]; centres 4, 12, ..., 180
RAND_RATIO = int(os.environ.get("RAND_RATIO", 5))   # randoms per galaxy used in the counts
# processes: the cores this job was given (SLURM allotment or the laptop's 4-thread cap)
NPROC = min(len(os.sched_getaffinity(0)), int(os.environ.get("OMP_NUM_THREADS", 4)))

cat = np.load(DATA / "ch13" / "cmass_south.npz")
xg, xr = cat["xg"].astype(float), cat["xr"].astype(float)
nkeep = min(len(xr), RAND_RATIO * len(xg))
xr = xr[:nkeep]                              # the stored randoms are already a random subsample
rg, rr = cat["reg_g"], cat["reg_r"][:nkeep]
NJK = int(max(rg.max(), rr.max()) + 1)
WEIGHTS = {                                  # (galaxy weight, random weight) for three versions
    "full": (cat["wg"], cat["wr"][:nkeep]),
    "fkp": (cat["wfkp"], cat["wr"][:nkeep]),
    "none": (np.ones(len(xg)), np.ones(nkeep)),
}
TG, TR = cKDTree(xg, leafsize=32), cKDTree(xr, leafsize=32)


def pairs(ta, tb, wa, wb):
    """Weighted pair counts sum_{i in a, j in b} w_i w_j in each bin; ordered pairs, self-pairs dropped."""
    c = ta.count_neighbors(tb, EDGES, weights=(wa, wb), cumulative=False)
    return c[1:]                             # c[0] holds distances <= 0 (the self-pairs)


def region_counts(args):
    """Counts that involve region k: C(k, all) and C(k, k) for DD, DR (both ways) and RR."""
    k, wname = args
    wgal, wran = WEIGHTS[wname]
    ig, ir = np.flatnonzero(rg == k), np.flatnonzero(rr == k)
    tgk, trk = cKDTree(xg[ig]), cKDTree(xr[ir])
    out = {
        "dd_ka": pairs(tgk, TG, wgal[ig], wgal), "dd_kk": pairs(tgk, tgk, wgal[ig], wgal[ig]),
        "dr_ka": pairs(tgk, TR, wgal[ig], wran), "dr_ak": pairs(TG, trk, wgal, wran[ir]),
        "dr_kk": pairs(tgk, trk, wgal[ig], wran[ir]),
    }
    if wname != "fkp":                       # "fkp" has the same random weights as "full"
        out["rr_ka"] = pairs(trk, TR, wran[ir], wran)
        out["rr_kk"] = pairs(trk, trk, wran[ir], wran[ir])
    out["wg"], out["wg2"] = wgal[ig].sum(), (wgal[ig] ** 2).sum()
    out["wr"], out["wr2"] = wran[ir].sum(), (wran[ir] ** 2).sum()
    return out


def landy_szalay(DD, DR, RR, Wg, Sg, Wr, Sr):
    """xi_LS = (DD - 2 DR + RR)/RR with each count divided by its total number of weighted pairs."""
    dd = DD / (Wg ** 2 - Sg)                 # ordered pairs i != j: (sum w)^2 - sum w^2
    dr = DR / (Wg * Wr)
    rr_ = RR / (Wr ** 2 - Sr)
    return (dd - 2 * dr + rr_) / rr_


def measure(wname, pool, rr_from=None):
    t0 = time.time()
    res = pool.map(region_counts, [(k, wname) for k in range(NJK)], chunksize=1)
    S = {key: np.array([q[key] for q in res]) for key in res[0]}
    if rr_from is not None:
        S["rr_ka"], S["rr_kk"] = rr_from
    DD, DR, RR = S["dd_ka"].sum(0), S["dr_ka"].sum(0), S["rr_ka"].sum(0)
    Wg, Sg, Wr, Sr = S["wg"].sum(), S["wg2"].sum(), S["wr"].sum(), S["wr2"].sum()
    xi = landy_szalay(DD, DR, RR, Wg, Sg, Wr, Sr)
    # leave region k out: remove every pair with at least one member in k
    DDk = DD - 2 * S["dd_ka"] + S["dd_kk"]
    DRk = DR - S["dr_ka"] - S["dr_ak"] + S["dr_kk"]
    RRk = RR - 2 * S["rr_ka"] + S["rr_kk"]
    xik = landy_szalay(DDk, DRk, RRk, (Wg - S["wg"])[:, None], (Sg - S["wg2"])[:, None],
                       (Wr - S["wr"])[:, None], (Sr - S["wr2"])[:, None])
    print(f"{wname}: {time.time() - t0:.0f} s")
    return xi, xik, DD, DR, RR, time.time() - t0, (S["rr_ka"], S["rr_kk"])


if __name__ == "__main__":
    s = 0.5 * (EDGES[1:] + EDGES[:-1])
    out = {"s": s, "edges": EDGES}
    with Pool(NPROC) as pool:
        rr_full = None
        for wname in ["full", "fkp", "none"]:
            xi, xik, DD, DR, RR, dt, rr = measure(wname, pool, rr_full if wname == "fkp" else None)
            if wname == "full":
                rr_full = rr
            out.update({f"xi_{wname}": xi, f"xik_{wname}": xik, f"DD_{wname}": DD, f"DR_{wname}": DR,
                        f"RR_{wname}": RR, f"t_{wname}": dt})
    # jackknife covariance (derived in the text): C = (N-1)/N sum_k (xi_k - mean)(xi_k - mean)^T
    xik = out["xik_full"]
    dev = xik - xik.mean(0)
    C = (NJK - 1) / NJK * dev.T @ dev
    out["C_jk"] = C
    np.savez(DATA / "ch13" / "cmass_xi.npz", **out, njk=NJK, rand_ratio=RAND_RATIO)
    i100 = int(np.argmin(np.abs(s - 100)))
    save_numbers("ch13", "22_cmass_xi", {
        "XiRandRatio": RAND_RATIO, "XiNran": f"{nkeep:,}".replace(",", "{,}"),
        "XiNproc": NPROC, "XiTime": f"{out['t_full']:.0f}",
        "XiNbins": len(s), "XiBin": "8",
        "XiAtHundred": f"{1e3 * out['xi_full'][i100]:.2f}",
        "XiErrHundred": f"{1e3 * np.sqrt(C[i100, i100]):.2f}",
        "XiPairsDD": float(out["DD_none"].sum()), "XiPairsRR": float(out["RR_none"].sum()),
    })
