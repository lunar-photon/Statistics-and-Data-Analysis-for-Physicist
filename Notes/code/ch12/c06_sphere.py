"""c06_sphere.py -- excursion-set topology of CMB skies: beam, noise and mask as operators.

Question: on the HEALPix sphere, how do a beam, white pixel noise and a mask change the
hot-spot and cold-spot counts beta0(nu) and the persistence diagram of a CMB temperature map,
and can Gaussian simulations passed through the same operators tell a Gaussian sky from a
slightly non-Gaussian sky that has the same power spectrum?
Computes (nside 128, l_max 383, fiducial Planck 2018 spectrum):
  * H0 persistence of the superlevel sets (hot spots, 8 neighbours) and of the sublevel sets
    (cold spots, 4 edge-sharing neighbours) by union-find on healpy.get_all_neighbours,
    compiled with numba; Betti curves read off the diagrams; checks against connected
    components (lib_sphere.betti_sphere) and against gudhi (SimplexTree, nside 32);
  * beam: FWHM 40', 80', 160' (100 skies each), against the Gaussian-kinematic scaling with
    lambda = sigma_1^2 / (2 sigma_0^2);
  * noise: white noise sigma_N / sigma_S = 0, 0.2, 0.5 on top of the 80' beam, and the noisy
    map smoothed again by 60' (100 skies each); persistence of the extra features;
  * mask: |b| < 20 deg band plus 100 point-source holes of radius 1 deg (100 skies);
  * a Planck-style test: 600 + 400 Gaussian skies (beam 80', noise 0.2, mask) for the null mean,
    covariance and chi^2 distribution; 'data' skies: Gaussian, and local-type
    T = s + a (s^2 - sigma^2)/sigma applied before the beam, renormalised per multipole to the
    same spectrum, for a = 0.02, 0.05 and 0.15 (100 skies each, for the power); the same test
    with a binned pseudo-C_l of the masked map.
Writes: figures/ch12/c_sphere_ops.pdf, figures/ch12/c_sphere_noise.pdf,
        figures/ch12/c_sphere_test.pdf, results/ch12/c06_sphere.tex, results/ch12/c06_gudhi.tex (if gudhi is installed),
        data/ch12/c_sphere.npz (cache of all summaries; delete to recompute)
"""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch08"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from numba import njit
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, DATA
from camb_fiducial import load_fiducial
from lib_cmbsim import synalm, gaussian_beam
from lib_masks import band, holes, hole_centres
from lib_sphere import betti_sphere, sigma_tau

NSIDE, LMAX = 128, 383
NU = np.linspace(-3, 3, 13)
NPIX = hp.nside2npix(NSIDE)
OUT = DATA / "ch12" / "c_sphere.npz"
rng = rng_for("ch12", "c06_sphere")

ell, cl_fid = load_fiducial("TT")
CL = cl_fid[: LMAX + 1].copy()
CL[:2] = 0.0                                                 # no monopole, no dipole


# ------------------------------------------------------------------ persistence by union-find
@njit(cache=True)
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]                        # path halving
        a = parent[a]
    return a


@njit(cache=True)
def h0_superlevel(u, nbr, active):
    """(birth, death) pairs of the islands of {u >= t} as t decreases (elder rule).

    nbr: (npix, k) neighbour table (-1 = none); active: pixels that are observed.
    Essential islands (one per observed region) get death = -inf.
    """
    n = u.size
    order = np.argsort(-u)
    parent = -np.ones(n, np.int64)
    birth = np.zeros(n)
    out_b = np.empty(n)
    out_d = np.empty(n)
    m = 0
    for p in order:
        if not active[p]:
            continue
        parent[p] = p
        birth[p] = u[p]
        for j in range(nbr.shape[1]):
            q = nbr[p, j]
            if q < 0 or parent[q] < 0:
                continue
            rp = _find(parent, p)
            rq = _find(parent, q)
            if rp == rq:
                continue
            if birth[rp] < birth[rq]:                        # the younger island dies
                young, old = rp, rq
            else:
                young, old = rq, rp
            if birth[young] > u[p]:                          # skip zero-persistence pairs
                out_b[m] = birth[young]
                out_d[m] = u[p]
                m += 1
            parent[young] = old
    for p in range(n):                                       # survivors: essential classes
        if active[p] and parent[p] == p:
            out_b[m] = birth[p]
            out_d[m] = -np.inf
            m += 1
    return out_b[:m], out_d[:m]


NB_ALL = hp.get_all_neighbours(NSIDE, np.arange(NPIX)).T.copy()   # (npix, 8): SW W NW N NE E SE S
NB_EDGE = NB_ALL[:, [0, 2, 4, 6]].copy()                           # SW NW NE SE share an edge


def diagrams(u, active):
    """Hot-spot diagram of u (8 neighbours) and cold-spot diagram (4 neighbours), in sigma units."""
    hb, hd = h0_superlevel(u, NB_ALL, active)
    cb, cd = h0_superlevel(-u, NB_EDGE, active)
    return (hb, hd), (-cb, -cd)                              # cold: born at a low value, dies higher


def betti_curves(hot, cold):
    hb, hd = hot
    cb, cd = cold
    nh = np.array([np.sum((hb >= nu) & (hd < nu)) for nu in NU])          # islands of {u >= nu}
    nc = np.array([np.sum((cb <= nu) & (cd > nu)) for nu in NU])          # lakes of {u <= nu}
    return nh, nc


def standardise(m, active):
    x = m[active]
    return (m - x.mean()) / x.std()


PBINS = np.linspace(0.0, 2.0, 41)                        # persistence bins (sigma units)


def summarise(m, active, nshort=0.1):
    u = standardise(m, active)
    hot, cold = diagrams(u, active)
    nh, nc = betti_curves(hot, cold)
    pers = hot[0] - hot[1]
    fin = np.isfinite(pers)
    hist = np.histogram(pers[fin], bins=PBINS)[0]
    return nh, nc, np.sum(pers[fin] < nshort), fin.sum(), hot, hist


# ------------------------------------------------------------------ the operators
def beam(fwhm):
    return gaussian_beam(fwhm, LMAX)


def sky_alm(r):
    return synalm(CL, LMAX, r)


SIG_S80 = np.sqrt(np.sum((2 * np.arange(LMAX + 1) + 1) * CL * beam(80.0) ** 2) / (4 * np.pi))


def observe(alm, fwhm=80.0, noise=0.0, r=None, resmooth=None):
    m = hp.alm2map(hp.almxfl(alm, beam(fwhm)), NSIDE, lmax=LMAX)
    if noise > 0:
        m = m + noise * SIG_S80 * r.standard_normal(NPIX)
    if resmooth:
        a = hp.map2alm(m, lmax=LMAX, iter=1)
        m = hp.alm2map(hp.almxfl(a, beam(resmooth)), NSIDE, lmax=LMAX)
    return m


MASK = (band(NSIDE, 20.0) * holes(NSIDE, hole_centres(100, rng_for("ch12", "c06_mask")), 1.0)) > 0.5
FULL = np.ones(NPIX, bool)


def ng_alm(r, a, R=None):
    """Local-type sky: s + a (s^2 - sigma^2)/sigma before the beam, renormalised per multipole."""
    s = hp.alm2map(sky_alm(r), NSIDE, lmax=LMAX)
    sg = s.std()
    phi = s + a * (s ** 2 - sg ** 2) / sg
    alm = hp.map2alm(phi, lmax=LMAX, iter=1)
    return alm if R is None else hp.almxfl(alm, R)


AMPS = (0.02, 0.05, 0.15)                                    # local-type amplitudes a
NNULL, NCOV, NDATA = 1000, 600, 100
EDGES = np.unique(np.geomspace(2, LMAX, 13).astype(int))


def pcl(m, active):
    c = hp.anafast(np.where(active, m, 0.0), lmax=LMAX, iter=1)
    return np.array([c[a:b].mean() for a, b in zip(EDGES[:-1], EDGES[1:])])


# ------------------------------------------------------------------ the Monte Carlo
def run():
    t0 = time.time()
    res = {}
    # beam
    for fw in (40.0, 80.0, 160.0):
        r = rng_for("ch12", "c06_beam", int(fw))
        S = [summarise(observe(sky_alm(r), fw), FULL) for _ in range(100)]
        res[f"beam{int(fw)}_h"] = np.array([s[0] for s in S])
        res[f"beam{int(fw)}_c"] = np.array([s[1] for s in S])
    print("beam", time.time() - t0)
    # noise (same skies and noise draws for every level: common random numbers)
    for lev, rs in ((0.0, None), (0.2, None), (0.5, None), (0.5, 60.0)):
        r = rng_for("ch12", "c06_noise")
        S = []
        for _ in range(100):
            alm = sky_alm(r)
            S.append(summarise(observe(alm, 80.0, lev, r, rs), FULL))
        tag = f"noise{int(100 * lev)}" + ("s" if rs else "")
        res[tag + "_h"] = np.array([s[0] for s in S])
        res[tag + "_c"] = np.array([s[1] for s in S])
        res[tag + "_short"] = np.array([s[2] for s in S])
        res[tag + "_nfeat"] = np.array([s[3] for s in S])
        res[tag + "_phist"] = np.array([s[5] for s in S])
        hb, hd = S[0][4]
        res[tag + "_pd"] = np.stack([hb, np.where(np.isfinite(hd), hd, -4.0)], 1)
    print("noise", time.time() - t0)
    # mask (the same skies as the 80' beam row of the noise study: CRN)
    r = rng_for("ch12", "c06_noise")
    S = [summarise(observe(sky_alm(r), 80.0), MASK) for _ in range(100)]
    res["mask_h"] = np.array([s[0] for s in S])
    res["mask_c"] = np.array([s[1] for s in S])
    print("mask", time.time() - t0)
    # null for the test: beam 80', noise 0.2, mask
    r = rng_for("ch12", "c06_null")
    H, C, P = [], [], []
    for _ in range(NNULL):
        m = observe(sky_alm(r), 80.0, 0.2, r)
        nh, nc, *_ = summarise(m, MASK)
        H.append(nh)
        C.append(nc)
        P.append(pcl(m, MASK))
    res["null_h"], res["null_c"], res["null_pcl"] = np.array(H), np.array(C), np.array(P)
    print("null", time.time() - t0)
    # renormalisation of the non-Gaussian skies: R_l = sqrt(C_l^s / C_l^phi), both spectra measured
    # on the SAME 100 Gaussian skies s (common random numbers), so the ratio is nearly noise-free
    for a in AMPS:
        r = rng_for("ch12", "c06_renorm", int(100 * a))
        cs, cp = np.zeros(LMAX + 1), np.zeros(LMAX + 1)
        for _ in range(100):
            sm = hp.alm2map(sky_alm(r), NSIDE, lmax=LMAX)
            sg = sm.std()
            cs += hp.anafast(sm, lmax=LMAX, iter=1)
            cp += hp.anafast(sm + a * (sm ** 2 - sg ** 2) / sg, lmax=LMAX, iter=1)
        R = np.sqrt(np.where(cp > 0, cs / np.where(cp > 0, cp, 1), 0.0))
        r = rng_for("ch12", "c06_data", int(100 * a))
        H, C, P = [], [], []
        for _ in range(NDATA):
            alm = ng_alm(r, a, R)
            m = hp.alm2map(hp.almxfl(alm, beam(80.0)), NSIDE, lmax=LMAX) + 0.2 * SIG_S80 * r.standard_normal(NPIX)
            nh, nc, *_ = summarise(m, MASK)
            H.append(nh)
            C.append(nc)
            P.append(pcl(m, MASK))
        res[f"ng{int(100 * a)}_h"], res[f"ng{int(100 * a)}_c"] = np.array(H), np.array(C)
        res[f"ng{int(100 * a)}_pcl"] = np.array(P)
    print("ng", time.time() - t0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **res)
    return res


def checks():
    """Union-find against connected components (nside 128) and against gudhi (nside 32)."""
    r = rng_for("ch12", "c06_checks")
    m = standardise(observe(sky_alm(r), 80.0, 0.2, r), FULL)
    hot, cold = diagrams(m, FULL)
    nh, nc = betti_curves(hot, cold)
    bad = 0
    for k, nu in enumerate(NU):
        h, c = betti_sphere(m >= nu, NSIDE)
        bad += (h != nh[k]) + (c != nc[k])
    # gudhi on a small sphere: lower-star filtration of -u on the 8-neighbour graph
    try:
        import gudhi
    except ImportError:                                      # not installed on every machine
        print("[checks] gudhi not available: results/ch12/c06_gudhi.tex is kept from the last run with gudhi")
        return int(bad), len(NU) * 2
    ns = 32
    npx = hp.nside2npix(ns)
    u = hp.alm2map(hp.almxfl(synalm(CL[:3 * ns], 3 * ns - 1, r), gaussian_beam(300.0, 3 * ns - 1)), ns)
    u = (u - u.mean()) / u.std()
    nb = hp.get_all_neighbours(ns, np.arange(npx)).T
    st = gudhi.SimplexTree()
    for p in range(npx):
        st.insert([p], filtration=-u[p])
    for p in range(npx):
        for q in nb[p]:
            if q > p:
                st.insert([p, q], filtration=max(-u[p], -u[q]))
    st.compute_persistence(persistence_dim_max=False)
    g = np.array(st.persistence_intervals_in_dimension(0))
    g = g[np.isfinite(g[:, 1]) & (g[:, 1] > g[:, 0])]
    b, d = h0_superlevel(u, nb.copy(), np.ones(npx, bool))
    fin = np.isfinite(d)
    ours = np.sort(np.stack([-b[fin], -d[fin]], 1), axis=0)
    theirs = np.sort(g, axis=0)
    gdiff = np.max(np.abs(ours - theirs)) if len(ours) == len(theirs) else np.inf
    save_numbers("ch12", "c06_gudhi", {
        "CpGudhiN": len(ours), "CpGudhiNg": len(theirs),
        "CpGudhiDiff": f"{gdiff:.1e}" if np.isfinite(gdiff) else "inf"})
    return int(bad), len(NU) * 2


def pvalue(ref, obs):
    return (1 + (ref[None, :] >= np.atleast_1d(obs)[:, None]).sum(1)) / (1 + len(ref))


def main():
    res = dict(np.load(OUT)) if OUT.exists() else run()
    _ = summarise(observe(sky_alm(rng_for("ch12", "c06_warm")), 80.0), FULL)   # compile the numba code first
    t0 = time.time()
    _ = summarise(observe(sky_alm(rng), 80.0), FULL)
    t_one = time.time() - t0
    bad, nchk = checks()

    # --- beam: peak hot-spot count versus lambda
    lam = {}
    for fw in (40, 80, 160):
        s2, tau = sigma_tau(CL * beam(float(fw)) ** 2)
        lam[fw] = tau / s2
    peak = {fw: res[f"beam{fw}_h"].mean(0).max() for fw in (40, 80, 160)}
    ratio_obs = peak[40] / peak[160]
    ratio_lam = lam[40] / lam[160]
    k1 = list(NU).index(1.0)
    gkf1 = {fw: 4 * np.pi * lam[fw] * 1.0 * np.exp(-0.5) / (2 * np.pi) ** 1.5 for fw in (40, 80, 160)}
    chi1 = {fw: (res[f"beam{fw}_h"][:, k1] - res[f"beam{fw}_c"][:, k1] + 1).mean() for fw in (40, 80, 160)}

    # --- mask
    fsky = MASK.mean()
    km = list(NU).index(1.5)
    mask_ratio = res["mask_h"].mean(0)[km] / res["noise0_h"].mean(0)[km]

    # --- the test
    def vecs(tag):
        return np.concatenate([res[tag + "_h"], res[tag + "_c"]], 1)

    Nall = vecs("null")
    ok = Nall.std(0) > 1e-9
    Nall = Nall[:, ok]
    Ncov, Nref = Nall[:NCOV], Nall[NCOV:]
    n, p = Ncov.shape
    mu = Ncov.mean(0)
    hart = (n - p - 2) / (n - 1)
    Ci = np.linalg.inv(np.cov(Ncov.T)) * hart
    chi2 = lambda X: np.einsum("...i,ij,...j->...", X - mu, Ci, X - mu)  # noqa: E731
    ref = chi2(Nref)
    # calibration: ranking each reference sky among the others is a permutation and cannot fail, so
    # split the reference skies instead: half are the reference, the other half are tested against it
    nh_ref = len(ref) // 2
    p_null = pvalue(ref[:nh_ref], ref[nh_ref:])
    from scipy import stats
    ks_null = stats.ks_2samp(ref[nh_ref:], ref[:nh_ref]).pvalue
    sd_null = np.sqrt(0.05 * 0.95 / (len(ref) - nh_ref) + 0.05 * 0.95 / (nh_ref + 1))
    out = {}
    for tag in ("ng2", "ng5", "ng15"):
        X = vecs(tag)[:, ok]
        pv = pvalue(ref, chi2(X))
        out[tag] = (np.median(pv), (pv < 0.05).mean())
    # pseudo-C_l test
    Pall = res["null_pcl"]
    Pc, Pr = Pall[:NCOV], Pall[NCOV:]
    mup = Pc.mean(0)
    Cip = np.linalg.inv(np.cov(Pc.T)) * (NCOV - Pc.shape[1] - 2) / (NCOV - 1)
    chi2p = lambda X: np.einsum("...i,ij,...j->...", X - mup, Cip, X - mup)  # noqa: E731
    refp = chi2p(Pr)
    outp = {}
    pcl_ratio = {}
    for tag in ("ng2", "ng5", "ng15"):
        pcl_ratio[tag] = np.max(np.abs(res[tag + "_pcl"].mean(0) / mup - 1) / (Pc.std(0) / mup / np.sqrt(NDATA)))
        pv = pvalue(refp, chi2p(res[tag + "_pcl"]))
        outp[tag] = (np.median(pv), (pv < 0.05).mean())
    # the 'data' skies for the bracket plot: first Gaussian reference sky and first NG sky
    data_g = Nref[0]
    data_ng = vecs("ng15")[0, :][ok]
    p_data_g = pvalue(np.delete(ref, 0), ref[0])[0]
    p_data_ng = pvalue(ref, chi2(data_ng))[0]
    p_data_ng5 = pvalue(ref, chi2(vecs("ng5")[0, :][ok]))[0]

    # ----------------------------------------------------------- figures
    setup(9.0, 3.0)
    fig, ax = plt.subplots(1, 3)
    for fw, c in zip((40, 80, 160), SERIES):
        ax[0].plot(NU, res[f"beam{fw}_h"].mean(0), "-o", ms=2.5, color=c, label=f"FWHM {fw}$'$")
        ax[0].plot(NU, res[f"beam{fw}_c"].mean(0), "--", color=c, lw=1)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1, None)
    ax[0].set_xlabel(r"threshold $\nu$")
    ax[0].set_ylabel("mean count (full sky)")
    ax[0].set_title("beam: hot (solid), cold (dashed)", fontsize=9)
    ax[0].legend(fontsize=7)
    for tag, c, lab in (("noise0", SERIES[0], "no noise"), ("noise20", SERIES[1], r"$\sigma_N=0.2\sigma_S$"),
                        ("noise50", SERIES[2], r"$\sigma_N=0.5\sigma_S$"),
                        ("noise50s", SERIES[3], r"$0.5\sigma_S$, smoothed $60'$")):
        ax[1].plot(NU, res[tag + "_h"].mean(0), "-o", ms=2.5, color=c, label=lab)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1, None)
    ax[1].set_xlabel(r"threshold $\nu$")
    ax[1].set_title(r"noise: hot spots $\beta_0(\nu)$, beam $80'$", fontsize=9)
    ax[1].legend(fontsize=7)
    full = res["noise0_h"].mean(0)
    msk = res["mask_h"].mean(0)
    good = full > 3
    ax[2].plot(NU[good], msk[good] / full[good], "-o", ms=2.5, color=SERIES[0], label="hot spots")
    fullc = res["noise0_c"].mean(0)
    mskc = res["mask_c"].mean(0)
    goodc = fullc > 3
    ax[2].plot(NU[goodc], mskc[goodc] / fullc[goodc], "--s", ms=2.5, color=SERIES[1], label="cold spots")
    ax[2].axhline(fsky, color="k", ls=":", lw=1, label=r"$f_{\rm sky}$")
    ax[2].set_xlabel(r"threshold $\nu$")
    ax[2].set_title("mask: masked / full-sky count", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "ch12", "c_sphere_ops")

    setup(8.0, 3.2)
    fig, ax = plt.subplots(1, 2)
    for tag, c, lab in (("noise50", SERIES[1], r"$\sigma_N=0.5\sigma_S$"), ("noise0", SERIES[0], "no noise")):
        pd = res[tag + "_pd"]
        ax[0].plot(pd[:, 1], pd[:, 0], ".", ms=1.5 if tag == "noise50" else 3, color=c, label=lab,
                   alpha=0.6 if tag == "noise50" else 1)
    ax[0].plot([-4, 4], [-4, 4], color=INK2, lw=0.6)
    ax[0].set_xlim(-4.2, 3)
    ax[0].set_ylim(-3, 4.5)
    ax[0].set_xlabel(r"death level $d$ (hot spot merges)")
    ax[0].set_ylabel(r"birth level $b$ (peak height)")
    ax[0].set_title("hot-spot diagram of one sky", fontsize=9)
    ax[0].legend(fontsize=7, markerscale=3)
    tags = ["noise0", "noise20", "noise50", "noise50s"]
    xl = ["no noise", r"$0.2\sigma_S$", r"$0.5\sigma_S$", r"$0.5\sigma_S$" + "\n" + r"$+\,60'$"]
    xb = np.arange(len(tags))
    ax[1].bar(xb - 0.2, [res[t + "_nfeat"].mean() for t in tags], 0.4, color=SERIES[0], label="all finite hot-spot pairs")
    ax[1].bar(xb + 0.2, [res[t + "_short"].mean() for t in tags], 0.4, color=SERIES[1], label=r"persistence $<0.1\sigma$")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xb)
    ax[1].set_xticklabels(xl, fontsize=7)
    ax[1].set_ylabel("mean number per sky")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "ch12", "c_sphere_noise")

    setup(8.6, 3.0)
    fig, ax = plt.subplots(1, 2)
    sd = np.sqrt(np.diag(np.cov(Ncov.T)))
    nh = int(ok[:13].sum())
    labs = np.concatenate([NU, NU])[ok]
    for a, d, t in ((ax[0], data_g, "a Gaussian 'data' sky"), (ax[1], data_ng, r"local-type sky, $a=0.15$")):
        for sl, mk, lab in ((slice(0, nh), "o-", "hot spots"), (slice(nh, None), "s--", "cold spots")):
            a.plot(labs[sl], ((d - mu) / sd)[sl], mk, ms=3, label=lab)
        a.fill_between([-3.2, 3.2], -2, 2, color=INK2, alpha=0.12, lw=0)
        a.fill_between([-3.2, 3.2], -1, 1, color=INK2, alpha=0.18, lw=0)
        a.set_xlim(-3.2, 3.2)
        a.set_xlabel(r"threshold $\nu$")
        a.set_title(t, fontsize=9)
    ax[0].set_ylabel(r"(count $-$ sim. mean) / sim. sd")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "ch12", "c_sphere_test")

    save_numbers("ch12", "c06_sphere", {
        "CpNside": NSIDE, "CpLmax": LMAX, "CpPixArcmin": f"{hp.nside2resol(NSIDE, arcmin=True):.0f}",
        "CpTone": f"{1000 * t_one:.0f}", "CpChkBad": bad, "CpChkN": nchk,
        "CpLamFourty": f"{lam[40]:.0f}", "CpLamOneSixty": f"{lam[160]:.0f}",
        "CpPeakFourty": f"{peak[40]:.0f}", "CpPeakEighty": f"{peak[80]:.0f}", "CpPeakOneSixty": f"{peak[160]:.0f}",
        "CpRatioObs": f"{ratio_obs:.1f}", "CpRatioLam": f"{ratio_lam:.1f}",
        "CpChiOneEighty": f"{chi1[80]:.0f}", "CpGkfOneEighty": f"{gkf1[80]:.0f}",
        "CpChiOneFourty": f"{chi1[40]:.0f}", "CpGkfOneFourty": f"{gkf1[40]:.0f}",
        "CpNfeatZero": f"{res['noise0_nfeat'].mean():.0f}", "CpNfeatTwenty": f"{res['noise20_nfeat'].mean():.0f}",
        "CpNfeatFifty": f"{res['noise50_nfeat'].mean():.0f}", "CpNfeatFiftyS": f"{res['noise50s_nfeat'].mean():.0f}",
        "CpShortZero": f"{res['noise0_short'].mean():.0f}", "CpShortTwenty": f"{res['noise20_short'].mean():.0f}",
        "CpShortFifty": f"{res['noise50_short'].mean():.0f}", "CpShortFiftyS": f"{res['noise50s_short'].mean():.0f}",
        "CpHotOneZero": f"{res['noise0_h'].mean(0)[k1]:.0f}", "CpHotOneFifty": f"{res['noise50_h'].mean(0)[k1]:.0f}",
        "CpHotOneFiftyS": f"{res['noise50s_h'].mean(0)[k1]:.0f}",
        "CpFsky": f"{fsky:.3f}", "CpMaskRatio": f"{mask_ratio:.3f}",
        "CpNcov": n, "CpNref": len(ref), "CpP": p, "CpHart": f"{hart:.3f}",
        "CpCalib": f"{100 * (p_null < 0.05).mean():.1f}", "CpCalibN": nh_ref,
        "CpCalibSd": f"{100 * sd_null:.1f}", "CpCalibKS": f"{ks_null:.2f}",
        "CpNnull": NNULL, "CpNdata": NDATA, "CpNrenorm": 100,
        "CpMedTwo": f"{out['ng2'][0]:.2f}", "CpPowTwo": f"{100 * out['ng2'][1]:.0f}",
        "CpPclMedTwo": f"{outp['ng2'][0]:.2f}", "CpPclPowTwo": f"{100 * outp['ng2'][1]:.0f}",
        "CpPclPullMax": f"{max(pcl_ratio.values()):.1f}",
        "CpMedFive": f"{out['ng5'][0]:.3f}", "CpPowFive": f"{100 * out['ng5'][1]:.0f}",
        "CpMedFifteen": f"{out['ng15'][0]:.3f}", "CpPowFifteen": f"{100 * out['ng15'][1]:.0f}",
        "CpPclMedFive": f"{outp['ng5'][0]:.2f}", "CpPclPowFive": f"{100 * outp['ng5'][1]:.0f}",
        "CpPclMedFifteen": f"{outp['ng15'][0]:.2f}", "CpPclPowFifteen": f"{100 * outp['ng15'][1]:.0f}",
        "CpPdataG": f"{p_data_g:.2f}", "CpPdataNG": f"{p_data_ng:.3f}", "CpPdataNGfive": f"{p_data_ng5:.2f}",
    })
    print("checks", bad, nchk, "t_one", t_one)
    print("lam", lam, "peaks", peak, ratio_obs, ratio_lam, "chi1", chi1, gkf1)
    print("fsky", fsky, "mask ratio", mask_ratio, "calib", (p_null < 0.05).mean())
    print("pcl pulls", pcl_ratio)
    print("topo", out, "pcl", outp, "data p", p_data_g, p_data_ng, p_data_ng5)


if __name__ == "__main__":
    main()
