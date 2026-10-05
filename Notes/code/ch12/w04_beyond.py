"""Beyond two points: peaks, Minkowski functionals, Betti numbers and persistent Betti numbers.

Question: a KiDS-like convergence map is noisy and skewed.  How well does each summary of it
(power spectrum, peak counts, Minkowski functionals, Betti curves, persistent Betti numbers)
measure (Omega_m, sigma_8), how much does each add to the power spectrum, and does the gain
vanish, as it must, when the field is Gaussian?

Computes, on 6.4 x 6.4 deg periodic patches (1.5 arcmin pixels, n = 6.2 per arcmin^2,
sigma_e = 0.27, Gaussian smoothing 3 arcmin, thresholds in units of the smoothed noise rms):
  * the toy maps of Heydenreich et al. (2021, their Fig. 4): peaks, Betti numbers and H0 diagrams,
    by the planar union-find of lib_persist and by the periodic one of lib_wl (and by gudhi where
    it is installed);
  * for lognormal fields (and, as a control, Gaussian fields with the same C_ell):
    NFID fiducial maps -> covariance of every summary;
    NDER seeds x 4 shifted cosmologies (Omega_m, sigma_8 by +/-5%) with the SAME white noise and
    the SAME shape noise for all four (common random numbers) -> derivatives of the means;
  * the Fisher matrix of every summary and of combinations with the power spectrum, with the
    Hartlap factor, scaled to a 1000 deg^2 survey; sigma(S_8), sigma(Omega_m), figure of merit;
    three estimates: plain (biased high by the noise of the simulated derivatives), with that
    noise bias subtracted, and compressed with one half of the seeds and evaluated with the other
    (biased low);
  * checks: beta_0 against scipy-style component labelling, chi = beta_0 - beta_1 against the
    cubical Euler characteristic, the Betti and persistent Betti numbers of the fiducial maps
    against an earlier run of the same seeds with gudhi's PeriodicCubicalComplex
    (data/ch12/w_beyond_gudhi_ref.npz, if present), peak counts with and without shape noise (Kratochvil et al. 2010),
    the stability of the Fisher matrix between two halves of the derivative seeds.

Writes: data/ch12/w_beyond_sims.npz, figures/ch12/w_beyond_response.pdf,
figures/ch12/w_beyond_fisher.pdf, figures/ch12/w_beyond_peaks.pdf, results/ch12/w04_beyond.tex
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA
import lib_wl as wl
import lib_topo as topo
import lib_persist as lp

CACHE = DATA / "ch12" / "w_beyond_sims.npz"
NFID = {"ln": 1000, "g": 600}          # fiducial maps (covariance)
NDER = {"ln": 800, "g": 400}           # seeds for the derivatives (each gives 4 maps)
DSTEP = 0.10                           # derivative steps: Omega_m, sigma_8 moved by +/-10% (w04a)
NUS = np.arange(-1.5, 3.01, 0.5)       # thresholds for Minkowski functionals and Betti curves
PEAK_EDGES = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 12.0])
PB_GRID = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])   # persistent Betti numbers on pairs of this grid
SHIFTS = ("om_p", "om_m", "s8_p", "s8_m")
BLOCKS = ("cl", "pk", "mf", "betti", "pbetti")


# ------------------------------------------------------------------ the toy maps (Heydenreich Fig. 4)
TOY = [np.zeros((7, 10)), np.zeros((7, 10)), np.zeros((7, 10))]
TOY[0][1, 2], TOY[0][2, 7], TOY[0][5, 1], TOY[0][5, 8] = 4, 1, 2, 3
TOY[1][1:6, 2:7] = [[2, 2, 2, 1, 1], [2, 4, 2, 3, 1], [2, 2, 1, 1, 1], [0, 0, 1, 2, 1], [1, 0, 1, 1, 1]]
TOY[2][1:4, 1:4] = [[2, 2, 2], [2, 4, 2], [2, 2, 2]]
TOY[2][3:6, 6:9] = [[1, 1, 1], [1, 3, 1], [1, 1, 1]]


def toy_report():
    """Peaks, beta_0 at nu = 1..4, and H0 pairs of the three toy maps, two (or three) ways."""
    try:
        import gudhi
    except ImportError:
        gudhi = None
    out = []
    for f in TOY:
        mx = np.zeros_like(f)
        pad = np.pad(f, 1, constant_values=-1)
        nb = np.max([pad[1 + a:8 + a, 1 + b:11 + b] for a in (-1, 0, 1) for b in (-1, 0, 1)
                     if (a, b) != (0, 0)], axis=0)
        peaks = sorted(f[f > nb].astype(int).tolist(), reverse=True)
        b0 = [topo.n_components(f >= nu, 8) for nu in (1, 2, 3, 4)]
        mine = lp.islands(f)                                   # union-find, superlevel, planar
        mine = sorted([(int(b), d) for b, d in mine], reverse=True)
        per = wl.persistence_pairs(f)[0]                       # union-find, superlevel, periodic
        per = sorted([(int(b), d) for b, d in per], reverse=True)
        gd = None
        if gudhi is not None:
            cc = gudhi.CubicalComplex(top_dimensional_cells=-f)
            cc.compute_persistence()
            gd = np.array(cc.persistence_intervals_in_dimension(0))
            gd = sorted([(int(-b), (-d if np.isfinite(d) else -np.inf)) for b, d in gd], reverse=True)
        out.append(dict(peaks=peaks, b0=b0, mine=mine, periodic=per, gudhi=gd))
    return out


# ------------------------------------------------------------------ one map -> every summary
def summaries(kE, sig_n):
    """Data vectors of one noisy kappa_E map (unsmoothed, periodic)."""
    cl = wl.band_power(kE, wl.CL_EDGES)
    u = wl.smooth(kE, wl.THETA_G) / sig_n                     # S/N map
    pk, _ = wl.peaks(u, PEAK_EDGES)
    v0, v1, v2 = wl.minkowski(u, NUS)
    h0, h1 = wl.persistence_pairs(u)
    betti = np.concatenate([wl.betti_from_pairs(h0, NUS), wl.betti_from_pairs(h1, NUS)])
    pbetti = np.concatenate([wl.persistent_betti(h0, PB_GRID), wl.persistent_betti(h1, PB_GRID)])
    return dict(cl=cl, pk=pk, mf=np.concatenate([v0, v1, v2]), betti=betti, pbetti=pbetti), u, (h0, h1)


def model(sp, name, kind):
    P = wl.cl_on_grid(sp["ells"], sp[f"cl_{name}"])
    lam = float(sp[f"lam_{name}"])
    if kind == "g":
        return lambda w: wl.gaussian_map(P, w)
    PG, _ = wl.lognormal_spectrum(P, lam)
    return lambda w: wl.lognormal_map(PG, lam, w)


def observe(kappa, noise, s_pix):
    g1, g2 = wl.shear_from_kappa(kappa)
    kE, _ = wl.kaiser_squires(g1 + s_pix * noise[0], g2 + s_pix * noise[1])
    return kE


def simulate(sp, sp10, kind, sig_n):
    rng = rng_for("ch12", f"w04_{kind}")
    s_pix = wl.sigma_pix()
    fid = model(sp, "fid", kind)
    shifted = {k: model(sp10, k, kind) for k in SHIFTS}
    F = {b: [] for b in BLOCKS}
    D = {k: {b: [] for b in BLOCKS} for k in SHIFTS}
    for _ in range(NFID[kind]):
        w = rng.standard_normal((wl.N, wl.N))
        n = rng.standard_normal((2, wl.N, wl.N))
        s, _, _ = summaries(observe(fid(w), n, s_pix), sig_n)
        for b in BLOCKS:
            F[b].append(s[b])
    for _ in range(NDER[kind]):
        w = rng.standard_normal((wl.N, wl.N))
        n = rng.standard_normal((2, wl.N, wl.N))
        for k in SHIFTS:                                       # common random numbers
            s, _, _ = summaries(observe(shifted[k](w), n, s_pix), sig_n)
            for b in BLOCKS:
                D[k][b].append(s[b])
    out = {}
    for b in BLOCKS:
        out[f"{kind}_fid_{b}"] = np.array(F[b], float)
        for k in SHIFTS:
            out[f"{kind}_{k}_{b}"] = np.array(D[k][b], float)
    return out


# ------------------------------------------------------------------ Fisher matrices
PB_DIAG = [k for k, (i, j) in enumerate((i, j) for i in range(len(PB_GRID)) for j in range(i + 1)) if i == j]


def block(sims, kind, which, b):
    """One block of the data vector; 'pboff' = persistent Betti numbers without the diagonal
    (lo = hi), which repeats the plain Betti numbers and, through chi = beta_0 - beta_1, the MFs."""
    if b == "pboff":
        x = sims[f"{kind}_{which}_pbetti"]
        n = x.shape[1] // 2
        off = [k for k in range(n) if k not in PB_DIAG]
        return x[:, off + [k + n for k in off]]
    return sims[f"{kind}_{which}_{b}"]


def fisher(sims, kind, blocks, h):
    """Fisher matrices of the concatenated blocks, per patch.

    C from the fiducial maps (Hartlap-corrected inverse).  Per-seed derivatives
    D_k = [s(theta + h) - s(theta - h)] / 2h (common random numbers).  Returns
      F_raw  = dbar^T C^-1 dbar                        (biased high by the derivative noise)
      F_deb  = F_raw - tr(C^-1 Sigma_ij) / N_der       (that bias subtracted)
      F_cmp  = compressed with half A, evaluated with half B, averaged over A <-> B (biased low)
    """
    fid = np.hstack([block(sims, kind, "fid", b) for b in blocks])
    keep = fid.std(0) > 0                                       # drop components that never change
    fid = fid[:, keep]
    sd = fid.std(0)
    fid = fid / sd
    C = np.cov(fid, rowvar=False)
    n, p = fid.shape
    hart = (n - p - 2) / (n - 1)
    Ci = hart * np.linalg.inv(C)
    D = []
    for (kp, km), hi in ((("om_p", "om_m"), h[0]), (("s8_p", "s8_m"), h[1])):
        a = np.hstack([block(sims, kind, kp, b) for b in blocks])[:, keep] / sd
        c = np.hstack([block(sims, kind, km, b) for b in blocks])[:, keep] / sd
        D.append((a - c) / (2 * hi))
    D = np.array(D)                                             # (2, N_der, p)
    nd = D.shape[1]
    d = D.mean(1)
    F_raw = d @ Ci @ d.T
    bias = np.array([[np.trace(Ci @ np.cov(D[i].T, D[j].T)[:p, p:]) / nd for j in range(2)] for i in range(2)])
    F_deb = F_raw - bias
    F_cmp = np.zeros((2, 2))
    halves = (slice(0, nd // 2), slice(nd // 2, nd))
    for A, B in (halves, halves[::-1]):
        dA, dB = D[:, A].mean(1), D[:, B].mean(1)
        Amat = dA @ Ci                                          # compression vectors (2, p)
        Ct = Amat @ (Ci_inv := np.linalg.inv(Ci)) @ Amat.T      # covariance of the compressed pair
        J = Amat @ dB.T                                         # its response, from the other half
        F_cmp += 0.5 * J.T @ np.linalg.inv(Ct) @ J
    return dict(raw=F_raw, deb=F_deb, cmp=F_cmp, bias=bias, p=p, hart=hart)


def errors(F, th, scale):
    if np.any(np.linalg.eigvalsh(F) <= 0):
        return dict(cov=np.full((2, 2), np.nan), sS8=np.nan, sOm=np.nan, sS=np.nan, fom=np.nan, r=np.nan)
    cov = np.linalg.inv(F * scale)
    S8 = th[1] * np.sqrt(th[0] / 0.3)
    g = np.array([0.5 * S8 / th[0], S8 / th[1]])
    return dict(cov=cov, sS8=np.sqrt(g @ cov @ g), sOm=np.sqrt(cov[0, 0]), sS=np.sqrt(cov[1, 1]),
                fom=1 / np.sqrt(np.linalg.det(cov)), r=cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def main():
    setup()
    t0 = time.time()
    sp = dict(np.load(DATA / "ch12" / "w_cl.npz"))
    sp10 = dict(np.load(DATA / "ch12" / "w_cl10.npz"))          # +/-10% cosmologies (w04a_spectra.py)
    sig_n, _ = wl.sigma_noise_smoothed()
    th = np.array([wl.OM_FID, wl.S8_FID])
    h = DSTEP * th
    scale = wl.AREA_DEG2 / wl.A_PATCH_DEG2                       # independent patches in 1000 deg^2

    # ---------------------------------------------------------------- toy maps
    toy = toy_report()
    for i, t in enumerate(toy):
        print("toy", i + 1, t)
    toy_ok = all(t["mine"] == t["periodic"] and (t["gudhi"] is None or t["mine"] == t["gudhi"])
                 for t in toy)

    # ---------------------------------------------------------------- one map: checks
    rng = rng_for("ch12", "w04_checks")
    P = wl.cl_on_grid(sp["ells"], sp["cl_fid"])
    lam = float(sp["lam_fid"])
    PG, _ = wl.lognormal_spectrum(P, lam)
    w = rng.standard_normal((wl.N, wl.N))
    n = rng.standard_normal((2, wl.N, wl.N))
    kap = wl.lognormal_map(PG, lam, w)
    kE = observe(kap, n, wl.sigma_pix())
    s, u, (h0, h1) = summaries(kE, sig_n)
    b0 = wl.betti_from_pairs(h0, NUS)
    b1 = wl.betti_from_pairs(h1, NUS)
    lab = np.array([topo.n_components(u >= nu, 8, periodic=True) for nu in NUS])
    chi = np.array([topo.euler_cubical(u >= nu, 8, periodic=True) for nu in NUS])
    hi = NUS >= 1.0
    dev_b0 = int(np.abs(lab - b0).max())
    dev_chi = int(np.abs(chi - (b0 - b1))[hi].max())
    # peaks with and without shape noise (Kratochvil et al. 2010): 50 maps
    npk_noisy, npk_clean, hist_noisy, hist_clean = [], [], [], []
    edges_k = np.linspace(-2, 7, 37)
    for _ in range(50):
        w = rng.standard_normal((wl.N, wl.N))
        n = rng.standard_normal((2, wl.N, wl.N))
        kap = wl.lognormal_map(PG, lam, w)
        clean = wl.smooth(kap, wl.THETA_G) / sig_n
        noisy = wl.smooth(observe(kap, n, wl.sigma_pix()), wl.THETA_G) / sig_n
        hc, _ = wl.peaks(clean, edges_k)
        hn, _ = wl.peaks(noisy, edges_k)
        _, allc = wl.peaks(clean, np.array([-1e9, 1e9]))
        _, alln = wl.peaks(noisy, np.array([-1e9, 1e9]))
        npk_clean.append(len(allc)); npk_noisy.append(len(alln))
        hist_clean.append(hc); hist_noisy.append(hn)
    boost = np.mean(npk_noisy) / np.mean(npk_clean)

    # ---------------------------------------------------------------- the simulations (cached)
    if CACHE.exists():
        sims = dict(np.load(CACHE))
    else:
        sims = {}
        for kind in ("ln", "g"):
            sims.update(simulate(sp, sp10, kind, sig_n))
            print(kind, "done", time.time() - t0)
        np.savez(CACHE, **sims)

    # ---------------------------------------------------------------- cross-check with gudhi (earlier run)
    ref_path = DATA / "ch12" / "w_beyond_gudhi_ref.npz"
    gudhi_dev, gudhi_n = "n/a", 0
    if ref_path.exists():
        ref = np.load(ref_path)
        dev = 0
        for kind in ("ln", "g"):
            for b in ("betti", "pbetti"):
                k = f"{kind}_fid_{b}"
                dev = max(dev, int(np.abs(ref[k] - sims[k]).max()))
                gudhi_n += ref[k].shape[0]
        gudhi_dev = dev
        gudhi_n //= 2
    print("gudhi reference: max deviation", gudhi_dev, "over", gudhi_n, "maps")

    # ---------------------------------------------------------------- Fisher for every combination
    combos = {"cl": ("cl",), "pk": ("pk",), "mf": ("mf",), "betti": ("betti",), "pbetti": ("pbetti",),
              "cl+pk": ("cl", "pk"), "cl+mf": ("cl", "mf"), "cl+betti": ("cl", "betti"),
              "cl+pbetti": ("cl", "pbetti"), "all": ("cl", "pk", "mf", "pboff")}
    res, F_all = {}, {}
    for kind in ("ln", "g"):
        for name, blocks in combos.items():
            f = fisher(sims, kind, blocks, h)
            F_all[(kind, name)] = f
            e = {k: errors(f[k], th, scale) for k in ("raw", "deb", "cmp")}
            e.update(p=f["p"], hart=f["hart"], biasfrac=np.max(np.diag(f["bias"]) / np.diag(f["raw"])))
            res[(kind, name)] = e
            print(kind, name, "p", f["p"], "hart %.3f" % f["hart"],
                  " ".join("%s: sS8 %.4f sOm %.3f fom %.0f" % (k, e[k]["sS8"], e[k]["sOm"], e[k]["fom"])
                           for k in ("raw", "deb", "cmp")), "bias/F %.3f" % e["biasfrac"])
    # peaks split by height: which peaks carry the information?
    pk_sets = {"low": slice(0, 3), "mid": slice(3, 7), "high": slice(7, 11)}  # nu<0.5, 0.5-2.5, >2.5
    pk_res = {}
    for nm, sl in pk_sets.items():
        sims_sub = {k: (v[:, sl] if k.endswith("_pk") else v) for k, v in sims.items()}
        pk_res[nm] = errors(fisher(sims_sub, "ln", ("pk",), h)["deb"], th, scale)
        print("peaks", nm, pk_res[nm]["sS8"], pk_res[nm]["fom"])

    # ---------------------------------------------------------------- figure 1: responses
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=True)
    panels = [("pk", "peak counts", 0.5 * (PEAK_EDGES[1:] + PEAK_EDGES[:-1]), slice(0, 11), r"peak height $\nu$"),
              ("mf", r"Euler characteristic $V_2$", NUS, slice(20, 30), r"threshold $\nu$"),
              ("betti", r"islands $\beta_0$", NUS, slice(0, 10), r"threshold $\nu$"),
              ("betti", r"lakes $\beta_1$", NUS, slice(10, 20), r"threshold $\nu$")]
    for a, (b, title, x, sl, xl) in zip(ax.ravel(), panels):
        fid = sims[f"ln_fid_{b}"][:, sl]
        sd = fid.std(0)
        for (kp, km), lab_, c in ((("s8_p", "s8_m"), r"$\partial/\partial\ln\sigma_8$", SERIES[0]),
                                  (("om_p", "om_m"), r"$\partial/\partial\ln\Omega_m$", SERIES[1])):
            d = (sims[f"ln_{kp}_{b}"][:, sl] - sims[f"ln_{km}_{b}"][:, sl]).mean(0) / (2 * DSTEP)
            a.plot(x, np.where(sd > 0, d / np.where(sd > 0, sd, 1), 0), "-o", ms=3, color=c, label=lab_)
        a.axhline(0, color="k", lw=0.6)
        a.set_title(title, fontsize=9)
        a.set_xlabel(xl)
    ax[0, 0].set_ylabel("response / scatter of one patch")
    ax[1, 0].set_ylabel("response / scatter of one patch")
    ax[0, 0].legend(fontsize=8)
    savefig(fig, "ch12", "w_beyond_response")

    # ---------------------------------------------------------------- figure 2: ellipses (noise-bias subtracted)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), constrained_layout=True, sharey=True)
    show = [("cl", SERIES[0], r"$C_\ell$"), ("pk", SERIES[1], "peaks"), ("mf", SERIES[3], "Minkowski"),
            ("pbetti", SERIES[2], "persistent Betti"), ("all", "k", "all combined")]
    for a, kind, ttl in zip(ax, ("ln", "g"), ("lognormal maps", "Gaussian maps (control)")):
        for name, c, lab_ in show:
            cov = res[(kind, name)]["deb"]["cov"]
            if not np.all(np.isfinite(cov)):
                continue
            wv, vv = np.linalg.eigh(cov)
            ang = np.degrees(np.arctan2(vv[1, 1], vv[0, 1]))
            a.add_patch(Ellipse(th, 2 * np.sqrt(2.30 * wv[1]), 2 * np.sqrt(2.30 * wv[0]), angle=ang,
                                fill=False, color=c, lw=1.3, ls="-" if name != "all" else "--", label=lab_))
        xx = np.linspace(0.15, 0.5, 100)
        S8f = th[1] * np.sqrt(th[0] / 0.3)
        a.plot(xx, S8f * np.sqrt(0.3 / xx), ":", color="0.5", lw=0.8)
        a.plot(*th, "k+")
        a.set_xlim(0.18, 0.46); a.set_ylim(0.62, 1.02)
        a.set_xlabel(r"$\Omega_m$"); a.set_title(ttl, fontsize=9)
    ax[0].set_ylabel(r"$\sigma_8$")
    ax[0].legend(fontsize=7.5, loc="upper right")
    savefig(fig, "ch12", "w_beyond_fisher")

    # ---------------------------------------------------------------- figure 3: peaks with and without noise
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    xc = 0.5 * (edges_k[1:] + edges_k[:-1])
    ax.step(xc, np.mean(hist_clean, 0), where="mid", color=SERIES[0], label="noise-free, smoothed")
    ax.step(xc, np.mean(hist_noisy, 0), where="mid", color=SERIES[1], label="with shape noise")
    ax.set_xlabel(r"peak height $\nu=\kappa/\sigma_{\rm noise}$")
    ax.set_ylabel(r"peaks per $41\,{\rm deg}^2$ per bin")
    ax.legend(fontsize=8)
    savefig(fig, "ch12", "w_beyond_peaks")

    # ---------------------------------------------------------------- numbers
    nums = {
        "WToyOK": "agree" if toy_ok else "DISAGREE",
        "WGudhiDev": gudhi_dev, "WGudhiN": gudhi_n,
        "WDevBzero": dev_b0, "WDevChi": dev_chi,
        "WNpkClean": f"{np.mean(npk_clean):.0f}", "WNpkNoisy": f"{np.mean(npk_noisy):.0f}",
        "WPkBoost": f"{100 * (boost - 1):.0f}",
        "WNfidLN": NFID["ln"], "WNderLN": NDER["ln"], "WNfidG": NFID["g"], "WNderG": NDER["g"],
        "WNmapsTot": NFID["ln"] + 4 * NDER["ln"] + NFID["g"] + 4 * NDER["g"],
        "WScale": f"{scale:.1f}", "WDstep": f"{100 * DSTEP:.0f}",
        "WSigNoiseSm": f"{sig_n:.4f}",
        "WPall": res[("ln", "all")]["p"], "WHartAll": f"{res[('ln', 'all')]['hart']:.2f}",
        "WRunMin": f"{(time.time() - t0) / 60:.1f}",
        "WPkLowSE": f"{pk_res['low']['sS8']:.3f}", "WPkMidSE": f"{pk_res['mid']['sS8']:.3f}",
        "WPkHighSE": f"{pk_res['high']['sS8']:.3f}",
        "WPkLowFom": f"{pk_res['low']['fom']:.0f}", "WPkMidFom": f"{pk_res['mid']['fom']:.0f}",
        "WPkHighFom": f"{pk_res['high']['fom']:.0f}",
    }
    tag = {"cl": "Cl", "pk": "Pk", "mf": "Mf", "betti": "Betti", "pbetti": "Pbetti", "cl+pk": "ClPk",
           "cl+mf": "ClMf", "cl+betti": "ClBetti", "cl+pbetti": "ClPbetti", "all": "All"}
    est = {"deb": "", "raw": "Raw", "cmp": "Cmp"}
    for kind, K in (("ln", "L"), ("g", "G")):
        ref = res[(kind, "cl")]["deb"]
        for name, t in tag.items():
            e = res[(kind, name)]
            nums[f"W{K}{t}P"] = e["p"]
            nums[f"W{K}{t}Bias"] = f"{100 * e['biasfrac']:.1f}"
            for k, suf in est.items():
                ee = e[k]
                nums[f"W{K}{t}SE{suf}"] = f"{ee['sS8']:.4f}"
                nums[f"W{K}{t}Om{suf}"] = f"{ee['sOm']:.3f}"
                nums[f"W{K}{t}Fom{suf}"] = f"{ee['fom']:.0f}"
                nums[f"W{K}{t}Gain{suf}"] = f"{ee['fom'] / ref['fom']:.2f}"
                nums[f"W{K}{t}SEgain{suf}"] = f"{100 * (1 - ee['sS8'] / ref['sS8']):.0f}"
            nums[f"W{K}{t}R"] = f"{e['deb']['r']:.3f}"
    L = {k: v["deb"] for k, v in res.items()}
    nums["WLAllOverPk"] = f"{L[('ln', 'all')]['fom'] / L[('ln', 'pk')]['fom']:.2f}"
    nums["WLPkOverCl"] = f"{L[('ln', 'pk')]['fom'] / L[('ln', 'cl')]['fom']:.2f}"
    nums["WLClPkOverPk"] = f"{L[('ln', 'cl+pk')]['fom'] / L[('ln', 'pk')]['fom']:.2f}"
    nums["WLClPkOverCl"] = f"{L[('ln', 'cl+pk')]['fom'] / L[('ln', 'cl')]['fom']:.2f}"
    nums["WLPbOverPk"] = f"{L[('ln', 'pbetti')]['fom'] / L[('ln', 'pk')]['fom']:.2f}"
    nums["WLPbOverPkSE"] = f"{100 * (1 - L[('ln', 'pbetti')]['sS8'] / L[('ln', 'pk')]['sS8']):.0f}"
    nums["WLBettiOverPk"] = f"{L[('ln', 'betti')]['fom'] / L[('ln', 'pk')]['fom']:.2f}"
    nums["WLPbOverBetti"] = f"{L[('ln', 'pbetti')]['fom'] / L[('ln', 'betti')]['fom']:.2f}"
    # figures of merit in the convention of Dietrich & Hartlap (2010): 1 / area of the 95% ellipse
    # (Delta chi^2 = 5.99 for two parameters), for their 180 deg^2 survey (FoM scales with the area)
    for name, t in (("cl", "Cl"), ("pk", "Pk"), ("cl+pk", "ClPk")):
        nums[f"WL{t}FomDH"] = f"{L[('ln', name)]['fom'] * (180 / wl.AREA_DEG2) / (np.pi * 5.991):.0f}"
    nums["WLPkOverClSE"] = f"{100 * (1 - L[('ln', 'pk')]['sS8'] / L[('ln', 'cl')]['sS8']):.0f}"
    nums["WLBettiOverPkSE"] = f"{100 * (1 - L[('ln', 'betti')]['sS8'] / L[('ln', 'pk')]['sS8']):.0f}"
    save_numbers("ch12", "w04_beyond", nums)
    print(nums)
    print("minutes", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
