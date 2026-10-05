"""b08_charge_stats.py -- the charge statistics of polarization singularities (Vachaspati-Lue, Huterer-Vachaspati).

Question: the net index q enclosed by a loop of length L fluctuates around zero.  Does its
root-mean-square grow as L^(1/2), as Vachaspati and Lue (2003) argued, with the exponents
nu = 0.54 (l_max = 100) and 0.45 (l_max = 500) that Huterer and Vachaspati (2005) measured?
Is it the pairing of opposite indices that makes the exponent 1/2?  And are singularities of
opposite index closer to each other than those of equal index (their mean nearest-neighbour
distances at l_max = 100: 3.24 deg and 2.69 deg)?
Computes:
 (A) their method on the full sky: E-mode HEALPix maps (N_side = 256, as theirs) with the
     spectrum of their cosmology cut at l_max; the winding of arg(Q+iU) along every ring of
     the northern hemisphere = the index enclosed by the polar cap; sigma(q) over independent
     maps against the ring length; power-law fit over their range.
 (B) flat periodic 40 x 40 deg patches: the singularities themselves; q inside circles with
     the true indices and with the indices shuffled among the singularities; nearest-neighbour
     distances by index.
Writes: figures/ch12/b_charge_stats.pdf, results/ch12/b08_charge_stats.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA
from lib_persist import flat_qu, winding

rng = rng_for("ch12", "b08_charge_stats")
ee = np.load(DATA / "ch12" / "b_hv_spectra.npz")["EE_noreion"]   # written by b07_hv_counts.py

# ---------------------------------------------------------------- (A) full sky, rings
NSIDE, NMAP = 256, 100
s_grid = 0.01 + 0.02 * np.arange(50)                       # their abscissa: 0.01, 0.03, ..., 0.99
rings = np.arange(1, 2 * NSIDE)                            # northern-hemisphere rings
start, npr, cth, _, _ = hp.ringinfo(NSIDE, rings)
sinth = np.sqrt(1 - cth ** 2)
pick = np.array([np.argmin(np.abs(sinth - s)) for s in s_grid])   # ring closest to each sin(theta)
ring_pix = [np.arange(start[k], start[k] + npr[k]) for k in pick]


def ring_charges(Q, U):
    """Index enclosed by each polar cap = winding of arg(Q+iU) along its boundary ring / 4pi
    (up to a constant from the turning of the local axes around the pole)."""
    out = []
    for idx in ring_pix:
        a = np.angle(Q[idx] + 1j * U[idx])
        d = np.angle(np.exp(1j * (np.roll(a, -1) - a)))     # steps the short way round
        out.append(np.rint(d.sum() / (2 * np.pi)) / 2)
    return np.array(out)


def full_sky(lmax):
    cl = np.zeros(lmax + 1)
    cl[2:] = ee[2: lmax + 1]
    zero = np.zeros(hp.Alm.getsize(lmax), complex)
    ell, em = hp.Alm.getlm(lmax)
    sd = np.sqrt(cl[ell])
    q = []
    for m in range(NMAP):
        re, im = rng.standard_normal(ell.size), rng.standard_normal(ell.size)
        almE = np.where(em == 0, sd * re, sd * (re + 1j * im) / np.sqrt(2))   # E_lm with our rng
        _, Q, U = hp.alm2map([zero, almE, zero], NSIDE, lmax=lmax, pol=True)
        q.append(ring_charges(Q, U))
    q = np.array(q)
    sig = q.std(axis=0)                                     # spread over maps, ring by ring

    def slope(qq):
        sd = qq.std(axis=0)
        ok = sd > 0                                         # a few small caps may hold no charge
        return np.polyfit(np.log(s_grid[ok]), np.log(sd[ok]), 1)[0]

    nu = slope(q)
    # error: bootstrap over the maps (resample the NMAP maps with replacement, refit the slope); its own
    # stream, so the map draws and the flat-patch part below keep their random numbers
    rb = rng_for("ch12", "b08_boot", lmax)
    nu_err = np.std([slope(q[rb.integers(0, NMAP, NMAP)]) for _ in range(400)], ddof=1)
    return sig, nu, nu_err, q.mean(axis=0).mean()


fs = {}
for lmax in (100, 500):
    fs[lmax] = full_sky(lmax)
    print(f"full sky l_max={lmax}: nu = {fs[lmax][1]:.3f} +- {fs[lmax][2]:.3f}, "
          f"sigma(q) at sin(theta)=0.01, 1: {fs[lmax][0][0]:.2f}, {fs[lmax][0][-1]:.2f};"
          f" mean q {fs[lmax][3]:.2f}")

# ---------------------------------------------------------------- (B) flat patches
SIDE, NCEN = 40.0, 40
Lc = np.logspace(-2, np.log10(1.2), 14)                    # circle circumferences [rad]


def flat(lmax, npix, nsim):
    pix = np.deg2rad(SIDE) / npix
    qt, qs = np.zeros((nsim, NCEN, len(Lc))), np.zeros((nsim, NCEN, len(Lc)))
    same, opp, near, nsing = [], [], [], 0
    for s in range(nsim):
        Q, U, _ = flat_qu(ee, npix, SIDE, rng, lmax=lmax)
        w = winding(Q, U)
        i, j = np.nonzero(w)
        pos = np.column_stack([i, j]) + 0.5                  # plaquette centres, pixel units
        q = w[i, j]
        nsing += len(q)
        qsh = rng.permutation(q)                             # same positions, shuffled indices
        cen = rng.uniform(0, npix, (NCEN, 2))
        d = np.abs(cen[:, None, :] - pos[None, :, :])
        d = np.minimum(d, npix - d)                          # periodic distance
        r = np.hypot(d[..., 0], d[..., 1]) * pix
        for k, L in enumerate(Lc):
            inside = r < L / (2 * np.pi)
            qt[s, :, k] = inside @ q
            qs[s, :, k] = inside @ qsh
        tp, tm = cKDTree(pos[q > 0], boxsize=npix), cKDTree(pos[q < 0], boxsize=npix)
        dpp = tp.query(pos[q > 0], k=2)[0][:, 1]             # k=2: the first hit is the point itself
        dmm = tm.query(pos[q < 0], k=2)[0][:, 1]
        dpm = tm.query(pos[q > 0], k=1)[0]
        dmp = tp.query(pos[q < 0], k=1)[0]
        deg = np.rad2deg(pix)
        same.append(np.r_[dpp, dmm] * deg)
        opp.append(np.r_[dpm, dmp] * deg)
        near.append(np.r_[np.minimum(dpp, dpm), np.minimum(dmm, dmp)] * deg)
    sig_t = qt.reshape(-1, len(Lc)).std(axis=0)
    sig_s = qs.reshape(-1, len(Lc)).std(axis=0)
    big = Lc > 0.2                                          # loops much longer than a pair
    nu_t = np.polyfit(np.log(Lc[big]), np.log(sig_t[big]), 1)[0]
    nu_s = np.polyfit(np.log(Lc[big]), np.log(sig_s[big]), 1)[0]
    density = nsing / (nsim * np.rad2deg(np.deg2rad(SIDE)) ** 2)   # per square degree
    return dict(sig_t=sig_t, sig_s=sig_s, nu_t=nu_t, nu_s=nu_s, same=np.concatenate(same),
                opp=np.concatenate(opp), near=np.concatenate(near), density=density)


fl = {}
for lmax, npix, nsim in ((100, 256, 40), (500, 512, 12)):
    fl[lmax] = r = flat(lmax, npix, nsim)
    print(f"flat l_max={lmax}: nu (L>0.2) true {r['nu_t']:.2f}, shuffled {r['nu_s']:.2f}; NN same"
          f" {r['same'].mean():.2f}, opposite {r['opp'].mean():.2f}, any {r['near'].mean():.2f} deg;"
          f" density {r['density']:.3f}/deg^2")

# ---------------------------------------------------------------- figure
setup(7.4, 3.0)
fig, ax = plt.subplots(1, 2)
hv = {100: (0.54, 0.55, SERIES[0]), 500: (0.45, 1.75, SERIES[1])}  # nu, their fit at 0.01 (read off)
for lmax in (100, 500):
    nu_hv, a_hv, c = hv[lmax]
    sig, nu, nu_err, _ = fs[lmax]
    ax[0].loglog(s_grid, sig, "o", ms=3, color=c,
                 label=rf"$\ell_{{\max}}={lmax}$: ours $\nu={nu:.2f}$")
    ok = sig > 0
    a_us = np.exp(np.polyfit(np.log(s_grid[ok]), np.log(sig[ok]), 1)[1])
    ax[0].loglog(s_grid, a_us * s_grid ** nu, "-", color=c, lw=0.8)
    ax[0].loglog(s_grid, a_hv * (s_grid / 0.01) ** nu_hv, "--", color=c, lw=1,
                 label=rf"H\&V fit (read off): $\nu={nu_hv}$")
ax[0].set_xlabel(r"ring length / $2\pi$  $(=\sin\theta)$")
ax[0].set_ylabel(r"rms enclosed index $\sigma(q)$")
ax[0].legend(fontsize=6.5, loc="upper left")
r = fl[100]
bins = np.linspace(0, 8, 41)
ax[1].hist(r["same"], bins, density=True, histtype="step", color="k", label="same index")
ax[1].hist(r["opp"], bins, density=True, histtype="step", color=SERIES[1], ls="--",
           label="opposite index")
ax[1].axvline(3.24, color="k", lw=0.8, alpha=0.6)
ax[1].axvline(2.69, color=SERIES[1], lw=0.8, alpha=0.6)
ax[1].set_xlabel("distance to nearest neighbour [deg]")
ax[1].set_ylabel("density")
ax[1].set_title(r"$\ell_{\max}=100$; vertical lines: H\&V means", fontsize=9)
ax[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch12", "b_charge_stats")

setup(4.2, 3.0)
fig, ax = plt.subplots()
for lmax, c in ((100, SERIES[0]), (500, SERIES[1])):
    r = fl[lmax]
    ax.loglog(Lc, r["sig_t"], "o-", ms=3, color=c, label=rf"$\ell_{{\max}}={lmax}$, true indices")
    ax.loglog(Lc, r["sig_s"], "x--", ms=4, color=c, label="indices shuffled")
ax.set_xlabel(r"circumference $L$ of the circle [rad]")
ax.set_ylabel(r"$\sigma(q)$")
ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch12", "b_charge_shuffle")

save_numbers("ch12", "b08_charge_stats", {
    "PbQnside": NSIDE, "PbQnmap": NMAP,
    "PbNuHundred": fs[100][1], "PbNuFiveHundred": fs[500][1],
    "PbNuErrHundred": fs[100][2], "PbNuErrFiveHundred": fs[500][2],
    "PbNuDiffFive": f"{fs[500][1] - 0.45:.2f}",                    # ours minus their 0.45 at l_max = 500
    "PbNuPullFive": f"{(fs[500][1] - 0.45) / np.hypot(fs[500][2], 0.01):.1f}",
    "PbNuPullHundred": f"{abs(fs[100][1] - 0.54) / np.hypot(fs[100][2], 0.01):.1f}",
    "PbSigLowHundred": fs[100][0][0], "PbSigHighHundred": fs[100][0][-1],
    "PbSigLowFiveHundred": fs[500][0][0], "PbSigHighFiveHundred": fs[500][0][-1],
    "PbQside": SIDE, "PbQncen": NCEN,
    "PbNuFlatHundred": fl[100]["nu_t"], "PbNuFlatFiveHundred": fl[500]["nu_t"],
    "PbNuShufHundred": fl[100]["nu_s"], "PbNuShufFiveHundred": fl[500]["nu_s"],
    "PbNNsame": fl[100]["same"].mean(), "PbNNopp": fl[100]["opp"].mean(),
    "PbNNall": fl[100]["near"].mean(), "PbNNcount": len(fl[100]["near"]),
    "PbNNratio": fl[100]["same"].mean() / fl[100]["opp"].mean(),
    "PbNNdensity": fl[100]["density"],
    "PbNNrandom": 0.5 / np.sqrt(fl[100]["density"]),
})
