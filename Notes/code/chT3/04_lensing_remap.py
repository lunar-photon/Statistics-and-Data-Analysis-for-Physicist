"""04_lensing_remap.py -- does CMB lensing change the topology of the temperature and polarization maps?

Question: lensing moves each point of the last-scattering image by a small, smooth deflection.
Which morphological numbers survive that remapping unchanged, and which do not?

Computes (20 patches of 10 x 10 deg, 1024^2 pixels of 0.59 arcmin):
  unlensed T, Q, U from CAMB's unlensed spectra; a lensing potential phi from C_ell^{phi phi};
  deflection alpha = grad phi; lensed maps X~(x) = X(x + alpha(x)) by cubic-spline interpolation.
  Case A (pure remapping): smooth the unlensed sky with a 5 arcmin beam first, then lens.
  Case B (the real order): lens the unsmoothed sky, then apply the 5 arcmin beam.
  For each: the Euler characteristic of {T > nu sigma_0} at nu = -2..2, the number of polarization
  singularities, and (case A) whether every lensed singularity sits where the deflection says its
  unlensed parent was, with the same index.
Writes: data/chT3/camb_unlensed_phi.npz (cache), figures/chT3/lensing_remap.pdf,
        results/chT3/04_lensing_remap.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from common import setup, savefig, save_numbers, rng_for, SERIES
from lib_t3 import (ARCMIN, beam, ell_grid, cl_on_grid, gaussian_flat, gradient, smooth,
                    euler_torus, singularities, camb_unlensed_phi)

setup()
rng = rng_for("chT3", "04_lensing_remap")
N, dx, FWHM, NREAL = 1024, 10 * 60 / 1024 * ARCMIN, 5.0, 20
ell, TT, EE, PP = camb_unlensed_phi()
TT, EE, PP = (np.where(ell <= 4000, c, 0.0) for c in (TT, EE, PP))   # nothing left beyond 4000
lg, lx, ly = ell_grid(N, dx)
ph = np.arctan2(ly, lx)
nus = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")


def remap(m, ax, ay):
    """m(x + alpha) on the periodic grid (cubic spline)."""
    return map_coordinates(m, [I + ax / dx, J + ay / dx], order=3, mode="grid-wrap")


def topo(T, Q, U, s0):
    chi = np.array([euler_torus(T > n * s0) for n in nus])
    i, j, q = singularities(Q, U)
    return chi, (i, j, q)


rows, match, rms_def, kap, detmin, s1ratio = [], [], [], [], [], []
keep = None
for r in range(NREAL):
    T = gaussian_flat(TT, N, dx, rng)
    Ek = gaussian_flat(EE, N, dx, rng, fourier=True)
    Q = np.fft.ifft2(Ek * np.cos(2 * ph)).real
    U = np.fft.ifft2(Ek * np.sin(2 * ph)).real
    phi = gaussian_flat(PP, N, dx, rng)
    ax, ay = gradient(phi, dx)                               # deflection field, radians
    axx, axy = gradient(ax, dx); ayx, ayy = gradient(ay, dx)
    rms_def.append(np.sqrt(np.mean(ax**2 + ay**2)) / ARCMIN)
    kap.append(np.std(-0.5 * (axx + ayy)))
    detmin.append(np.min((1 + axx) * (1 + ayy) - axy * ayx))
    # case A: smooth sky, then the pure remapping
    Ts, Qs, Us = (smooth(m, FWHM, dx) for m in (T, Q, U))
    s0 = Ts.std()
    TA, QA, UA = (remap(m, ax, ay) for m in (Ts, Qs, Us))
    # case B: lens the raw sky, then the beam
    TB, QB, UB = (smooth(remap(m, ax, ay), FWHM, dx) for m in (T, Q, U))
    c0, s_0 = topo(Ts, Qs, Us, s0)
    cA, s_A = topo(TA, QA, UA, s0)
    cB, s_B = topo(TB, QB, UB, s0)
    rows.append(np.r_[c0, s_0[2].size, cA, s_A[2].size, cB, s_B[2].size])
    g0 = gradient(Ts, dx); gA = gradient(TA, dx)
    s1ratio.append(np.mean(gA[0]**2 + gA[1]**2) / np.mean(g0[0]**2 + g0[1]**2))
    # matching: lensed singularity at plaquette centre x has its parent at x + alpha(x)
    i, j, q = s_A
    xc, yc = i + 0.5, j + 0.5
    axc = map_coordinates(ax, [xc, yc], order=1, mode="grid-wrap") / dx
    ayc = map_coordinates(ay, [xc, yc], order=1, mode="grid-wrap") / dx
    src = np.mod(np.c_[xc + axc, yc + ayc], N)
    p0 = np.c_[s_0[0] + 0.5, s_0[1] + 0.5]
    d, k = cKDTree(p0, boxsize=N).query(src)
    ok = (d < 1.5) & (s_0[2][k] == q)          # zero positions are known to a plaquette
    match.append(ok.mean())
    if r == 0:
        keep = dict(T=Ts, p0=p0, q0=s_0[2], pA=np.c_[xc, yc], qA=q, src=src)

rows = np.array(rows)
c0, n0 = rows[:, 0:5], rows[:, 5]
cA, nA = rows[:, 6:11], rows[:, 11]
cB, nB = rows[:, 12:17], rows[:, 17]
peak = np.abs(c0).mean(0).max()                            # changes in units of the largest mean |chi|
relA = (cA - c0).mean(0) / peak
relB = (cB - c0).mean(0) / peak
nrelA = (nA - n0).sum() / n0.sum()
nrelB = (nB - n0).sum() / n0.sum()
# errors of these means from the patch-to-patch scatter of the paired differences
sq = np.sqrt(NREAL)
relA_err = (cA - c0).std(0, ddof=1) / sq / peak
relB_err = (cB - c0).std(0, ddof=1) / sq / peak
nrelA_err = (nA - n0).std(ddof=1) / sq / n0.mean()
nrelB_err = (nB - n0).std(ddof=1) / sq / n0.mean()
iB = np.argmax(np.abs(relB))                                 # threshold of the largest case-B change
gauss_pred = np.mean(s1ratio) - 1                            # Gaussian formula: chi scales as sigma_1^2

# ---- figure: (a) a 1 x 1 deg zoom of parents and children, (b) relative changes
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
a = axs[0]
z = int(60 * ARCMIN / dx)                                    # 1 degree in pixels
sl = (slice(0, z), slice(0, z))
pix = dx / ARCMIN
a.imshow((keep["T"][sl] / keep["T"].std()).T, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3,
         extent=[0, z * pix, 0, z * pix], alpha=0.5)
for pts, qq, filled in [(keep["p0"], keep["q0"], False), (keep["pA"], keep["qA"], True)]:
    inside = (pts[:, 0] < z) & (pts[:, 1] < z)
    for sgn, col, mk in [(1, SERIES[1], "o"), (-1, SERIES[6], "s")]:
        s = inside & (qq * sgn > 0)
        a.plot(pts[s, 0] * pix, pts[s, 1] * pix, mk, ms=6 if not filled else 3.5,
               mfc=(col if filled else "none"), mec=col, mew=1.3)
ins = (keep["pA"][:, 0] < z) & (keep["pA"][:, 1] < z)
for (xa, ya), (xs, ys) in zip(keep["pA"][ins], keep["src"][ins]):
    if abs(xs - xa) < z / 2 and abs(ys - ya) < z / 2:
        a.plot([xa * pix, xs * pix], [ya * pix, ys * pix], color="0.3", lw=0.7)
a.set_xlim(0, z * pix); a.set_ylim(0, z * pix); a.grid(False)
a.set_xlabel("arcmin"); a.set_ylabel("arcmin")
a.set_title("open: unlensed, filled: lensed")
b = axs[1]
xpos = np.arange(6)
labs = [r"$\chi(\nu{=}%d)$" % n for n in nus] + ["singularities"]
b.bar(xpos - 0.2, 100 * np.r_[relA, nrelA], 0.4, yerr=100 * np.r_[relA_err, nrelA_err], capsize=2,
      color=SERIES[0], label="A: smooth sky, then lens")
b.bar(xpos + 0.2, 100 * np.r_[relB, nrelB], 0.4, yerr=100 * np.r_[relB_err, nrelB_err], capsize=2,
      color=SERIES[1], label="B: lens, then beam")
b.axhline(100 * gauss_pred, color="k", ls="--", lw=1.2, label=r"Gaussian formula with lensed $\sigma_1$")
b.axhline(0, color="k", lw=0.6)
b.set_xticks(xpos); b.set_xticklabels(labs, fontsize=8, rotation=35, ha="right")
b.set_ylabel(r"change [% of max $|\chi|$ or of the count]"); b.legend(fontsize=8, loc="lower left")
savefig(fig, "chT3", "lensing_remap")

save_numbers("chT3", "04_lensing_remap", {
    "TdNreal": NREAL, "TdPixArcmin": f"{dx / ARCMIN:.2f}", "TdFwhm": f"{FWHM:.0f}",
    "TdDefl": f"{np.mean(rms_def):.2f}", "TdKappa": f"{np.mean(kap):.3f}", "TdDetMin": f"{np.min(detmin):.2f}",
    "TdMatchPct": f"{100 * np.mean(match):.1f}", "TdMatchMinPct": f"{100 * np.min(match):.1f}",
    "TdSingPerPatch": f"{n0.mean():.0f}",
    "TdRelAMaxPct": f"{100 * np.max(np.abs(np.r_[relA, nrelA])):.2f}",
    "TdRelAErrMaxPct": f"{100 * np.max(np.r_[relA_err, nrelA_err]):.2f}",
    "TdRelBSingPct": f"{100 * nrelB:.2f}", "TdRelBSingErrPct": f"{100 * nrelB_err:.2f}",
    "TdRelBChiMaxPct": f"{100 * np.max(np.abs(relB)):.2f}", "TdRelBChiErrPct": f"{100 * relB_err[iB]:.2f}",
    "TdGaussPredPct": f"{100 * gauss_pred:.1f}",
})
print(f"defl rms {np.mean(rms_def):.2f}', kappa rms {np.mean(kap):.3f}, min det A {np.min(detmin):.3f}")
print("chi unlensed", c0.mean(0), "A", cA.mean(0), "B", cB.mean(0))
print(f"singularities {n0.mean():.0f} A {nA.mean():.0f} B {nB.mean():.0f}; matched {100*np.mean(match):.2f}% "
      f"(min {100*np.min(match):.2f}%); relA {relA}, nrelA {nrelA:.4f}; relB {relB}, nrelB {nrelB:.4f}; "
      f"Gaussian-formula change {100*gauss_pred:.2f}%")
