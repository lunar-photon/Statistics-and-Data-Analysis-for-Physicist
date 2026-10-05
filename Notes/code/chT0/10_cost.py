"""How long will it run?  Measure a few small sizes, fit the power law, extrapolate.

Question:  how does the run time of the book's four typical heavy operations grow with the size of
           the problem, and can a few quick timings predict the cost of a big run?
Computes:  best-of-three wall times for
             (a) the FFT of N numbers (N = 2^10 ... 2^22),
           all on a single core,
             (b) the Cholesky factor of an n x n covariance matrix (n = 250 ... 3000, then 6000),
             (c) counting the pairs of N random points closer than r, by brute force (all pairs,
                 in blocks) and with a k-d tree (scipy.spatial.cKDTree),
             (d) a HEALPix map -> a_lm transform (healpy.map2alm) at N_side = 16 ... 256, then 512;
           the slope of log(time) against log(size) from the larger sizes, a prediction for a size
           four times beyond the largest one fitted, and that size actually timed (for (b) and (d)).
Writes:    figures/chT0/10_cost.pdf, results/chT0/10_cost.tex, results/chT0/10_cost_out.txt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"          # one core: the time then counts operations, not how well they share 4 cores
import time
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.spatial import cKDTree
from common import setup, savefig, save_numbers, rng_for, SERIES, NOTES

rng = rng_for("chT0", "10_cost")
nums, lines = {}, []


def say(text):
    print(text)
    lines.append(text)


def best_time(f, repeats=3):
    """Shortest of a few runs: the least disturbed by whatever else the machine is doing."""
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


def slope(sizes, times, last=4):
    """Least-squares slope and intercept of log10(time) against log10(size), from the largest sizes."""
    b, a = np.polyfit(np.log10(sizes[-last:]), np.log10(times[-last:]), 1)
    return b, a


# (a) FFT ------------------------------------------------------------------------------------------
N_fft = 2 ** np.arange(10, 23, 2)
t_fft = np.array([best_time(lambda: np.fft.fft(x)) for x in (rng.standard_normal(n) for n in N_fft)])

# (b) Cholesky ----------------------------------------------------------------------------------------
def spd(n):
    G = rng.standard_normal((n, n))
    return G @ G.T / n + np.eye(n)


n_ch = np.array([250, 500, 1000, 1500, 2000, 3000])
t_ch = np.array([best_time(lambda: np.linalg.cholesky(A)) for A in (spd(n) for n in n_ch)])
n_big_ch = 6000
A_big = spd(n_big_ch)
t_big_ch = best_time(lambda: np.linalg.cholesky(A_big))

# (c) pair counts ------------------------------------------------------------------------------------
r_max = 0.02


def pairs_brute(p):
    """All pairs, compared in blocks of 250 rows so the distance table fits in memory."""
    total = 0
    for i in range(0, len(p), 250):
        d2 = np.sum((p[i:i + 250, None, :] - p[None, :, :]) ** 2, axis=-1)
        total += np.count_nonzero(d2 < r_max ** 2)
    return (total - len(p)) // 2                     # drop each point paired with itself, count each pair once


def pairs_tree(p):
    tree = cKDTree(p)
    return (tree.count_neighbors(tree, r_max) - len(p)) // 2


N_pc = np.array([1000, 2000, 4000, 8000, 16000])
t_brute, t_tree, agree = [], [], True
for n in N_pc:
    p = rng.random((n, 3))
    t_brute.append(best_time(lambda: pairs_brute(p), repeats=1 if n > 4000 else 3))
    t_tree.append(best_time(lambda: pairs_tree(p)))
    agree &= pairs_brute(p) == pairs_tree(p)
t_brute, t_tree = np.array(t_brute), np.array(t_tree)
N_tree_big = np.array([64000, 256000])
t_tree_big = np.array([best_time(lambda: pairs_tree(rng.random((n, 3))), repeats=1) for n in N_tree_big])

# (d) spherical harmonic transform ---------------------------------------------------------------------
nside_s = np.array([16, 32, 64, 128, 256])
t_sht = []
for ns in nside_s:
    m = rng.standard_normal(hp.nside2npix(ns))
    t_sht.append(best_time(lambda: hp.map2alm(m, lmax=3 * ns - 1, iter=0)))
t_sht = np.array(t_sht)
m512 = rng.standard_normal(hp.nside2npix(512))
t_sht_512 = best_time(lambda: hp.map2alm(m512, lmax=3 * 512 - 1, iter=0))

# fits and predictions ---------------------------------------------------------------------------
b_fft, a_fft = slope(N_fft, t_fft)
b_ch, a_ch = slope(n_ch, t_ch)
b_br, a_br = slope(N_pc, t_brute, last=3)
b_tr, a_tr = slope(N_pc, t_tree, last=3)
b_sh, a_sh = slope(nside_s, t_sht, last=3)
pred_ch = 10 ** (a_ch + b_ch * np.log10(n_big_ch))
pred_sh = 10 ** (a_sh + b_sh * np.log10(512))
pred_br_big = 10 ** (a_br + b_br * np.log10(1e6))     # brute force for a million galaxies
npix256 = 12 * 256 ** 2
pred_ch_planck = 10 ** (a_ch + b_ch * np.log10(npix256))   # Cholesky of an N_pix x N_pix matrix, nside 256
pred_ch_planck3 = t_big_ch * (npix256 / n_big_ch) ** 3     # the same with the counted exponent 3
say(f"FFT:       slope {b_fft:.2f}   (N log N alone would give {1 + 1 / np.log(N_fft[-2]):.2f})")
say(f"Cholesky:  slope {b_ch:.2f};  n = {n_big_ch}: predicted {pred_ch:.2f} s, measured {t_big_ch:.2f} s")
say(f"pairs:     brute-force slope {b_br:.2f}, k-d tree slope {b_tr:.2f};  same counts: {agree}")
say(f"           brute force for 10^6 points, predicted {pred_br_big / 3600:.1f} h;  tree at 256000 points "
    f"{t_tree_big[-1]:.2f} s")
say(f"map2alm:   slope in N_side {b_sh:.2f};  N_side = 512: predicted {pred_sh:.2f} s, measured {t_sht_512:.2f} s")
say(f"Cholesky of the 786432 x 786432 pixel covariance at N_side = 256: fitted slope {pred_ch_planck / 86400:.0f} days, "
    f"exponent 3 {pred_ch_planck3 / 86400:.0f} days, memory {8 * npix256 ** 2 / 1e12:.1f} TB")
nums.update(TzSlopeFft=f"{b_fft:.2f}", TzSlopeChol=f"{b_ch:.2f}", TzSlopeBrute=f"{b_br:.2f}",
            TzSlopeTree=f"{b_tr:.2f}", TzSlopeSht=f"{b_sh:.2f}",
            TzCholPred=f"{pred_ch:.2f}", TzCholMeas=f"{t_big_ch:.2f}", TzCholBigN=n_big_ch,
            TzCholTwoK=f"{t_ch[-1]:.3f}",
            TzShtPred=f"{pred_sh:.2f}", TzShtMeas=f"{t_sht_512:.2f}",
            TzShtTwoFiveSix=f"{t_sht[-1]:.3f}",
            TzBruteMillionH=f"{pred_br_big / 3600:.1f}", TzTreeBig=f"{t_tree_big[-1]:.2f}",
            TzBruteSixteenK=f"{t_brute[-1]:.2f}", TzTreeSixteenK=f"{t_tree[-1]:.3f}",
            TzFftBig=f"{1e3 * t_fft[-1]:.0f}", TzFftBigN=int(N_fft[-1]),
            TzCholNsideDays=f"{pred_ch_planck / 86400:.0f}", TzCholNsideDaysThree=f"{pred_ch_planck3 / 86400:.0f}",
            TzFftNlogN=f"{1 + 1 / np.log(N_fft[-2]):.2f}", TzPairsAgree="yes" if agree else "no")

# figure -----------------------------------------------------------------------------------------
setup(8.6, 3.6)
fig, axes = plt.subplots(1, 2)
ax = axes[0]
for x, y, b, a, c, lab in [(N_fft, t_fft, b_fft, a_fft, SERIES[0], "FFT, $N$"),
                           (N_pc, t_brute, b_br, a_br, SERIES[1], "pairs, brute force, $N$"),
                           (np.r_[N_pc, N_tree_big], np.r_[t_tree, t_tree_big], b_tr, a_tr, SERIES[2],
                            "pairs, k-d tree, $N$")]:
    ax.loglog(x, y, "o", color=c, ms=4)
    xx = np.geomspace(x[0], x[-1] * 4, 50)
    ax.loglog(xx, 10 ** (a + b * np.log10(xx)), color=c, lw=1, label=f"{lab}: slope {b:.2f}")
ax.set_xlabel("number of points $N$")
ax.set_ylabel("wall time (s)")
ax.set_ylim(top=1e5)                             # headroom: the legend must not hide the fitted lines
ax.legend(fontsize=7, loc="upper left")
ax = axes[1]
for x, y, b, a, c, lab, xb, yb in [(n_ch, t_ch, b_ch, a_ch, SERIES[3], "Cholesky, $n$", n_big_ch, t_big_ch),
                                   (nside_s * 8, t_sht, b_sh, a_sh, SERIES[4], r"map2alm, $8N_{\rm side}$",
                                    512 * 8, t_sht_512)]:
    ax.loglog(x, y, "o", color=c, ms=4)
    xx = np.geomspace(x[0], xb * 1.2, 50)
    scale = 8 if "map2alm" in lab else 1
    ax.loglog(xx, 10 ** (a + b * np.log10(xx / scale)), color=c, lw=1, label=f"{lab}: slope {b:.2f}")
    ax.loglog([xb], [yb], "*", color=c, ms=10, mec="k", mew=0.5)
ax.set_xlabel(r"matrix size $n$, or $8N_{\rm side}$")
from matplotlib.ticker import NullFormatter
_lo, _hi = ax.get_xlim()
_tk = [t for t in (100, 200, 500, 1000, 2000, 5000, 10000) if _lo <= t <= _hi]
ax.set_xticks(_tk)
ax.set_xticklabels([f"{t:g}" for t in _tk])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylabel("wall time (s)")
ax.legend(fontsize=7, loc="upper left")
ax.text(0.98, 0.04, "stars: measured after the prediction", transform=ax.transAxes, ha="right", fontsize=7,
        color="0.35")
fig.tight_layout()
savefig(fig, "chT0", "10_cost")

(NOTES / "results" / "chT0" / "10_cost_out.txt").write_text("\n".join(lines) + "\n")
save_numbers("chT0", "10_cost", nums)
