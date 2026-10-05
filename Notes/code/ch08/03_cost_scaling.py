"""03_cost_scaling.py -- why the exact pixel likelihood is out of reach and the pseudo-C_l is not.

Question: how long does one evaluation of the exact Gaussian likelihood of a map take (it needs
the Cholesky factor of the N_pix x N_pix pixel covariance), compared with one spherical-harmonic
transform (the cost of a pseudo-C_l)?
Computes: wall-clock time of numpy's Cholesky for N = 1000..6000 (fit a N^3 law) and of
healpy.map2alm for nside = 32..256 (fit a N_pix^(3/2) law); extrapolations to nside 256, 512, 2048;
memory of the dense covariance.
Writes: figures/ch08/cost_scaling.pdf, results/ch08/03_cost_scaling.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES

setup()
rng = rng_for("ch08", "03_cost_scaling")


def best_time(f, repeat=3):
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return min(t)


# dense likelihood: Cholesky of a positive-definite N x N matrix
Ns = np.array([1000, 2000, 3000, 4500, 6000])
t_chol = []
for n in Ns:
    A = rng.standard_normal((n, n)); C = A @ A.T / n + np.eye(n)
    t_chol.append(best_time(lambda: np.linalg.cholesky(C)))
t_chol = np.array(t_chol)
a3 = t_chol[-1] / Ns[-1] ** 3          # t = a3 N^3, normalised at the largest N (closest to asymptotic)

# spherical-harmonic transform at l_max = 2 nside
nsides = np.array([32, 64, 128, 256])
t_sht = []
for ns in nsides:
    m = rng.standard_normal(hp.nside2npix(ns))
    t_sht.append(best_time(lambda: hp.map2alm(m, lmax=2 * ns, iter=0)))
t_sht = np.array(t_sht)
npx = 12 * nsides ** 2
a15 = t_sht[-1] / npx[-1] ** 1.5       # t = a15 N_pix^(3/2), normalised at nside 256

YEAR = 3.156e7
target = {256: 12 * 256 ** 2, 512: 12 * 512 ** 2, 2048: 12 * 2048 ** 2}
like_years = {k: a3 * v ** 3 / YEAR for k, v in target.items()}
sht_sec = {k: a15 * v ** 1.5 for k, v in target.items()}
mem_tb = {k: 8.0 * v ** 2 / 1e12 for k, v in target.items()}

fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.loglog(Ns, t_chol, "o", color=SERIES[0], label="Cholesky (measured)")
ax.loglog(npx, t_sht, "s", color=SERIES[1], label="map2alm (measured)")
xx = np.geomspace(1e3, 6e7, 100)
ax.loglog(xx, a3 * xx ** 3, color=SERIES[0], lw=1.2, label=r"$\propto N^3$")
ax.loglog(xx, a15 * xx ** 1.5, color=SERIES[1], lw=1.2, label=r"$\propto N_{\rm pix}^{3/2}$")
for k, v in target.items():
    ax.axvline(v, color="0.6", ls=":", lw=1)
    ax.text(v * 1.08, 3e0, rf"$N_{{\rm side}}={k}$", rotation=90, fontsize=8, color="0.35", va="bottom")
ax.axhline(YEAR, color="0.4", ls="--", lw=1)
ax.text(1.3e3, YEAR * 2.0, "one year", fontsize=8, color="0.3")
ax.set_xlabel(r"number of pixels $N$")
ax.set_ylabel("seconds per evaluation")
ax.set_ylim(1e-5, 1e15)
ax.legend(loc="upper left", fontsize=8)
savefig(fig, "ch08", "cost_scaling")


def sci(x, nd=1):
    m, e = f"{x:.{nd}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


save_numbers("ch08", "03_cost_scaling", {
    "EightACholSix": f"{t_chol[-1]:.2f}",
    "EightALikeDaysTwoFiveSix": f"{like_years[256] * 365.25:.0f}",
    "EightALikeYrFiveTwelve": f"{like_years[512]:.0f}",
    "EightALikeYrPlanck": sci(like_years[2048]),
    "EightAShtTwoFiveSix": f"{sht_sec[256]:.2f}",
    "EightAShtPlanck": f"{sht_sec[2048]:.0f}",
    "EightAMemTwoFiveSix": f"{mem_tb[256]:.1f}",
    "EightAMemPlanck": f"{mem_tb[2048]:.0f}",
})
print("chol", dict(zip(Ns, t_chol)), "sht", dict(zip(nsides, t_sht)))
print("like years", like_years, "sht s", sht_sec, "mem TB", mem_tb)
