"""When the Poisson assumptions fail (dead time) and the large-lambda Gaussian limit.

Question:  (a) a counter that is blind for tau after each recorded hit breaks
           the 'no memory' postulate.  What happens to the recorded rate
           (theory: non-paralysable m = n/(1+n tau), paralysable m = n e^(-n tau))
           and to the scatter of the counts (Fano factor Var/mean; theory
           1/(1+n tau)^2 and 1 - 2 n tau e^(-n tau))?
           (b) how close is Poisson(lam) to a Gaussian N(lam, lam) as lam grows,
           and how badly does the Gaussian do in the far tail?
Computes:  one long stationary Poisson stream of true hits at n = 5e4 /s,
           tau = 5 us, cut into 4000 windows of 10 ms; recorded counts for both
           dead-time models; recorded rate versus true rate for a range of n;
           max |Poisson - Gaussian| for lam = 2..400; a 4-sigma tail at lam = 25.
Writes:    figures/ch02/08_deadtime.pdf, figures/ch02/08_poisson_gauss.pdf,
           results/ch02/08_deadtime_gauss.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

setup(6.4, 2.9)
rng = rng_for("ch02", "08_deadtime_gauss")
tau, Twin, windows = 5e-6, 1e-2, 4000


def true_hits(n, duration):
    """Arrival times of a Poisson stream of rate n on [0, duration): exponential gaps."""
    gaps = -np.log(rng.uniform(size=int(n * duration + 10 * np.sqrt(n * duration) + 10))) / n
    t = np.cumsum(gaps)
    return t[t < duration]


def record(t, paralysable):
    """Times the counter records. Non-paralysable: blind for tau after each RECORDED hit.
    Paralysable: blind for tau after EVERY hit, so a hit is recorded only if the
    previous true hit is at least tau earlier."""
    if paralysable:
        gap_before = np.diff(t, prepend=-np.inf)
        return t[gap_before >= tau]
    kept, last = [], -np.inf
    for ti in t:                       # a plain loop: the state is 'when did the last recorded hit start?'
        if ti - last >= tau:
            kept.append(ti)
            last = ti
    return np.array(kept)


def window_counts(times):
    return np.bincount((times / Twin).astype(int), minlength=windows)[:windows]


# ---- (a1) one stream at n = 5e4 /s, counted in 10 ms windows --------------------
n_true = 5e4
t = true_hits(n_true, windows * Twin)
c_raw = window_counts(t)
c_np = window_counts(record(t, False))
c_p = window_counts(record(t, True))
fano = lambda c: float(c.var(ddof=1) / c.mean())
fano_se = np.sqrt(2 / (windows - 1))            # sd of a Fano factor of a Poisson sample, relative

# ---- (a2) recorded rate as a function of the true rate ---------------------------
n_grid = np.array([1e4, 3e4, 6e4, 1e5, 1.5e5, 2e5, 3e5, 4e5, 5e5])
m_np, m_p = [], []
for n in n_grid:
    tt = true_hits(n, 0.05)
    m_np.append(record(tt, False).size / 0.05)
    m_p.append(record(tt, True).size / 0.05)

fig, (ax1, ax2) = plt.subplots(1, 2)
nn = np.linspace(0, 5e5, 400)
ax1.plot(nn * tau, nn * tau, color="0.6", lw=1.0, label="no dead time")
ax1.plot(n_grid * tau, np.array(m_np) * tau, "o", color=SERIES[0], label="non-paralysable")
ax1.plot(n_grid * tau, np.array(m_p) * tau, "s", color=SERIES[1], label="paralysable")
theory_line(ax1, nn * tau, nn * tau / (1 + nn * tau))
theory_line(ax1, nn * tau, nn * tau * np.exp(-nn * tau), label="_nolegend_")
ax1.set_xlabel(r"true rate $\times\,\tau$  ($n\tau$)"); ax1.set_ylabel(r"recorded rate $\times\,\tau$  ($m\tau$)")
ax1.set_ylim(0, 1.05); ax1.legend(fontsize=7.5, loc="upper left")

k = np.arange(330, 470)
ax2.hist(c_np, bins=np.arange(330, 471, 4), density=True, color=SERIES[0], alpha=0.45, label="recorded (non-par.)")
ax2.plot(k, poisson.pmf(k, c_np.mean()), "k--", lw=1.3, label="Poisson, same mean")
ax2.set_xlabel("recorded counts per 10 ms"); ax2.set_ylabel("probability")
ax2.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
savefig(fig, "ch02", "08_deadtime")

# ---- (b) Poisson -> Gaussian ------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2)
maxdiff = {}
for c, L in zip(SERIES, [4, 25, 100]):
    kk = np.arange(0, int(L + 5 * np.sqrt(L)) + 1)
    z = (kk - L) / np.sqrt(L)
    sel = np.abs(z) <= 4.5
    ax1.plot(z[sel], np.sqrt(L) * poisson.pmf(kk[sel], L), "o", color=c, ms=3, label=f"$\\lambda={L}$")
zz = np.linspace(-4.5, 4.5, 300)
theory_line(ax1, zz, norm.pdf(zz), label="$\\mathcal{N}(0,1)$")
ax1.set_xlabel(r"$(k-\lambda)/\sqrt{\lambda}$"); ax1.set_ylabel(r"$\sqrt{\lambda}\,P(k)$")
ax1.legend(fontsize=7.5)

lams = np.array([2, 4, 8, 16, 25, 50, 100, 200, 400])
for L in lams:
    kk = np.arange(0, int(L + 10 * np.sqrt(L)) + 10)
    maxdiff[int(L)] = np.max(np.abs(poisson.pmf(kk, L) - norm.pdf(kk, L, np.sqrt(L))))
md = np.array([maxdiff[int(L)] for L in lams])
ax2.loglog(lams, md, "o", color=SERIES[0], label=r"max$_k\,|P_{\rm Pois}-p_{\rm Gauss}|$")
theory_line(ax2, lams, md[-1] * lams[-1] / lams, label=r"$\propto 1/\lambda$")
ax2.set_xlabel(r"$\lambda$"); ax2.set_ylabel("largest difference")
ax2.legend(fontsize=7.5)
fig.tight_layout()
savefig(fig, "ch02", "08_poisson_gauss")

# a 4-sigma upward fluctuation on a background of 25: exact versus Gaussian
L, kcut = 25, 45
tail_pois = poisson.sf(kcut - 1, L)                      # P(N >= 45)
tail_gauss = norm.sf((kcut - L) / np.sqrt(L))            # naive Gaussian, no continuity correction
tail_gauss_cc = norm.sf((kcut - 0.5 - L) / np.sqrt(L))   # with continuity correction

q = np.exp(-n_true * tau)
save_numbers("ch02", "08_deadtime_gauss", {
    "twoaDTTrueMean": n_true * Twin,
    "twoaDTRawMean": float(c_raw.mean()), "twoaDTRawFano": fano(c_raw),
    "twoaDTFanoSE": fano_se,
    "twoaDTnpMean": float(c_np.mean()), "twoaDTnpTh": n_true * Twin / (1 + n_true * tau),
    "twoaDTnpFano": fano(c_np), "twoaDTnpFanoTh": 1 / (1 + n_true * tau) ** 2,
    "twoaDTpMean": float(c_p.mean()), "twoaDTpTh": n_true * Twin * q,
    "twoaDTpFano": fano(c_p), "twoaDTpFanoTh": 1 - 2 * n_true * tau * q,
    "twoaDTnHat": float(c_np.mean() / Twin / (1 - c_np.mean() / Twin * tau)) / 1e4,   # in units of 1e4 /s
    # distances from the predictions in units of the scatter F sqrt(2/(k-1)), signed
    "twoaDTRawPull": f"{(fano(c_raw) - 1) / fano_se:+.1f}",
    "twoaDTnpPull": f"{(fano(c_np) - 1 / (1 + n_true * tau) ** 2) / (fano_se / (1 + n_true * tau) ** 2):+.1f}",
    "twoaDTpPull": f"{(fano(c_p) - (1 - 2 * n_true * tau * q)) / (fano_se * (1 - 2 * n_true * tau * q)):+.1f}",
    # |n_hat - n| in units of its error sqrt(n (1 + n tau) / T), T = windows * Twin
    "twoaDTnHatPull": f"{abs(c_np.mean() / Twin / (1 - c_np.mean() / Twin * tau) - n_true) / np.sqrt(n_true * (1 + n_true * tau) / (windows * Twin)):.1f}",
    "twoaGaussDiffFour": maxdiff[4], "twoaGaussDiffTwentyfive": maxdiff[25], "twoaGaussDiffHundred": maxdiff[100],
    "twoaGaussDiffFourHundred": maxdiff[400],
    "twoaTailPois": tail_pois, "twoaTailGauss": tail_gauss, "twoaTailGaussCC": tail_gauss_cc,
    "twoaTailRatio": tail_pois / tail_gauss,
})
