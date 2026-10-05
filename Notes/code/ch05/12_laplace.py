"""12_laplace.py -- the Gaussian (Laplace) approximation to a posterior, and where it fails.

Question: near its peak, ln p(theta|d) is a parabola.  How good is the Gaussian built
from the peak and the curvature, sigma = (-d^2 L/dtheta^2)^(-1/2)?

Computes:
  * the coin: 9 heads in 32 tosses and 1 head in 10 tosses, flat prior: the exact
    Beta posterior against the Gaussian with mean R/N and sigma = sqrt(H0(1-H0)/N);
    the probability mass the Gaussian puts at theta < 0;
  * Gull's lighthouse (Sivia 2.4): flashes along the coast from a lighthouse at
    alpha = 1 km along the shore and beta = 1 km out to sea; grid posterior for alpha
    after N = 1, 2, 3, 8, 64, 512 flashes, the sample mean of the positions (useless,
    Cauchy data), and the Laplace width at N = 64 against the true posterior sd.

Writes: figures/ch05/laplace_coin.pdf, figures/ch05/lighthouse.pdf,
        results/ch05/12_laplace.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch05", "12_laplace")
setup()

# ---------------------------------------------------------------- coin
th = np.linspace(-0.2, 1, 2401)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
coin = {}
for a, (R, N) in zip(ax, [(9, 32), (1, 10)]):
    H0 = R / N
    s = np.sqrt(H0 * (1 - H0) / N)
    exact = stats.beta(R + 1, N - R + 1)
    a.plot(th, exact.pdf(th), color=SERIES[0], label="exact Beta")
    theory_line(a, th, stats.norm(H0, s).pdf(th), label="Gaussian approx.")
    a.axvline(0, color=INK2, lw=0.6)
    a.set_xlabel(r"$\theta$"); a.set_title(f"{R} head{'s' if R != 1 else ''} in {N} tosses", fontsize=8)
    a.set_yticks([]); a.legend(fontsize=7)
    coin[(R, N)] = dict(H0=H0, s=s, neg=stats.norm(H0, s).cdf(0),
                        mean=exact.mean(), sd=exact.std())
fig.tight_layout()
savefig(fig, "ch05", "laplace_coin")

# ---------------------------------------------------------------- lighthouse
ALPHA, BETA = 1.0, 1.0
theta_k = rng.uniform(-np.pi / 2, np.pi / 2, 512)      # azimuths of the flashes
x_k = ALPHA + BETA * np.tan(theta_k)                   # where they hit the coast
alpha = np.linspace(-5, 5, 20001); da = alpha[1] - alpha[0]


def post_alpha(xs):
    L = -np.sum(np.log(BETA**2 + (xs[None, :] - alpha[:, None]) ** 2), axis=1)
    p = np.exp(L - L.max())
    return p / (p.sum() * da)


shown = [1, 2, 3, 8, 64, 512]
fig, axs = plt.subplots(2, 3, figsize=(7.6, 3.8))
res = {}
for a, n in zip(axs.flat, shown):
    p = post_alpha(x_k[:n])
    a.plot(alpha, p, color=SERIES[0])
    xbar = x_k[:n].mean()
    a.axvline(np.clip(xbar, -5, 5), color=SERIES[1], lw=1, ls="--")
    if n <= 8:
        a.plot(x_k[:n], np.full(n, 1.05 * p.max()), "o", mfc="none", color=INK2, ms=3)
    a.set_xlim(-5, 5); a.set_yticks([])
    a.set_title(f"N = {n}, sample mean {xbar:.2f}", fontsize=8)
    mean = np.sum(alpha * p) * da
    res[n] = dict(mode=alpha[np.argmax(p)], sd=np.sqrt(np.sum((alpha - mean) ** 2 * p) * da), xbar=xbar)
for a in axs[1]:
    a.set_xlabel(r"lighthouse position $\alpha$ (km)")
fig.tight_layout()
savefig(fig, "ch05", "lighthouse")

# Laplace width at N = 64: -d2L/dalpha2 at the mode, L = -sum ln(beta^2 + (x-alpha)^2)
x64, a0 = x_k[:64], res[64]["mode"]
u = x64 - a0
d2L = np.sum(2 * (u**2 - BETA**2) / (BETA**2 + u**2) ** 2)     # d^2 L / d alpha^2
lap64 = 1 / np.sqrt(-d2L)

c1, c2 = coin[(9, 32)], coin[(1, 10)]
save_numbers("ch05", "12_laplace", {
    "FiveBLapHa": c1["H0"], "FiveBLapSa": c1["s"], "FiveBLapMeanA": c1["mean"], "FiveBLapSdA": c1["sd"],
    "FiveBLapHb": c2["H0"], "FiveBLapSb": c2["s"], "FiveBLapNegB": c2["neg"],
    "FiveBLapMeanB": c2["mean"], "FiveBLapSdB": c2["sd"],
    "FiveBLhModeFive": res[512]["mode"], "FiveBLhSdFive": res[512]["sd"], "FiveBLhXbarFive": res[512]["xbar"],
    "FiveBLhModeSix": res[64]["mode"], "FiveBLhSdSix": res[64]["sd"], "FiveBLhLapSix": lap64,
    "FiveBLhXbarSix": res[64]["xbar"], "FiveBLhSqrtTwoN": np.sqrt(2 / 512),
})
