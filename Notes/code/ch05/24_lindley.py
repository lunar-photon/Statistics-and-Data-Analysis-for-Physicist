"""24_lindley.py -- a fixed p-value, a growing sample, and a Bayes factor that turns around.

Question: an effect is measured at "1.96 sigma" (two-sided p = 0.05).  How
probable is the null hypothesis mu = 0 afterwards, and does the answer depend on
the sample size n at which the 1.96 sigma was reached?

Model: xbar ~ N(mu, 1/n).  H0: mu = 0.  H1: mu ~ N(0, b^2).  Then
  B01 = sqrt(1 + n b^2) exp( -z^2/2 * n b^2 / (1 + n b^2) ),  z = sqrt(n) xbar.

Computes
  * P(H0 | data) against n at fixed z = 1.96 and z = 2.576 (Jeffreys-Lindley),
  * the prior-free lower bound P(H0|data) >= 1 / (1 + exp(z^2/2)),
  * the calibration B10 <= -1 / (e p ln p) of p-values against Bayes factors,
  * ln B01 over the (information, significance) plane.

Writes: figures/ch05/lindley.pdf, figures/ch05/evidence_plane.pdf,
        results/ch05/24_lindley.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup()


def lnB01(lam, ratio):
    """Gaussian nested models: lam = |theta_hat|/sigma, ratio = Sigma/sigma (prior width / likelihood width)."""
    r2 = ratio**2
    return 0.5 * np.log(1 + r2) - 0.5 * lam**2 * r2 / (1 + r2)


n = np.logspace(0, 7, 300)
b = 1.0
fig, ax = plt.subplots(figsize=(5.6, 3.0))
vals = {}
for c, (z, lab) in zip(SERIES, [(1.96, r"$p=0.05$ ($z=1.96$)"), (2.576, r"$p=0.01$ ($z=2.58$)")]):
    B = np.exp(lnB01(z, np.sqrt(n) * b))
    P0 = B / (1 + B)
    ax.semilogx(n, P0, color=c, label=lab)
    ax.axhline(1 / (1 + np.exp(z**2 / 2)), color=c, ls=":", lw=0.9)
    vals[z] = lambda m, z=z: (lambda B: B / (1 + B))(np.exp(lnB01(z, np.sqrt(m) * b)))
ax.set_xlabel("sample size $n$"); ax.set_ylabel(r"$P(H_0\,|\,\mathrm{data})$")
ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="upper left")
savefig(fig, "ch05", "lindley")

# n at which P(H0|data) = 1/2 for z = 1.96
n_half = n[np.argmin(abs(vals[1.96](n) - 0.5))]

# information--significance plane (Trotta 2008, Fig. 3 style)
I10 = np.linspace(-1, 3, 300); lam = np.linspace(0, 6, 300)
LI, LL = np.meshgrid(I10, lam)
Z = lnB01(LL, 10**LI)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
cs = ax.contourf(LI, LL, Z, levels=[-40, -5, -2.5, -1, 1, 2.5, 5], colors=[SERIES[1], "#f3a47e", "#f9d6c3", "white",
                                                                             "#c9ddf5", "#7fb0ea"], extend="neither")
ax.contour(LI, LL, Z, levels=[-5, -2.5, -1, 1, 2.5], colors="k", linewidths=0.6)
ax.set_xlabel(r"information $I_{10}=\log_{10}(\Sigma/\sigma)$")
ax.set_ylabel(r"significance $\lambda=|\hat\theta|/\sigma$")
for x0, y0, t in [(2.4, 0.9, "simpler model\nfavoured"), (1.8, 5.2, "extra parameter\nfavoured"),
                  (-0.6, 1.2, "inconclusive")]:
    ax.text(x0, y0, t, fontsize=8, ha="center")
savefig(fig, "ch05", "evidence_plane")

pv = np.array([0.05, 0.01, 0.003, 0.001, 3e-4, 5.7e-7])
Bmax = -1 / (np.e * pv * np.log(pv))
zs = stats.norm.isf(pv / 2)
out = {"FiveCLinPzeroN": vals[1.96](1.0), "FiveCLinPzeroNk": vals[1.96](1e3), "FiveCLinPzeroNm": vals[1.96](1e6),
       "FiveCLinPzeroNkk": vals[2.576](1e6),
       "FiveCLinNhalf": n_half, "FiveCLinBound": 1 / (1 + np.exp(1.96**2 / 2)),
       "FiveCLinBoundLR": np.exp(1.96**2 / 2)}
names = ["a", "b", "c", "d", "e", "f"]
for nm, p, B, zz in zip(names, pv, Bmax, zs):
    out[f"FiveCSel{nm}B"] = int(round(B)) if B > 1000 else B; out[f"FiveCSel{nm}lnB"] = np.log(B); out[f"FiveCSel{nm}Z"] = zz
save_numbers("ch05", "24_lindley", out)
for k, v in out.items():
    print(k, v)
