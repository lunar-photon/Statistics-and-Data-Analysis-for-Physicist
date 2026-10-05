"""Four instruments, four sampling distributions.

Question: what does the distribution that each instrument's inference rests on look
like, and where does it differ from a Gaussian?

Computes:
  (a) collider: a background-only Poisson spectrum in 40 mass bins, scanned with a
      one-bin signal at every mass; the local significance Z(m) and its maximum;
  (b) CMB: the exact full-sky likelihood of C_ell given C_hat = C_fid at ell = 2, 10, 50,
      as a function of C_ell / C_fid;
  (c) gravitational waves: the distribution of the unknown-phase statistic rho^2 in noise
      (chi^2_2) and with a signal of rho = 4 (noncentral chi^2_2), with a 1e-3 threshold;
  (d) simulated summaries: the distribution of r^T C_hat^{-1} r for p = 10 summaries and
      n = 40 simulations, Hartlap-corrected, against chi^2_10.

Writes: figures/ch14/four_instruments.pdf, results/ch14/03_four_instruments.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = rng_for("ch14", "03_four_instruments")
out = {}

# (a) background-only scan: the largest local excess appears by chance somewhere
nb = 40
m = np.linspace(100, 160, nb)
bkg = 200 * np.exp(-(m - 100) / 30)
counts = rng.poisson(bkg)
def zloc(n, b):
    # one-sided local significance of an excess in one bin, Z = sqrt(q0) with sign
    q0 = 2 * (n * np.log(np.maximum(n, 1e-300) / b) - (n - b))
    return np.sign(n - b) * np.sqrt(np.maximum(q0, 0))
Z = zloc(counts, bkg)
out.update(ThFiNbins=f"{nb}", ThFiZmax=f"{Z.max():.2f}", ThFiMmax=f"{m[np.argmax(Z)]:.0f}")
# how often does a background-only scan of 40 bins reach that local Z somewhere?
nt = 200_000
cs = rng.poisson(bkg, size=(nt, nb))
zmax_t = zloc(cs, bkg).max(axis=1)
out.update(ThFiPglob=f"{np.mean(zmax_t >= Z.max()):.2f}",
           ThFiPloc=f"{stats.norm.sf(Z.max()):.3f}",
           ThFiPthree=f"{np.mean(zmax_t >= 3.0):.3f}", ThFiPthreeLoc=f"{stats.norm.sf(3.0):.5f}")

# (b) exact full-sky C_ell likelihood, C_hat = C_fid
x = np.linspace(0.05, 4, 800)
like_b = {}
for ell in (2, 10, 50):
    nu = 2 * ell + 1
    lnL = -0.5 * nu * (1 / x + np.log(x))
    like_b[ell] = np.exp(lnL - lnL.max())
    # where the likelihood peaks and its 68% likelihood width in x
out.update(ThFiNuTwo="5")

# (c) matched filter with unknown phase
rho = 4.0
thr = 2 * np.log(1e3)
p_det = stats.ncx2.sf(thr, 2, rho**2)
out.update(ThFiRho="4", ThFiThr=f"{thr:.2f}", ThFiPdet=f"{p_det:.2f}")

# (d) chi^2 with a simulated covariance, p = 10, n = 40
p, n = 10, 40
ntr = 20000
vals = np.empty(ntr)
for t in range(ntr):
    X = rng.normal(size=(n, p))
    C = np.cov(X, rowvar=False)
    r = rng.normal(size=p)
    vals[t] = (n - p - 2) / (n - 1) * (r @ np.linalg.solve(C, r))
crit = stats.chi2.isf(0.05, p)
out.update(ThFiP="10", ThFiN="40", ThFiFa=f"{np.mean(vals > crit):.3f}",
           ThFiMean=f"{vals.mean():.2f}")

save_numbers("ch14", "03_four_instruments", out)

setup(7.0, 5.2)
fig, ax = plt.subplots(2, 2)
a = ax[0, 0]
a.step(m, Z, where="mid", color=SERIES[0])
a.axhline(0, color="k", lw=0.6)
a.axhline(3, color=SERIES[1], lw=0.8, ls=":")
a.set_xlabel("mass bin (GeV)"); a.set_ylabel("local $Z$")
a.set_title("(a) collider: a background-only scan")
a = ax[0, 1]
for ell, c in zip((2, 10, 50), SERIES):
    a.plot(x, like_b[ell], color=c, label=rf"$\ell={ell}$")
a.axvline(1, color="k", lw=0.6, ls=":")
a.set_xlabel(r"$C_\ell/\widehat C_\ell$"); a.set_ylabel(r"$\mathcal{L}/\mathcal{L}_{\max}$")
a.set_title(r"(b) CMB: the exact $C_\ell$ likelihood"); a.legend()
a = ax[1, 0]
xx = np.linspace(0, 40, 600)
a.plot(xx, stats.chi2.pdf(xx, 2), color=SERIES[0], label="noise: $\\chi^2_2$")
a.plot(xx, stats.ncx2.pdf(xx, 2, rho**2), color=SERIES[1], label=r"signal, $\rho=4$")
a.axvline(thr, color="k", lw=0.8, ls="--", label=r"false alarm $10^{-3}$")
a.set_yscale("log"); a.set_ylim(1e-5, 0.6)
a.set_xlabel(r"$\rho^2=z_c^2+z_s^2$"); a.set_ylabel("density")
a.set_title("(c) GW: matched filter, unknown phase"); a.legend(fontsize=7.5)
a = ax[1, 1]
bins = np.linspace(0, 45, 60)
a.hist(vals, bins=bins, density=True, color=SERIES[2], alpha=0.5, label="simulated $\\hat{C}$, Hartlap")
theory_line(a, bins, stats.chi2.pdf(bins, p), label=r"$\chi^2_{10}$")
a.axvline(crit, color=SERIES[1], lw=0.8, ls=":")
a.set_xlabel(r"$\mathbf{r}^{\mathsf{T}}\hat{C}^{-1}\mathbf{r}$, $p=10$, $n=40$"); a.set_ylabel("density")
a.set_title("(d) simulated summaries: the tail grows"); a.legend(fontsize=7.5)
fig.tight_layout()
savefig(fig, "ch14", "four_instruments")
