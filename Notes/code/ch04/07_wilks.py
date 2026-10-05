"""07_wilks.py -- is -2 ln(lambda) really chi-squared under H0?

Question: (i) Casella & Berger's Poisson check: n = 25 counts with lambda0 = 5,
-2 ln lambda = 2n[(lambda0 - lhat) - lhat ln(lambda0/lhat)].  Do its percentiles
match chi^2_1?  (ii) a straight-line-plus-quadratic fit with known Gaussian noise:
testing c = 0 (one constraint) and b = c = 0 (two constraints) gives
-2 ln lambda = chi^2_min(H0) - chi^2_min(H1); is it chi^2_1 and chi^2_2 exactly?
(iii) a counting experiment n ~ Pois(s + b) with known b and the physical
constraint s >= 0: is q0 = -2 ln lambda distributed as (1/2) delta(0) + (1/2) chi^2_1
(Chernoff), so that Z = sqrt(q0)?  How good is that at b = 3.2 and b = 100?

Writes: figures/ch04/wilks_checks.pdf, figures/ch04/wilks_boundary.pdf,
        results/ch04/07_wilks.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "07_wilks")
setup()
M = 100_000

# ---------- (i) Poisson LRT, lambda0 = 5, n = 25 ----------
lam0, n = 5.0, 25
lhat = rng.poisson(lam0, size=(M, n)).mean(1)
q_pois = 2 * n * ((lam0 - lhat) - lhat * np.log(lam0 / lhat))
perc = [0.80, 0.90, 0.95, 0.99]
sim_q = np.quantile(q_pois, perc)
chi_q = stats.chi2.ppf(perc, 1)

# ---------- (ii) nested linear-Gaussian fits ----------
x = np.linspace(-1, 1, 20)
sig = 0.5
y = 1.0 + 0.0 * x + 0.0 * x**2 + sig * rng.standard_normal((M, x.size))   # H0 true: b = c = 0
def chi2min(cols):
    X = np.column_stack(cols) / sig
    Yw = y / sig
    beta, *_ = np.linalg.lstsq(X, Yw.T, rcond=None)
    r = Yw - (X @ beta).T
    return (r**2).sum(1)
c0 = chi2min([np.ones_like(x)])
c1 = chi2min([np.ones_like(x), x])
c2 = chi2min([np.ones_like(x), x, x**2])
q_one = c1 - c2           # H0: c = 0
q_two = c0 - c2           # H0: b = c = 0

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
bins = np.linspace(0, 8, 65)
axs[0].hist(q_pois, bins=bins, density=True, color=SERIES[0], alpha=0.6, label=r"$-2\ln\lambda$, Poisson")
g = np.linspace(0.05, 8, 300)
theory_line(axs[0], g, stats.chi2.pdf(g, 1), label=r"$\chi^2_1$")
axs[0].set_ylim(0, 1.2); axs[0].set_xlabel(r"$-2\ln\lambda$"); axs[0].legend(fontsize=7)
axs[1].hist(q_one, bins=bins, density=True, histtype="step", lw=1.5, color=SERIES[1], label=r"$H_0: c=0$")
axs[1].hist(q_two, bins=bins, density=True, histtype="step", lw=1.5, color=SERIES[2], label=r"$H_0: b=c=0$")
theory_line(axs[1], g, stats.chi2.pdf(g, 1), label=r"$\chi^2_1$")
axs[1].plot(g, stats.chi2.pdf(g, 2), color="k", ls=":", lw=1.4, label=r"$\chi^2_2$")
axs[1].set_ylim(0, 1.2); axs[1].set_xlabel(r"$\Delta\chi^2=-2\ln\lambda$"); axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch04", "wilks_checks")

# ---------- (iii) counting experiment with s >= 0 ----------
def q0(nobs, b):
    nobs = np.asarray(nobs, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 2 * (nobs * np.log(nobs / b) - (nobs - b))
    return np.where(nobs > b, q, 0.0)

fig, ax = plt.subplots(figsize=(5.6, 3.0))
Zg = np.linspace(0, 4.5, 200)
res = {}
for j, b in enumerate([3.2, 100.0]):
    qq = q0(rng.poisson(b, size=2_000_000), b)
    sf = np.array([np.mean(qq >= z**2) for z in Zg])
    ax.plot(Zg, sf, color=SERIES[j], label=f"toys, $b={b:g}$", drawstyle="steps-post")
    res[b] = (np.mean(qq == 0), np.mean(qq >= 9))
theory_line(ax, Zg, stats.norm.sf(Zg), label=r"$\frac{1}{2}\chi^2_1$ tail $=\Phi(-Z)$")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
ax.set_xlabel(r"$Z=\sqrt{q_0}$"); ax.set_ylabel(r"$P(\sqrt{q_0}\geq Z\mid H_0)$")
ax.legend(fontsize=8)
savefig(fig, "ch04", "wilks_boundary")

# Cowan's peak: 11 on 3.2 -- Wilks significance vs exact Poisson significance
Zw = np.sqrt(q0(11, 3.2))
Zex = stats.norm.isf(stats.poisson.sf(10, 3.2))
# Asimov median discovery significance vs s/sqrt(b)
def ZA(s, b):
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
s_ex, b_ex = 5.0, 3.2
s_big, b_big = 10.0, 100.0
# check Asimov against the median of toy significances for s = 10, b = 100
qq = q0(rng.poisson(s_big + b_big, size=400_000), b_big)
Zmed = np.median(np.sqrt(qq))

save_numbers("ch04", "07_wilks", {
    "FourAWkSimEighty": f"{sim_q[0]:.3f}", "FourAWkSimNinety": f"{sim_q[1]:.3f}",
    "FourAWkSimNinetyfive": f"{sim_q[2]:.3f}", "FourAWkSimNinetynine": f"{sim_q[3]:.3f}",
    "FourAWkChiEighty": f"{chi_q[0]:.3f}", "FourAWkChiNinety": f"{chi_q[1]:.3f}",
    "FourAWkChiNinetyfive": f"{chi_q[2]:.3f}", "FourAWkChiNinetynine": f"{chi_q[3]:.3f}",
    "FourAWkMeanOne": f"{q_one.mean():.3f}", "FourAWkMeanTwo": f"{q_two.mean():.3f}",
    "FourAWkVarOne": f"{q_one.var():.3f}", "FourAWkVarTwo": f"{q_two.var():.3f}",
    "FourAWkZeroSmall": f"{res[3.2][0]:.3f}", "FourAWkZeroBig": f"{res[100.0][0]:.3f}",
    "FourAWkThreeSmall": res[3.2][1], "FourAWkThreeBig": res[100.0][1],
    "FourAWkThreeTh": stats.norm.sf(3),
    "FourAWkZwilks": f"{Zw:.2f}", "FourAWkZexact": f"{Zex:.2f}",
    "FourAWkZAsmall": f"{ZA(s_ex, b_ex):.2f}", "FourAWkZsbSmall": f"{s_ex / np.sqrt(b_ex):.2f}",
    "FourAWkZAbig": f"{ZA(s_big, b_big):.3f}", "FourAWkZsbBig": f"{s_big / np.sqrt(b_big):.3f}",
    "FourAWkZmedBig": f"{Zmed:.3f}",
})
print(sim_q, chi_q, q_one.mean(), q_two.mean(), q_one.var(), q_two.var(), res, Zw, Zex,
      ZA(s_ex, b_ex), s_ex / np.sqrt(b_ex), ZA(s_big, b_big), Zmed)
