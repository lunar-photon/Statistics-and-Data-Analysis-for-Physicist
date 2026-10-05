"""Student's t and Snedecor's F, built from Gaussians and checked by simulation.

Question answered
    (1) With n = 5 Gaussian readings and the noise level sigma unknown, we form
        T = (xbar - mu) / (S / sqrt(n)).  Is T distributed as Student's t with
        n - 1 = 4 degrees of freedom, and how badly does the N(0,1) table
        underestimate its tails?
    (2) Lambert's picture: a Gaussian whose variance is itself random,
        sigma^2 ~ Inv-Gamma(nu/2, nu/2), is a Student t_nu.  Does sampling agree?
    (3) The ratio of two independent sample variances from Gaussian parents,
        (S_X^2/sigma_X^2)/(S_Y^2/sigma_Y^2), is claimed to be F_{n-1, m-1}.
        Does it follow the F pdf, with mean (m-1)/(m-3)?  Is T^2 ~ F_{1, n-1}?

What it computes
    Monte Carlo histograms and tail fractions for (1)-(3), compared with the
    analytic pdfs from scipy.stats.

What it writes
    figures/ch02/t_student.pdf, figures/ch02/f_ratio.pdf,
    results/ch02/26_student_t_F.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup()
rng = rng_for("ch02", "26_student_t_F")
NREP = 400_000

# ---------------------------------------------------------------- (1) the t statistic
n, mu, sigma = 5, 9.81, 0.05                  # five pendulum readings of g, noise 0.05
x = rng.normal(mu, sigma, (NREP, n))
xbar = x.mean(axis=1)
S = x.std(axis=1, ddof=1)
T = (xbar - mu) / (S / np.sqrt(n))            # sigma replaced by its estimate
Z = (xbar - mu) / (sigma / np.sqrt(n))        # sigma known: exactly N(0,1)

frac_T_196 = np.mean(np.abs(T) > 1.96)        # "2-sigma" by the Gaussian table
frac_Z_196 = np.mean(np.abs(Z) > 1.96)
t975 = stats.t.ppf(0.975, n - 1)              # the correct 95% two-sided point
frac_T_t975 = np.mean(np.abs(T) > t975)
frac_T_3 = np.mean(np.abs(T) > 3.0)

# ---------------------------------------------------------------- (2) scale mixture of Gaussians
nu = 3
# sigma^2 ~ Inv-Gamma(nu/2, rate nu/2)  <=>  1/sigma^2 ~ Gamma(shape nu/2, scale 2/nu)
prec = rng.gamma(shape=nu / 2, scale=2 / nu, size=NREP)
x_mix = rng.standard_normal(NREP) / np.sqrt(prec)
# the "textbook" construction U / sqrt(V/nu) with V ~ chi^2_nu
x_uv = rng.standard_normal(NREP) / np.sqrt(rng.chisquare(nu, NREP) / nu)
ks_mix = stats.kstest(x_mix, stats.t(nu).cdf).statistic
ks_uv = stats.kstest(x_uv, stats.t(nu).cdf).statistic

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
ax = axes[0]
bins = np.linspace(-6, 6, 121)
ax.hist(T, bins=bins, density=True, color=SERIES[0], alpha=0.6,
        label=r"$T=(\bar x-\mu)/(S/\sqrt{n})$")
tt = np.linspace(-6, 6, 600)
theory_line(ax, tt, stats.t.pdf(tt, n - 1), label=rf"Student $t_{{{n - 1}}}$")
ax.plot(tt, stats.norm.pdf(tt), color=SERIES[1], lw=1.4, label=r"$N(0,1)$")
ax.set_yscale("log")
ax.set_ylim(1e-4, 0.6)
ax.set_xlabel("$t$")
ax.set_ylabel("density")
ax.set_title(f"(a) $n={n}$ readings, $\\sigma$ estimated")
ax.legend(fontsize=7, loc="lower center")

ax = axes[1]
bins = np.linspace(-8, 8, 161)
ax.hist(x_mix, bins=bins, density=True, color=SERIES[2], alpha=0.6,
        label=r"$x\mid\sigma^2\sim N(0,\sigma^2)$, random $\sigma^2$")
ax.hist(x_uv, bins=bins, density=True, histtype="step", color=SERIES[0], lw=1.2,
        label=r"$U/\sqrt{V/\nu}$")
tt = np.linspace(-8, 8, 600)
theory_line(ax, tt, stats.t.pdf(tt, nu), label=rf"$t_{{{nu}}}$")
ax.set_yscale("log")
ax.set_ylim(1e-4, 0.6)
ax.set_xlabel("$x$")
ax.set_title(rf"(b) two constructions of $t_{{{nu}}}$")
ax.legend(fontsize=7, loc="lower center")
savefig(fig, "ch02", "t_student")

# ---------------------------------------------------------------- (3) the F ratio
nX, nY = 6, 11
sX, sY = 1.5, 0.4                       # different true widths: they cancel in the ratio
SX2 = rng.normal(0, sX, (NREP, nX)).var(axis=1, ddof=1)
SY2 = rng.normal(3, sY, (NREP, nY)).var(axis=1, ddof=1)
Fr = (SX2 / sX ** 2) / (SY2 / sY ** 2)
d1, d2 = nX - 1, nY - 1
F_mean_th = d2 / (d2 - 2)
F95 = stats.f.ppf(0.95, d1, d2)
frac_F95 = np.mean(Fr > F95)
T2 = T ** 2                              # T^2 should be F_{1, n-1}
ks_T2 = stats.kstest(T2[:100_000], stats.f(1, n - 1).cdf).statistic

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
ax = axes[0]
bins = np.linspace(0, 8, 161)
ax.hist(Fr, bins=bins, density=True, color=SERIES[0], alpha=0.6,
        label=r"$(S_X^2/\sigma_X^2)/(S_Y^2/\sigma_Y^2)$")
ff = np.linspace(0.005, 8, 600)
theory_line(ax, ff, stats.f.pdf(ff, d1, d2), label=rf"$F_{{{d1},{d2}}}$")
ax.axvline(F95, color=SERIES[1], lw=1.2, ls=":", label="95th percentile")
ax.set_xlabel("variance ratio")
ax.set_ylabel("density")
ax.set_title(f"(a) $n_X={nX}$, $n_Y={nY}$ Gaussian readings")
ax.legend(fontsize=7)
ax = axes[1]
bins = np.linspace(0, 12, 121)
ax.hist(T2, bins=bins, density=True, color=SERIES[2], alpha=0.6, label=r"$T^2$ from panel (a) of the $t$ figure")
ff = np.linspace(0.02, 12, 600)
theory_line(ax, ff, stats.f.pdf(ff, 1, n - 1), label=rf"$F_{{1,{n - 1}}}$")
ax.set_yscale("log")
ax.set_ylim(1e-4, 5)
ax.set_xlabel("$t^2$")
ax.set_title(r"(b) the square of a $t$ is an $F$")
ax.legend(fontsize=7)
savefig(fig, "ch02", "f_ratio")

save_numbers("ch02", "26_student_t_F", {
    "tbTn": n, "tbTdof": n - 1, "tbTnrep": NREP,
    "tbTfracOneNineSix": f"{frac_T_196:.4f}",
    "tbTfracOneNineSixTh": f"{2 * stats.t.sf(1.96, n - 1):.4f}",
    "tbZfracOneNineSix": f"{frac_Z_196:.4f}",
    "tbTquant": f"{t975:.3f}",
    "tbTfracQuant": f"{frac_T_t975:.4f}",
    "tbTfracThree": f"{frac_T_3:.4f}",
    "tbTfracThreeTh": f"{2 * stats.t.sf(3.0, n - 1):.4f}",
    "tbTfracThreeGauss": f"{2 * stats.norm.sf(3.0):.4f}",
    "tbTmixNu": nu,
    "tbTksMix": f"{ks_mix:.4f}", "tbTksUV": f"{ks_uv:.4f}",
    "tbFdOne": d1, "tbFdTwo": d2,
    "tbFmean": f"{Fr.mean():.3f}", "tbFmeanTh": f"{F_mean_th:.3f}",
    "tbFninetyfive": f"{F95:.3f}", "tbFfracNinetyfive": f"{frac_F95:.4f}",
    "tbFksTsq": f"{ks_T2:.4f}",
})
