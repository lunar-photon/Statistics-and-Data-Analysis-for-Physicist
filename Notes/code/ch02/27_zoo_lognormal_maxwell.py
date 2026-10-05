"""Two generating stories from the zoo: multiplication makes log-normals, lengths of Gaussian vectors make Rayleigh and Maxwell.

Question answered
    (1) A photomultiplier multiplies each photo-electron through 10 dynodes; each
        dynode multiplies by a random factor g_i ~ Uniform(3, 5).  Is the total
        gain G = prod g_i log-normal, as the CLT applied to ln G predicts?  Is its
        mean exp(mu + s^2/2) rather than exp(mu)?
    (2) The speed of an argon atom at T = 300 K is the length of a vector of three
        independent Gaussian velocity components.  Does it follow the
        Maxwell-Boltzmann law with mean speed sqrt(8 k T / (pi m))?  The length of a
        2-D Gaussian vector (the amplitude of a complex Gaussian mode) should be
        Rayleigh, and its square exponential.

What it computes
    Monte Carlo samples of G, of 2-D and 3-D Gaussian vector lengths, and their
    comparison with the analytic log-normal, Rayleigh, Maxwell and exponential pdfs.

What it writes
    figures/ch02/zoo_lognormal.pdf, figures/ch02/zoo_rayleigh_maxwell.pdf,
    results/ch02/27_zoo_lognormal_maxwell.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

setup()
rng = rng_for("ch02", "27_zoo_lognormal_maxwell")
NREP = 400_000

# ---------------------------------------------------------------- (1) PMT gain
n_dyn = 10
g = rng.uniform(3.0, 5.0, (NREP, n_dyn))
G = g.prod(axis=1)
lnG = np.log(G)
# CLT prediction for ln G: sum of n_dyn iid ln g_i
# E ln g and Var ln g for g ~ U(3,5), computed by quadrature-free closed form
a, b = 3.0, 5.0
E_ln = (b * np.log(b) - b - (a * np.log(a) - a)) / (b - a)
E_ln2 = (b * (np.log(b) ** 2 - 2 * np.log(b) + 2) - a * (np.log(a) ** 2 - 2 * np.log(a) + 2)) / (b - a)
mu_ln = n_dyn * E_ln
s_ln = np.sqrt(n_dyn * (E_ln2 - E_ln ** 2))
mean_G_th_ln = np.exp(mu_ln + s_ln ** 2 / 2)       # log-normal mean
mean_G_exact = 4.0 ** n_dyn                          # E prod g_i = (E g)^n exactly
median_G = np.median(G)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
ax = axes[0]
bins = np.linspace(0, 4.0e6, 161)
ax.hist(G, bins=bins, density=True, color=SERIES[0], alpha=0.6, label="simulated gain $G$")
gg = np.linspace(1e3, 4.0e6, 800)
theory_line(ax, gg, stats.lognorm.pdf(gg, s=s_ln, scale=np.exp(mu_ln)), label="log-normal (CLT on $\\ln G$)")
ax.axvline(np.exp(mu_ln), color=SERIES[1], ls=":", lw=1.3, label="median $e^{\\mu}$")
ax.axvline(G.mean(), color=SERIES[2], ls="-.", lw=1.3, label="mean")
ax.set_xlabel("gain $G$")
ax.set_ylabel("density")
ax.set_title(f"(a) product of {n_dyn} dynode factors")
ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-6, -6))
ax.legend(fontsize=7)
ax = axes[1]
bins = np.linspace(mu_ln - 5 * s_ln, mu_ln + 5 * s_ln, 121)
ax.hist(lnG, bins=bins, density=True, color=SERIES[0], alpha=0.6, label="$\\ln G$")
ll = np.linspace(bins[0], bins[-1], 500)
theory_line(ax, ll, stats.norm.pdf(ll, mu_ln, s_ln), label="Gaussian")
ax.set_xlabel("$\\ln G$")
ax.set_title("(b) the logarithm is Gaussian")
ax.legend(fontsize=7)
savefig(fig, "ch02", "zoo_lognormal")

# ---------------------------------------------------------------- (2) Rayleigh and Maxwell
kB = 1.380649e-23
m_Ar = 39.948 * 1.66053906660e-27
T = 300.0
sv = np.sqrt(kB * T / m_Ar)                          # std of each velocity component (m/s)
v3 = rng.normal(0, sv, (NREP, 3))
speed = np.linalg.norm(v3, axis=1)
v_mean_th = np.sqrt(8 * kB * T / (np.pi * m_Ar))
v_mode_th = np.sqrt(2 * kB * T / m_Ar)
v_rms_th = np.sqrt(3 * kB * T / m_Ar)

z2 = rng.standard_normal((NREP, 2))                 # re and im parts of a complex mode
amp = np.linalg.norm(z2, axis=1)
power = amp ** 2

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
ax = axes[0]
bins = np.linspace(0, 5, 101)
ax.hist(amp, bins=bins, density=True, color=SERIES[0], alpha=0.55, label="$|z|$, 2-D (amplitude)")
ax.hist(power / 2, bins=bins, density=True, histtype="step", color=SERIES[1], lw=1.3,
        label="$|z|^2/2$ (power)")
rr = np.linspace(0, 5, 400)
theory_line(ax, rr, stats.rayleigh.pdf(rr), label="Rayleigh")
ax.plot(rr, np.exp(-rr), color="0.35", ls=":", lw=1.4, label="exponential")
ax.set_xlabel("value")
ax.set_ylabel("density")
ax.set_title("(a) a complex Gaussian mode")
ax.legend(fontsize=7)
ax = axes[1]
bins = np.linspace(0, 1200, 121)
ax.hist(speed, bins=bins, density=True, color=SERIES[2], alpha=0.6, label="$|\\mathbf{v}|$, 3-D")
vv = np.linspace(0, 1200, 500)
theory_line(ax, vv, stats.maxwell.pdf(vv, scale=sv), label="Maxwell--Boltzmann")
for val, lab, c in [(v_mode_th, "mode", SERIES[1]), (v_mean_th, "mean", SERIES[0]), (v_rms_th, "rms", SERIES[3])]:
    ax.axvline(val, color=c, ls=":", lw=1.2, label=lab)
ax.set_xlabel("speed (m/s)")
ax.set_title(f"(b) argon at $T={T:.0f}$ K")
ax.legend(fontsize=7)
savefig(fig, "ch02", "zoo_rayleigh_maxwell")

save_numbers("ch02", "27_zoo_lognormal_maxwell", {
    "tbLNndyn": n_dyn,
    "tbLNmu": f"{mu_ln:.3f}", "tbLNs": f"{s_ln:.4f}",
    "tbLNlnMean": f"{lnG.mean():.3f}", "tbLNlnSd": f"{lnG.std(ddof=1):.4f}",
    "tbLNmedian": f"{median_G / 1e6:.3f}", "tbLNmedianTh": f"{np.exp(mu_ln) / 1e6:.3f}",
    "tbLNmean": f"{G.mean() / 1e6:.3f}", "tbLNmeanTh": f"{mean_G_th_ln / 1e6:.3f}",
    "tbLNmeanExact": f"{mean_G_exact / 1e6:.3f}",
    "tbLNskew": f"{stats.skew(G):.3f}", "tbLNskewLn": f"{stats.skew(lnG):.3f}",
    "tbMBsv": f"{sv:.1f}",
    "tbMBmean": f"{speed.mean():.1f}", "tbMBmeanTh": f"{v_mean_th:.1f}",
    "tbMBmodeTh": f"{v_mode_th:.1f}", "tbMBrmsTh": f"{v_rms_th:.1f}",
    "tbMBrms": f"{np.sqrt(np.mean(speed ** 2)):.1f}",
    "tbRAYmean": f"{amp.mean():.4f}", "tbRAYmeanTh": f"{np.sqrt(np.pi / 2):.4f}",
    "tbPOWmean": f"{power.mean():.4f}", "tbPOWsd": f"{power.std(ddof=1):.4f}",
})
