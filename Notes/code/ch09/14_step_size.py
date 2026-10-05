"""How big should the steps be?  Acceptance rate, efficiency and the 0.234 rule.

Question 1: for a standard Gaussian target in D dimensions and Gaussian proposals of width
l / sqrt(D), how do the acceptance rate and the efficiency 1/tau depend on l?  Compare with the
large-D formulas  a(l) = 2 Phi(-l/2)  and  speed ∝ l^2 a(l)  (maximum at l = 2.38, a = 0.234).
Question 2: for a strongly correlated 2-D Gaussian (rho = 0.95), what do too-small, too-large
and well-chosen isotropic steps do, and what does a proposal shaped like the target change?
Writes figures/ch09/step_scaling.pdf, figures/ch09/step_correlated.pdf,
results/ch09/14_step_size.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_mcmc import metropolis, tau_int

rng = rng_for("ch09", "14_step_size")
nums = {}

# ---- optimum of the large-D theory ------------------------------------------------------------
a_th = lambda l: 2 * stats.norm.cdf(-l / 2)
l_opt = optimize.minimize_scalar(lambda l: -l**2 * a_th(l), bounds=(0.5, 5), method="bounded").x
nums["NineBStLopt"] = l_opt
nums["NineBStAopt"] = a_th(l_opt)

# ---- scan in D = 1, 10, 50 ---------------------------------------------------------------------
# Efficiency is measured by the expected squared jump distance per step, ESJD = <|x_{t+1}-x_t|^2>,
# which is far less noisy than 1/tau; for large D theory gives ESJD = l^2 a(l).
Ds = [1, 10, 50]
ells = np.geomspace(0.15, 12, 22)
n = 40_000
scan = {}
for D in Ds:
    logp = lambda x: -0.5 * x @ x
    acc, esjd = [], []
    for l in ells:
        x0 = rng.standard_normal(D)                       # start in equilibrium: no burn-in needed
        ch, _, a = metropolis(logp, x0, n, l / np.sqrt(D), rng)
        acc.append(a)
        esjd.append(np.mean(np.sum(np.diff(ch, axis=0) ** 2, axis=1)))
    scan[D] = (np.array(acc), np.array(esjd))
    best = np.argmax(scan[D][1])
    tag = {1: "One", 10: "Ten", 50: "Fifty"}[D]
    nums[f"NineBStBestAcc{tag}"] = scan[D][0][best]
    nums[f"NineBStBestL{tag}"] = ells[best]
    # tau of one coordinate at l = 2.38, in units of D (Dunkley et al.: tau ~ 3.3 D)
    ch, _, a = metropolis(logp, rng.standard_normal(D), 200_000, l_opt / np.sqrt(D), rng)
    nums[f"NineBStTauD{tag}"] = tau_int(ch[:, 0])[0] / D
    nums[f"NineBStAccOpt{tag}"] = a

setup(7.0, 2.9)
fig, (a1, a2) = plt.subplots(1, 2)
ll = np.geomspace(0.1, 14, 300)
for D, c in zip(Ds, SERIES):
    acc, esjd = scan[D]
    a1.plot(ells, acc, "o", color=c, ms=3.5, label=f"$D={D}$")
    a2.plot(acc, esjd, "o-", color=c, ms=3.5, lw=0.8, label=f"$D={D}$")
theory_line(a1, ll, a_th(ll), label=r"$2\Phi(-\ell/2)$")
theory_line(a2, a_th(ll), ll**2 * a_th(ll), label=r"$\ell^2\,2\Phi(-\ell/2)$")
a1.set_xscale("log")
a1.set_xlabel(r"step scale $\ell$ (proposal sd $\ell/\sqrt{D}$)")
a1.set_ylabel("acceptance rate")
a1.legend(fontsize=7)
a1.set_title("(a) acceptance", fontsize=9, loc="left")
a2.axvline(a_th(l_opt), color="k", ls=":", lw=0.9)
a2.text(a_th(l_opt) + 0.01, 0.05, "0.234", fontsize=8)
a2.set_xlabel("acceptance rate")
a2.set_ylabel("mean squared jump per step")
a2.set_title("(b) how far the walker moves", fontsize=9, loc="left")
a2.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "step_scaling")

# ---- correlated 2-D Gaussian -------------------------------------------------------------------
rho = 0.95
C = np.array([[1.0, rho], [rho, 1.0]])
Ci = np.linalg.inv(C)
logp2 = lambda x: -0.5 * x @ Ci @ x
n2 = 40_000
x0 = np.array([0.0, 0.0])
# scan of the isotropic step first: the best a round proposal can do
iso = []
for s in np.geomspace(0.1, 10, 16):
    ch, _, a = metropolis(logp2, x0, n2, s, rng)
    iso.append((max(tau_int(ch[:, i])[0] for i in range(2)), s, a))
best_iso = min(iso)
nums["NineBStIsoBestTau"], nums["NineBStIsoBestStep"], nums["NineBStIsoBestAcc"] = best_iso
cases = [("small", 0.05, None), ("large", 30.0, None), ("tuned", best_iso[1], None),
         ("shaped", 2.38 / np.sqrt(2), C)]
res = {}
for name, s, cov in cases:
    ch, _, a = metropolis(logp2, x0, n2, s, rng, cov=cov)
    taus = [tau_int(ch[:, i])[0] for i in range(2)]
    res[name] = (ch, a, max(taus))
    tag = name.capitalize()
    nums[f"NineBStCorr{tag}Acc"] = a
    nums[f"NineBStCorr{tag}Tau"] = max(taus)
    nums[f"NineBStCorr{tag}Step"] = s
nums["NineBStRho"] = rho
save_numbers("ch09", "14_step_size", nums)

setup(7.2, 4.8)
fig, axes = plt.subplots(2, 4, gridspec_kw={"height_ratios": [1.25, 1]})
g = np.linspace(-3.5, 3.5, 200)
G1, G2 = np.meshgrid(g, g)
Z = np.exp(-0.5 * (Ci[0, 0] * G1**2 + 2 * Ci[0, 1] * G1 * G2 + Ci[1, 1] * G2**2))
titles = {"small": "(a) tiny steps", "large": "(b) huge steps", "tuned": "(c) best round step",
          "shaped": "(d) shaped like target"}
for j, (name, s, cov) in enumerate(cases):
    ch, a, tau = res[name]
    ax = axes[0, j]
    ax.contour(G1, G2, Z, levels=[np.exp(-2), np.exp(-0.5)], colors="0.6", linewidths=0.8)
    ax.plot(ch[:400, 0], ch[:400, 1], "-o", color=SERIES[j], lw=0.5, ms=1.2)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_title(titles[name], fontsize=9)
    ax.set_xlabel(r"$\theta_1$")
    if j == 0:
        ax.set_ylabel(r"$\theta_2$")
    axt = axes[1, j]
    axt.plot(ch[:3000, 0], color=SERIES[j], lw=0.5)
    axt.set_ylim(-4, 4)
    axt.set_xlabel("step")
    axt.text(0.03, 0.05, f"acc {a:.2f}, $\\tau$ {tau:.0f}", transform=axt.transAxes, fontsize=7.5)
    if j == 0:
        axt.set_ylabel(r"$\theta_1$")
fig.tight_layout()
savefig(fig, "ch09", "step_correlated")
