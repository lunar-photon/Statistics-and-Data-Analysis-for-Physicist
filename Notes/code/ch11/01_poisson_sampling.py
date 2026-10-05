"""01_poisson_sampling.py -- galaxies as a Poisson sample of a continuous density field.

Question: if galaxies are dropped at random with probability proportional to n_bar (1 + delta),
does the power spectrum of the points equal P(k) + 1/n_bar, and does the variance of counts in
cells equal <N> + <N>^2 sigma_R^2, as the Cox-process derivation predicts?
Computes: NSIM lognormal fields (side 600 Mpc/h, 96^3 cells, b = 1.5 times the z = 0 linear
matter spectrum); for three mean densities, Poisson counts in every cell; the shell-averaged
power of the counts against the power of the underlying field plus 1/n_bar; a uniform field
(delta = 0) sampled the same way; counts-in-cells variances for cubes of side 6.25-50 Mpc/h.
Writes: figures/ch11/poisson_pk.pdf, figures/ch11/poisson_cic.pdf, results/ch11/01_poisson_sampling.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES
from lib_lss import linear_pk_z0, xi_box_from_pk, Lognormal, poisson_sample, KBins, power_raw

L, N, D3, B = 600.0, 96, 3, 1.5
NSIM = 10
NBARS = [1e-4, 1e-3, 1e-2]                  # (h/Mpc)^3
rng = rng_for("ch11", "01_poisson_sampling")

k_tab, pk_tab, _, _ = linear_pk_z0()
P = lambda k: B ** 2 * np.interp(k, k_tab, pk_tab)
ln = Lognormal(xi_box_from_pk(P, N, L, D3), N, L, D3)
edges = np.geomspace(1.5 * 2 * np.pi / L, np.pi * N / L, 22)
kb = KBins(N, L, D3, edges)
dV = (L / N) ** 3

p_field, p_pts, p_unif = [], {n: [] for n in NBARS}, []
p_eps = {n: [] for n in NBARS}
cic = {m: [] for m in (1, 2, 4, 8)}          # block sizes in cells -> (meanN, varN, var of field)
for s in range(NSIM):
    delta = ln.draw(rng)
    p_field.append(kb.avg(power_raw(delta, L)))
    for nb in NBARS:
        c = poisson_sample(1 + delta, nb, L, rng)
        dn = c / (nb * dV) - 1                  # density contrast of the points
        p_pts[nb].append(kb.avg(power_raw(dn, L)))
        p_eps[nb].append(kb.avg(power_raw(dn - delta, L)))   # the sampling noise alone
        if nb == 1e-3:
            for m in cic:                       # counts in cubes of m^3 cells
                cs = c.reshape(N // m, m, N // m, m, N // m, m).sum(axis=(1, 3, 5))
                ds = (1 + delta).reshape(N // m, m, N // m, m, N // m, m).mean(axis=(1, 3, 5)) - 1
                cic[m].append((cs.mean(), cs.var(), ds.var()))
    cu = poisson_sample(np.ones_like(delta), 1e-3, L, rng)
    p_unif.append(kb.avg(power_raw(cu / (1e-3 * dV) - 1, L)))

pf = np.mean(p_field, axis=0)
k = kb.kmean
setup(6.0, 3.9)
fig, ax = plt.subplots()
for i, nb in enumerate(NBARS):
    ax.loglog(k, np.mean(p_pts[nb], axis=0), color=SERIES[i], label=rf"points, $\bar n=10^{{{int(np.log10(nb))}}}$")
    ax.axhline(1 / nb, color=SERIES[i], lw=0.8, ls=":")
ax.loglog(k, np.mean(p_unif, axis=0), color=SERIES[3], label=r"uniform field, $\bar n=10^{-3}$")
ax.loglog(k, pf, color="0.35", lw=2.2, alpha=0.6, label="underlying field")
theory_line(ax, k, pf + 1 / NBARS[0], label=r"field $+1/\bar n$")
for nb in NBARS[1:]:
    ax.loglog(k, pf + 1 / nb, color="k", ls="--", lw=1.2)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
ax.set_ylabel(r"$\langle|\delta_{\mathbf{k}}|^2\rangle/V\ [(h^{-1}\mathrm{Mpc})^3]$")
ax.set_ylim(30, 3e5)
ax.legend(fontsize=8, ncol=2, loc="upper right")
savefig(fig, "ch11", "poisson_pk")

# shot-noise check: power of the sampling noise eps = delta_n - delta, times nbar (should be 1);
# P_points - P_field also contains the cross term 2 Re(delta eps^*), zero on average but noisy at low k
ratio = {nb: float(np.mean(np.mean(p_eps[nb], axis=0) * nb)) for nb in NBARS}
hik = k > 0.2
diff = {nb: float(np.mean(((np.mean(p_pts[nb], axis=0) - pf) * nb)[hik])) for nb in NBARS}
unif = float(np.mean(np.mean(p_unif, axis=0)) * 1e-3)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
mN, fano, pred = [], [], []
for m in cic:
    a = np.array(cic[m]).mean(axis=0)
    mN.append(a[0]); fano.append(a[1] / a[0]); pred.append(1 + a[0] * a[2])
ax.semilogx(mN, fano, "o", color=SERIES[0], label=r"measured $\mathrm{Var}(N)/\langle N\rangle$")
theory_line(ax, mN, pred, label=r"$1+\langle N\rangle\,\sigma_R^2$")
ax.axhline(1, color=SERIES[1], ls=":", label="pure Poisson")
for j, (x, y, m) in enumerate(zip(mN, fano, cic)):
    last = j == len(mN) - 1
    ax.annotate(rf"$R={m*L/N:g}$", (x, y), textcoords="offset points",
                xytext=(-8, 2) if last else ((0, 14) if j == 0 else (8, -12)), fontsize=8, ha="right" if last else "left")
ax.set_xlabel(r"mean count per cell $\langle N\rangle$")
ax.set_ylabel("variance / mean")
ax.legend(fontsize=8, loc="upper left")
savefig(fig, "ch11", "poisson_cic")

save_numbers("ch11", "01_poisson_sampling", {
    "PoisL": f"{L:g}", "PoisN": N, "PoisNsim": NSIM, "PoisBias": f"{B:g}",
    "PoisShotA": f"{ratio[1e-4]:.3f}", "PoisShotB": f"{ratio[1e-3]:.3f}", "PoisShotC": f"{ratio[1e-2]:.3f}",
    "PoisUnif": f"{unif:.3f}", "PoisDiffC": f"{diff[1e-2]:.3f}", "PoisDiffB": f"{diff[1e-3]:.3f}",
    "PoisCicMeanSmall": f"{mN[0]:.3f}", "PoisCicFanoSmall": f"{fano[0]:.3f}", "PoisCicPredSmall": f"{pred[0]:.3f}",
    "PoisCicMeanBig": f"{mN[-1]:.1f}", "PoisCicFanoBig": f"{fano[-1]:.2f}", "PoisCicPredBig": f"{pred[-1]:.2f}",
    "PoisClip": f"{100*ln.clipped:.2f}",
})
