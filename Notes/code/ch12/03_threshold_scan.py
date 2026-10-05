"""03_threshold_scan.py -- beta0, beta1 and chi as the threshold moves; monotone transforms.

Question: (a) for one realisation of a Gaussian field, how do the number of components,
the number of holes and chi change as the threshold nu is lowered?  (b) A monotone change
of the field values, f(phi) with f' > 0, maps the excursion set {phi > nu} onto
{f(phi) > f(nu)}.  Does labelling thresholds by the AREA FRACTION they enclose make chi
blind to f?  And does it stay blind for a field that is not a monotone function of a
Gaussian one?

Computes: the threshold table for one map; mean chi curves over 60 maps for a Gaussian
          field, its exponential (lognormal) and a chi-squared field.
Writes:   figures/ch12/threshold_scan.pdf, figures/ch12/monotone.pdf,
          results/ch12/03_threshold_scan.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES
from lib_fields import gaussian_field
from lib_topo import betti_plane, euler_cubical, gkf_plane, gkf_square, lam_from_pk

setup()
rng = rng_for("ch12", "03_threshold_scan")
N, R = 256, 4.0
P = lambda k: np.exp(-(k * R) ** 2)
lam = lam_from_pk(P, N, N)

# ---------------------------------------------------------------- (a) one map, many thresholds
phi = gaussian_field(P, N, N, 2, rng)
u = (phi - phi.mean()) / phi.std()
nus = np.round(np.arange(-2.5, 2.51, 0.25), 2)
tab = np.array([betti_plane(u > nu) for nu in nus])
chi = tab[:, 0] - tab[:, 1]
for nu, (b0, b1), c in zip(nus, tab, chi):
    print(f"{nu:6.2f} {b0:5d} {b1:5d} {c:6d}")
nums = {"ScanN": N, "ScanR": int(R), "ScanLam": round(lam, 4)}
for nu_pick, tag in [(-1.0, "Mone"), (0.0, "Zero"), (1.0, "One")]:
    i = int(np.where(nus == nu_pick)[0][0])
    nums[f"Scan{tag}Bzero"], nums[f"Scan{tag}Bone"], nums[f"Scan{tag}Chi"] = int(tab[i, 0]), int(tab[i, 1]), int(chi[i])
imax = int(np.argmax(tab[:, 0]))
nums["ScanBzeroMaxNu"] = nus[imax]
nums["ScanBzeroMax"] = int(tab[imax, 0])

fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(nus, tab[:, 0], "o-", color=SERIES[0], ms=3, label=r"$\beta_0$ (components)")
ax.plot(nus, tab[:, 1], "s-", color=SERIES[1], ms=3, label=r"$\beta_1$ (holes)")
ax.plot(nus, chi, "^-", color=SERIES[2], ms=3, label=r"$\chi=\beta_0-\beta_1$")
nn = np.linspace(-2.6, 2.6, 300)
theory_line(ax, nn, gkf_square(nn, lam, N), label=r"expected $\chi$, Gaussian field in a square")
ax.axhline(0, color="0.6", lw=0.6)
ax.set_xlabel(r"threshold $\nu$ (units of $\sigma$)")
ax.set_ylabel("count in the map")
ax.legend(fontsize=8)
savefig(fig, "ch12", "threshold_scan")

# ---------------------------------------------------------------- (b) monotone transforms and the area-fraction threshold
NS = 60
fs = gaussian_field(P, N, N, 2, rng, nsim=NS)
gs = gaussian_field(P, N, N, 2, rng, nsim=NS)            # a second, independent field for chi^2
nuA = np.linspace(-2.5, 2.5, 41)


def thresh_area(x, nua):
    """Threshold t with area fraction P(x > t) equal to the Gaussian tail Psi(nua)."""
    frac = 0.5 * (1 - erf(np.asarray(nua) / np.sqrt(2)))
    return np.quantile(x, 1 - frac)


curves = {k: np.zeros(len(nuA)) for k in ("g_raw", "ln_raw", "ln_area", "g_area", "c2_area")}
same_sets = True
for f, g in zip(fs, gs):
    a = (f - f.mean()) / f.std()
    b = (g - g.mean()) / g.std()
    ln = np.exp(0.8 * a)                                  # lognormal: a monotone function of a
    ln_s = (ln - ln.mean()) / ln.std()
    c2 = (a ** 2 + b ** 2 - 2) / 2                        # chi^2 with 2 dof, standardised: NOT monotone in a
    tA_g, tA_ln, tA_c2 = thresh_area(a, nuA), thresh_area(ln_s, nuA), thresh_area(c2, nuA)
    for j, nu in enumerate(nuA):
        curves["g_raw"][j] += euler_cubical(a > nu, 8, True)
        curves["ln_raw"][j] += euler_cubical(ln_s > nu, 8, True)
        sg, sl = a > tA_g[j], ln_s > tA_ln[j]
        same_sets &= bool(np.array_equal(sg, sl))
        curves["g_area"][j] += euler_cubical(sg, 8, True)
        curves["ln_area"][j] += euler_cubical(sl, 8, True)
        curves["c2_area"][j] += euler_cubical(c2 > tA_c2[j], 8, True)
for k in curves:
    curves[k] /= NS
print("lognormal sets identical to Gaussian sets at equal area fraction:", same_sets)
j1 = int(np.argmin(np.abs(nuA - 1.0)))
jm = int(np.argmin(np.abs(nuA + 1.0)))
nums.update({"MonoNsim": NS, "MonoSame": "yes" if same_sets else "no",
             "MonoGaussOne": round(curves["g_area"][j1], 1), "MonoGaussMone": round(curves["g_area"][jm], 1),
             "MonoLnRawOne": round(curves["ln_raw"][j1], 1), "MonoLnRawMone": round(curves["ln_raw"][jm], 1),
             "MonoCtwoOne": round(curves["c2_area"][j1], 1), "MonoCtwoMone": round(curves["c2_area"][jm], 1),
             "MonoTheoryOne": round(gkf_plane(1.0, lam)[2] * N * N, 1)})
save_numbers("ch12", "03_threshold_scan", nums)

fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
th = gkf_plane(nuA, lam)[2] * N * N
axs[0].plot(nuA, curves["g_raw"], color=SERIES[0], label=r"Gaussian $\phi$")
axs[0].plot(nuA, curves["ln_raw"], color=SERIES[1], label=r"lognormal $e^{0.8\phi}$")
axs[0].set_title(r"threshold in units of each field's own $\sigma$")
axs[0].set_xlabel(r"$\nu$")
axs[1].plot(nuA, curves["g_area"], color=SERIES[0], lw=2.5, alpha=0.5, label=r"Gaussian $\phi$")
axs[1].plot(nuA, curves["ln_area"], color=SERIES[1], ls=":", lw=1.8, label=r"lognormal $e^{0.8\phi}$")
axs[1].plot(nuA, curves["c2_area"], color=SERIES[2], label=r"$\chi^2_2$ field $(\phi^2+\psi^2)$")
axs[1].set_title("threshold labelled by the area fraction it encloses")
axs[1].set_xlabel(r"$\nu_A$, with $\Psi(\nu_A)=$ area fraction")
for ax in axs:
    theory_line(ax, nuA, th, label="Gaussian prediction")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.legend(fontsize=7.5)
axs[0].set_ylabel(r"mean $\chi$ per map")
savefig(fig, "ch12", "monotone")
