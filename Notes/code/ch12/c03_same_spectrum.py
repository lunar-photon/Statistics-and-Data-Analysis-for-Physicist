"""c03_same_spectrum.py -- two fields with one power spectrum, and what their topology sees.

Question: the local model u = W * R_eps * [g + eps (g^2 - 1)] has the same power spectrum for
every eps.  Is that true map by map (phase-randomised twin), how do the mean Euler
characteristic and Betti curves move with eps, and does Matsubara's first-order formula
predict the move of chi, for thresholds in units of sigma and for area-fraction thresholds?
Computes: one map at eps = 0.2 and its phase-randomised twin (same |FFT| to machine
precision); binned spectra of the eps = 0 and eps = 0.1 ensembles (cache of c02); mean curves
and their shift; 400 periodic maps at eps = +-0.05 (common g) for the derivative of chi on the
torus, against Matsubara's prediction with the skewness parameters measured on the maps.
Writes: figures/ch12/c_twin.pdf, figures/ch12/c_shift.pdf, figures/ch12/c_matsubara.pdf,
        results/ch12/c03_same_spectrum.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2, DATA, theory_line  # noqa: F401
from lib_fields import randomise_phases
from lib_fnl import Model, NU, standardise, gaussianise
from lib_topo import euler_cubical
from c02_mc_fnl import skew_params, N, SMOOTH

rng = rng_for("ch12", "c03_same_spectrum")
model = Model(N, SMOOTH)
z = np.load(DATA / "ch12" / "c_fnl_mc.npz")
eg = list(z["eps_grid"])
i0, ip, im = eg.index(0.0), eg.index(0.05), eg.index(-0.05)

# ---------------------------------------------------------------- 1. one map and its twin
u = model.draw(0.2, rng)
twin = randomise_phases(u, rng)
same = np.max(np.abs(np.abs(np.fft.fft2(u)) - np.abs(np.fft.fft2(twin)))) / np.abs(np.fft.fft2(u)).max()
sk_u, sk_t = np.mean(standardise(u) ** 3), np.mean(standardise(twin) ** 3)

setup(9.0, 3.0)
fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
vmax = 3.0
for a, m, t in ((ax[0], u, r"local model, $\epsilon=0.2$"), (ax[1], twin, "same $|\\tilde u_k|$, random phases")):
    a.imshow(standardise(m)[:128, :128], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    a.set_title(t)
    a.set_xticks([])
    a.set_yticks([])
    a.grid(False)
kc = np.sqrt(model.kedges[1:] * model.kedges[:-1])
P0 = z["null_pk"].mean(0)
P1 = z["grid_pk"][:, eg.index(0.1)].mean(0)
ax[2].loglog(kc, P0, "o-", color=SERIES[0], label=r"$\epsilon=0$ (mean of 1000)")
ax[2].loglog(kc, P1, "s", color=SERIES[1], mfc="none", label=r"$\epsilon=0.1$ (mean of 500)")
ax[2].loglog(kc, model.pk(u), ":", color=SERIES[2], label=r"one map, $\epsilon=0.2$")
ax[2].loglog(kc, model.pk(twin), "--", color=INK2, lw=1, label="its twin")
ax[2].set_xlabel(r"$k$ (units of $2\pi/256$ pixels)")
ax[2].set_ylabel(r"binned $|\tilde u_k|^2/N^2$")
ax[2].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch12", "c_twin")
pk_ratio = np.max(np.abs(P1 / P0 - 1))
pk_err = np.max(z["null_pk"].std(0) / P0 / np.sqrt(500))

# ---------------------------------------------------------------- 2. the mean curves move
def shift(key, j):
    G = z["grid_" + key]
    D = G[:, j] - G[:, i0]                              # same g: common random numbers
    return D.mean(0), D.std(0) / np.sqrt(len(D)), z["null_" + key].std(0)


setup(9.0, 3.0)
fig, ax = plt.subplots(1, 3, sharex=True)
for a, key, lab in zip(ax, ["chi", "b0", "b1"], [r"$\chi$", r"$\beta_0$", r"$\beta_1$"]):
    for key2, ls, c, l in ((key, "-", SERIES[0], r"thresholds in units of $\sigma$"),
                           (key + "A", "--", SERIES[1], "thresholds by area fraction")):
        d, se, sd = shift(key2, ip)
        a.plot(NU, d, ls, color=c, label=l)
        a.fill_between(NU, d - 2 * se, d + 2 * se, color=c, alpha=0.25, lw=0)
    sd = z["null_" + key].std(0)
    a.plot(NU, sd, ":", color=INK2, lw=1, label="scatter of one map")
    a.plot(NU, -sd, ":", color=INK2, lw=1)
    a.axhline(0, color=INK2, lw=0.6)
    a.set_title(r"mean shift of " + lab + r", $\epsilon=0.05$")
    a.set_xlabel(r"threshold $\nu$")
ax[0].set_ylim(-28, 15)
ax[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "ch12", "c_shift")
d_chi, se_chi, sd_chi = shift("chi", ip)
k1 = list(NU).index(1.0)
km = list(NU).index(-1.0)
k3 = list(NU).index(3.0)

# ---------------------------------------------------------------- 3. Matsubara on the torus
nseed, deps = 400, 0.05
dchi, dchiA, sks = [], [], []
for s in range(nseed):
    g = model.gauss(rng)
    res = {}
    for e in (+deps, -deps):
        x = standardise(model.draw(e, g=g))
        xg = gaussianise(x)
        res[e] = (np.array([euler_cubical(x >= nu, periodic=True) for nu in NU]),
                  np.array([euler_cubical(xg >= nu, periodic=True) for nu in NU]))
    dchi.append((res[deps][0] - res[-deps][0]) / (2 * deps))
    dchiA.append((res[deps][1] - res[-deps][1]) / (2 * deps))
dchi, dchiA = np.array(dchi), np.array(dchiA)

sk = z["grid_sk"]                                      # (seed, eps, [S0, S1, S2, lambda])
epsv = np.array(eg)
slope = np.polyfit(epsv, sk[:, :, :3].mean(0), 1)[0]   # dS^(a) sigma0 / d eps for a = 0, 1, 2
lam = sk[:, i0, 3].mean()
nuf = np.linspace(-3.2, 3.2, 200)
H = {1: nuf, 2: nuf ** 2 - 1, 4: nuf ** 4 - 6 * nuf ** 2 + 3}
amp = N * N * lam / (2 * np.pi) ** 1.5 * np.exp(-nuf ** 2 / 2)
pred = amp * (slope[0] / 6 * H[4] + 2 * slope[1] / 3 * H[2] + slope[2] / 3)
predA = amp * (2 * (slope[1] - slope[0]) / 3 * H[2] + (slope[2] - slope[0]) / 3)
gauss_chi = amp * H[1]

setup(8.0, 3.0)
fig, ax = plt.subplots(1, 2)
for a, D, P, t in ((ax[0], dchi, pred, r"thresholds in units of $\sigma$"),
                   (ax[1], dchiA, predA, "thresholds by area fraction")):
    a.errorbar(NU, D.mean(0), 2 * D.std(0) / np.sqrt(nseed), fmt="o", color=SERIES[0], ms=3,
               label=f"Monte Carlo, {nseed} pairs")
    a.plot(nuf, P, "--", color="k", label="Matsubara, first order")
    a.axhline(0, color=INK2, lw=0.6)
    a.set_title(t)
    a.set_xlabel(r"threshold $\nu$")
ax[0].set_ylabel(r"$\mathrm{d}\langle\chi\rangle/\mathrm{d}\epsilon$")
ax[0].set_ylim(top=1.3 * ax[0].get_ylim()[1])
ax[1].set_ylim(ax[0].get_ylim())
ax[0].legend(fontsize=7, loc="upper center")
fig.tight_layout()
savefig(fig, "ch12", "c_matsubara")

predK = np.interp(NU, nuf, pred)
predAK = np.interp(NU, nuf, predA)
peak = np.argmax(np.abs(predK))
save_numbers("ch12", "c03_same_spectrum", {
    "CsTwinSame": same, "CsSkewU": f"{sk_u:.3f}", "CsSkewTwin": f"{sk_t:.3f}",
    "CsPkRatio": f"{100 * pk_ratio:.1f}", "CsPkErr": f"{100 * pk_err:.1f}",
    "CsShiftOne": f"{d_chi[k1]:.2f}", "CsShiftOneSe": f"{se_chi[k1]:.2f}",
    "CsShiftMone": f"{d_chi[km]:.2f}", "CsShiftThree": f"{d_chi[k3]:.2f}",
    "CsSdOne": f"{sd_chi[k1]:.1f}", "CsChiOne": f"{z['null_chi'][:, k1].mean():.1f}",
    "CsSlopeZero": f"{slope[0]:.2f}", "CsSlopeOne": f"{slope[1]:.2f}", "CsSlopeTwo": f"{slope[2]:.2f}",
    "CsLam": f"{lam:.4f}", "CsCorrLen": f"{1 / np.sqrt(2 * lam):.1f}",
    "CsMatNu": f"{NU[peak]:.1f}", "CsMatMC": f"{dchi.mean(0)[peak]:.0f}",
    "CsMatMCse": f"{dchi.std(0)[peak] / np.sqrt(nseed):.0f}", "CsMatTh": f"{predK[peak]:.0f}",
    "CsMatAMaxMC": f"{np.abs(dchiA.mean(0)).max():.0f}", "CsMatAMaxTh": f"{np.abs(predAK).max():.0f}",
    "CsMatRatio": f"{np.abs(predAK).max() / np.abs(predK).max():.2f}",
    "CsGaussChiOne": f"{np.interp(1.0, nuf, gauss_chi):.1f}",
    "CsNseed": nseed,
})
print("twin", same, "pk", pk_ratio, pk_err, "slope", slope, "lam", lam)
print("dchi MC", dchi.mean(0).round(0), "\npred", predK.round(0))
print("dchiA MC", dchiA.mean(0).round(0), "\npredA", predAK.round(0))
