"""11_voids.py -- voids in a Zel'dovich universe: finder, size function, profiles, velocities.

Question: what does a spherical void finder return on a simulated matter field, how many voids
of each size are there compared with the two-barrier excursion-set prediction of Sheth & van de
Weygaert (2004), what is their stacked density profile and does the Hamaus, Sutter & Wandelt (2014)
formula fit it, and do the particles stream out of voids as linear continuity predicts?
Computes:
  (1) two-barrier random walks: fraction of walks that reach delta_v before delta_c against
      delta_c/(delta_c + |delta_v|), and the first-crossing distribution against the SvdW series;
  (2) the linear threshold delta_v that a spherical void needs to reach the non-linear contrast
      Delta = -0.8 (parametric spherical-void solution), and the expansion factor 1.71;
  (3) spherical voids (Delta(<R) = -0.8) in the Zel'dovich box of lib_web (600 Mpc/h, 192^3
      particles, z = 0): sizes, the size function against SvdW and the volume-conserving variant of
      Jennings et al. (2013), with the linear threshold of real gravity (-2.78) and of the
      Zel'dovich dynamics itself (-2.13);
  (4) watershed voids (basins of the density landscape, as in Hamaus et al. 2014): stacked density
      profiles in four radius bins, Hamaus et al. eq. (2) fitted to each;
  (5) stacked radial velocity profiles against the linear prediction v = -(1/3) f H r Delta(<r)
      and the spherical Zel'dovich one v = f H r [1 - (1 + Delta(<r))^(1/3)];
  (6) the number of connected underdense regions {delta_s < t} against the number of voids.
Writes: figures/ch11/void_barriers.pdf, figures/ch11/void_sizes.pdf, figures/ch11/void_profiles.pdf,
figures/ch11/void_topology.pdf, data/ch11/voids.npz, results/ch11/11_voids.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import brentq, curve_fit
from scipy.spatial import cKDTree
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line, DATA
from lib_web import (BOX_L, BOX_N, DELTA_C, RHO_M, zeldovich_box, cic, find_voids, first_crossings,
                     hamaus_profile, sigma_R, watershed_voids)
from lib_lss import growth

rng = rng_for("ch11", "11_voids")
THR = -0.8

# ---------------------------------------------------------------- (2) the linear void threshold
rho_nl = lambda t: 4.5 * (np.sinh(t) - t) ** 2 / (np.cosh(t) - 1) ** 3      # 1 + Delta (EdS)
d_lin = lambda t: -0.6 * 0.75 ** (2 / 3) * (np.sinh(t) - t) ** (2 / 3)
th = brentq(lambda t: rho_nl(t) - (1 + THR), 0.5, 10)
DV = float(d_lin(th))                                  # about -2.78
EXPAND = (1 + THR) ** (-1 / 3)                         # Eulerian / Lagrangian radius, 1.71
th_sc = 3.53                                          # shell crossing of a top-hat void
DV_sc, DNL_sc = float(d_lin(th_sc)), float(rho_nl(th_sc) - 1)
# in the Zel'dovich approximation a uniform sphere moves as r = q (1 - delta_lin/3), so
# 1 + Delta = (1 - delta_lin/3)^-3 and Delta = -0.8 needs delta_lin = -3 (5^(1/3) - 1)
DV_ZA = float(-3 * ((1 + THR) ** (-1 / 3) - 1))
DC_ZA = 3.0                                            # a Zel'dovich sphere collapses at delta_lin = 3


def svdw_SF(S, dv, dc, jmax=400):
    """S F(S): Sheth & van de Weygaert (2004) eq. (1), first down-crossings of dv avoiding dc."""
    D = abs(dv) / (dc + abs(dv))
    j = np.arange(1, jmax + 1)[:, None]
    x = np.atleast_1d(dv ** 2 / S)[None, :]
    return np.sum(j ** 2 * np.pi ** 2 * D ** 2 / x * np.sin(j * np.pi * D) / (j * np.pi)
                  * np.exp(-j ** 2 * np.pi ** 2 * D ** 2 / (2 * x)), axis=0)


# ---------------------------------------------------------------- (1) two-barrier random walks
S_grid = np.geomspace(1e-2, 400.0, 6000)
NW = 40000
mc = {}
for dc in (DELTA_C, 1.06):
    up, down = first_crossings(S_grid, NW, rng, barrier=dc, lower=DV)
    frac_v = float(np.mean(np.isfinite(down)))
    sb = np.geomspace(0.5, 300, 36)
    h, _ = np.histogram(down[np.isfinite(down)], bins=sb)
    sc = np.sqrt(sb[1:] * sb[:-1])
    mc[dc] = dict(frac=frac_v, frac_th=dc / (dc + abs(DV)), Sc=sc,
                  SF=h / NW / np.diff(np.log(sb)), err=np.sqrt(h) / NW / np.diff(np.log(sb)))
up1, down1 = first_crossings(S_grid, NW, rng, barrier=1e9, lower=DV)    # single barrier at delta_v
h1, _ = np.histogram(down1[np.isfinite(down1)], bins=sb)

# ---------------------------------------------------------------- (3) voids in the Zel'dovich box
L, N = BOX_L, BOX_N
x, psi, _ = zeldovich_box(L, N, rng_for("ch11", "za_box"))
rho = cic(x, N, L)
radii = np.arange(3.0, 60.0, 1.0)
RMIN = 8.0
cen, rv = find_voids(rho, L, radii, rmin=RMIN)
np.savez(DATA / "ch11" / "voids.npz", cen=cen, rv=rv)
nvoid = len(rv)
volfrac = float(np.sum(4 / 3 * np.pi * rv ** 3) / L ** 3)
rb = np.geomspace(RMIN, 45, 13)
hv, _ = np.histogram(rv, bins=rb)
rc = np.sqrt(rb[1:] * rb[:-1])
dndlnR = hv / L ** 3 / np.diff(np.log(rb))
dndlnR_err = np.sqrt(hv) / L ** 3 / np.diff(np.log(rb))
# predictions: Eulerian radius R = 1.71 R_L, S = sigma^2(R_L)
RE = np.geomspace(5, 60, 120)
RL = RE / EXPAND
S_RL = sigma_R(RL) ** 2
dlnS = np.abs(np.gradient(np.log(S_RL), np.log(RL)))
VL, VE = 4 / 3 * np.pi * RL ** 3, 4 / 3 * np.pi * RE ** 3
pred = {}
for key, dv, dc in [("grav", DV, DELTA_C), ("za", DV_ZA, DC_ZA)]:
    SF = svdw_SF(S_RL, dv, dc)
    pred[key] = dict(svdw=SF * dlnS / VL, vdn=SF * dlnS / VE)
at = lambda R, arr: float(np.interp(np.log(R), np.log(RE), arr))

# ---------------------------------------------------------------- (4) stacked profiles
tree = cKDTree(x, boxsize=L)
nbar = N ** 3 / L ** 3
_, fz = growth(0.0)
f0 = float(fz[0])
redges = np.linspace(0, 3, 31)
rmid = 0.5 * (redges[1:] + redges[:-1])
shellv = 4 / 3 * np.pi * (redges[1:] ** 3 - redges[:-1] ** 3)
bins_R = [10, 13, 16, 20, 30]
wcen, wR, wfloor = watershed_voids(rho, L)
nws = len(wR)
prof, vel = [], []
for i in range(nws):
    idx = tree.query_ball_point(wcen[i], 3 * wR[i])
    d = x[idx] - wcen[i]
    d -= L * np.round(d / L)
    rr = np.sqrt((d ** 2).sum(1))
    s = rr / wR[i]
    cnt, _ = np.histogram(s, bins=redges)
    prof.append(cnt / (nbar * shellv * wR[i] ** 3) - 1)
    vr = 100.0 * f0 * (psi[idx] * d).sum(1) / np.maximum(rr, 1e-9)     # km/s: v = H0 f Psi
    vs, _ = np.histogram(s, bins=redges, weights=vr)
    vel.append(np.where(cnt > 0, vs / np.maximum(cnt, 1), np.nan))
prof, vel = np.array(prof), np.array(vel)
fits, stacks = [], []
for a, b in zip(bins_R[:-1], bins_R[1:]):
    m = (wR >= a) & (wR < b)
    mean = prof[m].mean(0)
    err = prof[m].std(0) / np.sqrt(m.sum())
    expected = nbar * shellv * np.sum(wR[m] ** 3)          # particles expected in each stacked shell
    err = np.maximum(err, np.sqrt(np.maximum(1 + mean, 0.02) / expected))   # Poisson floor
    vmean = np.nanmean(vel[m], 0)
    verr = np.nanstd(vel[m], 0) / np.sqrt(m.sum())
    ok = rmid > 0.15
    p, cov = curve_fit(hamaus_profile, rmid[ok], mean[ok], p0=[-0.8, 0.9, 2.0, 8.0], sigma=err[ok],
                       absolute_sigma=True, maxfev=20000,
                       bounds=([-1.5, 0.3, 0.5, 1.0], [0.0, 3.0, 8.0, 25.0]))
    chi2 = float(np.sum(((mean[ok] - hamaus_profile(rmid[ok], *p)) / err[ok]) ** 2))
    # integrated contrast Delta(<r) of the stack and the linear velocity prediction
    Rbar = float(wR[m].mean())
    cum = np.cumsum((mean + 1) * shellv) / (4 / 3 * np.pi * redges[1:] ** 3) - 1
    v_lin = -(1 / 3) * f0 * 100.0 * (redges[1:] * Rbar) * cum
    v_za = f0 * 100.0 * (redges[1:] * Rbar) * (1 - np.cbrt(np.maximum(1 + cum, 1e-6)))
    fits.append(dict(n=int(m.sum()), Rbar=Rbar, p=p, perr=np.sqrt(np.diag(cov)), chi2=chi2, dof=int(ok.sum() - 4)))
    stacks.append(dict(mean=mean, err=err, v=vmean, verr=verr, vlin=v_lin, vza=v_za, cum=cum))

# ---------------------------------------------------------------- (6) connected underdense regions
# Sublevel sets {delta_s < t} of the density smoothed as for the watershed (1 cell): as t rises,
# new regions are born at local minima and merge at saddles. Count both.
sm = ndimage.gaussian_filter(rho, 1.0, mode="wrap") - 1
minima_vals = sm[sm == ndimage.minimum_filter(sm, size=3, mode="wrap")]


def n_components_periodic(mask):
    lab, n = ndimage.label(mask)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for ax in range(3):
        a = np.take(lab, 0, axis=ax).ravel()
        b = np.take(lab, -1, axis=ax).ravel()
        for p_, q_ in zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)]):
            rp, rq = find(p_), find(q_)
            if rp != rq:
                parent[rp] = rq
    return len({find(i) for i in range(1, n + 1)})


tvals = np.linspace(-0.95, 0.0, 20)
ncomp = np.array([n_components_periodic(sm < t) for t in tvals])
nmin = np.array([np.sum(minima_vals < t) for t in tvals])
vol_under = np.array([np.mean(sm < t) for t in tvals])

# ---- figure 1: two-barrier walks
setup(6.8, 2.8)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
nu_v = lambda S: DV ** 2 / S
for dc, col in [(DELTA_C, SERIES[0]), (1.06, SERIES[1])]:
    d = mc[dc]
    axs[0].errorbar(nu_v(d["Sc"]), d["SF"], d["err"], fmt="o", ms=3, color=col, label=rf"walks, $\delta_c={dc:.3g}$")
    SS = np.geomspace(0.5, 300, 300)
    axs[0].plot(nu_v(SS), svdw_SF(SS, DV, dc), color=col, lw=1.2)
SS = np.geomspace(0.5, 300, 300)
axs[0].plot(nu_v(SS), np.sqrt(nu_v(SS) / (2 * np.pi)) * np.exp(-nu_v(SS) / 2), color="0.55", ls=":",
            label=r"one barrier at $\delta_v$")
axs[0].errorbar(nu_v(sc), h1 / NW / np.diff(np.log(sb)), fmt="^", ms=3, color="0.55")
axs[0].set_xscale("log"); axs[0].set_xlim(0.03, 20); axs[0].set_ylim(0, 0.5)
axs[0].set_xlabel(r"$(\delta_v/\sigma)^2$"); axs[0].set_ylabel(r"$S\,\mathcal{F}(S)$ per $\ln S$")
axs[0].legend(fontsize=6.5)
# sample walks with both barriers
Sx = np.linspace(0, 8, 801)[1:]
for i, col in enumerate(SERIES[:4]):
    w = np.cumsum(rng.standard_normal(800) * np.sqrt(8 / 800))
    axs[1].plot(Sx, w, lw=0.8, color=col)
axs[1].axhline(DELTA_C, color="k", ls="--", lw=1); axs[1].axhline(DV, color="k", ls="--", lw=1)
axs[1].text(0.1, DELTA_C + 0.15, r"$\delta_c$", fontsize=9); axs[1].text(0.1, DV - 0.5, r"$\delta_v$", fontsize=9)
axs[1].set_xlabel(r"$S$"); axs[1].set_ylabel(r"$\delta(S)$"); axs[1].set_ylim(-4.5, 3.5)
fig.tight_layout()
savefig(fig, "ch11", "void_barriers")

# ---- figure 2: void size function
fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.errorbar(rc, dndlnR, dndlnR_err, fmt="o", ms=4, color=SERIES[0], label="spherical finder, Zel'dovich box")
ax.plot(RE, pred["za"]["svdw"], color=SERIES[1], label=r"SvdW, Zel'dovich thresholds")
ax.plot(RE, pred["za"]["vdn"], color=SERIES[2], label=r"volume-conserving, Zel'dovich thresholds")
ax.plot(RE, pred["grav"]["svdw"], color=SERIES[1], ls=":", label=r"SvdW, spherical-collapse thresholds")
ax.plot(RE, pred["grav"]["vdn"], color=SERIES[2], ls=":", label=r"volume-conserving, spherical collapse")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(6, 60); ax.set_ylim(1e-8, 1e-3)
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xticks([6, 10, 20, 30, 60]); ax.set_xticklabels(["6", "10", "20", "30", "60"])
ax.set_xlabel(r"void radius $R_v\ [h^{-1}\mathrm{Mpc}]$"); ax.set_ylabel(r"$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^3\mathrm{Mpc}^{-3}]$")
ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch11", "void_sizes")

# ---- figure 3: stacked density and velocity profiles
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
rr_f = np.linspace(0.05, 3, 300)
for i, (ft, st) in enumerate(zip(fits, stacks)):
    lab = rf"$\bar R_v={ft['Rbar']:.0f}$, $N={ft['n']}$"
    axs[0].errorbar(rmid, st["mean"], st["err"], fmt="o", ms=2.5, color=SERIES[i], label=lab)
    axs[0].plot(rr_f, hamaus_profile(rr_f, *ft["p"]), color=SERIES[i], lw=1)
    axs[1].errorbar(rmid, st["v"], st["verr"], fmt="o", ms=2.5, color=SERIES[i])
    axs[1].plot(redges[1:], st["vza"], color=SERIES[i], lw=1)
    axs[1].plot(redges[1:], st["vlin"], color=SERIES[i], lw=1, ls=":")
axs[0].axhline(0, color="0.6", lw=0.8)
axs[0].set_xlabel(r"$r/R_v$"); axs[0].set_ylabel(r"$\rho_v(r)/\bar\rho-1$")
axs[0].legend(fontsize=6.5, loc="lower right")
axs[1].axhline(0, color="0.6", lw=0.8)
axs[1].set_xlabel(r"$r/R_v$"); axs[1].set_ylabel(r"radial velocity $v_r$ [km/s]")
fig.tight_layout()
savefig(fig, "ch11", "void_profiles")

# ---- figure 4: underdense components against the threshold
fig, ax = plt.subplots(figsize=(4.8, 3.0))
ax.plot(tvals, nmin, "s-", ms=3, color=SERIES[1], label=r"local minima below $t$ (births)")
ax.plot(tvals, ncomp, "o-", ms=3, color=SERIES[0], label=r"connected regions of $\{\delta_s<t\}$")
ax.set_yscale("log")
ax.set_xlabel(r"threshold $t$"); ax.set_ylabel("number")
ax2 = ax.twinx()
ax2.plot(tvals, vol_under, color="0.6", lw=1, ls=":")
ax2.set_ylabel("volume fraction below $t$", color="0.45")
ax2.grid(False)
ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.14, 0.12))
fig.tight_layout()
savefig(fig, "ch11", "void_topology")

vals = {
    "VdThr": THR, "VdDv": DV, "VdExpand": EXPAND, "VdDvSc": DV_sc, "VdDnlSc": DNL_sc,
    "VdNwalk": NW, "VdFracMC": mc[DELTA_C]["frac"], "VdFracTh": mc[DELTA_C]["frac_th"],
    "VdFracMCb": mc[1.06]["frac"], "VdFracThb": mc[1.06]["frac_th"],
    "VdDvoid": abs(DV) / (DELTA_C + abs(DV)),
    "VdN": nvoid, "VdRmin": RMIN, "VdVolFrac": 100 * volfrac, "VdRmax": float(rv.max()),
    "VdRmed": float(np.median(rv)), "VdFzero": f0,
    "VdDvZA": DV_ZA, "VdDcZA": DC_ZA,
    "VdSvdwRatio": float(np.median((np.interp(np.log(rc), np.log(RE), pred["za"]["svdw"]) / dndlnR)[(rc > 9) & (rc < 19)])),
    "VdVdnRatio": float(np.median((np.interp(np.log(rc), np.log(RE), pred["za"]["vdn"]) / dndlnR)[(rc > 9) & (rc < 19)])),
    "VdGravRatioTwenty": float(at(20, pred["za"]["svdw"]) / at(20, pred["grav"]["svdw"])),
    "VdNws": nws, "VdRmedWs": float(np.median(wR)),
    "VdNcompEight": int(ncomp[np.argmin(abs(tvals + 0.8))]), "VdNminEight": int(nmin[np.argmin(abs(tvals + 0.8))]),
    "VdNcompFive": int(ncomp[np.argmin(abs(tvals + 0.5))]), "VdNminFive": int(nmin[np.argmin(abs(tvals + 0.5))]),
    "VdNcompZero": int(ncomp[-1]), "VdNminZero": int(nmin[-1]), "VdNcompMax": int(ncomp.max()),
    "VdTcompMax": float(tvals[np.argmax(ncomp)]), "VdVolUnderFive": float(vol_under[np.argmin(abs(tvals + 0.5))]),
}
names = ["A", "B", "C", "D"]
for nm, ft, st in zip(names, fits, stacks):
    vals.update({f"VdN{nm}": ft["n"], f"VdR{nm}": ft["Rbar"], f"VdDc{nm}": ft["p"][0], f"VdRs{nm}": ft["p"][1],
                 f"VdAl{nm}": ft["p"][2], f"VdBe{nm}": ft["p"][3], f"VdChi{nm}": ft["chi2"], f"VdDof{nm}": ft["dof"],
                 f"VdDcErr{nm}": ft["perr"][0], f"VdRsErr{nm}": ft["perr"][1], f"VdAlErr{nm}": ft["perr"][2],
                 f"VdBeErr{nm}": ft["perr"][3],
                 f"VdAlH{nm}": -2 * (ft["p"][1] - 2),
                 f"VdBeH{nm}": (17.5 * ft["p"][1] - 6.5) if ft["p"][1] < 0.91 else (-9.8 * ft["p"][1] + 18.4),
                 f"VdCumThree{nm}": float(st["cum"][-1]),
                 f"VdVmax{nm}": float(np.nanmax(st["v"])), f"VdVlinmax{nm}": float(np.nanmax(st["vlin"])),
                 f"VdVzamax{nm}": float(np.nanmax(st["vza"])),
                 f"VdWall{nm}": float(np.max(st["mean"][rmid > 0.8]))})
print("NONFINITE", {k: v for k, v in vals.items() if not np.isfinite(v)})
print("FITS", [(f["n"], f["Rbar"], f["p"], f["perr"], f["chi2"]) for f in fits])
save_numbers("ch11", "11_voids", vals)
print({k: v for k, v in vals.items()})
