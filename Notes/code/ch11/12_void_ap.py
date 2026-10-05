"""12_void_ap.py -- stacked voids as standard spheres: the Alcock-Paczynski test, and what
redshift-space distortions do to it.

Question: the average void is spherical. If we convert redshifts to distances with a wrong
cosmology, line-of-sight distances are stretched by q = (D_M H)_true / (D_M H)_fid. Can a stack of
a few thousand voids measure q, and how large is the competing elongation caused by the outflow
of matter from voids in redshift space?
Computes: the spherical voids of 11_voids.py (data/ch11/voids.npz) with 10 <= R_v < 25 Mpc/h in the
Zel'dovich box; stacked 2-D density in (r_perp, r_par)/R_v in (a) real space, (b) real space with
the line of sight stretched by q_true = 1.05, (c) redshift space s_z = z + f Psi_z, (d) both;
in 10 bins of |mu| the radius where the stacked density rises through -0.5, and the ellipse
r(mu) = R0 / sqrt(1 - mu^2 (1 - 1/q^2)) fitted to those radii (jackknife errors over 27 sub-cubes of the box); the linear and
spherical-Zel'dovich predictions of the redshift-space elongation; D_M H / c at z = 0.51 for the
fiducial cosmology against Hamaus et al. (2020), and its sensitivity to Omega_m.
Writes: figures/ch11/void_ap.pdf, results/ch11/12_void_ap.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA
from lib_web import BOX_L, BOX_N, zeldovich_box
from lib_lss import growth, E_of_z, OMM

L, N = BOX_L, BOX_N
x, psi, _ = zeldovich_box(L, N, rng_for("ch11", "za_box"))
v = np.load(DATA / "ch11" / "voids.npz")
sel = (v["rv"] >= 10) & (v["rv"] < 25)
cen, rv = v["cen"][sel], v["rv"][sel]
_, fz = growth(0.0)
f0 = float(fz[0])
Q_TRUE = 1.05
nbar = N ** 3 / L ** 3
RMAX, NR, NMU, NSUB = 2.0, 60, 10, 3
NJK = NSUB ** 3
re_ = np.linspace(0, RMAX, NR + 1)
mue = np.linspace(0, 1, NMU + 1)
muc = 0.5 * (mue[1:] + mue[:-1])
LEVEL = -0.5
# spatial jackknife: the group of a void is the sub-cube (NSUB^3 of them) that holds its centre, so that
# leaving a group out removes a region of the box, with the large-scale modes that the voids in it share
cell_ = np.clip((cen // (L / NSUB)).astype(int), 0, NSUB - 1)
jk = (cell_[:, 0] * NSUB + cell_[:, 1]) * NSUB + cell_[:, 2]


def stack(pos, Lz, q_geom):
    """Counts and expected counts in (r, |mu|) bins, r in units of R_v, per jackknife group.

    pos: particle positions in a box of sides (L, L, Lz); void centres are moved by q_geom along z.
    The fraction of directions with |mu| in a bin of width dmu is dmu, so a shell piece holds
    nbar (4 pi/3) (r2^3 - r1^3) R_v^3 dmu particles on average.
    """
    tree = cKDTree(pos, boxsize=[L, L, Lz])
    box = np.array([L, L, Lz])
    cnt = np.zeros((NJK, NR, NMU))
    exp = np.zeros((NJK, NR, NMU))
    n_local = len(pos) / (L * L * Lz)
    shell = 4 / 3 * np.pi * (re_[1:] ** 3 - re_[:-1] ** 3)[:, None] * np.diff(mue)[None, :]
    for g, c, r in zip(jk, cen, rv):
        cc = c * np.array([1, 1, q_geom])
        idx = tree.query_ball_point(cc, RMAX * r)
        d = pos[idx] - cc
        d -= box * np.round(d / box)
        rr = np.sqrt((d ** 2).sum(1))
        mu = np.abs(d[:, 2]) / np.maximum(rr, 1e-12)
        h, _, _ = np.histogram2d(rr / r, mu, bins=[re_, mue])
        cnt[g] += h
        exp[g] += n_local * shell * r ** 3
    return cnt, exp


def crossing_radii(cnt, exp):
    """Radius at which the stacked density in each mu bin first rises through LEVEL."""
    dens = cnt / exp - 1
    rc = 0.5 * (re_[1:] + re_[:-1])
    out = np.empty(NMU)
    for j in range(NMU):
        i = np.nonzero(dens[:, j] >= LEVEL)[0][0]
        out[j] = rc[i - 1] + (LEVEL - dens[i - 1, j]) / (dens[i, j] - dens[i - 1, j]) * (rc[i] - rc[i - 1])
    return out


def ellipse_q(rc_mu):
    """Fit r(mu) = R0 / sqrt(1 - mu^2 (1 - 1/q^2)): the contour of a sphere stretched by q along z."""
    sol = least_squares(lambda p: p[0] / np.sqrt(1 - muc ** 2 * (1 - 1 / p[1] ** 2)) - rc_mu, x0=[1.0, 1.0])
    return sol.x


def fit(cnt, exp):
    full = ellipse_q(crossing_radii(cnt.sum(0), exp.sum(0)))
    jks = np.array([ellipse_q(crossing_radii(cnt.sum(0) - cnt[g], exp.sum(0) - exp[g])) for g in range(NJK)])
    err = np.sqrt((NJK - 1) / NJK * np.sum((jks - jks.mean(0)) ** 2, axis=0))
    return full, err, cnt.sum(0) / exp.sum(0) - 1, crossing_radii(cnt.sum(0), exp.sum(0))


# (a) real space, (b) stretched line of sight, (c) redshift space, (d) both
x_rsd = x.copy()
x_rsd[:, 2] = (x[:, 2] + f0 * psi[:, 2]) % L
cases = {
    "real": (x, L, 1.0),
    "ap": (x * np.array([1, 1, Q_TRUE]), Q_TRUE * L, Q_TRUE),
    "rsd": (x_rsd, L, 1.0),
    "both": (x_rsd * np.array([1, 1, Q_TRUE]), Q_TRUE * L, Q_TRUE),
}
res = {}
for key, (pos, Lz, qg) in cases.items():
    cnt, exp = stack(pos, Lz, qg)
    res[key] = fit(cnt, exp)
    print(key, res[key][0], res[key][1])

# predicted redshift-space elongation at r = R_v, where Delta(<R_v) = -0.8 by construction
q_rsd_lin = 1 + f0 * 0.8 / 3
q_rsd_za = 1 + f0 * (1 - 0.2 ** (1 / 3))
# predicted shape of the -0.5 contour: take the real-space stack averaged over mu. Along the line of
# sight (mu = 1) each shell moves by f Psi_r, Psi_r = r [1 - (1 + Delta(<r))^(1/3)] (spherical
# Zel'dovich) or -r Delta(<r)/3 (linear), and the density is divided by ds/dr (number conservation).
# Across it (mu = 0) nothing moves, but the line-of-sight stretching of the map, 1 + f dPsi_z/dz =
# 1 + f Psi_r/r there, still thins the void. The axis ratio is the ratio of the two radii.
prof = res["real"][2].mean(axis=1)                    # rho/rho_bar - 1 in r bins (mu-averaged)
rc_ = 0.5 * (re_[1:] + re_[:-1])
cum = np.cumsum((prof + 1) * (re_[1:] ** 3 - re_[:-1] ** 3)) / re_[1:] ** 3 - 1
cum_c = np.interp(rc_, re_[1:], cum)
pred_q = {}
for key, psi_r in [("za", rc_ * (1 - np.cbrt(np.maximum(1 + cum_c, 1e-6)))), ("lin", -rc_ * cum_c / 3)]:
    s_los = rc_ + f0 * psi_r
    dens_s = (1 + prof) / np.gradient(s_los, rc_) - 1
    i = np.nonzero(dens_s >= LEVEL)[0][0]
    s_c = s_los[i - 1] + (LEVEL - dens_s[i - 1]) / (dens_s[i] - dens_s[i - 1]) * (s_los[i] - s_los[i - 1])
    j = np.nonzero(prof >= LEVEL)[0][0]
    r_c = rc_[j - 1] + (LEVEL - prof[j - 1]) / (prof[j] - prof[j - 1]) * (rc_[j] - rc_[j - 1])
    dens_p = (1 + prof) / (1 + f0 * psi_r / rc_) - 1
    k_ = np.nonzero(dens_p >= LEVEL)[0][0]
    s_p = rc_[k_ - 1] + (LEVEL - dens_p[k_ - 1]) / (dens_p[k_] - dens_p[k_ - 1]) * (rc_[k_] - rc_[k_ - 1])
    pred_q[key] = (s_c / r_c, s_p / r_c, s_c / s_p)     # mu = 1 vs real, mu = 0 vs real, axis ratio
print("predicted contour elongation", pred_q)

# D_M H / c at z = 0.51 (flat LCDM), and its sensitivity to Omega_m
def dmh(z, om):
    zz = np.linspace(0, z, 4001)
    return float(E_of_z(z, om) * np.trapezoid(1 / E_of_z(zz, om), zz))


Z_B = 0.51
dmh_fid = dmh(Z_B, OMM)
dlnq_dom = (np.log(dmh(Z_B, OMM + 0.01)) - np.log(dmh(Z_B, OMM - 0.01))) / 0.02
sig_om_from_q = res["ap"][1][1] / Q_TRUE / abs(dlnq_dom)

# ---- figure: the four stacks, mapped back to (r_perp, r_par)
setup(7.0, 2.6)
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
titles = {"real": "real space", "ap": rf"line of sight $\times{Q_TRUE}$", "rsd": "redshift space", "both": "both"}
g = np.linspace(-RMAX, RMAX, 161)
GX, GZ = np.meshgrid(g, g, indexing="xy")
RR = np.sqrt(GX ** 2 + GZ ** 2)
MU = np.abs(GZ) / np.maximum(RR, 1e-9)
rc = 0.5 * (re_[1:] + re_[:-1])
for a, key in zip(axs, ["real", "ap", "rsd", "both"]):
    dens = res[key][2]
    ir = np.clip(np.searchsorted(re_, RR) - 1, 0, NR - 1)
    im = np.clip(np.searchsorted(mue, MU) - 1, 0, NMU - 1)
    img = np.where(RR < RMAX, dens[ir, im], np.nan)
    a.imshow(img, origin="lower", extent=[-RMAX, RMAX, -RMAX, RMAX], cmap="RdBu_r", vmin=-1, vmax=1)
    q, R0 = res[key][0][1], res[key][0][0]
    t = np.linspace(0, 2 * np.pi, 200)
    a.plot(R0 * np.cos(t), R0 * q * np.sin(t), color="k", lw=0.8)
    a.plot(R0 * np.cos(t), R0 * np.sin(t), color="0.4", lw=0.6, ls=":")
    a.set_title(titles[key] + "\n" + rf"$q={q:.3f}$", fontsize=8)
    a.set_xlabel(r"$r_\perp/R_v$", fontsize=8); a.grid(False)
    a.set_aspect("equal")
axs[0].set_ylabel(r"$r_\parallel/R_v$", fontsize=8)
fig.tight_layout()
savefig(fig, "ch11", "void_ap")

save_numbers("ch11", "12_void_ap", {
    "ApNvoid": int(sel.sum()), "ApQtrue": Q_TRUE, "ApFzero": f0,
    "ApQreal": res["real"][0][1], "ApQrealErr": res["real"][1][1],
    "ApQap": res["ap"][0][1], "ApQapErr": res["ap"][1][1],
    "ApQrsd": res["rsd"][0][1], "ApQrsdErr": res["rsd"][1][1],
    "ApQboth": res["both"][0][1], "ApQbothErr": res["both"][1][1],
    "ApQprod": res["ap"][0][1] * res["rsd"][0][1], "ApLevel": LEVEL, "ApNjk": NJK,
    "ApRzero": res["real"][0][0],
    "ApQrsdLin": q_rsd_lin, "ApQrsdZA": q_rsd_za, "ApQcontZA": pred_q["za"][2], "ApQcontLin": pred_q["lin"][2],
    "ApParZA": pred_q["za"][0], "ApPerpZA": pred_q["za"][1], "ApParLin": pred_q["lin"][0], "ApPerpLin": pred_q["lin"][1],
    "ApParMeas": float(res["rsd"][3][-1] / res["real"][3][-1]), "ApPerpMeas": float(res["rsd"][3][0] / res["real"][3][0]),
    "ApQapPct": f"{100 * res['ap'][1][1] / res['ap'][0][1]:.1f}",
    "ApZ": Z_B, "ApDMH": dmh_fid, "ApDlnqDom": dlnq_dom, "ApSigOm": sig_om_from_q,
    "ApDMHHam": 0.588, "ApDMHHamErr": 0.004,
})
