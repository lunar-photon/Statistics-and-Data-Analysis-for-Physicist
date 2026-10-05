"""06_rsd_kaiser.py -- redshift-space distortions: does (1 + f mu^2)^2 come out of a simulation?

Question: if every particle is moved along the line of sight by its own peculiar velocity,
s = x + (u_z / aH) z_hat, is the power of each Fourier mode multiplied by (1 + f mu^2)^2 on large
scales, as the linear continuity-equation argument (Kaiser 1987) predicts, and are the monopole and
quadrupole (1 + 2f/3 + f^2/5) P and (4f/3 + 4f^2/7) P?
Computes: NSIM Gaussian linear fields at z = 1 (side 1000 Mpc/h, 128^3), Zel'dovich particles
x = q + Psi(q), Psi_k = i k delta_k / k^2, whose velocity is u = a H f Psi, so the redshift-space
shift is f Psi_z; cloud-in-cell densities in real and redshift space; the ratio of the two powers
mode by mode against mu; the multipoles P_0, P_2 against the Kaiser formulas; an estimate of f from
the large-scale modes; f sigma_8(z) of the Planck 2018 fiducial model against the BOSS DR12
consensus values (Alam et al. 2017, Table 7).
Writes: figures/ch11/rsd_slice.pdf, figures/ch11/rsd_kaiser.pdf, figures/ch11/rsd_fsigma8.pdf,
results/ch11/06_rsd_kaiser.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_lss import linear_pk_z0, growth, KBins
from lib_fields import kgrid

L, N, Z = 1000.0, 128, 1.0
NSIM = 24
rng = rng_for("ch11", "06_rsd_kaiser")
k_tab, pk_tab, sigma8, _ = linear_pk_z0()
Dz, fz = growth(Z)
D, f = float(Dz[0]), float(fz[0])
P = lambda kk: D ** 2 * np.interp(kk, k_tab, pk_tab)
cell = L / N
kmag, comps = kgrid(N, L, 3)
with np.errstate(divide="ignore", invalid="ignore"):
    amp = np.where(kmag > 0, np.sqrt(P(np.where(kmag > 0, kmag, 1.0)) / cell ** 3), 0.0)
    inv_k2 = np.where(kmag > 0, 1 / kmag ** 2, 0.0)
q = (np.indices((N, N, N)).reshape(3, -1).T + 0.5) * cell        # Lagrangian (grid) positions
# cloud-in-cell window on the mesh, to be divided out of the measured power
wcic = np.ones_like(kmag)
for c in comps:
    wcic *= np.sinc(c * cell / (2 * np.pi)) ** 2
edges = np.arange(0.005, 0.305, 0.01)
kb = KBins(N, L, 3, edges)


def cic(pos):
    """Cloud-in-cell density contrast of particles at pos (periodic box)."""
    g = pos / cell - 0.5
    i0 = np.floor(g).astype(int)
    w1 = g - i0
    rho = np.zeros(N ** 3)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (w1[:, 0] if dx else 1 - w1[:, 0]) * (w1[:, 1] if dy else 1 - w1[:, 1]) * (w1[:, 2] if dz else 1 - w1[:, 2])
                idx = (((i0[:, 0] + dx) % N) * N + (i0[:, 1] + dy) % N) * N + (i0[:, 2] + dz) % N
                rho += np.bincount(idx, weights=w, minlength=N ** 3)
    rho = rho.reshape(N, N, N)
    return rho / rho.mean() - 1


def power(delta):
    return np.abs(np.fft.rfftn(delta) * cell ** 3) ** 2 / L ** 3 / wcic ** 2


ratio_mu, mus, ks = [], None, None
mult_r, mult_s, mult_k, plin_bins = [], [], [], None
mu2_grid = kb.mu.reshape(kmag.shape) ** 2
psi_rms = []
for s in range(NSIM):
    dk = np.fft.rfftn(rng.standard_normal((N, N, N))) * amp           # linear delta_k on the grid
    psi = np.stack([np.fft.irfftn(1j * c * inv_k2 * dk, s=(N, N, N), axes=(0, 1, 2)).ravel() for c in comps], axis=1)
    psi_rms.append(np.sqrt(np.mean(psi ** 2)))
    x = (q + psi) % L
    sz = q + psi
    sz[:, 2] += f * psi[:, 2]                                          # s = x + (u_z/aH) z_hat, u = aHf Psi
    sz %= L
    pr, ps = power(cic(x)), power(cic(sz))
    mult_r.append(kb.multipoles(pr))
    mult_s.append(kb.multipoles(ps))
    mult_k.append(kb.multipoles((1 + f * mu2_grid) ** 2 * pr))    # Kaiser applied to this box's own modes
    sel = (kmag > 0) & (kmag < 0.08)
    ratio_mu.append(np.sqrt(ps[sel] / pr[sel]))                       # should be 1 + f mu^2 mode by mode
    if mus is None:
        mus = np.abs(kb.mu.reshape(kmag.shape)[sel])
    if s == 0:
        x0, s0 = x, sz
plin_bins = kb.avg(P(np.where(kmag > 0, kmag, 1.0)))
kk = kb.kmean
mr, ms, mk = np.mean(mult_r, axis=0), np.mean(mult_s, axis=0), np.mean(mult_k, axis=0)   # (3, nbins): P0, P2, P4
rmu = np.concatenate(ratio_mu)
mu_all = np.tile(mus, NSIM)

# estimate f: least squares of sqrt(P_s/P_r) - 1 = f mu^2 over the large-scale modes
f_hat = float(np.sum(mu_all ** 2 * (rmu - 1)) / np.sum(mu_all ** 4))
per_sim = [float(np.sum(mus ** 2 * (r_ - 1)) / np.sum(mus ** 4)) for r_ in ratio_mu]
low = kk < 0.06
k0 = 1 + 2 * f / 3 + f ** 2 / 5
k2 = 4 * f / 3 + 4 * f ** 2 / 7
R0 = float(np.mean((ms[0] / mr[0])[low]))
R2 = float(np.mean((ms[1] / mr[0])[low]))
R2k = float(np.mean((mk[1] / mr[0])[low]))       # the same average of the Kaiser model over the same modes
R0k = float(np.mean((mk[0] / mr[0])[low]))

# ---- figure 1: a thin slab in real and redshift space
setup(6.4, 3.2)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
slab = (x0[:, 0] > 480) & (x0[:, 0] < 500)
slab_s = (s0[:, 0] > 480) & (s0[:, 0] < 500)
for a, p, sl, t in [(axs[0], x0, slab, "real space"), (axs[1], s0, slab_s, "redshift space")]:
    a.scatter(p[sl, 1], p[sl, 2], s=0.15, color=SERIES[0], rasterized=True)
    a.set_xlim(0, 400); a.set_ylim(0, 400); a.set_aspect("equal"); a.set_title(t); a.grid(False)
    a.set_xlabel(r"$y\ [h^{-1}\mathrm{Mpc}]$")
axs[0].set_ylabel(r"$z$ (line of sight) $[h^{-1}\mathrm{Mpc}]$")
fig.tight_layout()
savefig(fig, "ch11", "rsd_slice")

# ---- figure 2: the Kaiser factor mode by mode, and the multipoles
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
mb = np.linspace(0, 1, 11)
idx = np.digitize(mu_all, mb) - 1
mc = 0.5 * (mb[1:] + mb[:-1])
avg = np.array([np.mean(rmu[idx == i] ** 2) for i in range(10)])
sd = np.array([np.std(rmu[idx == i] ** 2) / np.sqrt(np.sum(idx == i)) for i in range(10)])
axs[0].errorbar(mc, avg, sd, fmt="o", ms=4, color=SERIES[0], label=r"$P_s/P_r$, modes with $k<0.08$")
mm = np.linspace(0, 1, 100)
theory_line(axs[0], mm, (1 + f * mm ** 2) ** 2, label=r"$(1+f\mu^2)^2$")
axs[0].set_xlabel(r"$\mu=\hat{k}\cdot\hat{z}$")
axs[0].set_ylabel("power ratio")
axs[0].legend(fontsize=7)
axs[1].plot(kk, ms[0] / mr[0], "o", ms=3, color=SERIES[0], label=r"$P_0^s/P^r$")
axs[1].plot(kk, ms[1] / mr[0], "s", ms=3, color=SERIES[1], label=r"$P_2^s/P^r$")
axs[1].plot(kk, ms[2] / mr[0], "^", ms=3, color=SERIES[2], label=r"$P_4^s/P^r$")
for i in range(3):
    axs[1].plot(kk, mk[i] / mr[0], color="k", ls="--", lw=1.0)
for v in (k0, k2, 8 * f ** 2 / 35):
    axs[1].axhline(v, color="0.6", ls=":", lw=1.0)
axs[1].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
axs[1].set_ylabel("multipole / real-space power")
axs[1].legend(fontsize=7, loc="center right")
fig.tight_layout()
savefig(fig, "ch11", "rsd_kaiser")

# ---- figure 3: f sigma_8(z) of the fiducial model against BOSS DR12 (Alam et al. 2017, Table 7, BAO+FS)
zs = np.linspace(0.0, 1.5, 151)
Dg, fg = growth(zs)
fs8 = fg * sigma8 * Dg
boss = [(0.38, 0.497, 0.039, 0.024), (0.51, 0.458, 0.035, 0.015), (0.61, 0.436, 0.034, 0.009)]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
theory_line(ax, zs, fs8, label=r"Planck 2018 $\Lambda$CDM")
for z_, v, st, sy in boss:
    ax.errorbar(z_, v, np.hypot(st, sy), fmt="o", ms=4, color=SERIES[0])
ax.plot([], [], "o", color=SERIES[0], label="BOSS DR12 (stat+sys)")
ax.set_xlabel("redshift $z$")
ax.set_ylabel(r"$f\sigma_8(z)$")
ax.legend(fontsize=8)
savefig(fig, "ch11", "rsd_fsigma8")
pred = {z_: float(np.interp(z_, zs, fs8)) for z_, *_ in boss}
pulls = [(v - pred[z_]) / np.hypot(st, sy) for z_, v, st, sy in boss]

save_numbers("ch11", "06_rsd_kaiser", {
    "RsdL": f"{L:g}", "RsdN": N, "RsdNsim": NSIM, "RsdZ": f"{Z:g}", "RsdF": f"{f:.3f}", "RsdD": f"{D:.3f}",
    "RsdFhat": f"{f_hat:.3f}", "RsdFsd": f"{np.std(per_sim, ddof=1):.3f}",
    "RsdKzero": f"{k0:.3f}", "RsdKtwo": f"{k2:.3f}", "RsdKfour": f"{8*f**2/35:.3f}",
    "RsdRzero": f"{R0:.3f}", "RsdRtwo": f"{R2:.3f}", "RsdRtwoK": f"{R2k:.3f}", "RsdRzeroK": f"{R0k:.3f}", "RsdPsi": f"{np.mean(psi_rms):.1f}",
    "RsdSigEightZ": f"{sigma8*D:.3f}",
    "RsdFsA": f"{pred[0.38]:.3f}", "RsdFsB": f"{pred[0.51]:.3f}", "RsdFsC": f"{pred[0.61]:.3f}",
    "RsdFerr": f"{np.std(per_sim, ddof=1)/np.sqrt(NSIM):.3f}",
    "RsdFpull": f"{(f_hat - f)/(np.std(per_sim, ddof=1)/np.sqrt(NSIM)):.1f}",
    "RsdSmear": f"{f*np.mean(psi_rms):.1f}", "RsdDampK": f"{(0.2*f*np.mean(psi_rms))**2:.2f}",
    "RsdFoff": f"{f_hat - f:+.3f}",
    "RsdRtwoShort": f"{100 * (1 - R2 / R2k):.1f}",          # measured quadrupole below the same-mode Kaiser value [%]
    "RsdRtwoKex": f"{100 * abs(R2k / k2 - 1):.0f}",          # same-mode Kaiser value against the continuum [%]
    "RsdPullA": f"{pulls[0]:.1f}", "RsdPullB": f"{pulls[1]:.1f}", "RsdPullC": f"{pulls[2]:.1f}",
})
print(f"discrete Kaiser R0={R0k:.3f} R2={R2k:.3f}")
print(f"f={f:.3f} f_hat={f_hat:.3f}+-{np.std(per_sim, ddof=1):.3f}; R0={R0:.3f} vs {k0:.3f}; R2={R2:.3f} vs {k2:.3f}; "
      f"fs8 pred {pred}; pulls {pulls}")
