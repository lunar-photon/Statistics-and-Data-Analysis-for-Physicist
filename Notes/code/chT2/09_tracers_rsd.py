"""Galaxies as biased tracers, and the Kaiser squashing of redshift space.

Question: (1) if "galaxies" form where the local density (a smoothed Gaussian field plus an
independent small-scale part the grid cannot resolve) exceeds nu standard deviations, how much more clustered are they than the matter, and does the measured bias
match b_L = phi(nu) / [sigma_R (1 - Phi(nu))]? (2) After the tracers move with the matter
(Zel'dovich displacement), does the bias become 1 + b_L? (3) Seen in redshift space, with
the line-of-sight displacement multiplied by (1 + f), is the power along the line of sight
enhanced by (1 + beta mu^2)^2, beta = f / b?
Computes: one Gaussian random field in a periodic box (linear P(k) at z = 1 from CAMB, cached by
code/chT1/02_pk_xi.py), threshold tracers for several nu, Poisson-sampled galaxies,
Zel'dovich-moved matter and tracers in real and redshift space, and their power spectra.
Writes: figures/chT2/tracers_rsd.pdf, results/chT2/09_tracers_rsd.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line, DATA

L, N = 1000.0, 128                     # box side [Mpc/h], cells per side
R = 10.0                               # Gaussian smoothing of the resolved large-scale field [Mpc/h]
SIG_SMALL = 1.5                        # rms of the unresolved small-scale density (independent per cell)
NBAR = 3e-4                            # galaxy number density [(h/Mpc)^3]
OM, ZBOX = 0.3153, 1.0                 # the box is a snapshot at z = 1 (linear P(k) at z = 1)
OMZ = OM * (1 + ZBOX) ** 3 / (OM * (1 + ZBOX) ** 3 + 1 - OM)
F_GROW = OMZ ** 0.55                   # linear growth rate f at z = 1
KMAX = 0.05                            # largest k used in the fits [h/Mpc]
NUS = [0.5, 1.0, 1.5, 2.0, 2.5]

pk = np.load(DATA / "chT1" / "pk.npz")
kk, plin = pk["k"], pk["pk_lin"][1]

rng = rng_for("chT2", "09_tracers_rsd")
kf = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
kz1 = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
KX, KY, KZ = np.meshgrid(kf, kf, kz1, indexing="ij")
K = np.sqrt(KX**2 + KY**2 + KZ**2)
K[0, 0, 0] = 1.0

# --- a Gaussian field with power P(k): white noise times sqrt(P / cell volume)
cellv = (L / N) ** 3
white = np.fft.rfftn(rng.normal(0, 1, (N, N, N)))
dk = white * np.sqrt(np.interp(K, kk, plin) / cellv)
dk[0, 0, 0] = 0.0
delta = np.fft.irfftn(dk, s=(N, N, N))
WR = np.exp(-0.5 * (K * R) ** 2)
deltaR = np.fft.irfftn(dk * WR, s=(N, N, N))
sigR = deltaR.std()
# local density that decides where galaxies form: large-scale part + small-scale "peak" part
dsel = deltaR + rng.normal(0, SIG_SMALL, (N, N, N))
sigT = dsel.std()

# Zel'dovich displacement psi_k = i k delta_k / k^2 (so that -div psi = delta)
psi = [np.fft.irfftn(1j * Kc / K**2 * dk, s=(N, N, N)) for Kc in (KX, KY, KZ)]


def paint(pos, w=None):
    """Cloud-in-cell assignment of points (Mpc/h) to the grid; returns overdensity."""
    g = pos / (L / N)
    i0 = np.floor(g).astype(int)
    d = g - i0
    rho = np.zeros(N**3)
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                wt = (np.where(ox, d[:, 0], 1 - d[:, 0]) * np.where(oy, d[:, 1], 1 - d[:, 1])
                      * np.where(oz, d[:, 2], 1 - d[:, 2]))
                if w is not None:
                    wt = wt * w
                flat = (((i0[:, 0] + ox) % N) * N + (i0[:, 1] + oy) % N) * N + (i0[:, 2] + oz) % N
                rho += np.bincount(flat, weights=wt, minlength=N**3)
    rho = rho.reshape(N, N, N)
    return rho / rho.mean() - 1


kb = np.linspace(2 * np.pi / L, 0.2, 26)


def power(a, b=None, mu_edges=None):
    """Binned (cross-)power spectrum; with mu_edges, also binned in mu = |k_z|/k."""
    fa = np.fft.rfftn(a) * cellv
    fb = fa if b is None else np.fft.rfftn(b) * cellv
    p = (fa * np.conj(fb)).real / L**3
    if mu_edges is None:
        idx = np.digitize(K.ravel(), kb)
        num = np.bincount(idx, weights=p.ravel(), minlength=kb.size + 1)
        den = np.bincount(idx, minlength=kb.size + 1)
        return 0.5 * (kb[1:] + kb[:-1]), (num / np.maximum(den, 1))[1:kb.size]
    mu = np.abs(KZ) / K
    sel = (K < KMAX) & (K > 2 * np.pi / L * 1.5)
    out = []
    for a0, a1 in zip(mu_edges[:-1], mu_edges[1:]):
        m = sel & (mu >= a0) & (mu < a1)
        out.append(p[m].mean())
    return np.array(out)


def bias_cross(a, m, w=1.0):
    """Large-scale bias from the cross spectrum, mode by mode: mean and error of
    Re(a_k m_k*) / (w_k |m_k|^2) over the modes with k < KMAX (error = scatter / sqrt(modes))."""
    fa, fm = np.fft.rfftn(a), np.fft.rfftn(m)
    sel = (K < KMAX) & (K > 1e-3)
    r = ((fa * np.conj(fm)).real / (np.abs(fm) ** 2 * w))[sel]
    return r.mean(), r.std() / np.sqrt(r.size)


grid = (np.indices((N, N, N)).reshape(3, -1).T + 0.5) * (L / N)      # Lagrangian positions
dispx = np.stack([c.ravel() for c in psi], axis=1)

# matter, moved with the Zel'dovich displacement (real space and redshift space)
dm_real = paint(grid + dispx)
los = np.array([0, 0, 1.0])
dm_red = paint(grid + dispx + F_GROW * dispx[:, 2:3] * los)
kc, pmm = power(delta)
_, pmm_e = power(dm_real)

bL_meas, bE_meas, bL_th, bL_err, bE_err = [], [], [], [], []
fit = (kc < KMAX)
for nu in NUS:
    ind = (dsel > nu * sigT).astype(float)
    dt = ind / ind.mean() - 1
    b, e = bias_cross(dt, delta, WR)
    bL_meas.append(b); bL_err.append(e)
    bL_th.append(norm.pdf(nu) / (sigT * norm.sf(nu)))
    # Eulerian: move the tracer cells with the matter
    w = ind.ravel()
    sel = w > 0
    dte = paint(grid[sel] + dispx[sel])
    b, e = bias_cross(dte, dm_real)
    bE_meas.append(b); bE_err.append(e)
bL_meas, bE_meas, bL_th, bL_err, bE_err = map(np.array, (bL_meas, bE_meas, bL_th, bL_err, bE_err))

# --- one galaxy sample (nu = 1.5): Poisson-sample the tracer cells, then real vs redshift space
NU_G = 1.5
ind = (dsel > NU_G * sigT).ravel()
lam = NBAR * L**3 / ind.sum()
ngal = rng.poisson(lam, ind.sum())
src = np.repeat(np.flatnonzero(ind), ngal)
jitter = rng.uniform(-0.5, 0.5, (src.size, 3)) * (L / N)
gpos = grid[src] + jitter + dispx[src]
g_real = paint(gpos)
g_red = paint(gpos + F_GROW * dispx[src, 2:3] * los)
_, pgg = power(g_real)
_, pgm = power(g_real, dm_real)
bG, bG_err = bias_cross(g_real, dm_real)
shot = L**3 / src.size                                   # Poisson term 1/nbar
onehalo = L**3 * np.sum(ngal * (ngal - 1.0)) / src.size**2   # pairs that share a cell ("one-halo")
noise = shot + onehalo
bG_auto_naive = np.sqrt(np.mean((pgg[fit] - shot) / pmm_e[fit]))
bG_auto = np.sqrt(np.mean((pgg[fit] - noise) / pmm_e[fit]))
beta = F_GROW / bG
mu_e = np.linspace(0, 1, 11)
mu_c = 0.5 * (mu_e[1:] + mu_e[:-1])
ratio_g = (power(g_red, mu_edges=mu_e) - noise) / (power(g_real, mu_edges=mu_e) - noise)
ratio_g_naive = (power(g_red, mu_edges=mu_e) - shot) / (power(g_real, mu_edges=mu_e) - shot)
ratio_m = power(dm_red, mu_edges=mu_e) / power(dm_real, mu_edges=mu_e)
# the highest-mu bin holds few modes: repeat the matter part on fresh boxes to see its scatter
rm_last = [ratio_m[-1]]
for b in range(10):
    rb = rng_for("chT2", "09_tracers_rsd_boxes", stream=b)
    dkb = np.fft.rfftn(rb.normal(0, 1, (N, N, N))) * np.sqrt(np.interp(K, kk, plin) / cellv)
    dkb[0, 0, 0] = 0.0
    dispb = np.stack([np.fft.irfftn(1j * Kc / K**2 * dkb, s=(N, N, N)).ravel() for Kc in (KX, KY, KZ)], axis=1)
    rmb = (power(paint(grid + dispb + F_GROW * dispb[:, 2:3] * los), mu_edges=mu_e)
           / power(paint(grid + dispb), mu_edges=mu_e))
    rm_last.append(rmb[-1])
rm_last = np.array(rm_last)
kaiser_g = (1 + beta * mu_c**2) ** 2
mono_g = np.mean(ratio_g)                       # mu bins are equal width: average = monopole ratio

# --- figure
setup(7.2, 2.6)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
sl = 0
ax[0].imshow(deltaR[:, :, sl].T, origin="lower", extent=[0, L, 0, L], cmap="Greys", vmin=-2.5 * sigR, vmax=2.5 * sigR)
inslab = (grid[src, 2] < L / N)
ax[0].scatter(grid[src][inslab, 0], grid[src][inslab, 1], s=1.2, color=SERIES[1], lw=0)
ax[0].set(xlabel=r"$x$ [$h^{-1}$Mpc]", ylabel=r"$y$ [$h^{-1}$Mpc]", xlim=(0, 400), ylim=(0, 400))
ax[0].grid(False)
nn = np.linspace(0.3, 2.7, 100)
theory_line(ax[1], nn, norm.pdf(nn) / (sigT * norm.sf(nn)), label=r"$b_L=\varphi(\nu)/[\sigma(1-\Phi(\nu))]$")
ax[1].plot(nn, 1 + norm.pdf(nn) / (sigT * norm.sf(nn)), color=SERIES[0], lw=1, label=r"$1+b_L$")
ax[1].plot(nn, nn / sigT, color="0.6", lw=0.8, ls=":", label=r"$\nu/\sigma$")
ax[1].errorbar(NUS, bL_meas, yerr=bL_err, fmt="o", color="k", ms=4, label="before the move")
ax[1].errorbar(NUS, bE_meas, yerr=bE_err, fmt="s", color=SERIES[0], ms=4, label="after the move")
ax[1].set(xlabel=r"threshold $\nu$", ylabel="large-scale bias")
ax[1].legend(fontsize=6, frameon=False, loc="upper left")
ax[2].plot(mu_c, ratio_g, "o", color=SERIES[1], ms=4, label=rf"galaxies, $b={bG:.2f}$")
ax[2].plot(mu_c, ratio_m, "s", color=SERIES[0], ms=3.5, label=r"matter, $b=1$")
mm = np.linspace(0, 1, 100)
theory_line(ax[2], mm, (1 + beta * mm**2) ** 2, label=r"$(1+\beta\mu^2)^2$")
ax[2].plot(mm, (1 + F_GROW * mm**2) ** 2, color="k", ls="--", lw=1)
ax[2].set(xlabel=r"$\mu=k_\parallel/k$", ylabel=r"$P_s(k,\mu)/P_r(k)$")
ax[2].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout()
savefig(fig, "chT2", "tracers_rsd")

i2 = NUS.index(2.0)
save_numbers("chT2", "09_tracers_rsd", {
    "TwtL": int(L), "TwtN": N, "TwtR": int(R), "TwtSigR": f"{sigR:.3f}", "TwtKmax": KMAX,
    "TwtNuTwo": "2", "TwtBLthTwo": f"{bL_th[i2]:.2f}", "TwtBLmTwo": f"{bL_meas[i2]:.2f}",
    "TwtBEthTwo": f"{1 + bL_th[i2]:.2f}", "TwtBEmTwo": f"{bE_meas[i2]:.2f}",
    "TwtBLthOne": f"{bL_th[1]:.2f}", "TwtBLmOne": f"{bL_meas[1]:.2f}", "TwtBLerrOne": f"{bL_err[1]:.2f}",
    "TwtNuHighTwo": f"{2.0 / sigT:.2f}", "TwtNuHighLast": f"{2.5 / sigT:.2f}",
    "TwtNbar": "3\\times10^{-4}", "TwtNgal": f"{src.size}", "TwtShot": f"{shot:.0f}",
    "TwtBG": f"{bG:.2f}", "TwtBGerr": f"{bG_err:.2f}", "TwtBGauto": f"{bG_auto:.2f}",
    "TwtBGautoNaive": f"{bG_auto_naive:.2f}", "TwtOneHalo": f"{onehalo:.0f}", "TwtLam": f"{lam:.2f}",
    "TwtRatioParNaive": f"{ratio_g_naive[-1]:.2f}",
    "TwtBLerrTwo": f"{bL_err[i2]:.2f}", "TwtBEerrTwo": f"{bE_err[i2]:.2f}",
    "TwtBLerrLast": f"{bL_err[-1]:.2f}", "TwtBEerrLast": f"{bE_err[-1]:.2f}", "TwtBGth": f"{1 + norm.pdf(NU_G) / (sigT * norm.sf(NU_G)):.2f}",
    "TwtSigT": f"{sigT:.3f}", "TwtSigSmall": SIG_SMALL, "TwtNuG": NU_G, "TwtZ": int(ZBOX),
    "TwtBLthLast": f"{bL_th[-1]:.2f}", "TwtBLmLast": f"{bL_meas[-1]:.2f}",
    "TwtBEthLast": f"{1 + bL_th[-1]:.2f}", "TwtBEmLast": f"{bE_meas[-1]:.2f}",
    "TwtF": f"{F_GROW:.3f}", "TwtBeta": f"{beta:.3f}",
    "TwtMono": f"{mono_g:.3f}", "TwtMonoTh": f"{1 + 2 * beta / 3 + beta**2 / 5:.3f}",
    "TwtRatioPar": f"{ratio_g[-1]:.2f}", "TwtRatioParTh": f"{kaiser_g[-1]:.2f}",
    "TwtRatioMPar": f"{ratio_m[-1]:.2f}", "TwtRatioMParTh": f"{(1 + F_GROW * mu_c[-1]**2)**2:.2f}",
    "TwtNbox": rm_last.size, "TwtRatioMParMean": f"{rm_last.mean():.2f}",
    "TwtRatioMParErr": f"{rm_last.std(ddof=1) / np.sqrt(rm_last.size):.2f}",
})
print("sigma_R", sigR, "sigma_T", sigT, "bL meas", bL_meas, "th", bL_th, "bE meas", bE_meas)
print("errors", bL_err, bE_err)
print(f"galaxies {src.size}, b cross {bG:.3f}+-{bG_err:.3f}, b auto {bG_auto:.3f} (naive {bG_auto_naive:.3f}), beta {beta:.3f}, mono {mono_g:.3f} "
      f"vs {1 + 2*beta/3 + beta**2/5:.3f}; ratio_g {np.round(ratio_g, 2)}; ratio_m {np.round(ratio_m, 2)}")
