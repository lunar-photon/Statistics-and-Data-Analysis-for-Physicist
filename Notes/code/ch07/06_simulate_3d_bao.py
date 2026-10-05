"""06_simulate_3d_bao.py -- a 3-D Gaussian matter field with the CAMB linear P(k), and its BAO bump.

Question: if the matter field were exactly Gaussian with the linear Planck-2018 P(k),
how well would one box of side 2000 Mpc/h measure P(k) and xi(r), and can we see the
baryon acoustic bump near 105 Mpc/h?
Computes: linear P(k) at z = 0 (cached by chT1, or recomputed with CAMB if absent);
NSIM = 4 boxes of 256^3 cells (cell 7.8 Mpc/h); shell-averaged P_hat(k) and
xi_hat(r) of each; the theory xi(r) = int dk/2pi^2 k^2 P(k) sin(kr)/kr and the exact
box expectation (1/V) sum_k P(k) e^{ik.r}; the scatter of r^2 xi_hat between boxes.
Writes: figures/ch07/simulate_3d_bao.pdf, results/ch07/06_simulate_3d_bao.tex,
data/ch07/pk_lin_z0.npz (cache)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES, DATA

import numpy as np
import matplotlib.pyplot as plt
from lib_fields import gaussian_field, measure_pk, measure_xi, xi_box, radial_average, xi_from_pk


def linear_pk():
    """k [h/Mpc], P [(Mpc/h)^3] of the fiducial linear matter spectrum at z = 0."""
    cache = DATA / "ch07" / "pk_lin_z0.npz"
    if cache.exists():
        z = np.load(cache); return z["k"], z["pk"]
    src = DATA / "chT1" / "pk.npz"
    if src.exists():
        z = np.load(src); k, pk = z["k"], z["pk_lin"][0]
    else:
        import camb
        from camb_fiducial import FIDUCIAL
        p = camb.set_params(**FIDUCIAL); p.set_matter_power(redshifts=[0.0], kmax=60.0)
        k, _, pk = camb.get_results(p).get_matter_power_spectrum(minkh=1e-4, maxkh=50, npoints=2000)
        pk = pk[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, k=k, pk=pk)
    return k, pk


kc, pkc = linear_pk()
P = lambda k: np.exp(np.interp(np.log(k), np.log(kc), np.log(pkc)))
N, L, NSIM = 256, 2000.0, 4
rng = rng_for("ch07", "06_simulate_3d_bao")
setup(9.0, 3.4)
fig, ax = plt.subplots(1, 3, gridspec_kw={"width_ratios": [0.8, 1, 1]})
out = {"Box": L, "Ncell": N, "Cell": L / N, "Nsim": NSIM, "KNy": np.pi * N / L, "Kf": 2 * np.pi / L}
xis, ratios = [], []
for s in range(NSIM):
    f = gaussian_field(P, N, L, d=3, rng=rng)
    if s == 0:
        ax[0].imshow(f[:64, :64, 0], cmap="RdBu_r", origin="lower", extent=[0, L / 4, 0, L / 4],
                     vmin=-3 * f.std(), vmax=3 * f.std())
        ax[0].set_title(r"slice of $\delta$ (corner)"); ax[0].set_xlabel("Mpc/$h$"); ax[0].grid(False)
        out["SigCell"] = f.std()
    k, ph, nm, pt = measure_pk(f, L, nbins=25, log=True, P=P)
    ax[1].loglog(k, ph, "o", ms=2.5, color=SERIES[s], alpha=0.8)
    ratios.append(ph / pt)
    r, xh = measure_xi(f, L, nbins=50, rmax=200)
    xis.append(xh)
    del f
theory_line(ax[1], kc[(kc > 2e-3) & (kc < 0.5)], P(kc[(kc > 2e-3) & (kc < 0.5)]), label="CAMB linear")
ax[1].set_xlabel(r"$k$ ($h$/Mpc)"); ax[1].set_ylabel(r"$\hat P(k)$ (Mpc/$h$)$^3$"); ax[1].legend()
xis = np.array(xis)
_, xb = radial_average(xi_box(P, N, L, 3), L, nbins=50, rmax=200)
rt = np.linspace(20, 200, 300)
xt = xi_from_pk(P, rt, d=3, kmax=5.0, nk=200000)
for s in range(NSIM):
    ax[2].plot(r, r**2 * xis[s], "-", color=SERIES[s], lw=1, alpha=0.8)
theory_line(ax[2], rt, rt**2 * xt, label=r"$\xi(r)$, continuum")
ax[2].plot(r, r**2 * xb, ":", color="k", lw=1.4, label="box expectation")
ax[2].set_xlim(20, 200); ax[2].set_ylim(-20, 45)
ax[2].set_xlabel(r"$r$ (Mpc/$h$)"); ax[2].set_ylabel(r"$r^2\hat\xi(r)$ (Mpc/$h$)$^2$"); ax[2].legend(fontsize=7)
sel = (r > 80) & (r < 130)
out["XiScatter"] = np.mean(np.std(r[sel]**2 * xis[:, sel], axis=0, ddof=1))
out["XiPeakTheo"] = np.max(rt[(rt > 80) & (rt < 130)]**2 * xt[(rt > 80) & (rt < 130)])
out["RPeak"] = rt[(rt > 80) & (rt < 130)][np.argmax(rt[(rt > 80) & (rt < 130)]**2 * xt[(rt > 80) & (rt < 130)])]
ratios = np.array(ratios)
out["RatioLow"] = ratios[:, 0].mean()
out["RatioAll"] = ratios.mean()
out["NmodesLow"] = nm[0]
fig.tight_layout()
savefig(fig, "ch07", "simulate_3d_bao")
save_numbers("ch07", "06_simulate_3d_bao", {f"SevA{k}": v for k, v in out.items()})
