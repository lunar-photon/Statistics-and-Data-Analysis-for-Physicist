"""09_peculiar_velocities.py -- how fast do galaxies move, according to linear theory and P(k)?

Question: what rms peculiar velocity does the CAMB linear P(k) predict today, which scales
produce it, and how are the velocities of two galaxies a distance r apart correlated?

Computes
  * sigma_v (3-D and 1-D) = H0 f sqrt( int P(k) dk / 2 pi^2 ) [1-D: divide by sqrt 3];
  * the cumulative share of sigma_v^2 and of sigma_8^2 from wavenumbers below k;
  * the velocity correlation functions psi_par(r), psi_perp(r) (Gorski 1988);
  * a check on Gaussian random boxes: velocities from v_k = i H0 f k delta_k / k^2,
    one-point variance and psi_par, psi_perp measured on the grid against the grid-mode sums.
Writes figures/ch11/pecvel_theory.pdf, figures/ch11/pecvel_box.pdf, results/ch11/09_peculiar_velocities.tex.
Laptop scale: 4 boxes of 1000 Mpc/h with 256^3 cells (about 1 GB peak, ~1 minute).
"""
import pathlib
import sys

import numpy as np
from scipy.special import spherical_jn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import SERIES, rng_for, save_numbers, savefig, setup, theory_line  # noqa: E402
from lib_lss import growth, linear_pk_z0  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CH, NAME = "ch11", "09_peculiar_velocities"
H100 = 100.0  # H0 in km/s per Mpc/h


def sigma_v_theory(k, P, f, kmin=0.0, kmax=np.inf):
    """3-D rms velocity [km/s]: (H0 f)^2 int_kmin^kmax P dk / (2 pi^2), integrated in ln k."""
    m = (k >= kmin) & (k <= kmax)
    return H100 * f * np.sqrt(np.trapezoid(k[m] * P[m], np.log(k[m])) / (2 * np.pi ** 2))


def psi_theory(r, k, P, f):
    """psi_perp(r) = (H0 f)^2/(2 pi^2) int P j1(kr)/(kr) dk;  psi_par = ... int P [j0 - 2 j1/(kr)] dk."""
    lk = np.log(k)
    out_par, out_perp = [], []
    for rr in np.atleast_1d(r):
        x = k * rr
        j0 = spherical_jn(0, x)
        j1x = np.where(x > 1e-6, spherical_jn(1, x) / np.where(x > 1e-6, x, 1), 1.0 / 3.0)
        pref = (H100 * f) ** 2 / (2 * np.pi ** 2)
        out_perp.append(pref * np.trapezoid(k * P * j1x, lk))
        out_par.append(pref * np.trapezoid(k * P * (j0 - 2 * j1x), lk))
    return np.array(out_par), np.array(out_perp)


def main():
    setup()
    k, P, s8, _ = linear_pk_z0()
    f = float(growth(0.0)[1][0])
    sv3 = sigma_v_theory(k, P, f)
    sv1 = sv3 / np.sqrt(3)
    # cumulative share of sigma_v^2 and sigma_8^2 below k
    lk = np.log(k)
    cum_v = np.concatenate([[0], np.cumsum(0.5 * (k[1:] * P[1:] + k[:-1] * P[:-1]) * np.diff(lk))])
    cum_v /= cum_v[-1]
    x8 = 8 * k
    W8 = 3 * spherical_jn(1, x8) / x8
    g8 = k ** 3 * P * W8 ** 2
    cum_d = np.concatenate([[0], np.cumsum(0.5 * (g8[1:] + g8[:-1]) * np.diff(lk))])
    cum_d /= cum_d[-1]
    k_half_v = float(np.interp(0.5, cum_v, k))
    k_half_d = float(np.interp(0.5, cum_d, k))
    share_v_gt100 = float(np.interp(2 * np.pi / 100, k, cum_v))   # modes with wavelength > 100 Mpc/h

    # ---------------- Gaussian boxes
    L, N, nbox = 1000.0, 256, 4
    D = L / N
    kf = 2 * np.pi / L
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=D)
    kz = 2 * np.pi * np.fft.rfftfreq(N, d=D)
    KX, KY, KZ = np.meshgrid(k1, k1, kz, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    Kmag = np.sqrt(K2)
    Pg = np.where(Kmag > 0, np.interp(np.where(Kmag > 0, Kmag, 1.0), k, P), 0.0)
    V = L ** 3
    wts = np.full(Kmag.shape, 2.0)
    wts[..., 0] = 1.0
    wts[..., -1] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_k2 = np.where(K2 > 0, 1.0 / K2, 0.0)
    # exact expectation on the grid: <v_x^2> = (1/V) sum_k (H f)^2 P k_x^2/k^4
    sv1_grid = H100 * f * np.sqrt(np.sum(wts * Pg * KX ** 2 * inv_k2 ** 2) / V)
    sv1_cont_trunc = sigma_v_theory(k, P, f, kmin=kf, kmax=np.pi / D) / np.sqrt(3)
    # grid psi_par(r) (separation along x, component x) and psi_perp(r) (component y)
    nsep = np.arange(0, 41)
    rsep = nsep * D
    def grid_psi(comp):
        spec = (H100 * f) ** 2 * Pg * comp ** 2 * inv_k2 ** 2 / D ** 3
        corr = np.fft.irfftn(spec, s=(N, N, N), axes=(0, 1, 2))
        return corr[nsep, 0, 0]
    psi_par_grid, psi_perp_grid = grid_psi(KX), grid_psi(KY)

    rng = rng_for(CH, NAME)
    var_meas, par_meas, perp_meas, hist_v = [], [], [], []
    amp = np.sqrt(Pg / D ** 3)
    for b in range(nbox):
        w = rng.standard_normal((N, N, N))
        dk = np.fft.rfftn(w) * amp                 # delta_k / D^3 in the lib_lss convention
        vx = np.fft.irfftn(1j * H100 * f * KX * inv_k2 * dk, s=(N, N, N), axes=(0, 1, 2))
        vy = np.fft.irfftn(1j * H100 * f * KY * inv_k2 * dk, s=(N, N, N), axes=(0, 1, 2))
        var_meas.append(0.5 * (vx.var() + vy.var()))
        par_meas.append([np.mean(vx * np.roll(vx, -n, axis=0)) for n in nsep])
        perp_meas.append([np.mean(vy * np.roll(vy, -n, axis=0)) for n in nsep])
        hist_v.append(vx[::4, ::4, ::4].ravel().copy())
        del w, dk, vx, vy
    par_meas, perp_meas = np.array(par_meas), np.array(perp_meas)
    sv1_box = np.sqrt(np.mean(var_meas))
    sv1_box_scatter = np.std(np.sqrt(var_meas))
    hv = np.concatenate(hist_v)

    rth = np.linspace(1, 250, 250)
    ppar, pperp = psi_theory(rth, k, P, f)

    # ---------------- figure 1: theory
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax[0].semilogx(k, cum_v, color=SERIES[0], label=r"velocity variance $\sigma_v^2$")
    ax[0].semilogx(k, cum_d, color=SERIES[1], label=r"density variance $\sigma_8^2$")
    ax[0].axhline(0.5, color="0.6", lw=0.7, ls=":")
    ax[0].set_xlim(1e-3, 3)
    ax[0].set_xlabel(r"$k$ [$h$/Mpc]")
    ax[0].set_ylabel(r"share from wavenumbers below $k$")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].plot(rth, np.sqrt(np.clip(ppar, 0, None)), color=SERIES[0],
               label=r"$\psi_\parallel^{1/2}$ (components along the separation)")
    ax[1].plot(rth, np.sqrt(np.clip(pperp, 0, None)), color=SERIES[2],
               label=r"$\psi_\perp^{1/2}$ (components across it)")
    ax[1].axhline(sv1, color="0.5", lw=0.8, ls="--", label=r"$\sigma_{v,1\mathrm{D}}$ at $r=0$")
    ax[1].set_xlabel(r"separation $r$ [Mpc/$h$]")
    ax[1].set_ylabel("km/s")
    ax[1].set_ylim(0, 1.5 * sv1)
    ax[1].legend(fontsize=7.5, loc="upper right")
    savefig(fig, CH, "pecvel_theory")

    # ---------------- figure 2: box check
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    bins = np.linspace(-1200, 1200, 81)
    ax[0].hist(hv, bins=bins, density=True, histtype="step", color=SERIES[0], lw=1.4,
               label=r"$v_x$ in the boxes")
    xx = np.linspace(-1200, 1200, 400)
    theory_line(ax[0], xx, np.exp(-0.5 * (xx / sv1_grid) ** 2) / (np.sqrt(2 * np.pi) * sv1_grid),
                label=rf"Gaussian, $\sigma$ = {sv1_grid:.0f} km/s")
    ax[0].set_xlabel(r"$v_x$ [km/s]")
    ax[0].set_ylabel("density")
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.3)
    ax[0].legend(fontsize=8, loc="upper right")
    m_par, e_par = par_meas.mean(0), par_meas.std(0) / np.sqrt(nbox)
    m_perp, e_perp = perp_meas.mean(0), perp_meas.std(0) / np.sqrt(nbox)
    ax[1].errorbar(rsep, m_par / 1e4, e_par / 1e4, fmt="o", ms=3, color=SERIES[0], label=r"$\psi_\parallel$ boxes")
    ax[1].errorbar(rsep, m_perp / 1e4, e_perp / 1e4, fmt="s", ms=3, color=SERIES[2], label=r"$\psi_\perp$ boxes")
    theory_line(ax[1], rsep, psi_par_grid / 1e4, label="grid-mode sums")
    ax[1].plot(rsep, psi_perp_grid / 1e4, color="k", ls="--", lw=1.4)
    ax[1].set_xlabel(r"separation $r$ [Mpc/$h$]")
    ax[1].set_ylabel(r"$\psi(r)$ [$10^4$ km$^2$/s$^2$]")
    ax[1].legend(fontsize=8)
    savefig(fig, CH, "pecvel_box")

    i100 = int(np.argmin(np.abs(rth - 100)))
    save_numbers(CH, NAME, {
        "FBPvF": f"{f:.3f}", "FBPvSigEight": f"{s8:.3f}",
        "FBPvSigThree": f"{sv3:.0f}", "FBPvSigOne": f"{sv1:.0f}",
        "FBPvKhalfV": f"{k_half_v:.3f}", "FBPvKhalfD": f"{k_half_d:.2f}",
        "FBPvLhalfV": f"{2*np.pi/k_half_v:.0f}", "FBPvLhalfD": f"{2*np.pi/k_half_d:.0f}",
        "FBPvShareHundred": f"{100*share_v_gt100:.0f}",
        "FBPvBoxL": f"{L:.0f}", "FBPvBoxN": N, "FBPvNbox": nbox,
        "FBPvSigOneGrid": f"{sv1_grid:.0f}", "FBPvSigOneTrunc": f"{sv1_cont_trunc:.0f}",
        "FBPvSigOneBox": f"{sv1_box:.0f}", "FBPvSigOneBoxScatter": f"{sv1_box_scatter:.0f}",
        "FBPvParHundred": f"{np.sqrt(max(ppar[i100],0)):.0f}", "FBPvPerpHundred": f"{np.sqrt(max(pperp[i100],0)):.0f}",
    })
    print(f"f={f:.3f} sv3={sv3:.0f} sv1={sv1:.0f} khalf_v={k_half_v:.3f} khalf_d={k_half_d:.2f} "
          f"grid sv1={sv1_grid:.0f} trunc={sv1_cont_trunc:.0f} box={sv1_box:.0f}+-{sv1_box_scatter:.0f}; "
          f"psi(100): par {np.sqrt(max(ppar[i100],0)):.0f} perp {np.sqrt(max(pperp[i100],0)):.0f}")


if __name__ == "__main__":
    main()
