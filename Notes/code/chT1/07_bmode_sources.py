"""Where B modes come from: the angular shape and the frequency dependence of each source.

Question: if a CMB experiment sees B-mode power, which fingerprints tell gravitational waves,
lensing, Galactic dust, synchrotron, a rotation of the polarization plane and Faraday
rotation apart?

Computes:
  * tensor BB for r = 0.01 (CAMB, unlensed tensor modes): the reionization bump at l < 10 and
    the recombination bump near l ~ 80;
  * lensing BB (the fiducial lensed BB with r = 0);
  * dust and synchrotron BB in the cleanest ~1% of the sky, with the BICEP/Keck 2018 model:
    D_l = A (l/80)^alpha, dust A = 4.4 muK^2 at 353 GHz, alpha = -0.42, modified black body
    beta_d = 1.49, T_d = 19.6 K; synchrotron A < 1.4 muK^2 at 23 GHz (95% upper limit),
    alpha = -0.6, beta_s = -3.1;  all converted to CMB temperature units;
  * the BB made by a rotation beta = 0.35 deg, sin^2(2 beta) C_EE;
  * at l = 80, the rms B amplitude sqrt(D_l) of each source against frequency, and the
    rotation angle against frequency for birefringence (constant) and Faraday rotation
    (proportional to 1/nu^2, normalised to 1 deg at 30 GHz for a 1 nG primordial field).

Writes: data/chT1/t1c_tensor.npz (cache), figures/chT1/bmode_sources.pdf,
        figures/chT1/bmode_frequency.pdf, results/chT1/07_bmode_sources.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, DATA
from camb_fiducial import FIDUCIAL, load_fiducial

LMAX = 3000
R = 0.01
CACHE = DATA / "chT1" / "t1c_tensor.npz"
T_CMB, H_OVER_K = 2.7255, 0.0479924      # K, K/GHz
NU0 = 150.0                               # GHz, where the spectra are drawn
DUST = dict(A=4.4, nu0=353.0, alpha=-0.42, beta=1.49, Td=19.6)       # BK18, D_l at l = 80
SYNC = dict(A=1.4, nu0=23.0, alpha=-0.6, beta=-3.1)                  # BK18 95% upper limit
BETA_DEG = 0.35


def tensor_bb():
    if CACHE.exists():
        z = np.load(CACHE)
        return z["ell"], z["BBt"]
    import camb
    p = camb.set_params(**FIDUCIAL, r=R, WantTensors=True, max_l_tensor=1500,
                        max_eta_k_tensor=3000, lmax=LMAX)
    res = camb.get_results(p)
    t = res.get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=True, lmax=LMAX)["tensor"]
    ell = np.arange(t.shape[0])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ell=ell, BBt=t[:, 2])
    return ell, t[:, 2]


def rj_to_cmb(nu):
    """dT_CMB / dT_RJ at frequency nu [GHz]."""
    x = H_OVER_K * nu / T_CMB
    return np.expm1(x) ** 2 / (x**2 * np.exp(x))


def dust_sed(nu):
    """Dust amplitude in CMB units relative to its value at 353 GHz (amplitude, not power)."""
    def rj(n):
        return n ** (DUST["beta"] + 1) / np.expm1(H_OVER_K * n / DUST["Td"])
    return rj(nu) * rj_to_cmb(nu) / (rj(DUST["nu0"]) * rj_to_cmb(DUST["nu0"]))


def sync_sed(nu):
    return (nu / SYNC["nu0"]) ** SYNC["beta"] * rj_to_cmb(nu) / rj_to_cmb(SYNC["nu0"])


def main():
    ell_t, bbt = tensor_bb()
    L, BB = load_fiducial("BB")
    EE = load_fiducial("EE")[1]
    L = L[: LMAX + 1]; BB = BB[: LMAX + 1]; EE = EE[: LMAX + 1]
    D = L * (L + 1) / (2 * np.pi)
    Dt = np.zeros(LMAX + 1); n = min(bbt.size, LMAX + 1)
    Dt[:n] = (ell_t * (ell_t + 1) / (2 * np.pi) * bbt)[:n]
    Dlens = D * BB
    beta = np.deg2rad(BETA_DEG)
    Drot = D * np.sin(2 * beta) ** 2 * EE
    shape_d = np.where(L >= 2, (np.maximum(L, 1) / 80.0) ** DUST["alpha"], 0)
    shape_s = np.where(L >= 2, (np.maximum(L, 1) / 80.0) ** SYNC["alpha"], 0)
    Ddust = DUST["A"] * dust_sed(NU0) ** 2 * shape_d
    Dsync = SYNC["A"] * sync_sed(NU0) ** 2 * shape_s

    # ---- figure 1: the l shapes at 150 GHz ----
    setup(6.4, 3.9)
    fig, ax = plt.subplots()
    m = L >= 2
    mt = m & (Dt > 0)
    ax.loglog(L[mt], Dt[mt], color=SERIES[0], label=r"gravitational waves, $r=%.2f$" % R)
    ax.loglog(L[m], Dlens[m], color=SERIES[1], label="lensing of $E$")
    ax.loglog(L[m], Ddust[m], color=SERIES[3], label="dust (cleanest 1% of sky)")
    ax.loglog(L[m], Dsync[m], color=SERIES[4], ls="--", label="synchrotron (upper limit)")
    ax.loglog(L[m], Drot[m], color=SERIES[2], label=r"rotation, $\beta=%.2f^\circ$" % BETA_DEG)
    ax.set_xlim(2, LMAX); ax.set_ylim(1e-6, 1)
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$D_\ell^{BB}\ [\mu\mathrm{K}^2]$ at 150 GHz")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    savefig(fig, "chT1", "bmode_sources")

    # ---- figure 2: frequency dependence at l = 80 ----
    nu = np.geomspace(20, 450, 300)
    setup(7.2, 3.2)
    fig, (a1, a2) = plt.subplots(1, 2)
    a1.loglog(nu, np.full_like(nu, np.sqrt(Dt[80])), color=SERIES[0], label=r"GW, $r=%.2f$" % R)
    a1.loglog(nu, np.full_like(nu, np.sqrt(Dlens[80])), color=SERIES[1], label="lensing")
    a1.loglog(nu, np.sqrt(DUST["A"]) * dust_sed(nu), color=SERIES[3], label="dust")
    a1.loglog(nu, np.sqrt(SYNC["A"]) * sync_sed(nu), color=SERIES[4], ls="--", label="synchrotron (limit)")
    a1.set_xlabel(r"$\nu$ [GHz]"); a1.set_ylabel(r"$\sqrt{D_{80}^{BB}}\ [\mu\mathrm{K}_{\rm CMB}]$")
    a1.set_title(r"(a) $B$-mode amplitude at $\ell=80$")
    a1.legend(fontsize=8, loc="upper center")
    a2.loglog(nu, 1.0 * (30.0 / nu) ** 2, color=SERIES[6], label=r"Faraday, $\propto\nu^{-2}$")
    a2.loglog(nu, np.full_like(nu, BETA_DEG), color=SERIES[2], label=r"birefringence $\beta$")
    a2.loglog(nu, np.full_like(nu, 0.28), color="0.35", ls=":", label=r"calibration error $\alpha$")
    a2.set_xlabel(r"$\nu$ [GHz]"); a2.set_ylabel("rotation angle [deg]")
    a2.set_title("(b) rotation angle")
    a2.legend(fontsize=8, loc="lower left")
    for ax in (a1, a2):
        ax.set_xticks([20, 30, 50, 100, 200, 400])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    savefig(fig, "chT1", "bmode_frequency")

    # ---- numbers ----
    lo = (L >= 2) & (L <= 12)
    mid = (L >= 30) & (L <= 300)
    i_reion = int(L[lo][np.argmax(Dt[lo])])
    i_rec = int(L[mid][np.argmax(Dt[mid])])
    dust_ratio = (dust_sed(353.0) / dust_sed(150.0)) ** 2
    sync_ratio = (sync_sed(30.0) / sync_sed(150.0)) ** 2
    tot_fg = np.sqrt(DUST["A"]) * dust_sed(nu) + np.sqrt(SYNC["A"]) * sync_sed(nu)
    nu_min = nu[np.argmin(tot_fg)]
    save_numbers("chT1", "07_bmode_sources", {
        "TOneCr": R,
        "TOneCReionL": i_reion,
        "TOneCRecombL": i_rec,
        "TOneCDtRec": float(f"{Dt[i_rec]:.2e}"),
        "TOneCDlensEighty": float(f"{Dlens[80]:.2e}"),
        "TOneCDdustEighty": round(float(Ddust[80]), 3),
        "TOneCDustOverLens": round(float(Ddust[80] / Dlens[80]), 1),
        "TOneCDustOverGW": round(float(Ddust[80] / Dt[80]), 1),
        "TOneCDustRatio": round(float(dust_ratio), 0),
        "TOneCSyncRatio": round(float(sync_ratio), 0),
        "TOneCFgMinNu": int(round(nu_min / 5.0) * 5),
        "TOneCFaradayOneFifty": round(1.0 * (30.0 / 150.0) ** 2, 3),
        "TOneCDrotEighty": float(f"{Drot[80]:.2e}"),
    })
    print("tensor bumps at", i_reion, i_rec, "dust/lens at 80:", Ddust[80] / Dlens[80])


if __name__ == "__main__":
    main()
