"""Rotation versus lensing on simulated flat patches: both make B, only rotation makes EB and TB.

Question: a uniform rotation of the polarization plane and gravitational lensing both turn
part of E into B. Do they also leave the same EB and TB behind? And does the observed-spectrum
relation C_EB = (1/2) tan(4 beta) (C_EE - C_BB) hold sky by sky or only on average?

Computes, on a periodic 17 x 17 deg patch (256^2 pixels of 4 arcmin), NSIM Gaussian skies
with the unlensed T, E (correlated through TE) and B = 0, and an independent Gaussian lensing
potential phi with CAMB's C_phiphi:
  * rotation: Q + iU -> exp(2 i beta)(Q + iU) with beta = 2 deg (large enough that its BB is
    of the size of the lensing BB);
  * lensing to first order: X(x) -> X(x) + grad phi . grad X for X = T, Q, U;
then the binned BB, EB and TB of every sky, their Monte Carlo means and errors of the means,
and the predictions: rotation sin^2(2b) C_EE, (1/2) sin(4b) C_EE, sin(2b) C_TE; lensing the
first-order integral of 06_rotated_spectra.py for BB and zero for EB, TB.

Writes: figures/chT1/rotation_vs_lensing.pdf, results/chT1/08_flat_rotation_lensing.tex
Needs:  data/chT1/t1c_unlensed.npz (written by 06_rotated_spectra.py)
"""
import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES, DATA

N = 256
PIX = np.deg2rad(4.0 / 60.0)          # pixel side [rad]
NSIM = 500
BETA = np.deg2rad(2.0)
EDGES = np.arange(40, 2041, 80)
spec = importlib.util.spec_from_file_location("rs", pathlib.Path(__file__).with_name("06_rotated_spectra.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)


def grids():
    k = 2 * np.pi * np.fft.fftfreq(N, d=PIX)
    lx, ly = np.meshgrid(k, k, indexing="xy")
    ell = np.hypot(lx, ly)
    phi = np.arctan2(ly, lx)
    return lx, ly, ell, phi


def main():
    z = np.load(DATA / "chT1" / "t1c_unlensed.npz")
    L = z["ell"]
    lx, ly, ell, ang = grids()
    c2, s2 = np.cos(2 * ang), np.sin(2 * ang)

    def on_grid(cl):
        out = np.interp(ell, L, cl, left=0.0, right=0.0)
        out[0, 0] = 0.0
        return out
    ctt, cee, cte, cpp = on_grid(z["TTu"]), on_grid(z["EEu"]), on_grid(z["TEu"]), on_grid(z["cpp"])
    a_t = np.sqrt(ctt) / PIX
    a_te = np.where(ctt > 0, cte / np.sqrt(np.where(ctt > 0, ctt, 1)), 0) / PIX
    a_e = np.sqrt(np.maximum(cee - np.where(ctt > 0, cte**2 / np.where(ctt > 0, ctt, 1), 0), 0)) / PIX
    a_p = np.sqrt(cpp) / PIX

    which = np.digitize(ell, EDGES) - 1
    nb = EDGES.size - 1
    inbin = [(which == b) for b in range(nb)]
    lcen = np.array([ell[m].mean() for m in inbin])
    norm = PIX**2 / N**2

    def binned(Xk, Yk):
        p = np.real(Xk * np.conj(Yk)) * norm
        return np.array([p[m].mean() for m in inbin])

    def eb(Qk, Uk):
        return c2 * Qk + s2 * Uk, -s2 * Qk + c2 * Uk

    def grad(fk):
        return np.real(np.fft.ifft2(1j * lx * fk)), np.real(np.fft.ifft2(1j * ly * fk))

    rng = rng_for("chT1", "08_flat_rotation_lensing")
    keys = ["rBB", "rEB", "rTB", "lBB", "lEB", "lTB"]
    S = {k: np.zeros(nb) for k in keys}
    S2 = {k: np.zeros(nb) for k in keys}
    ident = 0.0
    cb, sb = np.cos(2 * BETA), np.sin(2 * BETA)
    for _ in range(NSIM):
        w = rng.standard_normal((3, N, N))
        wt, we, wp = (np.fft.fft2(x) for x in w)
        Tk = a_t * wt
        Ek = a_te * wt + a_e * we
        Pk = a_p * wp
        Qk, Uk = c2 * Ek, s2 * Ek                     # B = 0 before anything happens
        # rotation of every stick by BETA
        Qr, Ur = cb * Qk - sb * Uk, sb * Qk + cb * Uk
        Er, Br = eb(Qr, Ur)
        ree, rbb, reb = binned(Er, Er), binned(Br, Br), binned(Er, Br)
        vals = {"rBB": rbb, "rEB": reb, "rTB": binned(Tk, Br)}
        ident = max(ident, np.max(np.abs(reb - 0.5 * np.tan(4 * BETA) * (ree - rbb))) / np.max(np.abs(reb)))
        # first-order lensing
        px, py = grad(Pk)
        maps = []
        for fk in (Tk, Qk, Uk):
            f = np.real(np.fft.ifft2(fk))
            fx, fy = grad(fk)
            maps.append(np.fft.fft2(f + px * fx + py * fy))
        Tl, Ql, Ul = maps
        El, Bl = eb(Ql, Ul)
        vals.update({"lBB": binned(Bl, Bl), "lEB": binned(El, Bl), "lTB": binned(Tl, Bl)})
        for k in keys:
            S[k] += vals[k]; S2[k] += vals[k] ** 2
    mean = {k: S[k] / NSIM for k in keys}
    sem = {k: np.sqrt((S2[k] / NSIM - mean[k] ** 2) / (NSIM - 1)) for k in keys}

    # predictions averaged over the modes of each bin
    def bin_th(c):
        return np.array([c[m].mean() for m in inbin])
    th_rbb = np.sin(2 * BETA) ** 2 * bin_th(cee)
    th_reb = 0.5 * np.sin(4 * BETA) * bin_th(cee)
    th_rtb = np.sin(2 * BETA) * bin_th(cte)
    th_lbb = rs.lensing_bb_first_order(lcen, z["cpp"], z["EEu"])

    # ---- figure ----
    D = lcen * (lcen + 1) / (2 * np.pi)
    setup(7.4, 3.2)
    fig, axs = plt.subplots(1, 3)
    for ax, (kr, kl, th_r, th_l, name) in zip(axs, [
            ("rBB", "lBB", th_rbb, th_lbb, "BB"), ("rEB", "lEB", th_reb, 0 * lcen, "EB"),
            ("rTB", "lTB", th_rtb, 0 * lcen, "TB")]):
        ax.errorbar(lcen, D * mean[kr], D * sem[kr], fmt="o", ms=3, color=SERIES[2], label=r"rotation $2^\circ$")
        ax.errorbar(lcen * 1.01, D * mean[kl], D * sem[kl], fmt="s", ms=3, color=SERIES[0], label="lensing")
        theory_line(ax, lcen, D * th_r, label="prediction")
        ax.plot(lcen, D * th_l, color="k", ls="--", lw=1.4)
        ax.set_xlabel(r"$\ell$"); ax.set_title(r"$D_\ell^{%s}\ [\mu\mathrm{K}^2]$" % name, fontsize=9)
        ax.axhline(0, color="0.6", lw=0.6)
    axs[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    savefig(fig, "chT1", "rotation_vs_lensing")

    # ---- numbers ----
    chi_eb = float(np.sum((mean["lEB"] / sem["lEB"]) ** 2))
    chi_tb = float(np.sum((mean["lTB"] / sem["lTB"]) ** 2))
    good = (lcen > 100) & (lcen < 1500)
    w = 1 / (sem["lBB"][good] / th_lbb[good]) ** 2
    lbb_ratio = float(np.sum(w * mean["lBB"][good] / th_lbb[good]) / np.sum(w))
    lbb_err = float(1 / np.sqrt(np.sum(w)))
    w2 = 1 / (sem["rEB"] / th_reb) ** 2
    reb_ratio = float(np.sum(w2 * mean["rEB"] / th_reb) / np.sum(w2))
    reb_err = float(1 / np.sqrt(np.sum(w2)))
    eb_frac = float(np.max(np.abs(D * mean["lEB"])) / np.max(np.abs(D * th_reb)))
    bb_comp = float(np.median(mean["rBB"][good] / mean["lBB"][good]))
    save_numbers("chT1", "08_flat_rotation_lensing", {
        "TOneCNsim": NSIM,
        "TOneCNbin": nb,
        "TOneCSimBetaDeg": 2,
        "TOneCChiEB": round(chi_eb, 1),
        "TOneCChiTB": round(chi_tb, 1),
        "TOneCLensBBRatio": f"{lbb_ratio:.3f}",
        "TOneCLensBBErr": f"{lbb_err:.3f}",
        "TOneCRotEBRatio": f"{reb_ratio:.4f}",
        "TOneCRotEBErr": f"{reb_err:.4f}",
        "TOneCEBFracPct": round(100 * eb_frac, 2),
        "TOneCBBRotOverLens": f"{bb_comp:.2f}",
        "TOneCIdentity": float(f"{ident:.1e}"),
    })
    print(f"chi2 EB {chi_eb:.1f} TB {chi_tb:.1f} (nbin {nb}); lens BB ratio {lbb_ratio:.3f}+-{lbb_err:.3f}; "
          f"rot EB ratio {reb_ratio:.3f}+-{reb_err:.3f}; EB frac {eb_frac:.4f}; identity {ident:.1e}")


if __name__ == "__main__":
    main()
