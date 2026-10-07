"""What a uniform rotation of the polarization plane does to the CMB spectra, next to lensing.

Question: if every polarization stick on the sky is turned by the same small angle beta, how
much power moves from EE into BB, how large are the new EB and TB spectra, and how does the
B mode made this way compare with the B mode that gravitational lensing makes from E?

Computes:
  * the rotated spectra for beta = 0.35 deg from the fiducial (lensed, r = 0) spectra:
      EE' = cos^2(2b) EE + sin^2(2b) BB,   BB' = sin^2(2b) EE + cos^2(2b) BB,
      EB' = sin(4b)/2 (EE - BB),           TE' = cos(2b) TE,   TB' = sin(2b) TE;
  * the lensing BB from CAMB (lensed BB with r = 0) and, as a check of the derivation in the
    text, the first-order flat-sky lensing integral
      C_BB(l) = int d^2l'/(2pi)^2 [l'.(l-l')]^2 sin^2(2(phi_l' - phi_l)) C_phiphi(|l-l'|) C_EE(l'),
    and its l -> 0 limit (1/4pi) int dl' l'^5 C_phiphi C_EE, from CAMB's unlensed EE and C_phiphi.

Writes: data/chT1/t1c_unlensed.npz (cache: unlensed TT, EE, TE, C_phiphi, lensed spectra),
        figures/chT1/rotated_spectra.pdf, figures/chT1/lensing_bb_check.pdf,
        results/chT1/06_rotated_spectra.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, theory_line, SERIES, DATA
from camb_fiducial import FIDUCIAL, load_fiducial

LMAX = 3000
CACHE = DATA / "chT1" / "t1c_unlensed.npz"
BETA_DEG = 0.35                      # the Minami-Komatsu (2020) central value
ARCMIN = np.pi / (180 * 60)


def camb_spectra():
    """Unlensed scalar TT, EE, TE, the lensing potential C_phiphi, and lensed spectra (r = 0)."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    import camb
    p = camb.set_params(**FIDUCIAL, lmax=LMAX + 500, lens_potential_accuracy=1)
    r = camb.get_results(p)
    cls = r.get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=True, lmax=LMAX)
    unl = cls["unlensed_scalar"]
    lens = cls["total"]
    pp = r.get_lens_potential_cls(lmax=LMAX, raw_cl=False)[:, 0]   # [L(L+1)]^2 C_L^phiphi / 2pi
    L = np.arange(LMAX + 1)
    cpp = np.zeros(LMAX + 1)
    cpp[2:] = pp[2:] * 2 * np.pi / (L[2:] * (L[2:] + 1.0)) ** 2
    out = dict(ell=L, TTu=unl[:, 0], EEu=unl[:, 1], TEu=unl[:, 3], cpp=cpp,
               TT=lens[:, 0], EE=lens[:, 1], BB=lens[:, 2], TE=lens[:, 3])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def rotate_spectra(beta, TT, EE, BB, TE):
    c2, s2, s4 = np.cos(2 * beta), np.sin(2 * beta), np.sin(4 * beta)
    return dict(EE=c2**2 * EE + s2**2 * BB, BB=s2**2 * EE + c2**2 * BB,
                EB=0.5 * s4 * (EE - BB), TE=c2 * TE, TB=s2 * TE)


def lensing_bb_first_order(ells, cpp, cee, lpmax=3000, nth=720):
    """First-order flat-sky lensing B mode from E (B = 0 before lensing)."""
    lp = np.arange(2, lpmax + 1, dtype=float)
    th = (np.arange(nth) + 0.5) * 2 * np.pi / nth
    dth = 2 * np.pi / nth
    ce = np.interp(lp, np.arange(cee.size), cee, right=0.0)
    out = []
    for l in ells:
        # l along the first axis, l' at angle th
        dot = l * lp[:, None] * np.cos(th)[None, :] - lp[:, None] ** 2     # l'.(l - l')
        mag = np.sqrt(np.maximum(l**2 + lp[:, None] ** 2 - 2 * l * lp[:, None] * np.cos(th)[None, :], 0))
        cphi = np.interp(mag, np.arange(cpp.size), cpp, left=0.0, right=0.0)
        integrand = dot**2 * np.sin(2 * th)[None, :] ** 2 * cphi * ce[:, None] * lp[:, None]
        out.append(integrand.sum() * dth * 1.0 / (2 * np.pi) ** 2)     # dl' = 1
    return np.array(out)


def main():
    s = camb_spectra()
    L = s["ell"]
    beta = np.deg2rad(BETA_DEG)
    TT, EE, BB, TE = (load_fiducial(k)[1][: L.size] for k in ("TT", "EE", "BB", "TE"))
    rot = rotate_spectra(beta, TT, EE, BB, TE)
    D = L * (L + 1) / (2 * np.pi)
    bb_rot = rot["BB"] - np.cos(2 * beta) ** 2 * BB          # the part of BB' moved over from EE

    # ---- figure 1: the rotated spectra next to lensing ----
    setup(7.2, 3.3)
    fig, (a1, a2) = plt.subplots(1, 2)
    m = L >= 2
    a1.loglog(L[m], D[m] * BB[m], color=SERIES[0], label=r"lensing $BB$ ($\beta=0$)")
    a1.loglog(L[m], D[m] * bb_rot[m], color=SERIES[1], label=r"$\sin^2 2\beta\,C^{EE}_\ell$")
    a1.loglog(L[m], D[m] * rot["BB"][m], color="0.2", lw=0.9, ls=":", label=r"total $BB'$")
    a1.set_xlabel(r"$\ell$"); a1.set_ylabel(r"$D_\ell^{BB}\ [\mu\mathrm{K}^2]$")
    a1.set_xlim(2, LMAX); a1.set_ylim(1e-6, 1)
    a1.legend(loc="lower right", fontsize=8)
    a1.set_title(r"(a) $BB$, $\beta=%.2f^\circ$" % BETA_DEG)
    a2.plot(L[m], D[m] * rot["EB"][m], color=SERIES[2], label=r"$EB'=\frac{1}{2}\sin4\beta\,(C^{EE}-C^{BB})$")
    a2.plot(L[m], D[m] * rot["TB"][m], color=SERIES[3], label=r"$TB'=\sin2\beta\,C^{TE}$")
    a2.axhline(0, color=SERIES[0], lw=1.2, label=r"lensing: $EB=TB=0$")
    a2.set_xscale("log"); a2.set_xlim(2, LMAX)
    a2.set_xlabel(r"$\ell$"); a2.set_ylabel(r"$D_\ell\ [\mu\mathrm{K}^2]$")
    a2.legend(loc="upper left", fontsize=8)
    a2.set_title(r"(b) the parity-odd spectra")
    fig.tight_layout()
    savefig(fig, "chT1", "rotated_spectra")

    # ---- figure 2: first-order lensing BB versus CAMB ----
    ells = np.unique(np.round(np.geomspace(10, 2500, 28)).astype(int))
    bb1 = lensing_bb_first_order(ells, s["cpp"], s["EEu"])
    lp = np.arange(2, LMAX + 1)
    white = np.sum(lp**5 * s["cpp"][lp] * np.interp(lp, L, s["EEu"])) / (4 * np.pi)
    setup(5.4, 3.3)
    fig, ax = plt.subplots()
    ax.loglog(L[m], D[m] * BB[m], color=SERIES[0], label="CAMB lensed $BB$")
    ax.loglog(ells, ells * (ells + 1) / (2 * np.pi) * bb1, "o", color=SERIES[1], ms=4,
              label="first-order integral")
    theory_line(ax, L[m], D[m] * white, label=r"white limit $\ell\to0$")
    ax.set_xlim(2, LMAX); ax.set_ylim(1e-6, 0.3)
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$D_\ell^{BB}\ [\mu\mathrm{K}^2]$")
    ax.legend(loc="lower right", fontsize=8)
    savefig(fig, "chT1", "lensing_bb_check")

    # ---- numbers ----
    def at(arr, l):
        return float(arr[l])
    i_bbpk = 2 + np.argmax((D * BB)[2:])
    i_eb = 2 + np.argmax(np.abs(D * rot["EB"])[2:])
    i_tb = 2 + np.argmax(np.abs(D * rot["TB"])[2:])
    i_ee = 30 + np.argmax((D * EE)[30:])
    sel = (ells >= 20) & (ells <= 1500)
    ratio1 = bb1[sel] / BB[ells[sel]]
    low = (L >= 10) & (L <= 50)
    save_numbers("chT1", "06_rotated_spectra", {
        "TOneCBetaDeg": BETA_DEG,
        "TOneCTwoBetaPct": round(100 * np.sin(2 * beta), 2),
        "TOneCSinSqTwoBeta": float(f"{np.sin(2 * beta) ** 2:.2e}"),
        "TOneCEEpeakL": int(i_ee),
        "TOneCDEEpeak": round(float((D * EE)[i_ee]), 1),
        "TOneCBBpeakL": int(i_bbpk),
        "TOneCDBBpeak": round(float((D * BB)[i_bbpk]), 3),
        "TOneCRotLensHundred": round(at(bb_rot, 100) / at(BB, 100), 3),
        "TOneCRotLensThousand": round(at(bb_rot, 1000) / at(BB, 1000), 3),
        "TOneCRotLensMax": round(float(np.max(bb_rot[2:2001] / BB[2:2001])), 3),
        "TOneCDEBmax": round(float(np.abs(D * rot["EB"])[i_eb]), 3),
        "TOneCEBmaxL": int(i_eb),
        "TOneCDTBmax": round(float(np.abs(D * rot["TB"])[i_tb]), 3),
        "TOneCTBmaxL": int(i_tb),
        "TOneCDBBrotPeak": float(f"{float(np.max((D * bb_rot)[2:])):.2e}"),
        "TOneCLensWhiteC": float(f"{white:.2e}"),
        "TOneCLensWhiteUKarcmin": round(float(np.sqrt(white) / ARCMIN), 1),
        "TOneCLensLowMean": float(f"{float(np.mean(BB[low])):.2e}"),
        "TOneCFirstOrderMin": round(float(ratio1.min()), 3),
        "TOneCFirstOrderMax": round(float(ratio1.max()), 3),
    })
    print("first-order/CAMB ratios:", np.round(bb1 / BB[ells], 3))
    print("white limit", white, "mean low-l CAMB BB", np.mean(BB[low]))


if __name__ == "__main__":
    main()
