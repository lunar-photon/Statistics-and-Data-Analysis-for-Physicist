"""35_hyy_atlas2012.py -- the 2012 ATLAS diphoton result next to ours, through one formula.

Question: ATLAS expected a 2.5 sigma diphoton excess from 10.7 fb^-1 at 7 and 8 TeV in 2012 and
observed 4.5 sigma; what does the leading-order expected significance
Z^2 = sum_c S_c^2 / (2 sqrt(pi) sigma_c beta_c) give from the numbers in their Table 4, and how do
their ingredients (signal per fb^-1, background density under the peak, mass resolution, number of
categories) compare with those of the 10 fb^-1 of 13 TeV open data?
Computes: the formula for every one of the 2 x 10 ATLAS categories and for their inclusive sample;
the same formula and the full Asimov value for our spectrum; the cross section times branching
ratio of the simulated 13 TeV samples (XSection branch); the selection efficiencies; the sqrt(L)
growth of the expected significance.
Inputs: ATLAS, Phys. Lett. B 716 (2012) 1, Table 4 and Table 7 (typed in below);
data/ch13/hyy_search.npz and results of 31-33.
Writes: figures/ch13/hyy_atlas2012.pdf, results/ch13/35_hyy_atlas2012.tex
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import numpy as np
import matplotlib.pyplot as plt
import uproot
from common import setup, savefig, save_numbers, SERIES, DATA
from lib_hyy import CENT, BW

# ATLAS 2012, Table 4: per category N_D (data, 100-160 GeV) and N_S (expected signal, m_H = 126.5 GeV)
# at 7 TeV and 8 TeV, and the FWHM of the signal peak (8 TeV) in GeV.
ATLAS_T4 = np.array([
    # N_D7, N_S7, N_D8, N_S8, FWHM
    [2054, 10.5, 2945, 14.2, 3.4],    # unconverted central, low pTt
    [97, 1.5, 173, 2.5, 3.2],         # unconverted central, high pTt
    [7129, 21.6, 12136, 30.9, 3.7],   # unconverted rest, low pTt
    [444, 2.8, 785, 5.2, 3.6],        # unconverted rest, high pTt
    [1493, 6.7, 2015, 8.9, 3.9],      # converted central, low pTt
    [77, 1.0, 113, 1.6, 3.5],         # converted central, high pTt
    [8313, 21.1, 11099, 26.9, 4.5],   # converted rest, low pTt
    [501, 2.7, 706, 4.5, 3.9],        # converted rest, high pTt
    [3591, 9.5, 5140, 12.8, 6.1],     # converted transition
    [89, 2.2, 139, 3.0, 3.7],         # 2-jet
])
ATLAS_INCL = (23788, 79.6, 35251, 110.5, 3.9)
ATLAS_L = (4.8, 5.9)                  # fb^-1 at 7 and 8 TeV
ATLAS_XS = (39.0, 50.0)               # fb, sigma x BR(H -> gamma gamma) at 126.5 GeV
ATLAS_EZ, ATLAS_Z, ATLAS_MU, ATLAS_MUERR = 2.5, 4.5, 1.8, 0.5   # Table 7, 7+8 TeV
FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))
WINDOW = 60.0                         # GeV, the ATLAS fit range 100-160
LUMI = 10.064                         # fb^-1, ours
MC = ["mc_343981.ggH125_gamgam", "mc_345041.VBFH125_gamgam", "mc_345318.WpH125J_Wincl_gamgam",
      "mc_345319.ZH125J_Zincl_gamgam"]


def z2(S, fwhm, beta):
    """Leading-order Asimov q0 of a Gaussian peak on a locally flat background."""
    return S ** 2 / (2 * np.sqrt(np.pi) * (fwhm / FWHM_TO_SIGMA) * beta)


def keep_fraction(sigma, deg, lo, hi, lam, m0, bw=0.25):
    """1 - rho^2: the fraction of the expected q0 that survives when a Bernstein background of
    degree deg is fitted together with a Gaussian peak (Fisher matrix, Gaussian limit)."""
    from math import comb
    from scipy.special import erf
    e = np.arange(lo, hi + 1e-9, bw)
    c = 0.5 * (e[1:] + e[:-1])
    b = np.exp(-(c - lo) / lam)
    s = np.diff(0.5 * (1 + erf((e - m0) / (np.sqrt(2) * sigma))))
    x = (c - lo) / (hi - lo)
    A = np.stack([s] + [comb(deg, j) * x ** j * (1 - x) ** (deg - j) for j in range(deg + 1)], 1)
    F = (A / b[:, None]).T @ A
    return 1.0 / (F[0, 0] * np.linalg.inv(F)[0, 0])


def main():
    setup()
    nums = {}
    # ---- ATLAS: background density under the peak from N_D, assuming a spectrum whose density at
    # 126.5 GeV equals its mean over 100-160 GeV (an exponential falling by ~4 across the window).
    r = 1.0
    zc = 0.0
    for nd7, ns7, nd8, ns8, fw in ATLAS_T4:
        zc += z2(ns7, fw, r * nd7 / WINDOW) + z2(ns8, fw, r * nd8 / WINDOW)
    nd7, ns7, nd8, ns8, fw = ATLAS_INCL
    zi = z2(ns7, fw, r * nd7 / WINDOW) + z2(ns8, fw, r * nd8 / WINDOW)
    S_atlas = ns7 + ns8
    nums.update(HyyAtlasZcat=round(np.sqrt(zc), 2), HyyAtlasZincl=round(np.sqrt(zi), 2),
                HyyAtlasS=round(S_atlas, 1), HyyAtlasND=nd7 + nd8,
                HyyAtlasSperL=round(S_atlas / sum(ATLAS_L), 1),
                HyyAtlasBetaPerL=round((nd7 + nd8) / WINDOW / sum(ATLAS_L), 0),
                HyyAtlasSigma=round(fw / FWHM_TO_SIGMA, 2),
                HyyAtlasProduced=round(sum(l * x for l, x in zip(ATLAS_L, ATLAS_XS)), 0),
                HyyAtlasEff=round(S_atlas / sum(l * x for l, x in zip(ATLAS_L, ATLAS_XS)), 2),
                HyyAtlasZperL=round(ATLAS_EZ / np.sqrt(sum(ATLAS_L)), 2),
                HyyAtlasMuPull=round((ATLAS_MU - 1) / ATLAS_MUERR, 1))

    # ---- ours
    res = {}
    for line in (pathlib.Path(__file__).resolve().parents[2] / "results/ch13/33_hyy_search.tex").read_text().splitlines():
        if "renewcommand" in line:
            k = line.split("\\renewcommand{\\")[1].split("}")[0]
            res[k] = line.rsplit("{", 1)[1].rstrip("}")
    mod = np.load(DATA / "ch13" / "hyy_model.npz")
    s_tot = float(mod["s_tot"])
    fwhm_ours = None
    from lib_hyy import cb_pdf
    x = np.linspace(110, 140, 30001)
    f = cb_pdf(x, *mod["pcb"])
    above = x[f >= f.max() / 2]
    fwhm_ours = above[-1] - above[0]
    beta = float(res["HyyBeta"])
    S_win = float(res["HyySwin"])
    z_ours_formula = np.sqrt(z2(S_win, fwhm_ours, beta))
    xs = 0.0
    for name in MC:
        with uproot.open(DATA / "ch13" / "atlas_gamgam" / f"{name}.GamGam.root") as fh:
            xs += float(fh["mini"]["XSection"].array(library="np", entry_stop=1)[0])
    produced = xs * 1000 * LUMI                     # pb -> fb, times fb^-1
    nums.update(HyyOurZformula=round(z_ours_formula, 2), HyyOurXS=round(xs * 1000, 0),
                HyyOurProduced=round(produced, 0), HyyOurEff=round(s_tot / produced, 2),
                HyyOurSperL=round(s_tot / LUMI, 1), HyyOurBetaPerL=round(beta / LUMI, 0),
                HyyOurFWHM=round(fwhm_ours, 2),
                HyyOurZperL=round(float(res["HyyZexpOneTwoFive"]) / np.sqrt(LUMI), 2),
                HyyXSratio=round(xs * 1000 / np.average(ATLAS_XS, weights=ATLAS_L), 1),
                HyyLumiForFive=round(LUMI * (5 / float(res["HyyZexpOneTwoFive"])) ** 2, 0),
                HyyAtlasLumiForFive=round(sum(ATLAS_L) * (5 / ATLAS_EZ) ** 2, 0))
    # the price of fitting the background shape, for an ATLAS-2012-like and for our configuration
    lam_atlas = 60.0 / np.log(3.9)      # their spectrum falls by about 3.9 from 100 to 160 GeV (Fig. 4)
    k_atlas = keep_fraction(ATLAS_INCL[4] / FWHM_TO_SIGMA, 4, 100.0, 160.0, lam_atlas, 126.5)
    k_ours = keep_fraction(fwhm_ours / FWHM_TO_SIGMA, 5, 105.0, 160.0, 27.5 / 0.853, 125.0)
    nums.update(HyyAtlasLam=round(lam_atlas, 0), HyyAtlasKeep=round(k_atlas, 2), HyyOurKeepGauss=round(k_ours, 2),
                HyyAtlasZpred=round(np.sqrt(zc * k_atlas), 2),
                HyyOurZpred=round(z_ours_formula * np.sqrt(k_ours), 2))
    # the look-elsewhere correction if only 120-130 GeV had been searched: the expected number of
    # upcrossings is proportional to the length of the range (roughly stationary process)
    from scipy.stats import norm
    toys = np.load(DATA / "ch13" / "hyy_toys.npz")
    srch = np.load(DATA / "ch13" / "hyy_search.npz")
    q_obs = float(srch["q_full"].max())
    EN100 = toys["nup"][:100].mean()
    p_narrow = norm.sf(np.sqrt(q_obs)) + EN100 * (10.0 / 40.0) * np.exp(-(q_obs - 0.5) / 2)
    nums.update(HyyPgvNarrow=round(p_narrow, 3), HyyZgvNarrow=round(norm.isf(p_narrow), 2))
    print(nums)

    # ---- figure: expected significance against luminosity, with the two observations
    L = np.linspace(0.5, 40, 300)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ze_ours = float(res["HyyZexpOneTwoFive"])
    ax.plot(L, ATLAS_EZ * np.sqrt(L / sum(ATLAS_L)), color=SERIES[0], ls="--",
            label=r"ATLAS 2012, 7+8 TeV: expected $\propto\sqrt{L}$")
    ax.plot(L, ze_ours * np.sqrt(L / LUMI), color=SERIES[1], ls="--",
            label=r"open data, 13 TeV: expected $\propto\sqrt{L}$")
    ax.plot([sum(ATLAS_L)], [ATLAS_Z], "o", color=SERIES[0], label="ATLAS 2012 observed")
    ax.plot([LUMI], [float(res["HyyZoneTwoFive"])], "s", color=SERIES[1], label="open data observed (125 GeV)")
    ax.axhline(5, color="0.5", lw=0.7, ls=":")
    ax.set_xlabel(r"integrated luminosity [fb$^{-1}$]")
    ax.set_ylabel(r"local significance $Z$ [$\sigma$]")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 9)
    ax.legend(fontsize=7, loc="upper left")
    savefig(fig, "ch13", "hyy_atlas2012")
    save_numbers("ch13", "35_hyy_atlas2012", nums)


if __name__ == "__main__":
    main()
