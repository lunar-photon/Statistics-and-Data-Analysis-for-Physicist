"""24_coupling_check.py -- two independent checks of the mode-coupling formula <C~_l> = sum M_ll' C_l'.

Question: is the 3j-symbol matrix M_ll' of 22_coupling.py really what the mask does to the
power spectrum?
Computes:
  (A) a direct, non-random check.  A sky whose power sits in one multipole l' is a sum of the
      2l'+1 real modes of that multipole with independent coefficients.  The mean pseudo-
      spectrum of such a sky is the sum, over the modes, of the pseudo-spectrum of each masked
      mode times its variance.  Doing those 2l'+1 transforms gives the column M_{l l'} with no
      3j symbol and no random number.  Done for l' = 20 and 100, for the 10 per cent cap and the
      band cut.
  (B) the Monte Carlo check.  The mean of the pseudo-spectra of the 2000 simulated skies of
      23_mc_masked.py against the prediction sum_l' M_ll' T_l'^2 C_l' + N fsky w2, for every
      mask, in units of the Monte Carlo error of the mean.
Writes: figures/ch08/coupling_direct.pdf, figures/ch08/mean_check.pdf,
        results/ch08/24_coupling_check.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, theory_line
from camb_fiducial import load_fiducial
import lib_cmbsim as cs
import lib_masks as lm

setup()
L = 767
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"
z = np.load(DATA / "masks.npz")
cz = np.load(DATA / "coupling.npz")
NSIDE = int(z["nside"])
ell = np.arange(L + 1)
nums = {}


# ---------------------------------------------------------------- (A) direct mask response
def direct_column(w, lp):
    """sum over the 2l'+1 real modes of multipole l' of (variance) x (pseudo-C_l of W * mode)."""
    nalm = hp.Alm.getsize(L)
    out = np.zeros(L + 1)
    for m in range(lp + 1):
        for part, var in ([(1.0, 1.0)] if m == 0 else [(1.0, 0.5), (1j, 0.5)]):
            a = np.zeros(nalm, complex)
            a[hp.Alm.getidx(L, lp, m)] = part          # one real degree of freedom of a_{l'm}
            mode = hp.alm2map(a, NSIDE, lmax=L)          # the mode as a pixel map
            out += var * hp.alm2cl(hp.map2alm(w * mode, lmax=L, iter=0))
    return out


fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.1), sharey=True)
worst = 0.0
for ax, (k, lab) in zip(axs, [("cap10", "(a) cap, $f_{\\rm sky}=0.1$"), ("band20", "(b) band cut $|b|<20^\\circ$")]):
    w = z["mask_" + k].astype(float)
    for lp, c in [(20, SERIES[0]), (100, SERIES[1])]:
        d = direct_column(w, lp)
        M = cz["M_" + k][:, lp]
        sel = M > 1e-6 * M.max()                         # compare where the column carries power
        rel = np.max(np.abs(d[sel] / M[sel] - 1))
        worst = max(worst, rel)
        print(f"{k} l'={lp}: max |direct/3j - 1| = {rel:.2e} over {sel.sum()} multipoles")
        ev = ell[(ell % 2) == (lp % 2)] if k == "band20" else ell   # band: other parity vanishes
        ev = ev[ev <= 200]
        ax.semilogy(ev, M[ev], color=c, lw=1.2, label=f"3j formula, $\\ell'={lp}$")
        ax.semilogy(ev[::3], d[ev[::3]], "o", ms=3, mfc="none", color=c, label=f"direct, $\\ell'={lp}$")
    ax.set_title(lab, fontsize=9)
    ax.set_xlabel(r"measured multipole $\ell$")
    ax.set_ylim(1e-8, 1)
axs[0].set_ylabel(r"$M_{\ell\ell'}$")
axs[1].legend(loc="lower left", fontsize=7)
savefig(fig, "ch08", "coupling_direct")
nums["EightBdirectWorst"] = lm.tex_sci(worst)

# ---------------------------------------------------------------- (B) Monte Carlo mean
sys.argv = sys.argv[:1]
import importlib
mc = importlib.import_module("23_mc_masked").load_all()
ell_f, cl = load_fiducial()
exp = cs.Experiment()
T2 = exp.transfer(L) ** 2
Nw = exp.nl(L)[0]                                       # white-noise power sigma^2 Omega_pix
NSIM = mc["sky"].shape[0]
KEYS = ["full", "cap10", "cap10apo", "cap30", "cap70", "band20", "band20apo", "holes"]
NAMES = {"full": "Full", "cap10": "CapTen", "cap10apo": "CapTenApo", "cap30": "CapThirty",
         "cap70": "CapSeventy", "band20": "Band", "band20apo": "BandApo", "holes": "Holes"}
LMAXCHK = 600                                           # well inside 3 nside - 1 (quadrature)
pull, pullb = {}, {}
edges = np.arange(2, LMAXCHK + 2, 20)
for k in KEYS:
    w = z["mask_" + k].astype(float)
    fsky, wi = lm.mask_moments(w)
    pred = cz["M_" + k] @ (T2 * cl[: L + 1]) + Nw * fsky * wi[2]
    x = mc["pcl_" + k].astype(float)
    mean, sem = x.mean(0), x.std(0, ddof=1) / np.sqrt(NSIM)
    p = (mean - pred) / sem
    pull[k] = p
    xb = cs.bin_spectrum(x - pred, edges)              # per sky, binned residual: errors from the sims
    pullb[k] = xb.mean(0) / (xb.std(0, ddof=1) / np.sqrt(NSIM))
    chi2 = np.mean(p[2:LMAXCHK + 1] ** 2)
    print(f"{k:10s} mean pull {p[2:LMAXCHK+1].mean():+.3f}  <pull^2> = {chi2:.3f}  "
          f"max |rel diff| {np.max(np.abs(mean/pred-1)[2:LMAXCHK+1]):.3f}")
    nums[f"EightBpullchi{NAMES[k]}"] = f"{chi2:.2f}"
    nums[f"EightBpullmean{NAMES[k]}"] = f"{p[2:LMAXCHK+1].mean():+.2f}"

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axs[0]
dl = ell * (ell + 1) / (2 * np.pi)
theory_line(ax, ell[2:], (dl * T2 * cl[: L + 1])[2:], label=r"$T_\ell^2 D_\ell$ (full sky)")
for (k, lab), c in zip([("cap10", "cap 0.1"), ("band20", "band"), ("holes", "holes")], SERIES):
    w = z["mask_" + k].astype(float)
    fsky, wi = lm.mask_moments(w)
    x = mc["pcl_" + k].astype(float).mean(0) - Nw * fsky * wi[2]
    ax.plot(ell[2:], (dl * x)[2:] / (fsky * wi[2]), color=c, lw=0.8, label=lab)
ax.set_xlim(2, 700)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$\ell(\ell+1)\langle\widetilde C_\ell-\widetilde N_\ell\rangle/(2\pi f_{\rm sky}w_2)$ [$\mu$K$^2$]")
ax.set_title("(a) mean pseudo-spectrum, rescaled", fontsize=9)
ax.legend(fontsize=7.5)
ax = axs[1]
lc = 0.5 * (edges[:-1] + edges[1:])
for (k, lab), c in zip([("cap10", "cap 0.1"), ("cap10apo", "cap 0.1 tapered"), ("band20", "band"),
                        ("holes", "holes")], SERIES):
    ax.plot(lc, pullb[k], "o-", ms=2.5, lw=0.8, color=c, label=lab)
ax.axhspan(-2, 2, color="0.9", zorder=0)
ax.axhline(0, color="0.4", lw=0.6)
ax.set_xlabel(r"multipole $\ell$ (bins of 20)")
ax.set_ylabel("(MC mean $-$ prediction) / MC error")
ax.set_ylim(-5, 5)
ax.set_title("(b) the 3j prediction against 2000 skies", fontsize=9)
ax.legend(fontsize=7.5, ncol=2, loc="upper right")
savefig(fig, "ch08", "mean_check")
nums["EightBLchk"] = LMAXCHK
save_numbers("ch08", "24_coupling_check", nums)
