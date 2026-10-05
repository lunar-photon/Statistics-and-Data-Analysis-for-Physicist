"""08_transfer.py -- what the beam and the pixels do to the degraded SMICA map, multipole by multipole.

Question: the nside-2048 SMICA map carries the transfer function B_l p_l(2048).  After the
harmonic degrade of 01_planck_data.py it carries B_l p_l(512).  How large are these factors over
the multipoles we analyse (30 <= l < 1020), and how large is the change the degrade applies?
Reads the beam and pixel windows saved by 01_planck_data.py (data/planck/smica_transfer_n512.npz)
and healpy's pixel window for nside 2048.
Writes: figures/ch13/transfer.pdf, results/ch13/08_transfer.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
import lib_planck as lp

setup()
z = np.load(lp.PLANCK / "smica_transfer_n512.npz")
beam, p512 = z["beam"], z["pixwin"]
L = beam.size - 1                                   # 3 * 512 - 1
ell = np.arange(L + 1)
p2048 = np.asarray(hp.pixwin(2048, lmax=L))
sb = lp.FWHM_SMICA * lp.ARCMIN / np.sqrt(8 * np.log(2))
gauss = np.exp(-0.5 * ell * (ell + 1) * sb ** 2)     # a 5' Gaussian beam, for comparison
T2 = (beam * p512) ** 2

nums = {}
for l in (500, 1019, 1535):
    tag = {500: "Five", 1019: "Ten", 1535: "Max"}[l]
    nums[f"TwelveABeam{tag}"] = f"{beam[l]:.3f}"
    nums[f"TwelveAPixFive{tag}"] = f"{p512[l]:.3f}"
    nums[f"TwelveAPixTwo{tag}"] = f"{p2048[l]:.4f}"
    nums[f"TwelveATsq{tag}"] = f"{T2[l]:.3f}"
    nums[f"TwelveADegrade{tag}"] = f"{p512[l] / p2048[l]:.3f}"

# a bandpower is not D_l at l_eff: the first and last of Planck's bins, applied to the fiducial
from camb_fiducial import load_fiducial
_, clf = load_fiducial()
Dl = np.arange(clf.size) * (np.arange(clf.size) + 1) * clf / (2 * np.pi)
for (a, b), tag in (((30, 60), "First"), ((990, 1020), "Last")):
    ls = np.arange(a, b)
    w = ls * (ls + 1.0) / np.sum(ls * (ls + 1.0))
    le = np.sum(w * ls)
    Db = le * (le + 1) * np.sum(w * clf[a:b]) / (2 * np.pi)
    nums[f"TwelveABin{tag}Leff"] = f"{le:.2f}"
    nums[f"TwelveABin{tag}Db"] = f"{Db:.0f}"
    nums[f"TwelveABin{tag}Dmean"] = f"{Dl[a:b].mean():.0f}"
    nums[f"TwelveABin{tag}Dat"] = f"{Dl[int(round(le))]:.0f}"
    nums[f"TwelveABin{tag}LL"] = f"{le * (le + 1):.0f}"
    nums[f"TwelveABin{tag}LLmean"] = f"{np.mean(ls * (ls + 1.0)):.0f}"
    nums[f"TwelveABin{tag}Factor"] = f"{le * (le + 1) / np.mean(ls * (ls + 1.0)):.4f}"

fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
ax = axs[0]
ax.plot(ell, beam, color=SERIES[0], lw=1.2, label=r"SMICA beam $B_\ell$")
ax.plot(ell, gauss, "k--", lw=0.8, label=r"$5'$ Gaussian")
ax.plot(ell, p2048, color=SERIES[2], lw=1.2, label=r"pixel window $p_\ell$, $N_{\rm side}=2048$")
ax.plot(ell, p512, color=SERIES[1], lw=1.2, label=r"pixel window $p_\ell$, $N_{\rm side}=512$")
ax.axvspan(1020, L, color="0.92", lw=0)
ax.set_xlim(0, L)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel("amplitude factor")
ax.set_title("(a) the factors that multiply $a_{\\ell m}$", fontsize=9)
ax.legend(fontsize=6.8, loc="lower left")
ax = axs[1]
ax.plot(ell, T2, color=SERIES[0], lw=1.2, label=r"$T_\ell^2=(B_\ell\,p_\ell)^2$, $N_{\rm side}=512$")
ax.plot(ell, (p512 / p2048) ** 2, color=SERIES[1], lw=1.2, label=r"degrade factor $(p_\ell^{512}/p_\ell^{2048})^2$")
ax.axvspan(1020, L, color="0.92", lw=0)
ax.set_xlim(0, L)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel("power factor")
ax.set_title("(b) what they do to the power", fontsize=9)
ax.legend(fontsize=6.8, loc="lower left")
fig.tight_layout()
savefig(fig, "ch13", "transfer")
save_numbers("ch13", "08_transfer", nums)
print(nums)
