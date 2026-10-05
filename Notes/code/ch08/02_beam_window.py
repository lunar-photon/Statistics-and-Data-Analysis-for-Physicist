"""02_beam_window.py -- the beam and pixel transfer functions, derived and checked.

Question: is the small-beam formula B_l = exp(-l(l+1) sigma_b^2/2) the exact Legendre transform
of a Gaussian beam on the sphere, and how does it compare with the HEALPix pixel window?
Computes: the exact B_l (numerical Legendre integral) for FWHM = 30 arcmin, 5 deg and 10 deg;
the small-beam formula; healpy.gauss_beam; the nside = 256 pixel window p_l; the multipole
where B_l^2 = 1/2.
Writes: figures/ch08/beam_window.pdf, results/ch08/02_beam_window.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, theory_line
import lib_cmbsim as cs

setup()
exp = cs.Experiment()
L = exp.lmax_sim
ell = np.arange(L + 1)
sb = exp.fwhm_arcmin * cs.ARCMIN / np.sqrt(8 * np.log(2))

b_exact = cs.gaussian_beam_exact(exp.fwhm_arcmin, L)
b_small = cs.gaussian_beam(exp.fwhm_arcmin, L)
b_hp = hp.gauss_beam(exp.fwhm_arcmin * cs.ARCMIN, lmax=L)
pl = cs.pixel_window(exp.nside, L)

# where does the small-beam formula fail?  big beams, compared where B_l > 1e-3
dev = {}
for fw_deg in (5.0, 10.0):
    be = cs.gaussian_beam_exact(fw_deg * 60, 200)
    bs = cs.gaussian_beam(fw_deg * 60, 200)
    ok = be > 1e-3
    dev[fw_deg] = np.max(np.abs(bs[ok] / be[ok] - 1))

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
th = np.linspace(0, 60, 400)                                    # arcmin
prof = np.exp(-0.5 * (th * cs.ARCMIN / sb) ** 2)
axs[0].plot(th, prof, color=SERIES[0], label="Gaussian beam")
axs[0].axvline(exp.fwhm_arcmin / 2, color="0.5", ls=":", lw=1)
axs[0].text(exp.fwhm_arcmin / 2 + 3, 0.6, "half width at\nhalf maximum", fontsize=8, color="0.3")
side = np.sqrt(exp.omega_pix) / cs.ARCMIN
axs[0].axvspan(0, side / 2, color=SERIES[1], alpha=0.15, label="half a pixel")
axs[0].set_xlabel(r"angle from the pointing $\theta$ [arcmin]")
axs[0].set_ylabel(r"$b(\theta)/b(0)$")
axs[0].legend(loc="upper right")

axs[1].plot(ell, b_exact, color=SERIES[0], label=r"beam $B_\ell$ (exact)")
theory_line(axs[1], ell, b_small, label=r"$e^{-\ell(\ell+1)\sigma_b^2/2}$")
axs[1].plot(ell, pl, color=SERIES[1], label=r"pixel $p_\ell$")
axs[1].plot(ell, b_exact * pl, color=SERIES[2], label=r"$B_\ell p_\ell$")
axs[1].axvline(exp.lmax, color="0.5", ls=":", lw=1)
axs[1].set_xlabel(r"multipole $\ell$")
axs[1].set_ylabel("transfer")
axs[1].set_ylim(0, 1.05)
axs[1].legend(loc="lower left", fontsize=8)
fig.tight_layout()
savefig(fig, "ch08", "beam_window")

l_half = int(np.argmin(np.abs(b_exact ** 2 - 0.5)))
save_numbers("ch08", "02_beam_window", {
    "EightABeamMaxDev": f"{np.max(np.abs(b_small - b_exact)):.0e}".replace("e-0", r"\times10^{-") + "}",
    "EightABeamHpDev": f"{np.max(np.abs(b_small - b_hp)):.0e}" if np.max(np.abs(b_small - b_hp)) > 0 else "0",
    "EightABeamDevFive": f"{100 * dev[5.0]:.1f}",
    "EightABeamDevTen": f"{100 * dev[10.0]:.1f}",
    "EightALHalf": l_half,
    "EightALHalfFormula": f"{np.sqrt(np.log(2)) / sb:.0f}",
    "EightABlmax": f"{b_exact[exp.lmax]:.3f}",
    "EightAPlmax": f"{pl[exp.lmax]:.3f}",
    "EightAPlTwoFiveSix": f"{pl[256]:.3f}",
    "EightATlmaxSq": f"{(b_exact[exp.lmax] * pl[exp.lmax]) ** 2:.4f}",
})
print("max dev small vs exact", np.max(np.abs(b_small - b_exact)), "5deg", dev[5.0], "10deg", dev[10.0],
      "l_half", l_half, "B(512)", b_exact[512], "p(512)", pl[512])
