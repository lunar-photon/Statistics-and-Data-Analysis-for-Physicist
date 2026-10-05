"""01_observed_maps.py -- one sky, before and after the instrument.

Question: what do a 30 arcmin beam, HEALPix pixels and 200 muK arcmin of white noise do to a
patch of the CMB sky, and what are the experiment's numbers (pixel noise, N_l, beam width)?
Computes: one sky from the CAMB fiducial C_l; the same sky beamed and pixelised; the observed
map with noise added; a 20 x 20 degree gnomonic patch of each; rms values.
Writes: figures/ch08/observed_maps.pdf, results/ch08/01_observed_maps.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for
from camb_fiducial import load_fiducial
import lib_cmbsim as cs

setup()
ell, cl = load_fiducial()
exp = cs.Experiment()
rng = rng_for("ch08", "01_observed_maps")

a = cs.synalm(cl[: exp.lmax_sim + 1], exp.lmax_sim, rng)
sky = hp.alm2map(a, exp.nside, lmax=exp.lmax_sim)            # the sky itself (band-limited)
obs_s = cs.observe_signal(a, exp)                             # through beam and pixel window
noise = cs.noise_map(exp, rng)
obs = obs_s + noise                                           # what the experiment records

kw = dict(rot=(30, -40), xsize=400, reso=3.0, return_projected_map=True, no_plot=True)
patches = [hp.gnomview(m, **kw) for m in (sky, obs_s, obs)]
titles = ["sky", "beam + pixel", "beam + pixel + noise"]
vmax = 3 * sky.std()
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.75))
for ax, p, t in zip(axs, patches, titles):
    im = ax.imshow(p, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[-10, 10, -10, 10])
    ax.set_title(t)
    ax.set_xticks([-10, 0, 10]); ax.set_yticks([-10, 0, 10])
    ax.grid(False)
    ax.set_xlabel("degrees")
axs[0].set_ylabel("degrees")
cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02)
cb.set_label(r"$\Delta T$ [$\mu$K]")
savefig(fig, "ch08", "observed_maps")

sb = exp.fwhm_arcmin / np.sqrt(8 * np.log(2))
save_numbers("ch08", "01_observed_maps", {
    "EightANside": exp.nside,
    "EightANpix": exp.npix,
    "EightAPixSide": f"{np.sqrt(exp.omega_pix) / cs.ARCMIN:.1f}",
    "EightAOmegaPix": f"{exp.omega_pix:.3e}".replace("e-0", r"\times10^{-") + "}",
    "EightAFwhm": f"{exp.fwhm_arcmin:.0f}",
    "EightASigmaBeam": f"{sb:.1f}",
    "EightADepth": f"{exp.depth_uK_arcmin:.0f}",
    "EightASigmaPix": f"{exp.sigma_pix:.1f}",
    "EightANell": f"{exp.nl()[0]:.2e}".replace("e-0", r"\times10^{-") + "}",
    "EightALmax": exp.lmax,
    "EightALmaxSim": exp.lmax_sim,
    "EightARmsSky": f"{sky.std():.0f}",
    "EightARmsBeamed": f"{obs_s.std():.0f}",
    "EightARmsNoise": f"{noise.std():.1f}",
    "EightARmsObs": f"{obs.std():.0f}",
})
print("sigma_pix", exp.sigma_pix, "N_l", exp.nl()[0], "rms", sky.std(), obs_s.std(), noise.std())
