"""13_research_scale.py -- what the full research version of the chapter-8 Monte Carlo would cost.

Question: the simulations of this chapter run at nside 256 (and 512), l <= 512 (and 1000), with
2000 (and 300) Gaussian skies.  A Planck-like analysis works at nside 2048, l <= 2500, with about
a thousand simulated skies.  How much more would that cost on this laptop, in time, memory and
disk, and what does the reduction cost us in information?
Computes: the measured time of one synthesis + analysis pair at nside 256, 512 and 1024, its
N_pix^{3/2} (= nside^3) extrapolation to nside 2048, the time of the full pipeline (4 transforms per
sky as in lib_cmbsim.simulate) for 1000 skies there, the memory of one map and the disk for 1000
maps; the fraction of the harmonic modes kept by l_max = 512, and the Monte Carlo error of an
error bar estimated from 2000 and from 300 simulations.
Writes: results/ch08/13_research_scale.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
from common import save_numbers, rng_for
import lib_cmbsim as cs

rng = rng_for("ch08", "13_research_scale")
times = {}
for nside in (256, 512, 1024):
    lmax = 2 * nside
    alm = cs.synalm(np.ones(3 * nside), 3 * nside - 1, rng)
    hp.alm2map(alm, nside, lmax=3 * nside - 1)                    # warm-up
    t0 = time.time()
    for _ in range(3):
        m = hp.alm2map(alm, nside, lmax=3 * nside - 1)
        hp.map2alm(m, lmax=lmax, iter=1)
    times[nside] = (time.time() - t0) / 3

t_pair_2048 = times[1024] * (2048 / 1024) ** 3                    # SHT cost ~ N_pix^{3/2} ~ nside^3
t_sky_2048 = 2 * t_pair_2048                                      # ~4 transforms per sky
n_full = 1000
hours_full = n_full * t_sky_2048 / 3600
npix_2048 = hp.nside2npix(2048)
mem_map_gb = npix_2048 * 8 / 1e9
disk_maps_gb = n_full * mem_map_gb
spec_mb = n_full * 2501 * 6 * 4 / 1e6                             # six float32 spectra per sky
modes = lambda L: (L + 1) ** 2 - 4                                # sum_{l=2}^{L} (2l+1)
mode_frac = modes(512) / modes(2500)

save_numbers("ch08", "13_research_scale", {
    "EightAScaleTwoFiveSix": f"{times[256]:.2f}",
    "EightAScaleFiveTwelve": f"{times[512]:.2f}",
    "EightAScaleTenTwentyFour": f"{times[1024]:.1f}",
    "EightAScaleTwentyFortyEight": f"{t_pair_2048:.0f}",
    "EightAScaleSkySec": f"{t_sky_2048:.0f}",
    "EightAScaleHours": f"{hours_full:.0f}",
    "EightAScaleMapGB": f"{mem_map_gb:.2f}",
    "EightAScaleDiskGB": f"{disk_maps_gb:.0f}",
    "EightAScaleSpecMB": f"{spec_mb:.0f}",
    "EightAScaleModeFrac": f"{100 * mode_frac:.1f}",
    "EightAScaleSNRloss": f"{1 / np.sqrt(mode_frac):.1f}",
    "EightAScaleMCerrTwoK": f"{100 / np.sqrt(2 * 1999):.1f}",
    "EightAScaleMCerrThree": f"{100 / np.sqrt(2 * 299):.1f}",
})
print(times, "pair 2048", t_pair_2048, "sky", t_sky_2048, "hours", hours_full,
      "map GB", mem_map_gb, "disk", disk_maps_gb, "modes", mode_frac)
