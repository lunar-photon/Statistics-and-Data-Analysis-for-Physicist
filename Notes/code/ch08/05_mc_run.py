"""05_mc_run.py -- the Monte Carlo of the full-sky, beamed, noisy experiment.

Question: what is the sampling distribution of the power-spectrum estimator when the sky is
seen through a beam, a pixel window and pixel noise?  The simulations answer it directly.
Computes: N_SIM skies drawn from the CAMB fiducial C_l, each observed by the toy experiment of
lib_cmbsim.Experiment (nside 256, FWHM 30 arcmin, 200 muK arcmin) four ways: full map, two
half maps (for the cross-spectrum), white noise alone, and inhomogeneous noise of the same
mean variance.  Spectra up to l = 512 are cached in data/ch08/mc_fullsky.npz.
Writes: data/ch08/mc_fullsky.npz, results/ch08/05_mc_run.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import save_numbers, rng_for
from camb_fiducial import load_fiducial
import lib_cmbsim as cs

N_SIM = 2000
ell, cl = load_fiducial()
exp = cs.Experiment()
varmap = cs.variance_pattern(exp.nside, contrast=4.0)


def make():
    rng = rng_for("ch08", "05_mc_run")
    t0 = time.time()
    d = cs.simulate(exp, cl, N_SIM, rng, varmap=varmap, progress=200)
    d["seconds"] = np.array(time.time() - t0)
    return d


sims = cs.cached("mc_fullsky", make)
secs = float(sims["seconds"])
save_numbers("ch08", "05_mc_run", {
    "EightANsim": N_SIM,
    "EightAMinutes": f"{secs / 60:.0f}",
    "EightASecPerSky": f"{secs / N_SIM:.2f}",
})
print(f"{N_SIM} skies in {secs / 60:.1f} min")
