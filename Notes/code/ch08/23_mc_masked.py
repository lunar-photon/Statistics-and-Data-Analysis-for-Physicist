"""23_mc_masked.py -- the Monte Carlo of the cut sky: one simulated sky, seen through eight masks.

Question: what is the sampling distribution of the pseudo-C_l (and of every estimator built
from it) for each mask?  No formula gives it in general; the simulations do.
Computes: N_SIM skies drawn from the CAMB fiducial C_l, each observed by the toy experiment of
lib_cmbsim.Experiment (nside 256, FWHM 30 arcmin, white noise 200 muK arcmin, pixel window),
then multiplied by each mask of 21_masks.py and transformed (pixel quadrature, iter = 0) up to
l = 767.  For the 10 per cent cap we also apply a toy "scan filter" -- subtract the mean of
every iso-latitude ring -- to signal and noise separately (transfer function, Hivon sec. 3.2).
Runs in independent chunks (own random stream each) so that an interrupted run resumes.
Usage: python3 code/ch08/23_mc_masked.py [chunk ...]      (no argument: every chunk, in turn)
Writes: data/ch08/mc_masked_<chunk>.npz, results/ch08/23_mc_masked.tex (after the last chunk)
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
from common import save_numbers, rng_for
from camb_fiducial import load_fiducial
import lib_cmbsim as cs

N_CHUNK, PER_CHUNK = 8, 250                # 2000 skies in all, ~2 min per chunk
L = 767
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"
KEYS = ["full", "cap10", "cap10apo", "cap30", "cap70", "band20", "band20apo", "holes"]

ell, cl = load_fiducial()
exp = cs.Experiment()
z = np.load(DATA / "masks.npz")
MASKS = {k: z["mask_" + k].astype(np.float64) for k in KEYS}

# rings of constant latitude: the toy scan filter removes each ring's mean inside the mask
theta, _ = hp.pix2ang(exp.nside, np.arange(exp.npix))
_, RING = np.unique(theta, return_inverse=True)
GOOD = MASKS["cap10"] > 0
NRING = RING.max() + 1


def ring_filter(x):
    """Subtract, ring by ring, the mean of the observed pixels; masked pixels stay zero."""
    s = np.bincount(RING[GOOD], weights=x[GOOD], minlength=NRING)
    c = np.bincount(RING[GOOD], minlength=NRING)
    mean = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
    return (x - mean[RING]) * GOOD


def pcl(m):
    return hp.alm2cl(hp.map2alm(m, lmax=L, iter=0))


def run_chunk(c):
    path = DATA / f"mc_masked_{c}.npz"
    if path.exists():
        return
    rng = rng_for("ch08", "23_mc_masked", stream=c)
    out = {k: np.empty((PER_CHUNK, L + 1), np.float32) for k in
           ["sky"] + ["pcl_" + k for k in KEYS] + ["filt_s", "filt_sn", "filt_n"]}
    t0 = time.time()
    for i in range(PER_CHUNK):
        a = cs.synalm(cl[: exp.lmax_sim + 1], exp.lmax_sim, rng)    # the sky, l <= 767
        s = cs.observe_signal(a, exp)                                # beam and pixel window
        n = cs.noise_map(exp, rng)                                   # white pixel noise
        out["sky"][i] = hp.alm2cl(a)
        d = s + n
        for k in KEYS:
            out["pcl_" + k][i] = pcl(MASKS[k] * d)
        a_s = hp.map2alm(ring_filter(MASKS["cap10"] * s), lmax=L, iter=0)
        a_n = hp.map2alm(ring_filter(MASKS["cap10"] * n), lmax=L, iter=0)
        out["filt_s"][i] = hp.alm2cl(a_s)
        out["filt_n"][i] = hp.alm2cl(a_n)
        out["filt_sn"][i] = hp.alm2cl(a_s + a_n)
        if (i + 1) % 100 == 0:
            print(f"  chunk {c}: {i + 1}/{PER_CHUNK} skies, {time.time() - t0:.0f} s", flush=True)
    out["seconds"] = np.array(time.time() - t0)
    np.savez(path, **out)
    print(f"[cache] wrote {path}")


def load_all():
    """Concatenate every chunk: dict of arrays (N_SIM, L+1)."""
    parts = [np.load(DATA / f"mc_masked_{c}.npz") for c in range(N_CHUNK)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k != "seconds"}


if __name__ == "__main__":
    chunks = [int(a) for a in sys.argv[1:]] or list(range(N_CHUNK))
    for c in chunks:
        run_chunk(c)
    if all((DATA / f"mc_masked_{c}.npz").exists() for c in range(N_CHUNK)):
        secs = sum(float(np.load(DATA / f"mc_masked_{c}.npz")["seconds"]) for c in range(N_CHUNK))
        save_numbers("ch08", "23_mc_masked", {
            "EightBNsim": N_CHUNK * PER_CHUNK,
            "EightBSecPerSky": f"{secs / (N_CHUNK * PER_CHUNK):.2f}",
            "EightBCPUMinutes": f"{secs / 60:.0f}",
            "EightBNmasks": len(KEYS),
        })
