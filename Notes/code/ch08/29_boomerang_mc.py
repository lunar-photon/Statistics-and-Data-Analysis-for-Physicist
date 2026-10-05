"""29_boomerang_mc.py -- the Monte Carlo of a Boomerang-like experiment (Hivon et al. 2002, sec. 4).

Question: can the MASTER test of Hivon et al. be repeated on a laptop?  Their set-up: an
elliptical patch of semi-axes 20 x 12 degrees (fsky = 1.8%), three apodisations, a 10 arcmin
beam, 7 arcmin pixels (nside 512), a timestream high-pass filter, and 250 signal-only, 450
noise-only and 1350 signal+noise simulations.
Laptop version (what is kept and what is simplified):
  * same patch, apodisations, beam, pixel size and simulation counts;
  * the timestream filter is replaced by its idealisation: scans along lines of constant
    latitude through a patch on the equator, high-pass filtered at a sharp cut, remove exactly
    the harmonic modes with |m| < M_CUT (Hivon App. B; M_CUT = 50 mimics their 100 mHz at 1 deg/s);
  * the noise is white and homogeneous, 350 muK arcmin (their 130 muK sqrt(s) detector over
    500 s/deg^2 of integration), generated in harmonic space and filtered like the sky.
  The full analysis simulates the time-ordered data with the measured 1/f noise spectrum and
  the real pointing, and makes maps from them; that costs about 1 CPU-day for 300 skies
  (their eq. 33) and is what the simplification avoids.
Computes, per chunk (6 chunks, own random stream each): 225 signal+noise skies, 42 signal-only
skies and 75 noise-only skies, each seen through the three windows; pseudo-C_l up to l = 1300.
Usage: python3 code/ch08/29_boomerang_mc.py [chunk ...]
Writes: data/ch08/boom_window.npz (windows and their spectra), data/ch08/boom_<chunk>.npz,
        results/ch08/29_boomerang_mc.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
from common import save_numbers, rng_for
from camb_fiducial import load_fiducial
import lib_cmbsim as cs
import lib_masks as lm

NSIDE, LSIM, LAN = 512, 1535, 1300
FWHM, DEPTH, M_CUT = 10.0, 350.0, 50
A_DEG, B_DEG = 20.0, 12.0
N_CHUNK, N_SN, N_S, N_N = 6, 225, 42, 75
PROFILES = ["tophat", "cosine", "gauss"]
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"

ell, cl = load_fiducial()
cl = cl[: LSIM + 1]
exp = cs.Experiment(nside=NSIDE, fwhm_arcmin=FWHM, depth_uK_arcmin=DEPTH, lmax=LAN)
T = exp.transfer(LSIM)                                     # beam x pixel window, l <= 1535
NL = exp.nl(LSIM)[0]                                       # white-noise power sigma^2 Omega_pix
lidx, midx = hp.Alm.getlm(LSIM)
KEEP = (midx >= M_CUT).astype(float)                       # the scan filter: drop |m| < M_CUT


def windows():
    path = DATA / "boom_window.npz"
    if path.exists():
        z = np.load(path)
        return {p: z["w_" + p] for p in PROFILES}
    W = {p: lm.ellipse(NSIDE, A_DEG, B_DEG, p) for p in PROFILES}
    np.savez(path, **{"w_" + p: w.astype(np.float32) for p, w in W.items()},
             **{"wl_" + p: lm.window_spectrum(w, LSIM) for p, w in W.items()})
    return W


def pcls(m, W):
    return np.stack([hp.alm2cl(hp.map2alm(W[p] * m, lmax=LAN, iter=0)) for p in PROFILES])


def run_chunk(c, W):
    path = DATA / f"boom_{c}.npz"
    if path.exists():
        return
    rng = rng_for("ch08", "29_boomerang_mc", stream=c)
    out = {"sn": np.empty((N_SN, 3, LAN + 1), np.float32), "s": np.empty((N_S, 3, LAN + 1), np.float32),
           "n": np.empty((N_N, 3, LAN + 1), np.float32)}
    t0 = time.time()
    for kind, n in [("s", N_S), ("n", N_N), ("sn", N_SN)]:
        for i in range(n):
            alm = np.zeros(lidx.size, complex)
            if kind in ("s", "sn"):
                alm += T[lidx] * cs.synalm(cl, LSIM, rng)
            if kind in ("n", "sn"):
                alm += cs.synalm(np.full(LSIM + 1, NL), LSIM, rng)
            m = hp.alm2map(alm * KEEP, NSIDE, lmax=LSIM)      # filtered map
            out[kind][i] = pcls(m, W)
        print(f"  chunk {c}: {kind} done, {time.time() - t0:.0f} s", flush=True)
    out["seconds"] = np.array(time.time() - t0)
    np.savez(path, **out)
    print(f"[cache] wrote {path}")


def load_all():
    parts = [np.load(DATA / f"boom_{c}.npz") for c in range(N_CHUNK)]
    return {k: np.concatenate([p[k] for p in parts]) for k in ["sn", "s", "n"]}


if __name__ == "__main__":
    W = windows()
    chunks = [int(a) for a in sys.argv[1:]] or list(range(N_CHUNK))
    for c in chunks:
        run_chunk(c, W)
    if all((DATA / f"boom_{c}.npz").exists() for c in range(N_CHUNK)):
        secs = sum(float(np.load(DATA / f"boom_{c}.npz")["seconds"]) for c in range(N_CHUNK))
        save_numbers("ch08", "29_boomerang_mc", {
            "EightBboomNsn": N_CHUNK * N_SN, "EightBboomNs": N_CHUNK * N_S, "EightBboomNn": N_CHUNK * N_N,
            "EightBboomMinutes": f"{secs / 60:.0f}", "EightBboomMcut": M_CUT, "EightBboomDepth": f"{DEPTH:.0f}",
            "EightBboomFwhm": f"{FWHM:.0f}", "EightBboomNside": NSIDE, "EightBboomLan": LAN,
            "EightBboomSigmaPix": f"{exp.sigma_pix:.0f}",
        })
