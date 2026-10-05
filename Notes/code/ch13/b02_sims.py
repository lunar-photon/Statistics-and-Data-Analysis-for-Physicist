"""b02_sims.py -- Gaussian skies passed through the same beam, pixels, noise and mask as the SMICA map.

Question: if the microwave sky were an isotropic Gaussian field with the best-fit spectrum, what
would the area, boundary length, Euler characteristic and Betti numbers of its excursion sets look
like, after exactly the processing the real map received?  One run produces one chunk of skies at
one resolution, so that every run stays under ten minutes.

Usage:    python3 code/ch13/b02_sims.py NSIDE CHUNK      (NSIDE in 64, 128, 256, 512)
Computes: for each sky: signal a_lm from the best-fit C_l times beam and pixel window; noise a_lm
          from the half-difference noise spectrum times the same re-beaming as the data; the
          monopole and dipole fitted outside the mask and removed; then every statistic of
          lib_skytopo.sky_stats at Planck's 12 thresholds (and the H0 persistence pairs at
          N_side 64 and 128).
Writes:   data/ch13/b_sims_NSIDE_CHUNK.npz
"""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
from common import rng_for, DATA
from camb_fiducial import load_fiducial
from lib_skytopo import windows, pixwin, remove_monodipole, mesh, neighbours, sky_stats

FWHM = {64: 160.0, 128: 80.0, 256: 40.0, 512: 20.0}
PER_CHUNK = {64: 1000, 128: 1000, 256: 500, 512: 100}
NSIDE, CHUNK = int(sys.argv[1]), int(sys.argv[2])
out = DATA / "ch13" / f"b_sims_{NSIDE}_{CHUNK}.npz"
if out.exists():
    print("exists:", out)
    sys.exit(0)

prep = np.load(DATA / "ch13" / "b_prepared.npz")
lmax = 3 * NSIDE - 1
win = windows(NSIDE, FWHM[NSIDE], lmax)
win_in = hp.gauss_beam(np.radians(5.0 / 60), lmax=lmax) * pixwin(2048, lmax)
_, cl = load_fiducial("TT")
cl_sig = cl[:lmax + 1] * win ** 2
cl_sig[:2] = 0.0
nl = np.convolve(prep["nl_hd"], np.ones(21) / 21, mode="same")[:lmax + 1]   # smoothed in l
nl[:2] = 0.0
cl_noise = nl * (win / win_in) ** 2                    # the data's re-beaming applied to the noise
obs = prep[f"obs_{NSIDE}"]
msh, nb8, nb4 = mesh(NSIDE), neighbours(NSIDE), neighbours(NSIDE, edge_only=True)
rng = rng_for("ch13", f"b02_sims_{NSIDE}_{CHUNK}")


def gauss_alm(c):
    """a_l0 ~ N(0, C_l); Re, Im of a_lm (m > 0) ~ N(0, C_l / 2)."""
    ell, m = hp.Alm.getlm(lmax)
    s = np.sqrt(c[ell])
    re, im = rng.standard_normal(ell.size), rng.standard_normal(ell.size)
    return np.where(m == 0, s * re, s * (re + 1j * im) / np.sqrt(2))


n = PER_CHUNK[NSIDE]
keys = ("v0", "v1", "v2", "chi", "b0", "b1", "b0c")
res = {k: np.zeros((n, 12)) for k in keys}
res.update(ratio=np.zeros(n), s0=np.zeros(n))
pb, pd, off = [], [], [0]
t0 = time.time()
for i in range(n):
    a = gauss_alm(cl_sig) + gauss_alm(cl_noise)
    a = remove_monodipole(a, obs, NSIDE, lmax)
    s = sky_stats(a, NSIDE, lmax, obs, msh, nb8, nb4, keep_pairs=NSIDE <= 128)
    for k in keys:
        res[k][i] = s[k]
    res["ratio"][i], res["s0"][i] = s["ratio"], s["s0"]
    if NSIDE <= 128:
        pb.append(s["pairs"][0])
        pd.append(s["pairs"][1])
        off.append(off[-1] + len(s["pairs"][0]))
    if i % 100 == 0:
        print(f"sky {i}  {time.time() - t0:.0f}s", flush=True)
if NSIDE <= 128:
    res.update(pair_b=np.concatenate(pb), pair_d=np.concatenate(pd), pair_off=np.array(off))
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, **res)
print("wrote", out, f"{time.time() - t0:.0f}s")
