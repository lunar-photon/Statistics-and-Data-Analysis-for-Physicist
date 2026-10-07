"""41_pol_data.py -- the SMICA polarization maps and their confidence mask, at nside 512.

Question: the SMICA Q and U maps of the 2018 release (full mission and the two half-missions)
have 50 million pixels.  What map at nside 512 would Planck have made of the same polarized
sky, and which pixels does the SMICA polarization confidence mask (PMASK) let us trust?
Steps:
  1. read Q, U (K_CMB -> muK) and PMASK (field 4 of the full-mission file);
  2. set Q = U = 0 in the masked pixels, so that nothing from the regions SMICA does not trust
     can be carried into the kept sky by the band limit (the numbers written below show that
     the masked pixels are not much brighter than the kept ones: both are dominated by noise);
  3. spin-2 transform, keep l <= 1535, swap the polarization pixel window of nside 2048 for
     that of nside 512, synthesise Q, U at nside 512;
  4. degrade PMASK: a large pixel is kept only if all 16 small pixels are kept.
Writes: data/ch13/smica_pol_n512.npz (Q, U of full, hm1, hm2 as float32; pmask; beams),
        results/ch13/41_pol_data.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
from astropy.io import fits
from common import save_numbers
import lib_planck as lp
import lib_biref as lb

t0 = time.time()
N, L = lb.NSIDE, lb.LS
nums = {}
full_src = lp.raw_path("full")
with fits.open(full_src) as f:
    for i, h in enumerate(f):
        print(i, h.name, getattr(h, "columns", None) and h.columns.names, flush=True)
    bcols = f[2].columns.names
    beamtab = f[2].data
pmask = hp.read_map(full_src, field=4, dtype=np.float64)
print("PMASK values:", np.unique(np.round(pmask, 3))[:10], flush=True)
hole = pmask < 0.5
nums["ThirteenEFskyPmaskFull"] = f"{1 - hole.mean():.3f}"

pw2048 = hp.pixwin(2048, pol=True, lmax=L)[1]
pw512 = hp.pixwin(N, pol=True, lmax=L)[1]
fl = np.zeros(L + 1)
fl[2:] = pw512[2:] / pw2048[2:]
out = {}
for kind in ("full", "hm1", "hm2"):
    src = lp.raw_path(kind)
    q = hp.read_map(src, field=1, dtype=np.float64) * 1e6
    u = hp.read_map(src, field=2, dtype=np.float64) * 1e6
    p = np.hypot(q, u)
    tag = {"full": "Full", "hm1": "HmOne", "hm2": "HmTwo"}[kind]
    nums[f"ThirteenEPmaxIn{tag}"] = f"{p[hole].max():.0f}"
    nums[f"ThirteenEPmaxOut{tag}"] = f"{p[~hole].max():.0f}"
    nums[f"ThirteenEPrmsOut{tag}"] = f"{np.sqrt(np.mean(p[~hole] ** 2) / 2):.1f}"
    q[hole] = 0.0
    u[hole] = 0.0
    alm = hp.map2alm([np.zeros_like(q), q, u], lmax=L, iter=1, pol=True)
    del q, u, p
    aE, aB = hp.almxfl(alm[1], fl), hp.almxfl(alm[2], fl)
    _, Q, U = hp.alm2map([alm[0] * 0, aE, aB], N, lmax=L, pol=True)
    out[f"{kind}_Q"], out[f"{kind}_U"] = Q.astype(np.float32), U.astype(np.float32)
    print(f"{kind}: done ({time.time() - t0:.0f} s)", flush=True)

frac = hp.ud_grade((~hole).astype(np.float64), N)
pm512 = (frac > 0.999).astype(np.float32)
out["pmask"] = pm512
nums["ThirteenEFskyPmaskFive"] = f"{pm512.mean():.3f}"
ell = np.arange(L + 1)
out["beam_T"] = beamtab["INT_BEAM"][: L + 1].astype(float)
out["beam_P"] = (beamtab["POL_BEAM"][: L + 1] if "POL_BEAM" in bcols else beamtab["INT_BEAM"][: L + 1]).astype(float)
out["pixwin_P"] = pw512
out["pixwin_T"] = hp.pixwin(N, lmax=L)
out["beam_T"][:2] = 1.0
out["beam_P"][:2] = 1.0
print("beam columns:", bcols, " B_P(1000) =", out["beam_P"][1000], " B_T(1000) =", out["beam_T"][1000])
nums["ThirteenEBeamPol"] = f"{out['beam_P'][1000]:.3f}"
nums["ThirteenEPixPol"] = f"{pw512[1000]:.3f}"
lb.DATA.mkdir(parents=True, exist_ok=True)
np.savez(lb.DATA / f"smica_pol_n{N}.npz", **out)
nums["ThirteenEPrepSeconds"] = f"{time.time() - t0:.0f}"
print(nums)
save_numbers("ch13", "41_pol_data", nums)
