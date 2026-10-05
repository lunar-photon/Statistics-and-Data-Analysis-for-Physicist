"""01_planck_data.py -- the Planck PR3 SMICA maps and the common mask, brought to laptop size.

Question: the public maps have 50 million pixels (nside 2048).  How do we make a smaller map
that is still an honest observation of the same sky, with a transfer function we know exactly?
Answer used here: fill the masked holes with Planck's inpainted CMB, transform the nside-2048 map to a_lm, replace the 2048 pixel window by the
pixel window of the target nside, and synthesise the map at that nside.  The degraded map then
carries the transfer function  T_l = B_l(SMICA effective beam, a 5' Gaussian) x p_l(nside), the same form as every
simulated map of chapter 8.  The mask is averaged over the 16 (or 4) small pixels inside each
large one and a large pixel is kept only if all of them are kept.

Inputs (downloaded once with curl from the IRSA Planck archive, release_3; the large files are
kept in planck_raw/ next to the Notes folder):
  all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_{full,hm1,hm2}.fits
  ancillary-data/masks/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits
  ancillary-data/cosmoparams/COM_PowerSpect_CMB-TT-{binned,full}_R3.01.txt
Writes: data/planck/smica_{full,hm1,hm2}_n{512,1024}.fits (muK, RING),
        data/planck/smica_transfer_n{512,1024}.npz, data/planck/common_mask_n{512,1024}.fits,
        results/ch13/01_planck_data.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
from astropy.io import fits
from common import save_numbers
import lib_planck as lp

NSIDES = (512, 1024)
LMAX = 3 * max(NSIDES) - 1
nums = {}
t0 = time.time()

# ---------------------------------------------------------------- the maps
p2048 = np.asarray(hp.pixwin(2048, lmax=LMAX))
full_src = lp.raw_path("full")
mask2048 = hp.read_map(lp.raw_path("mask"), dtype=np.float64) < 0.5      # True in the holes
inpainted = hp.read_map(full_src, field=5, dtype=np.float64) * 1e6                  # I_STOKES_INP
for kind in ("full", "hm1", "hm2"):
    if all(lp.degraded_path(kind, n).exists() for n in NSIDES):
        continue
    src = lp.raw_path(kind)
    m = hp.read_map(src, field=0, dtype=np.float64) * 1e6            # K_CMB -> muK, NESTED -> RING
    # the holes hold bright Galactic and source residuals; a sharp band limit would ring them out
    # over the kept sky, so fill the holes with Planck's inpainted CMB first (they stay masked)
    tag = {"full": "Full", "hm1": "HmOne", "hm2": "HmTwo"}[kind]
    nums[f"TwelveAHoleMax{tag}"] = f"{np.abs(m[mask2048]).max():.0f}"
    nums[f"TwelveAKeptMax{tag}"] = f"{np.abs(m[~mask2048]).max():.0f}"
    m[mask2048] = inpainted[mask2048]
    alm = hp.map2alm(m, lmax=LMAX, iter=1)
    del m
    for n in NSIDES:
        L = 3 * n - 1
        a = hp.resize_alm(alm, LMAX, LMAX, L, L)
        fl = np.asarray(hp.pixwin(n, lmax=L)) / p2048[: L + 1]          # swap the pixel windows
        hp.write_map(lp.degraded_path(kind, n), hp.alm2map(hp.almxfl(a, fl), n, lmax=L),
                     dtype=np.float32, overwrite=True)
    print(f"{kind}: degraded ({time.time() - t0:.0f} s)", flush=True)

for n in NSIDES:
    L = 3 * n - 1
    ell = np.arange(L + 1)
    sb = lp.FWHM_SMICA * lp.ARCMIN / np.sqrt(8 * np.log(2))
    gauss = np.exp(-0.5 * ell * (ell + 1) * sb ** 2)
    beam = fits.getdata(full_src, 2)["INT_BEAM"][: L + 1].astype(float)   # SMICA effective beam
    beam[:2] = 1.0                                     # the file leaves l = 0, 1 empty
    nums["TwelveABeamVsGauss"] = f"{100 * np.max(np.abs(beam[2:1200] / gauss[2:1200] - 1)):.2f}"
    np.savez(lp.PLANCK / f"smica_transfer_n{n}.npz", beam=beam, pixwin=np.asarray(hp.pixwin(n, lmax=L)))

# ---------------------------------------------------------------- the mask
m2048 = (~mask2048).astype(np.float64)
nums["TwelveAFskyMaskFull"] = f"{m2048.mean():.3f}"
for n in NSIDES:
    frac = hp.ud_grade(m2048, n)                     # mean of the small pixels inside each large one
    binary = (frac > 0.999).astype(np.float32)       # keep a large pixel only if all of it is kept
    hp.write_map(lp.PLANCK / f"common_mask_n{n}.fits", binary, dtype=np.float32, overwrite=True)
    nums[f"TwelveAFskyMask{'Five' if n == 512 else 'Ten'}"] = f"{binary.mean():.3f}"
nums["TwelveAPrepSeconds"] = f"{time.time() - t0:.0f}"
print(nums)
save_numbers("ch13", "01_planck_data", nums)
