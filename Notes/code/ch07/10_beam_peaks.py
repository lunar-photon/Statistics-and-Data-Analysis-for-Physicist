"""10_beam_peaks.py -- what a Gaussian beam does to the acoustic peaks of the CMB spectrum.

Question: a beam of full width at half maximum FWHM multiplies C_l by B_l^2 = exp[-l(l+1) sigma_b^2],
sigma_b = FWHM / sqrt(8 ln 2). Which acoustic peaks survive for beams of 7 deg (COBE-like),
1 deg, 30', 10' and 5' (Planck-like), and why does dividing the beam back out amplify the noise?
Computes: D_l = l(l+1) C_l / 2pi of the fiducial lensed TT spectrum and D_l B_l^2 for each beam;
the positions of the acoustic peaks; the multipole l_1/2 at which B_l^2 = 1/2 (exact root and the
approximation sqrt(ln 2)/sigma_b); the number of peaks with B_l^2 > 1/2; the surviving power at the first peaks for a
10' and a 13' beam (WMAP's sharpest channel) and for COBE at the first peak; and, for a white noise of
30 muK arcmin, the deconvolved noise N_l / B_l^2 and the multipole where it equals C_l.
Writes: figures/ch07/beam_peaks.pdf, results/ch07/10_beam_peaks.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES, INK, INK2

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from camb_fiducial import load_fiducial

LMAX = 2500
ell, C = load_fiducial("TT")
ell, C = ell[: LMAX + 1], C[: LMAX + 1]
D = ell * (ell + 1) * C / (2 * np.pi)

# acoustic peaks: local maxima of D_l above l = 100
idx, _ = find_peaks(D, prominence=5.0)
peaks = ell[idx][ell[idx] > 100]

ARCMIN = np.radians(1 / 60)
beams = [("7^\\circ", 420.0, "Cobe"), ("1^\\circ", 60.0, "Deg"), ("30'", 30.0, "Thirty"),
         ("10'", 10.0, "Ten"), ("5'", 5.0, "Five")]
DEPTH = 30.0                                   # muK arcmin, the same for every beam
Nl = (DEPTH * ARCMIN) ** 2

nums = {"SevABmPkNum": len(peaks), "SevABmPkDepth": f"{DEPTH:.0f}"}
for i, p in enumerate(peaks[:7]):
    nums["SevABmPk" + "ABCDEFG"[i]] = int(p)

def b2(fwhm_arcmin, l):
    sb = fwhm_arcmin * ARCMIN / np.sqrt(8 * np.log(2))
    return np.exp(-l * (l + 1) * sb**2)


# how much of each peak's power survives: 10' beam, a 13' beam (WMAP's sharpest channel), COBE at the first peak
nums["SevABmBlTenOne"] = f"{100 * b2(10.0, peaks[0]):.0f}"
nums["SevABmBlTenFive"] = f"{100 * b2(10.0, peaks[4]):.0f}"
sb13 = 13.0 * ARCMIN / np.sqrt(8 * np.log(2))
nums["SevABmLhalfWmap"] = f"{(-1 + np.sqrt(1 + 4 * np.log(2) / sb13**2)) / 2:.0f}"
for i, tag in enumerate(["One", "Two", "Three"]):
    nums[f"SevABmWmap{tag}"] = f"{100 * b2(13.0, peaks[i]):.0f}"
sbC = 420.0 * ARCMIN / np.sqrt(8 * np.log(2))
nums["SevABmCobeExp"] = f"{peaks[0] * (peaks[0] + 1) * sbC**2 / np.log(10):.0f}"

setup(9.0, 3.6)
fig, (a, b) = plt.subplots(1, 2, sharey=True)
a.semilogx(ell[2:], D[2:], color=INK, lw=1.2, label="sky")
b.semilogx(ell[2:], D[2:], color=INK, lw=1.2, label="sky")
for (lab, fw, tag), c in zip(beams, SERIES):
    sb = fw * ARCMIN / np.sqrt(8 * np.log(2))
    B2 = np.exp(-ell * (ell + 1) * sb**2)
    lhalf = (-1 + np.sqrt(1 + 4 * np.log(2) / sb**2)) / 2      # root of l(l+1) sigma^2 = ln 2
    npk = int(np.sum(np.exp(-peaks * (peaks + 1) * sb**2) > 0.5))
    with np.errstate(divide="ignore", over="ignore"):
        noise_sky = ell * (ell + 1) * Nl / B2 / (2 * np.pi)
    cross = ell[(ell > 2) & (noise_sky > D)]
    leq = int(cross[0]) if len(cross) else LMAX
    nums[f"SevABmLhalf{tag}"] = f"{lhalf:.0f}"
    nums[f"SevABmLhalfApprox{tag}"] = f"{np.sqrt(np.log(2)) / sb:.0f}"
    nums[f"SevABmNpk{tag}"] = npk
    nums[f"SevABmLeq{tag}"] = leq
    nums[f"SevABmSigma{tag}"] = f"{sb / ARCMIN:.3g}"
    a.semilogx(ell[2:], (D * B2)[2:], color=c, label=rf"FWHM ${lab}$")
    a.plot([lhalf], [np.interp(lhalf, ell, D * B2)], "o", color=c, ms=4)
    m = noise_sky < 2e4
    b.semilogx(ell[2:][m[2:]], (D + noise_sky)[2:][m[2:]], color=c, lw=1.2)
    b.semilogx(ell[2:][m[2:]], noise_sky[2:][m[2:]], color=c, lw=0.9, ls="--")
for p in peaks[:7]:
    for ax in (a, b):
        ax.axvline(p, color=INK2, lw=0.5, ls=":")
for i, p in enumerate(peaks[:7]):
    a.text(p, 6300, str(i + 1), ha="center", va="bottom", fontsize=7, color=INK2)
a.set_xlim(2, LMAX); a.set_ylim(0, 6800)
a.set_xlabel(r"$\ell$"); a.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
a.set_title(r"observed: $D_\ell B_\ell^2$ (dots: $B_\ell^2=1/2$)")
a.legend(fontsize=7, loc="upper left")
b.set_xlabel(r"$\ell$")
b.set_title(rf"deconvolved: $D_\ell$ + noise $N_\ell/B_\ell^2$ (dashed), {DEPTH:.0f}$\,\mu$K$'$")
save_numbers("ch07", "10_beam_peaks", nums)
for k, v in nums.items():
    print(k, v)
savefig(fig, "ch07", "beam_peaks")
