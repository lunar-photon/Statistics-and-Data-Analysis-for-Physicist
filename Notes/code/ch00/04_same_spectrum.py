"""Same power spectrum, different morphology: why the choice of summary matters.

Question: can two fields have *identical* amplitude and two-point statistics
(variance, P(k)) and still look completely different?  Which summaries tell them apart?
Computes: a clumpy field (isolated Gaussian blobs dropped at random positions) and its
phase-randomised twin (same |Fourier amplitudes|, random phases).  For both it measures
the variance, the power spectrum P(k), the skewness (a higher-order statistic) and the
number of connected components of the excursion set {phi > nu} as nu is scanned.
Writes:   figures/ch00/same_spectrum.pdf, results/ch00/04_same_spectrum.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from common import setup, savefig, save_numbers, rng_for, SERIES

NPIX, NBLOB, RBLOB = 256, 60, 5.0     # grid size, number of blobs, blob radius (pixels)
setup(7.0, 6.0)
rng = rng_for("ch00", "04_same_spectrum")

# ---- field A: clumps.  Drop points, smooth with a Gaussian of width RBLOB (periodic box) ----
pts = np.zeros((NPIX, NPIX))
ij = rng.integers(0, NPIX, size=(NBLOB, 2))
np.add.at(pts, (ij[:, 0], ij[:, 1]), 1.0)
A = ndimage.gaussian_filter(pts, RBLOB, mode="wrap")
A = (A - A.mean()) / A.std()                       # zero mean, unit variance

# ---- field B: keep every |A_k|, replace the phases by those of white noise ----
#      rfft2 of a real white-noise map has exactly the Hermitian symmetry a real field needs
Ak = np.fft.rfft2(A)
phase = np.exp(1j * np.angle(np.fft.rfft2(rng.standard_normal((NPIX, NPIX)))))
Bk = np.abs(Ak) * phase
Bk[0, 0] = Ak[0, 0]                                 # keep the (zero) mean
B = np.fft.irfft2(Bk, s=A.shape)


def power_spectrum(f, nbins=40):
    """Azimuthally averaged |f_k|^2 in bins of |k| (arbitrary normalisation)."""
    fk2 = np.abs(np.fft.fft2(f)) ** 2
    kx = np.fft.fftfreq(NPIX)[:, None]; ky = np.fft.fftfreq(NPIX)[None, :]
    kk = np.hypot(kx, ky).ravel()
    edges = np.linspace(0, 0.5, nbins + 1)
    idx = np.digitize(kk, edges)
    pk = np.array([fk2.ravel()[idx == i].mean() for i in range(1, nbins + 1)])
    return 0.5 * (edges[1:] + edges[:-1]), pk


def skew(f):
    return np.mean((f - f.mean()) ** 3) / f.std() ** 3


def n_components(f, nu):
    """beta_0 of the excursion set {f > nu} (4-connected pixels)."""
    return ndimage.label(f > nu)[1]


kc, PA = power_spectrum(A)
_, PB = power_spectrum(B)
nus = np.linspace(-1.5, 4.0, 45)
b0A = [n_components(A, nu) for nu in nus]
b0B = [n_components(B, nu) for nu in nus]

fig, ax = plt.subplots(2, 2)
for a, f, name in [(ax[0, 0], A, "A: clumps"), (ax[0, 1], B, "B: same $|\\tilde\\phi_k|$, random phases")]:
    a.imshow(f, cmap="RdBu_r", vmin=-3, vmax=3, origin="lower")
    a.contour(f, levels=[1.5], colors="k", linewidths=0.6)
    a.set(title=name, xticks=[], yticks=[]); a.grid(False)
keep = PA > 1e-6 * PA.max()                         # drop the round-off floor at high k
ax[1, 0].loglog(kc[keep], PA[keep], "o", color=SERIES[0], ms=4, label="A")
ax[1, 0].loglog(kc[keep], PB[keep], "-", color=SERIES[1], label="B")
ax[1, 0].set(xlabel="$|k|$ (cycles/pixel)", ylabel="$P(k)$", title="power spectra coincide",
             )
ax[1, 0].legend()
ax[1, 1].plot(nus, b0A, color=SERIES[0], label="A")
ax[1, 1].plot(nus, b0B, color=SERIES[1], label="B")
ax[1, 1].set(xlabel="threshold $\\nu$", ylabel="components $\\beta_0(\\nu)$",
             title="topology differs")
ax[1, 1].legend()
fig.tight_layout()
savefig(fig, "ch00", "same_spectrum")

good = PA > 1e-12 * PA.max()
save_numbers("ch00", "04_same_spectrum", {
    "SameVarA": A.var(), "SameVarB": B.var(),
    "SamePkMaxDev": np.max(np.abs(PB[good] / PA[good] - 1)),
    "SameSkewA": skew(A), "SameSkewB": skew(B),
    "SameBzeroA": n_components(A, 3.0), "SameBzeroB": n_components(B, 3.0),
    "SameBzeroAzero": n_components(A, 0.0), "SameBzeroBzero": n_components(B, 0.0),
    "SameMinA": A.min(),
    "SameNblob": NBLOB,
})
