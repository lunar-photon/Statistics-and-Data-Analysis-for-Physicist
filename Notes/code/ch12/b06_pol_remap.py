"""b06_pol_remap.py -- do polarization singularities survive a smooth remapping, and how do they fail?

Question: a toy "lensing" moves every point x of a polarization map to x + d(x), with d the
gradient of a smooth random potential.  While the map x -> x + d(x) is one-to-one, every
singularity of the remapped map should sit on the image of an original one, with the same
index.  When the displacement is strong enough to fold the sky over itself (det A < 0, with
A = 1 + grad d), extra singularities appear in pairs.  And instrument noise, which is not a
remapping, creates pairs too.  How large are these effects?
Computes: E-mode Q, U on periodic 10 x 10 deg patches (10 arcmin beam, fiducial C_l^EE);
displacement fields scaled to rms convergence kappa_rms = 0.03 ... 0.8; the remapped maps by
cubic interpolation; singularities before and after; matching through the displacement;
the index rule (index after = sign(det A) x index before); white noise of increasing level.
Writes: figures/ch12/b_remap.pdf, results/ch12/b06_pol_remap.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree
from common import setup, savefig, save_numbers, rng_for, SERIES
from camb_fiducial import load_fiducial
from lib_persist import flat_qu, winding

rng = rng_for("ch12", "b06_pol_remap")
_, clee = load_fiducial("EE")
SIDE, N, FWHM = 10.0, 512, 10.0                            # deg, pixels, arcmin
pix_arcmin = SIDE * 60 / N
sig_b = np.deg2rad(FWHM / 60) / np.sqrt(8 * np.log(2))
ell = np.arange(len(clee))
cl_beam = clee * np.exp(-ell ** 2 * sig_b ** 2)            # beam applied to the power


def potential(rng):
    """A smooth random potential psi (pixel units), coherent over ~1 deg, and its derivatives."""
    k = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    kc = 2 * np.pi / (N / 8)                                # correlation length ~ 1.25 deg
    amp = np.where(k2 > 0, np.exp(-k2 / kc ** 2) / np.maximum(k2, 1e-12), 0.0)
    pk = np.fft.fft2(rng.standard_normal((N, N))) * amp
    dx = np.real(np.fft.ifft2(1j * KX * pk))
    dy = np.real(np.fft.ifft2(1j * KY * pk))
    dxx = np.real(np.fft.ifft2(-KX * KX * pk))
    dyy = np.real(np.fft.ifft2(-KY * KY * pk))
    dxy = np.real(np.fft.ifft2(-KX * KY * pk))
    kappa_rms = (0.5 * (dxx + dyy)).std()
    return np.array([dx, dy]) / kappa_rms, np.array([dxx, dyy, dxy]) / kappa_rms


def sing(Q, U):
    w = winding(Q, U)
    i, j = np.nonzero(w)
    return np.column_stack([i, j]) + 0.5, w[i, j]


def remap(F, d):
    """F evaluated at x + d(x) (pixel units), periodic cubic-spline interpolation."""
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return ndimage.map_coordinates(F, [ii + d[0], jj + d[1]], order=3, mode="grid-wrap")


def at(F, pos):
    """Bilinear value of a periodic field at plaquette centres."""
    return ndimage.map_coordinates(F, [pos[:, 0], pos[:, 1]], order=1, mode="grid-wrap")


strengths = [0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
NREAL = 6
rows = []
example = None
for r in range(NREAL):
    Q, U, _ = flat_qu(cl_beam, N, SIDE, rng)
    p0, q0 = sing(Q, U)
    tree0 = cKDTree(p0, boxsize=N)
    d1, h1 = potential(rng)
    for s in strengths:
        d, h = s * d1, s * h1
        detA = (1 + h[0]) * (1 + h[1]) - h[2] ** 2          # Jacobian of x -> x + d(x)
        Ql, Ul = remap(Q, d), remap(U, d)
        p1, q1 = sing(Ql, Ul)
        src = (p1 + at(d[0], p1)[:, None] * [1, 0] + at(d[1], p1)[:, None] * [0, 1]) % N
        dist, k = tree0.query(src)
        matched = dist < 1.5
        sgn = np.sign(at(detA, p1))
        rule = matched & (q1 == sgn * q0[k])                 # index = sign(det A) x parent index
        # total index carried by each parent: the sum over its images must equal its own index
        tot = np.zeros(len(q0))
        np.add.at(tot, k[matched], q1[matched])
        conserved = np.mean(tot == q0)
        rows.append((s, len(q0), len(q1), matched.mean(), rule.mean(), conserved,
                     np.mean(detA < 0), np.mean(sgn[matched] < 0)))
        if r == 0 and s == 0.6:
            example = (Q, U, p0, q0, p1, q1, detA, d)
rows = np.array(rows)
S = np.array(strengths)
mean = np.array([rows[rows[:, 0] == s].mean(0) for s in S])
extra = mean[:, 2] / mean[:, 1] - 1
print("strength, N0, N1, matched, index rule, parents conserved, fold area, images in folds")
for m in mean:
    print("  ".join(f"{v:.4f}" for v in m))

# ---------------------------------------------------------------- noise: not a remapping
noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2]
nrows = []
for r in range(NREAL):
    Q, U, _ = flat_qu(cl_beam, N, SIDE, rng)
    sP = np.sqrt(Q.var() + U.var())
    sm = lambda F: ndimage.gaussian_filter(F, 2.0, mode="wrap")   # extra smoothing, 2 pixels
    base = {False: sing(Q, U), True: sing(sm(Q), sm(U))}           # clean references
    for nl in noise_levels:
        Qn = Q + nl * sP * rng.standard_normal((N, N))
        Un = U + nl * sP * rng.standard_normal((N, N))
        for smooth in (False, True):
            Qs, Us = (sm(Qn), sm(Un)) if smooth else (Qn, Un)
            p1, q1 = sing(Qs, Us)
            p0, q0 = base[smooth]
            dist, _ = cKDTree(p0, boxsize=N).query(p1)
            nrows.append((nl, smooth, len(q1) / len(q0), np.mean(dist < 1.5)))
nrows = np.array(nrows)
nmean = {sm: np.array([nrows[(nrows[:, 0] == nl) & (nrows[:, 1] == sm)][:, 2:].mean(0)
                       for nl in noise_levels]) for sm in (0, 1)}
print("noise level, N/N0 raw, matched raw, N/N0 smoothed, matched smoothed")
for nl, a, b in zip(noise_levels, nmean[0], nmean[1]):
    print(nl, a, b)

# ---------------------------------------------------------------- figure
setup(7.4, 2.7)
fig, ax = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 1.1, 1.1]))
Q, U, p0, q0, p1, q1, detA, d = example
z = slice(0, 128)                                          # a 2.5 x 2.5 deg corner
ax[0].imshow(np.where(detA[z, z] < 0, 1.0, 0.0).T, origin="lower", cmap="Greys", vmin=0, vmax=3,
             extent=[0, 128, 0, 128])
ax[0].contour(detA[z, z].T, levels=[0], colors="0.4", linewidths=0.6, extent=[0, 128, 0, 128])
for (pp, qq, filled) in ((p0, q0, False), (p1, q1, True)):
    inz = (pp[:, 0] < 128) & (pp[:, 1] < 128)
    for sign, mk in ((0.5, "o"), (-0.5, "s")):
        sel = inz & (qq == sign)
        ax[0].plot(pp[sel, 0], pp[sel, 1], mk, ms=4 if filled else 6,
                   mfc=(SERIES[0] if sign > 0 else SERIES[1]) if filled else "none",
                   mec=SERIES[0] if sign > 0 else SERIES[1], mew=0.8, ls="none")
ax[0].set_xlim(0, 128)
ax[0].set_ylim(0, 128)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title(r"$\kappa_{\rm rms}=0.6$: folds grey", fontsize=9)
ax[1].plot(S, 100 * extra, "o-", color=SERIES[0], label="extra singularities [%]")
ax[1].plot(S, 100 * mean[:, 6], "s--", color="0.4", label=r"area with $\det A<0$ [%]")
ax[1].plot(S, 100 * (1 - mean[:, 5]), "^-", color=SERIES[2], label="parents not accounted for [%]")
ax[1].set_xlabel(r"strength of the remapping, $\kappa_{\rm rms}$")
ax[1].set_ylabel("per cent")
ax[1].legend(fontsize=6.5, loc="upper left")
nl = np.array(noise_levels)
ax[2].plot(nl[1:], nmean[0][1:, 0], "o-", color=SERIES[1], label="raw noisy map")
ax[2].plot(nl[1:], nmean[1][1:, 0], "s-", color=SERIES[2], label="smoothed again")
ax[2].set_xlabel(r"noise per pixel / $\sigma_P$")
ax[2].set_ylabel("singularities / noiseless count")
ax[2].legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
savefig(fig, "ch12", "b_remap")

i03, i06 = strengths.index(0.03), strengths.index(0.6)
save_numbers("ch12", "b06_pol_remap", {
    "PbRmSide": SIDE, "PbRmN": N, "PbRmFwhm": FWHM, "PbRmPix": pix_arcmin, "PbRmReal": NREAL,
    "PbRmNzero": mean[0, 1],
    "PbRmMatchLow": 100 * mean[i03, 3], "PbRmRuleLow": 100 * mean[i03, 4],
    "PbRmExtraLow": 100 * extra[i03],
    "PbRmFirstFold": strengths[int(np.argmax(mean[:, 6] > 0))],
    "PbRmExtraHigh": 100 * extra[i06], "PbRmFoldHigh": 100 * mean[i06, 6],
    "PbRmMatchHigh": 100 * mean[i06, 3], "PbRmRuleHigh": 100 * mean[i06, 4],
    "PbRmConsHigh": 100 * mean[i06, 5], "PbRmConsLow": 100 * mean[i03, 5],
    "PbRmInFoldHigh": 100 * mean[i06, 7],
    "PbRmConsMin": 100 * mean[:, 5].min(),
    "PbNoiseRawFive": nmean[0][noise_levels.index(0.05), 0],
    "PbNoiseRawTwenty": nmean[0][noise_levels.index(0.2), 0],
    "PbNoiseSmFive": nmean[1][noise_levels.index(0.05), 0],
    "PbNoiseSmTwenty": nmean[1][noise_levels.index(0.2), 0],
    "PbNoiseMatchSmTwenty": 100 * nmean[1][noise_levels.index(0.2), 1],
    "PbNoiseMatchRawTwenty": 100 * nmean[0][noise_levels.index(0.2), 1],
})
