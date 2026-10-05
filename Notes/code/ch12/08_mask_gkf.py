"""08_mask_gkf.py -- what a mask does to the Euler characteristic, and how the formula absorbs it.

Question: real sky maps are analysed outside a Galactic mask.  The Gaussian kinematic formula
says the domain enters only through three numbers: its Euler characteristic L0, half its
boundary length L1 and its area L2.  For the sky outside a |b| < 20 deg band (two polar caps),
does  E chi = L0 Psi(nu) + L1 rho1(nu) + L2 rho2(nu)  match simulated skies, and how large is
the boundary term compared with the full-sky prediction scaled by the sky fraction?

Computes: 300 Gaussian CMB skies (fiducial C_l, 160' beam, nside 256: about nine pixels per
          correlation angle, so that the pixel counting is unbiased at the per-cent level); chi of the excursion set
          restricted to the unmasked pixels at 25 thresholds; the formula with the L_j of the
          two caps and the naive "f_sky x full sky" prediction.
Writes:   figures/ch12/mask_gkf.pdf, results/ch12/08_mask_gkf.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import erfc
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES
from camb_fiducial import load_fiducial
from lib_fields import gaussian_field_sphere
from lib_sphere import mesh, chi_sphere

setup()
rng = rng_for("ch12", "08_mask_gkf")
NSIDE, FWHM, NSIM, BCUT = 256, 160.0, 300, 20.0
LMAX = 511                                     # the 160' beam leaves nothing above l ~ 300
ell, cl = load_fiducial("TT")
cl = cl[:LMAX + 1].copy()
cl[:2] = 0.0
cl *= (hp.gauss_beam(np.radians(FWHM / 60), LMAX) * hp.pixwin(NSIDE)[:LMAX + 1]) ** 2
l = np.arange(LMAX + 1)
lam = np.sum((2 * l + 1) * cl * l * (l + 1) / 2) / np.sum((2 * l + 1) * cl)     # per steradian

# the mask: keep |b| > 20 deg, i.e. two polar caps of angular radius 70 deg
theta, _ = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
keep = np.abs(np.pi / 2 - theta) > np.radians(BCUT)
fsky = keep.mean()
a = np.radians(90 - BCUT)                       # cap radius
L0, L1, L2 = 2.0, 0.5 * 2 * 2 * np.pi * np.sin(a), 2 * 2 * np.pi * (1 - np.cos(a))

nus = np.linspace(-3, 3, 25)
psi = lambda x: 0.5 * erfc(x / np.sqrt(2))
rho1 = lambda x: np.sqrt(lam) / (2 * np.pi) * np.exp(-x ** 2 / 2)
rho2 = lambda x: lam / (2 * np.pi) ** 1.5 * x * np.exp(-x ** 2 / 2)
gkf_mask = lambda x: L0 * psi(x) + L1 * rho1(x) + L2 * rho2(x)
naive = lambda x: fsky * (2 * psi(x) + 4 * np.pi * rho2(x))

msh = mesh(NSIDE)
CHI = np.zeros((NSIM, len(nus)))
for s in range(NSIM):
    t, _ = gaussian_field_sphere(cl, NSIDE, rng, lmax=LMAX)
    u = (t - t[keep].mean()) / t[keep].std()   # standardised with the unmasked pixels only
    CHI[s] = [chi_sphere((u > nu) & keep, msh) for nu in nus]

m, se = CHI.mean(0), CHI.std(0, ddof=1) / np.sqrt(NSIM)
i0 = int(np.argmin(np.abs(nus)))
i1 = int(np.argmin(np.abs(nus - 1)))
nums = {"MkNside": NSIDE, "MkFwhm": int(FWHM), "MkNsim": NSIM, "MkBcut": int(BCUT),
        "MkFsky": round(100 * fsky, 1), "MkLam": round(lam, 0),
        "MkLone": round(L1, 2), "MkLtwo": round(L2, 2),
        "MkZeroMC": round(m[i0], 1), "MkZeroSe": round(se[i0], 1), "MkZeroTh": round(gkf_mask(0.0), 1),
        "MkZeroNaive": round(naive(0.0), 2), "MkEdgeZero": round(L1 * rho1(0.0), 1),
        "MkOneMC": round(m[i1], 0), "MkOneTh": round(gkf_mask(1.0), 0), "MkOneNaive": round(naive(1.0), 0),
        "MkOneSd": round(CHI[:, i1].std(ddof=1), 0),
        "MkMaxPull": round(float(np.max(np.abs((m - gkf_mask(nus)) / se))), 1)}
save_numbers("ch12", "08_mask_gkf", nums)
for k, v in nums.items():
    print(k, v)

fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.3))
xx = np.linspace(-3.2, 3.2, 300)
ax = axs[0]
ax.errorbar(nus, m, yerr=CHI.std(0, ddof=1), fmt="o", ms=3, color=SERIES[0], elinewidth=0.7,
            label=f"masked skies, mean of {NSIM} (bar: one sky)")
theory_line(ax, xx, gkf_mask(xx), label=r"$\mathcal{L}_0\rho_0+\mathcal{L}_1\rho_1+\mathcal{L}_2\rho_2$")
ax.plot(xx, naive(xx), color=SERIES[1], lw=1.2, ls=":", label=r"$f_{\rm sky}\times$ full-sky formula")
ax.set_xlabel(r"$\nu$")
ax.set_ylabel(r"$\chi$ of the unmasked sky")
ax.legend(fontsize=7, loc="upper left")
ax = axs[1]
ax.plot(nus, m - naive(nus), "o", ms=3, color=SERIES[0], label="simulations minus naive")
ax.plot(xx, L1 * rho1(xx) + L0 * psi(xx) - fsky * 2 * psi(xx), color="k", ls="--", lw=1.2,
        label=r"boundary term $\mathcal{L}_1\rho_1$ (+ small $\rho_0$ part)")
ax.set_xlabel(r"$\nu$")
ax.set_ylabel(r"excess $\chi$")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch12", "mask_gkf")
