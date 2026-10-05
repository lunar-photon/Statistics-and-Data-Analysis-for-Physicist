"""03_grb_source.py -- which source produced this gamma-ray photon?

Question: a gamma-ray telescope records photons (sky position r, energy E)
during a 100 s window after a gamma-ray burst.  Three sources can produce a
photon: the burst (GRB), a steady blazar 1.3 deg away, and the diffuse
background.  For each photon, how probable is each source?

Model (flat 10 x 10 deg sky patch, energies 0.1-100 GeV):
  prior      P(k) = mu_k / sum(mu)   (expected photon counts in the window)
  likelihood p(r, E | k) = p(E | k) * PSF(r - r_k ; sigma(E))   for the two point sources
             p(r, E | D) = p(E | D) / area                          for the diffuse background
  PSF: 2-D Gaussian, sigma(E) = 0.8 deg * (E / 1 GeV)^-0.8, floor 0.1 deg
  spectra: power laws dN/dE ~ E^-gamma, gamma = 2.0 (GRB), 2.4 (blazar), 2.6 (diffuse)

Computes: the posterior source probabilities for two example photons, the same
with the diffuse hypothesis forgotten, and a simulated window of photons coloured
by P(GRB | photon); the sum of these probabilities estimates the GRB photon count.

Writes: figures/ch05/grb_sky.pdf, results/ch05/03_grb_source.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch05", "03_grb_source")
setup()

HALF = 5.0                                   # patch is [-5, 5]^2 deg
AREA = (2 * HALF) ** 2
EMIN, EMAX = 0.1, 100.0                      # GeV
POS = {"GRB": np.array([0.0, 0.0]), "blazar": np.array([1.2, 0.5])}
GAMMA = {"GRB": 2.0, "blazar": 2.4, "diffuse": 2.6}
MU = {"GRB": 30.0, "blazar": 8.0, "diffuse": 40.0}   # expected photons in the window
SRC = ["GRB", "blazar", "diffuse"]


def sigma_psf(E):
    return np.maximum(0.8 * E ** -0.8, 0.1)


def p_energy(E, g):
    """Normalised power law E^-g on [EMIN, EMAX]."""
    norm = (EMAX ** (1 - g) - EMIN ** (1 - g)) / (1 - g)
    return E ** -g / norm


def psf(r, centre, E):
    s = sigma_psf(E)
    d2 = np.sum((r - centre) ** 2, axis=-1)
    return np.exp(-d2 / (2 * s ** 2)) / (2 * np.pi * s ** 2)


def likelihoods(r, E):
    """p(r, E | k) for each source k (columns in SRC order)."""
    return np.stack([p_energy(E, GAMMA["GRB"]) * psf(r, POS["GRB"], E),
                     p_energy(E, GAMMA["blazar"]) * psf(r, POS["blazar"], E),
                     p_energy(E, GAMMA["diffuse"]) / AREA], axis=-1)


def membership(r, E, use=(0, 1, 2)):
    prior = np.array([MU[s] for s in SRC])[list(use)]
    L = likelihoods(r, E)[..., list(use)]
    w = prior * L
    return w / w.sum(axis=-1, keepdims=True)


prior = np.array([MU[s] for s in SRC]) / sum(MU.values())

# two example photons
r1, E1 = np.array([0.7, 0.3]), 2.0      # between the two point sources, fairly hard
r2, E2 = np.array([-3.0, 2.0]), 0.3     # far from both, soft
m1 = membership(r1, E1)
m2 = membership(r2, E2)
m2_forgot = membership(r2, E2, use=(0, 1))       # diffuse background forgotten
L1 = likelihoods(r1, E1)

# simulate one window of photons from the model itself
def sample_energy(g, n):
    u = rng.uniform(size=n)                        # inverse-CDF sampling of a power law
    a, b = EMIN ** (1 - g), EMAX ** (1 - g)
    return (a + u * (b - a)) ** (1 / (1 - g))

phot_r, phot_E, phot_src = [], [], []
for k, s in enumerate(SRC):
    n = rng.poisson(MU[s])
    E = sample_energy(GAMMA[s], n)
    if s == "diffuse":
        r = rng.uniform(-HALF, HALF, size=(n, 2))
    else:
        r = POS[s] + rng.normal(size=(n, 2)) * sigma_psf(E)[:, None]
    phot_r.append(r); phot_E.append(E); phot_src += [k] * n
phot_r = np.concatenate(phot_r); phot_E = np.concatenate(phot_E); phot_src = np.array(phot_src)
inside = np.all(np.abs(phot_r) < HALF, axis=1)
phot_r, phot_E, phot_src = phot_r[inside], phot_E[inside], phot_src[inside]
w = membership(phot_r, phot_E)
n_grb_true = int(np.sum(phot_src == 0))
n_grb_est = w[:, 0].sum()

fig, ax = plt.subplots(figsize=(5.2, 4.4))
sc = ax.scatter(phot_r[:, 0], phot_r[:, 1], c=w[:, 0], cmap="Blues", vmin=0, vmax=1,
                s=10 + 12 * np.log10(phot_E / EMIN), edgecolor=INK2, linewidth=0.4)
for s, mk in [("GRB", "*"), ("blazar", "D")]:
    ax.plot(*POS[s], mk, color=SERIES[1], ms=9, mec="k", mew=0.5, label=s)
    ax.add_patch(plt.Circle(POS[s], sigma_psf(1.0), fill=False, ls="--", color=SERIES[1], lw=0.8))
ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal")
ax.set_xlabel("offset in RA (deg)"); ax.set_ylabel("offset in Dec (deg)")
cb = fig.colorbar(sc, ax=ax, shrink=0.85)
cb.set_label("P(GRB | photon)")
ax.legend(loc="lower left", fontsize=8)
savefig(fig, "ch05", "grb_sky")

save_numbers("ch05", "03_grb_source", {
    "FiveAGrbPriorG": prior[0], "FiveAGrbPriorB": prior[1], "FiveAGrbPriorD": prior[2],
    "FiveAGrbSigOne": sigma_psf(E1),
    "FiveAGrbLG": L1[0], "FiveAGrbLB": L1[1], "FiveAGrbLD": L1[2],
    "FiveAGrbMaG": m1[0], "FiveAGrbMaB": m1[1], "FiveAGrbMaD": m1[2],
    "FiveAGrbMbG": m2[0], "FiveAGrbMbB": m2[1], "FiveAGrbMbD": m2[2],
    "FiveAGrbForgotG": m2_forgot[0], "FiveAGrbForgotB": m2_forgot[1],
    "FiveAGrbNphot": len(phot_E), "FiveAGrbNtrue": n_grb_true, "FiveAGrbNest": n_grb_est,
})
