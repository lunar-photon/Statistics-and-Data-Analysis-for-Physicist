"""Photometric redshifts: where their error distribution comes from, and what it does to n(z).

Question: a galaxy's redshift is estimated from six broad-band fluxes by fitting redshifted
templates. What is the error distribution of z_phot - z: a Gaussian core of what width, and how
many catastrophic outliers (a Lyman break mistaken for the 4000 A break)? How much does a prior
on n(z) help? If we select a tomographic bin by z_phot, how far is its true n(z) from the naive
one, and is <z | z_phot> equal to z_phot when <z_phot | z> = z?
Computes: toy two-template SEDs (red: strong 4000 A break; blue: weak), top-hat ugrizy fluxes
with Gaussian noise for 20000 galaxies, the likelihood on a redshift grid (amplitude and type
marginalised), posteriors with a flat and with the n(z) prior, scatter / outlier statistics, a
tomographic bin, and the two conditional means.
Writes: figures/chT2/photoz.pdf, results/chT2/11_photoz.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES

BANDS = [(3200, 4000), (4000, 5500), (5500, 6900), (6900, 8200), (8200, 9200), (9200, 10500)]  # ugrizy, Angstrom
DEPTH = np.array([24.6, 25.4, 25.4, 25.2, 24.6, 24.0])     # 5-sigma limiting AB magnitudes
ZG = np.arange(0.005, 4.0, 0.005)
NGAL = 20000


PARS = {"red": (1.5, 0.35), "blue": (0.2, 0.65)}             # (slope, flux ratio across the 4000 A break)


def sed(lam_rest, slope, brk):
    """Toy f_nu: a power law with a break at 4000 A, Lyman-alpha forest below 1216 A, nothing below 912 A."""
    f = (lam_rest / 5000.0) ** slope * np.where(lam_rest > 4000, 1.0, brk)
    f = f * np.where(lam_rest > 1216, 1.0, 0.25)
    return f * (lam_rest > 912)


def band_fluxes(z, slope, brk):
    """Mean f_nu through each top-hat band for sources at redshifts z (slope, brk scalars or arrays)."""
    slope, brk = np.broadcast_to(slope, z.shape)[:, None], np.broadcast_to(brk, z.shape)[:, None]
    out = []
    for lo, hi in BANDS:
        lam = np.linspace(lo, hi, 60)
        out.append(sed(lam[None, :] / (1 + z[:, None]), slope, brk).mean(axis=1))
    return np.array(out).T                                   # (nz, 6)


TEMPL = {k: band_fluxes(ZG, *v) for k, v in PARS.items()}

# --- the galaxy population: n(z) ~ z^2 exp[-(z/z0)^1.5], 35% red, i-band magnitude 21-24.5
rng = rng_for("chT2", "11_photoz")
Z0 = 0.55
nz = ZG**2 * np.exp(-((ZG / Z0) ** 1.5))
nz /= nz.sum()
ztrue = rng.choice(ZG, NGAL, p=nz) + rng.uniform(-0.0025, 0.0025, NGAL)
red = rng.uniform(0, 1, NGAL) < 0.35
imag = rng.uniform(21.0, 24.5, NGAL)
sig = 10 ** (-0.4 * (DEPTH - 25.0)) / 5                     # flux unit: AB mag 25 = 1
# each real galaxy differs a little from its template (template mismatch)
slope = np.where(red, PARS["red"][0], PARS["blue"][0]) + rng.normal(0, 0.15, NGAL)
brk = np.clip(np.where(red, PARS["red"][1], PARS["blue"][1]) + rng.normal(0, 0.04, NGAL), 0.1, 1.0)
model = band_fluxes(ztrue, slope, brk)
amp = 10 ** (-0.4 * (imag - 25.0)) / model[:, 3]
fobs = amp[:, None] * model + rng.normal(0, 1, (NGAL, 6)) * sig[None, :]


def likelihood(f):
    """L(z) for each galaxy: Gaussian flux errors, amplitude marginalised (flat prior), summed over types."""
    w = 1 / sig**2
    chis, norms = [], []
    for T in TEMPL.values():
        Tw = T * w                                           # (nz, 6)
        a = f @ Tw.T                                         # sum f T / sigma^2   (ngal, nz)
        b = (T * Tw).sum(1)                                  # sum T^2 / sigma^2   (nz,)
        chis.append((f**2 * w).sum(1)[:, None] - a**2 / b[None, :])   # chi^2 at the best amplitude
        norms.append(1 / np.sqrt(b))                         # width of the amplitude integral
    cmin = np.minimum(*chis).min(1, keepdims=True)           # common offset, for numerical safety
    return sum(np.exp(-0.5 * (c - cmin)) * n[None, :] for c, n in zip(chis, norms))


post_flat = np.zeros((NGAL, ZG.size))
for s in range(0, NGAL, 2000):
    L = likelihood(fobs[s:s + 2000])
    post_flat[s:s + 2000] = L / L.sum(1, keepdims=True)
post_prior = post_flat * nz[None, :]
post_prior /= post_prior.sum(1, keepdims=True)
zp_flat = ZG[post_flat.argmax(1)]
zp_prior = ZG[post_prior.argmax(1)]


def stats(zp):
    dz = (zp - ztrue) / (1 + ztrue)
    out = np.abs(dz) > 0.15
    core = dz[~out]
    nmad = 1.4826 * np.median(np.abs(core - np.median(core)))
    return nmad, out.mean(), np.median(core)


nmad_f, out_f, med_f = stats(zp_flat)
nmad_p, out_p, med_p = stats(zp_prior)

# --- a tomographic bin selected on z_phot (prior version, as surveys do)
lo, hi = 0.8, 1.2
inb = (zp_prior >= lo) & (zp_prior < hi)
nz_true_bin = np.histogram(ztrue[inb], bins=ZG[::4])[0]
stack = post_prior[inb].sum(0)
mean_true = ztrue[inb].mean()
mean_zp = zp_prior[inb].mean()
mean_stack = (stack * ZG).sum() / stack.sum()
frac_out_bin = np.mean(np.abs(ztrue[inb] - 1.0) > 0.5)

# --- the two conditional means:  <z_phot | z>  versus  <z | z_phot>
e = np.arange(0.2, 2.01, 0.2)
c = 0.5 * (e[1:] + e[:-1])
m_zp_given_z = np.array([zp_prior[(ztrue >= a) & (ztrue < b)].mean() for a, b in zip(e[:-1], e[1:])])
m_z_given_zp = np.array([ztrue[(zp_prior >= a) & (zp_prior < b)].mean() for a, b in zip(e[:-1], e[1:])])

# a galaxy whose flat-prior posterior is bimodal (Lyman/4000 A confusion)
peaks = []
for i in range(NGAL):
    p = post_flat[i]
    lowz, highz = p[ZG < 1.0].sum(), p[ZG > 1.8].sum()
    if 0.2 < lowz < 0.8 and highz > 0.2 and ztrue[i] > 2.0:
        peaks.append(i)
    if len(peaks) > 5:
        break
ib = peaks[0] if peaks else int(np.argmax(np.abs(zp_flat - ztrue)))

setup(7.2, 2.6)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax[0].scatter(ztrue, zp_flat, s=0.6, color=SERIES[0], alpha=0.35, lw=0, rasterized=True)
zz = np.linspace(0, 4, 2)
for sgn in (-1, 1):
    ax[0].plot(zz, zz + sgn * 0.15 * (1 + zz), color="0.4", lw=0.7, ls="--")
ax[0].set(xlabel=r"true $z$", ylabel=r"$z_{\rm phot}$ (flat prior)", xlim=(0, 3.5), ylim=(0, 3.5))
ax[1].plot(ZG, post_flat[ib] / post_flat[ib].max(), color=SERIES[3], label="flat prior")
ax[1].plot(ZG, post_prior[ib] / post_prior[ib].max(), color=SERIES[0], label=r"$n(z)$ prior")
ax[1].axvline(ztrue[ib], color="k", lw=0.8, ls=":")
ax[1].set(xlabel=r"$z$", ylabel=r"$p(z\,|\,\mathrm{fluxes})$ [scaled]", xlim=(0, 4))
ax[1].legend(fontsize=7, frameon=False, loc="center right")
w = np.diff(ZG[::4])[0]
ax[2].bar(0.5 * (ZG[::4][1:] + ZG[::4][:-1]), nz_true_bin / (nz_true_bin.sum() * w), width=w,
          color="0.75", label=r"true $z$ in the bin")
ax[2].plot(ZG, stack / (stack.sum() * 0.005), color=SERIES[0], lw=1.3, label=r"stacked $p(z)$")
ax[2].axvspan(lo, hi, color=SERIES[1], alpha=0.12, lw=0)
ax[2].set(xlabel=r"$z$", ylabel=r"$n(z)$", xlim=(0, 3.5))
ax[2].set_ylim(0, ax[2].get_ylim()[1] * 1.3)
ax[2].legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout()
savefig(fig, "chT2", "photoz")

save_numbers("chT2", "11_photoz", {
    "TwpN": NGAL, "TwpZzero": Z0, "TwpNmadF": f"{nmad_f:.3f}", "TwpOutF": f"{100 * out_f:.1f}",
    "TwpNmadP": f"{nmad_p:.3f}", "TwpOutP": f"{100 * out_p:.1f}", "TwpMedP": f"{med_p:.4f}",
    "TwpBinLo": lo, "TwpBinHi": hi, "TwpBinN": int(inb.sum()),
    "TwpMeanTrue": f"{mean_true:.3f}", "TwpMeanZp": f"{mean_zp:.3f}", "TwpMeanStack": f"{mean_stack:.3f}",
    "TwpBinOut": f"{100 * frac_out_bin:.1f}", "TwpZb": f"{ztrue[ib]:.2f}",
    "TwpCondA": f"{m_zp_given_z[6]:.3f}", "TwpCondB": f"{m_z_given_zp[6]:.3f}", "TwpCondZ": f"{c[6]:.1f}",
    "TwpCondAlow": f"{m_zp_given_z[1]:.3f}", "TwpCondBlow": f"{m_z_given_zp[1]:.3f}", "TwpCondZlow": f"{c[1]:.1f}",
})
print(f"flat: nmad {nmad_f:.4f} out {out_f:.4f} med {med_f:.4f}; prior: nmad {nmad_p:.4f} out {out_p:.4f} med {med_p:.4f}")
print(f"bin {lo}-{hi}: N {inb.sum()}, <z_true> {mean_true:.3f}, <zp> {mean_zp:.3f}, stack {mean_stack:.3f}, far {frac_out_bin:.3f}")
print("<zp|z>", np.round(m_zp_given_z, 3), "\n<z|zp>", np.round(m_z_given_zp, 3), "\ncentres", c, "bimodal", ib, ztrue[ib])
