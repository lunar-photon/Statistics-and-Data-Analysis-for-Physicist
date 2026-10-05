"""09_mass_function.py -- the halo mass function from random walks.

Question: what fraction of the mass sits in haloes of each mass, if a halo forms wherever the
linear density, smoothed on the scale that holds its mass, first reaches the collapse threshold
delta_c? Does the Press-Schechter formula with its factor of 2 come out of a simulation of the
random walks, and how does it compare with the Sheth-Tormen fit to N-body haloes?
Computes:
  (1) 50,000 excursion-set walks with independent Gaussian steps (the sharp-k filter), their
      first up-crossings of delta_c, and the distribution of nu = delta_c / sqrt(S) at first
      crossing, against Press-Schechter; the fraction of walks above delta_c at a given S
      against the fraction that have crossed by then (the factor of 2);
  (2) walks read off NFIELD = 4 white-noise Gaussian random fields (160^3 cells each), each smoothed
      with a sharp-k filter and with a top-hat filter at 120 scales: the same field, two filters, two
      answers; errors from the field-to-field scatter;
  (3) Press-Schechter and Sheth-Tormen multiplicities in the variables of Sheth & Tormen (1999,
      Fig. 2), and dn/dlnM at z = 0 from the CAMB linear P(k).
Writes: figures/ch11/hmf_walks.pdf, figures/ch11/hmf_multiplicity.pdf, figures/ch11/hmf_nM.pdf,
results/ch11/09_mass_function.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from lib_web import (DELTA_C, ST_A, F_ps, F_st, first_crossings, dndlnM, m_star, sigma_M,
                     tophat_w, K_TAB, PK_TAB, RHO_M)
from lib_fields import kgrid

rng = rng_for("ch11", "09_mass_function")

# ---------------------------------------------------------------- (1) independent-step walks
S_grid = np.geomspace(1e-3, 60.0, 4000)
NW = 50000
s_up, _ = first_crossings(S_grid, NW, rng)
nu_fc = DELTA_C / np.sqrt(s_up[np.isfinite(s_up)])
nub = np.geomspace(0.3, 5.0, 31)
h, _ = np.histogram(nu_fc, bins=nub)
nuc = np.sqrt(nub[1:] * nub[:-1])
F_mc = h / NW / np.diff(nub)                       # fraction of walks (mass) per unit nu
F_mc_err = np.sqrt(h) / NW / np.diff(nub)
# the factor of 2 at one scale: fraction above delta_c at S versus fraction crossed by S
S_chk, NCHK = 4.0, 20000
s_chk, _ = first_crossings(np.linspace(S_chk / 2000, S_chk, 2000), NCHK, rng)
frac_crossed = float(np.mean(s_chk <= S_chk))
frac_above = float(np.mean(rng.standard_normal(NCHK) * np.sqrt(S_chk) >= DELTA_C))
nu_chk = DELTA_C / np.sqrt(S_chk)
example = np.cumsum(rng.standard_normal((5, 800)) * np.sqrt(S_chk / 800), axis=1)

# ---------------------------------------------------------------- (2) walks in Gaussian fields
# A white-noise field (P = const): sigma(R) falls as R^-3/2, so a factor 8 in nu needs only a
# factor 4 in R and the box holds many independent patches even at the largest R. For a sharp-k
# filter the first-crossing distribution does not depend on the spectrum, so the test is general.
Nf = 160
NFIELD = 4                                           # independent fields: the scatter between them is the error
kmag, _ = kgrid(Nf, float(Nf), 3)                    # unit cells
rng_fields = rng_for("ch11", "09_mass_function_fields")   # fields 2..NFIELD; field 1 keeps the original stream
fks = [np.fft.rfftn(rng.standard_normal((Nf, Nf, Nf)))]  # white noise, unit variance per cell
fks += [np.fft.rfftn(rng_fields.standard_normal((Nf, Nf, Nf))) for _ in range(NFIELD - 1)]


def walk_first_crossing(filters, fk, nu_min=0.35):
    """Smooth the same field with each filter in turn (S increasing); first up-crossing per cell.

    The amplitude is fixed afterwards: scaling the field by A is the same as a barrier delta_c/A,
    chosen so that the smallest scale reaches nu_min.
    """
    ds = [np.fft.irfftn(fk * W, s=(Nf,) * 3, axes=(0, 1, 2)).ravel().astype(np.float32) for W in filters]
    S_raw = np.array([d.var() for d in ds])
    b = nu_min * np.sqrt(S_raw[-1])                   # barrier in raw units
    first_S = np.full(Nf ** 3, np.inf)
    for d, S in zip(ds, S_raw):
        new = (d >= b) & ~np.isfinite(first_S)
        first_S[new] = S
    return DELTA_C / np.sqrt(first_S * DELTA_C ** 2 / b ** 2), S_raw


kc = np.geomspace(np.pi / 10.0, np.pi, 120)          # sharp-k: cut from wavelength 20 to 2 cells
Rs = np.geomspace(4.6, 1.15, 120)                    # top hat, radius in cells, same S range
F_each = {"sharp": [], "top": []}                    # F(nu) of each field separately
for fk in fks:
    nu_sharp, _ = walk_first_crossing([(kmag <= k_).astype(float) for k_ in kc], fk)
    nu_top, _ = walk_first_crossing([tophat_w(kmag * r) for r in Rs], fk)
    for key, nuf in [("sharp", nu_sharp), ("top", nu_top)]:
        nuf = nuf[np.isfinite(nuf) & (nuf > 0)]
        hh, _ = np.histogram(nuf, bins=nub)
        F_each[key].append(hh / Nf ** 3 / np.diff(nub))
# mean over fields, with the error of the mean from the field-to-field scatter (cells of one field are not
# independent, so Poisson errors on the cell counts would be far too small)
F_field = {key: (np.mean(v, axis=0), np.std(v, axis=0, ddof=1) / np.sqrt(NFIELD)) for key, v in F_each.items()}
nu_field_min = 0.35
nu_field_max = 3.5

# ---------------------------------------------------------------- (3) PS and ST; dn/dlnM
M = np.geomspace(1e9, 3e15, 300)
n_st, nu_M, sig_M = dndlnM(M, "st")
n_ps, _, _ = dndlnM(M, "ps")
mst = m_star()


def n_above(n, Mcut):
    sel = M >= Mcut
    return float(np.trapezoid(n[sel], np.log(M[sel])))


nu_fine = np.linspace(1e-4, 8, 20001)
massfrac_st = float(np.trapezoid(F_st(nu_fine), nu_fine))
nu12 = DELTA_C / sigma_M(1e12)[0]
sel12 = nu_fine >= nu12
mf12_ps = float(np.trapezoid(F_ps(nu_fine[sel12]), nu_fine[sel12]))
mf12_st = float(np.trapezoid(F_st(nu_fine[sel12]), nu_fine[sel12]))
ratio = lambda m: float(np.interp(np.log(m), np.log(M), n_st / n_ps))

# ---- figure 1: walks, and the factor of 2
setup(6.8, 2.8)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
Sx = np.linspace(S_chk / 800, S_chk, 800)
for i in range(5):
    axs[0].plot(Sx, example[i], lw=0.8, color=SERIES[i])
axs[0].axhline(DELTA_C, color="k", ls="--", lw=1)
axs[0].text(0.05, DELTA_C + 0.12, r"$\delta_c$", fontsize=9)
axs[0].set_xlabel(r"$S=\sigma^2(M)$  (large mass $\leftarrow\;\to$ small mass)")
axs[0].set_ylabel(r"$\delta(S)$")
axs[0].set_title("five walks", fontsize=9)
Sv = np.linspace(0.05, 8, 200)
axs[1].plot(Sv, 0.5 * erfc(DELTA_C / np.sqrt(2 * Sv)), color=SERIES[1], label=r"above $\delta_c$ at $S$")
axs[1].plot(Sv, erfc(DELTA_C / np.sqrt(2 * Sv)), color=SERIES[0], label=r"crossed $\delta_c$ by $S$")
axs[1].plot([S_chk], [frac_above], "o", color=SERIES[1], ms=5)
axs[1].plot([S_chk], [frac_crossed], "s", color=SERIES[0], ms=5)
axs[1].set_xlabel(r"$S$")
axs[1].set_ylabel("fraction of walks")
axs[1].legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch11", "hmf_walks")

# ---- figure 2: multiplicity, simulations against formulas (Sheth-Tormen Fig. 2 axes on the right)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
nn = np.linspace(0.2, 5.2, 300)
axs[0].errorbar(nuc, F_mc, F_mc_err, fmt="o", ms=3, color=SERIES[0], label="independent steps (50,000 walks)")
ok = (nuc > nu_field_min) & (nuc < nu_field_max)
axs[0].errorbar(nuc[ok], F_field["sharp"][0][ok], F_field["sharp"][1][ok], fmt="s", ms=3, color=SERIES[2],
                label=f"{NFIELD} fields, sharp-$k$ filter")
axs[0].errorbar(nuc[ok], F_field["top"][0][ok], F_field["top"][1][ok], fmt="^", ms=3, color=SERIES[1],
                label=f"{NFIELD} fields, top-hat filter")
theory_line(axs[0], nn, F_ps(nn), label=r"Press--Schechter $\sqrt{2/\pi}\,e^{-\nu^2/2}$")
axs[0].plot(nn, 0.5 * F_ps(nn), color="0.55", ls=":", lw=1.2, label="without the factor 2")
axs[0].set_yscale("log"); axs[0].set_ylim(2e-4, 1.5)
axs[0].set_xlabel(r"$\nu=\delta_c/\sigma$"); axs[0].set_ylabel(r"$F(\nu)$, mass fraction per unit $\nu$")
axs[0].legend(fontsize=6.2, loc="lower left")
nst = np.geomspace(0.1, 30, 300)          # Sheth & Tormen's variable nu' = (delta_c/sigma)^2
nu_ = np.sqrt(nst)
axs[1].loglog(nst, 0.5 * nu_ * F_ps(nu_), color=SERIES[0], ls=":", label="Press--Schechter")
axs[1].loglog(nst, 0.5 * nu_ * F_st(nu_), color=SERIES[1], label="Sheth--Tormen")
axs[1].loglog(nuc ** 2, 0.5 * nuc * F_mc, "o", ms=3, color=SERIES[0], label="random walks")
axs[1].set_xlim(0.1, 30); axs[1].set_ylim(1e-3, 0.4)
axs[1].set_xlabel(r"$(\delta_c/\sigma)^2$"); axs[1].set_ylabel(r"$\nu' f(\nu')$ in ST99 Fig.~2 units")
axs[1].legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "ch11", "hmf_multiplicity")

# ---- figure 3: dn/dlnM
fig, axs = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
axs[0].loglog(M, n_ps, color=SERIES[0], ls=":", label="Press--Schechter")
axs[0].loglog(M, n_st, color=SERIES[1], label="Sheth--Tormen")
axs[0].axvline(mst, color="0.6", lw=0.8, ls="--")
axs[0].text(mst * 1.2, 1e-7, r"$M_*$", fontsize=8)
axs[0].set_ylim(1e-9, 1e0)
axs[0].set_ylabel(r"$\mathrm{d}n/\mathrm{d}\ln M\ [h^3\mathrm{Mpc}^{-3}]$")
axs[0].legend(fontsize=8)
axs[1].semilogx(M, n_st / n_ps, color=SERIES[1])
axs[1].axhline(1, color="0.6", lw=0.8)
axs[1].set_ylabel("ST / PS")
axs[1].set_xlabel(r"$M\ [h^{-1}M_\odot]$")
fig.tight_layout()
savefig(fig, "ch11", "hmf_nM")

# agreement of the independent-step walks with PS over the well-sampled bins
good = h > 50
chi2 = float(np.sum(((F_mc - F_ps(nuc)) / F_mc_err)[good] ** 2))
# the factor of 2: binomial standard errors of the two fractions and of their ratio
se_above = np.sqrt(frac_above * (1 - frac_above) / NCHK)
se_crossed = np.sqrt(frac_crossed * (1 - frac_crossed) / NCHK)
ratio2 = frac_crossed / frac_above
ratio2_err = ratio2 * np.hypot(se_above / frac_above, se_crossed / frac_crossed)
# the two filters at nu ~ 2.5, field by field
i25 = int(np.argmin(abs(nuc - 2.5)))
r25 = {key: np.array([F[i25] for F in v]) / F_ps(nuc[i25]) for key, v in F_each.items()}
save_numbers("ch11", "09_mass_function", {
    "HmfNchk": NCHK, "HmfSeAbove": f"{se_above:.4f}", "HmfSeCrossed": f"{se_crossed:.4f}",
    "HmfRatioMeas": f"{ratio2:.3f}", "HmfRatioMeasErr": f"{ratio2_err:.3f}",
    "HmfNfield": NFIELD,
    "HmfSharpRatioTwoErr": f"{r25['sharp'].std(ddof=1) / np.sqrt(NFIELD):.2f}",
    "HmfTopRatioTwoErr": f"{r25['top'].std(ddof=1) / np.sqrt(NFIELD):.2f}",
    "HmfSharpRatioTwoSd": f"{r25['sharp'].std(ddof=1):.2f}",
    "HmfNwalk": NW, "HmfNstep": len(S_grid), "HmfSchk": S_chk, "HmfNuchk": nu_chk,
    "HmfAbove": frac_above, "HmfCrossed": frac_crossed,
    "HmfAboveTh": 0.5 * erfc(nu_chk / np.sqrt(2)), "HmfCrossedTh": erfc(nu_chk / np.sqrt(2)),
    "HmfChi": chi2, "HmfNbins": int(good.sum()),
    "HmfFieldN": Nf, "HmfNuFieldMin": nu_field_min,
    "HmfTopRatioTwo": f"{r25['top'].mean():.2f}",
    "HmfSharpRatioTwo": f"{r25['sharp'].mean():.2f}",
    "HmfNuTwo": float(nuc[np.argmin(abs(nuc - 2.5))]),
    "HmfAst": ST_A, "HmfMassFracST": massfrac_st,
    "HmfMstar": mst / 1e12, "HmfRhoM": RHO_M / 1e10,
    "HmfNuTwelve": nu12, "HmfMfTwelvePS": mf12_ps, "HmfMfTwelveST": mf12_st,
    "HmfNfourteenPS": n_above(n_ps, 1e14), "HmfNfourteenST": n_above(n_st, 1e14),
    "HmfNfifteenPS": n_above(n_ps, 1e15), "HmfNfifteenST": n_above(n_st, 1e15),
    "HmfRatioFifteen": ratio(1e15), "HmfRatioEleven": ratio(1e11), "HmfRatioThirteen": ratio(1e13),
    "HmfSigFifteen": float(sigma_M(1e15)[0]), "HmfSigEleven": float(sigma_M(1e11)[0]),
})
print("chi2", chi2, good.sum(), "above/crossed", frac_above, frac_crossed, "Mstar", mst,
      "top/PS at 2.5", F_field["top"][0][np.argmin(abs(nuc - 2.5))] / F_ps(2.5))
