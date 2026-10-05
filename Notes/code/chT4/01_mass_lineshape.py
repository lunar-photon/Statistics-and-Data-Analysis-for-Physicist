"""01_mass_lineshape.py -- why a decaying particle shows up as a peak, and how wide the peak is.

Question: the two photons of H -> gamma gamma come out with energies that depend on how the Higgs
boson was moving, so why does their invariant mass always come out at 125 GeV? And once the
calorimeter measures the photon energies with a finite resolution, how wide is the peak, compared
with the natural width of the particle?

Computes:
  (a) 200 000 toy H -> gamma gamma decays: isotropic in the Higgs rest frame, then boosted with a
      toy transverse momentum (exponential, mean 25 GeV) and a toy rapidity (Gaussian, width 1.5).
      The invariant mass is recomputed from the lab-frame (pT, eta, phi) of the two photons.
  (b) the geometric acceptance of an ATLAS-like photon selection on the same toy events;
  (c) the photon energies smeared with the ATLAS design resolution 10%/sqrt(E) (+) 0.7%, the
      resulting mass width, and the error-propagation prediction for it;
  (d) line shapes: Breit-Wigner, Gaussian resolution and their convolution (Voigt) for the Z and
      for the Higgs; a Crystal Ball function against a Gaussian;
  (e) what three systematic effects do to a Crystal Ball signal template (sigma_CB = 1.6 GeV):
      a photon energy-scale shift of +-0.3%, a resolution change of +-14% and a yield change of
      +-11%, and the fraction of the signal inside a +-2 GeV window in each case.
Writes: figures/chT4/mass_from_photons.pdf, figures/chT4/line_shapes.pdf,
        figures/chT4/syst_variations.pdf, results/chT4/01_mass_lineshape.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile, erf
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

setup()
rng = rng_for("chT4", "01_mass_lineshape")
MH, N = 125.0, 200_000

# ---- (a) decays at rest: two back-to-back photons of energy MH/2, isotropic direction
cth = rng.uniform(-1, 1, N)
phi = rng.uniform(0, 2 * np.pi, N)
sth = np.sqrt(1 - cth**2)
n = np.stack([sth * np.cos(phi), sth * np.sin(phi), cth], axis=1)      # unit vector, photon 1
p1_rest = np.column_stack([np.full(N, MH / 2), MH / 2 * n])            # (E, px, py, pz)
p2_rest = np.column_stack([np.full(N, MH / 2), -MH / 2 * n])

# ---- the Higgs boson in the lab: toy pT and rapidity (a gluon-fusion-like spread, not a prediction)
ptH = rng.exponential(25.0, N)
phiH = rng.uniform(0, 2 * np.pi, N)
yH = rng.normal(0.0, 1.5, N)
mT = np.sqrt(MH**2 + ptH**2)                                           # transverse mass
PH = np.column_stack([mT * np.cosh(yH), ptH * np.cos(phiH), ptH * np.sin(phiH), mT * np.sinh(yH)])


def boost(p, P, M):
    """Lorentz-boost four-vectors p (given in the rest frame of P) into the frame where P is measured."""
    gamma = P[:, 0] / M
    bvec = P[:, 1:] / P[:, :1]                                         # velocity of the parent
    b2 = np.sum(bvec**2, axis=1)
    bp = np.sum(bvec * p[:, 1:], axis=1)
    E = gamma * (p[:, 0] + bp)
    coef = np.where(b2 > 0, (gamma - 1) * bp / np.where(b2 > 0, b2, 1), 0.0) + gamma * p[:, 0]
    return np.column_stack([E, p[:, 1:] + coef[:, None] * bvec])


p1, p2 = boost(p1_rest, PH, MH), boost(p2_rest, PH, MH)


def pt_eta_phi(p):
    pt = np.hypot(p[:, 1], p[:, 2])
    eta = np.arcsinh(p[:, 3] / pt)                                     # eta = asinh(pz/pT), massless
    return pt, eta, np.arctan2(p[:, 2], p[:, 1])


pt1, eta1, phi1 = pt_eta_phi(p1)
pt2, eta2, phi2 = pt_eta_phi(p2)
m_lab = np.sqrt(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)))   # PDG (49.45), massless
dev = np.max(np.abs(m_lab - MH))
E_lo, E_hi = np.percentile(np.concatenate([p1[:, 0], p2[:, 0]]), [5, 95])

# ---- (b) geometric/kinematic acceptance of an ATLAS-2012-like diphoton selection
lead, sub = np.maximum(pt1, pt2), np.minimum(pt1, pt2)
def in_calo(eta):
    a = np.abs(eta)
    return (a < 2.37) & ~((a > 1.37) & (a < 1.52))
acc_eta = in_calo(eta1) & in_calo(eta2)
acc = acc_eta & (lead > 40) & (sub > 30)
A_eta, A_all = acc_eta.mean(), acc.mean()
A_err = np.sqrt(A_all * (1 - A_all) / N)

# ---- (c) smear the photon energies; angles are kept exact (their effect is small, see text)
a_stoch, c_const = 0.10, 0.007
def rel_res(E):
    return np.sqrt(a_stoch**2 / E + c_const**2)
r1, r2 = rel_res(p1[:, 0]), rel_res(p2[:, 0])
k1 = 1 + r1 * rng.standard_normal(N)
k2 = 1 + r2 * rng.standard_normal(N)
m_meas = m_lab * np.sqrt(k1 * k2)                                      # m^2 is proportional to E1 E2
sel = acc
sig_meas = np.std(m_meas[sel])
sig_pred = MH * np.sqrt(np.mean(0.25 * (r1[sel]**2 + r2[sel]**2)))   # propagation, averaged over events
q16, q84 = np.percentile(m_meas[sel], [16, 84])
sig_eff = 0.5 * (q84 - q16)
r_typ = rel_res(MH / 2)

# ---- figure 1: photon energies are spread out, the pair mass is not
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
bins = np.linspace(0, 400, 81)
ax[0].hist(p1[:, 0], bins=bins, histtype="stepfilled", alpha=0.35, color=SERIES[0], label=r"photon 1, $E_1$")
ax[0].hist(p2[:, 0], bins=bins, histtype="step", color=SERIES[1], label=r"photon 2, $E_2$")
ax[0].axvline(MH / 2, color="k", ls=":", lw=1)
ax[0].text(MH / 2 + 5, ax[0].get_ylim()[1] * 0.85, r"$M/2$ at rest", fontsize=8)
ax[0].set_xlabel("photon energy in the lab [GeV]"); ax[0].set_ylabel("decays per 5 GeV")
ax[0].set_title("(a) energies depend on the boost")
ax[0].legend(loc="upper right")
mb = np.linspace(115, 135, 101)
ax[1].hist(m_meas[sel], bins=mb, histtype="stepfilled", alpha=0.35, color=SERIES[2],
           label="smeared energies")
xs = np.linspace(115, 135, 400)
w = mb[1] - mb[0]
theory_line(ax[1], xs, sel.sum() * w * np.exp(-0.5 * ((xs - MH) / sig_pred)**2) / (np.sqrt(2 * np.pi) * sig_pred),
            label="Gaussian, propagated width")
ax[1].axvline(MH, color=SERIES[1], lw=2, label="exact energies: every event")
ax[1].set_xlabel(r"$m_{\gamma\gamma}$ [GeV]"); ax[1].set_ylabel("decays per 0.2 GeV")
ax[1].set_title("(b) the pair mass is the same in every event")
ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.45)   # head-room so the legend clears the peak
ax[1].legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
savefig(fig, "chT4", "mass_from_photons")

# ---- (d) line shapes
def bw(x, M, G):
    return (G / 2) / np.pi / ((x - M)**2 + G**2 / 4)

def fwhm(x, y):
    """Full width at half maximum, with the two crossings found by linear interpolation."""
    h = y.max() / 2
    i = np.flatnonzero(y >= h)
    lo, hi = i[0], i[-1]
    xl = x[lo - 1] + (h - y[lo - 1]) * (x[lo] - x[lo - 1]) / (y[lo] - y[lo - 1])
    xr = x[hi] + (y[hi] - h) * (x[hi + 1] - x[hi]) / (y[hi] - y[hi + 1])
    return xr - xl

MZ, GZ, sZ = 91.1876, 2.4952, 1.5
xz = np.linspace(80, 102, 4001)
bz, gz, vz = bw(xz, MZ, GZ), np.exp(-0.5 * ((xz - MZ) / sZ)**2) / (np.sqrt(2 * np.pi) * sZ), voigt_profile(xz - MZ, sZ, GZ / 2)
GH, sH = 0.0041, 1.6
xh = np.linspace(115, 135, 40001)
vh = voigt_profile(xh - MH, sH, GH / 2)
gh = np.exp(-0.5 * ((xh - MH) / sH)**2) / (np.sqrt(2 * np.pi) * sH)
fw_bz, fw_gz, fw_vz = fwhm(xz, bz), fwhm(xz, gz), fwhm(xz, vz)
fw_vh, fw_gh = fwhm(xh, vh), fwhm(xh, gh)
vh_rel = np.max(np.abs(vh - gh)) / gh.max()

# Crystal Ball: Gaussian core for t > -alpha, power-law tail below; A, B from continuity of f and f'
alpha, ncb = 1.5, 5.0
Acb = (ncb / alpha)**ncb * np.exp(-alpha**2 / 2)
Bcb = ncb / alpha - alpha
def cb_shape(t):
    return np.where(t > -alpha, np.exp(-t**2 / 2), Acb * (Bcb - t)**(-ncb))
core = np.sqrt(np.pi / 2) * (1 + erf(alpha / np.sqrt(2)))             # integral of the core
tail = Acb * (Bcb + alpha)**(1 - ncb) / (ncb - 1)                    # integral of the tail
Ncb = 1 / (core + tail)
tail_frac = tail * Ncb
xc = np.linspace(-8, 5, 2001)
cb = Ncb * cb_shape(xc)
g0 = np.exp(-xc**2 / 2) / np.sqrt(2 * np.pi)
fw_cb = fwhm(xc, cb)

fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
ax[0].plot(xz, bz, color=SERIES[0], label=rf"Breit--Wigner, $\Gamma={GZ:.2f}$")
ax[0].plot(xz, gz, color=SERIES[1], label=rf"resolution, $\sigma={sZ}$")
ax[0].plot(xz, vz, color=SERIES[2], label="convolution (Voigt)")
ax[0].set_xlabel(r"$m_{ee}$ [GeV]"); ax[0].set_ylabel("density [1/GeV]")
ax[0].set_title(r"(a) Z: width and resolution comparable"); ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.55); ax[0].legend(fontsize=7.5, loc="upper left")
ax[1].plot(xh, gh, color=SERIES[1], label=rf"resolution, $\sigma={sH}$")
ax[1].plot(xh, vh, color=SERIES[2], ls="--", label=rf"Voigt with $\Gamma_H={GH*1e3:.1f}$ MeV")
ax[1].set_xlabel(r"$m_{\gamma\gamma}$ [GeV]"); ax[1].set_title("(b) Higgs: resolution only")
ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.55); ax[1].legend(fontsize=7.5, loc="upper left")
ax[2].semilogy(xc, g0, color=SERIES[1], label="Gaussian")
ax[2].semilogy(xc, cb, color=SERIES[3], label=rf"Crystal Ball, $\alpha={alpha}$, $n={ncb:g}$")
ax[2].axvline(-alpha, color="k", ls=":", lw=1)
ax[2].set_ylim(1e-6, 1); ax[2].set_xlabel(r"$t=(m-\bar m)/\sigma$")
ax[2].set_title("(c) a low-side tail"); ax[2].legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
savefig(fig, "chT4", "line_shapes")

# ---- (e) systematic variations of a signal template (Crystal Ball core 1.6 GeV, as ATLAS 2012)
SCB, ESC, RES, YLD = 1.6, 0.003, 0.14, 0.11
xm = np.linspace(110, 135, 2501)


def cb_mass(x, mean, sig):
    """Normalised Crystal Ball density in the mass x [1/GeV]."""
    return Ncb * cb_shape((x - mean) / sig) / sig


def win_frac(mean, sig, half=2.0):
    sel = np.abs(xm - MH) < half
    return np.trapezoid(cb_mass(xm, mean, sig)[sel], xm[sel])


nom = cb_mass(xm, MH, SCB)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
for a_, (lab, up, dn) in zip(ax, [
        (rf"energy scale $\pm{100*ESC:.1f}\%$", cb_mass(xm, MH * (1 + ESC), SCB), cb_mass(xm, MH * (1 - ESC), SCB)),
        (rf"resolution $\pm{100*RES:.0f}\%$", cb_mass(xm, MH, SCB * (1 + RES)), cb_mass(xm, MH, SCB * (1 - RES))),
        (rf"yield $\pm{100*YLD:.0f}\%$", (1 + YLD) * nom, (1 - YLD) * nom)]):
    a_.plot(xm, nom, color="k", lw=1.4, label="nominal")
    a_.plot(xm, up, color=SERIES[1], lw=1.2, label="$+1$ s.d.")
    a_.plot(xm, dn, color=SERIES[0], lw=1.2, ls="--", label="$-1$ s.d.")
    a_.set_xlim(117, 131); a_.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]"); a_.set_title(lab)
ax[0].set_ylabel("signal density [1/GeV]"); ax[0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
savefig(fig, "chT4", "syst_variations")
wf_nom, wf_up, wf_dn = win_frac(MH, SCB), win_frac(MH, SCB * (1 + RES)), win_frac(MH, SCB * (1 - RES))
wf_shift = win_frac(MH * (1 + ESC), SCB)

save_numbers("chT4", "01_mass_lineshape", {
    "TfWinNom": 100 * wf_nom, "TfWinResUp": 100 * wf_up, "TfWinResDn": 100 * wf_dn,
    "TfWinShift": 100 * wf_shift, "TfShiftGeV": MH * ESC,
    "TfNdecay": N, "TfMaxDev": dev, "TfElo": E_lo, "TfEhi": E_hi,
    "TfAeta": A_eta, "TfAall": A_all, "TfAerr": A_err,
    "TfSigMeas": sig_meas, "TfSigPred": sig_pred, "TfSigEff": sig_eff,
    "TfRtyp": 100 * r_typ, "TfRtypM": 100 * r_typ / np.sqrt(2),
    "TfFwBZ": fw_bz, "TfFwGZ": fw_gz, "TfFwVZ": fw_vz, "TfFwVH": fw_vh, "TfFwGH": fw_gh,
    "TfVHrel": vh_rel, "TfAcb": Acb, "TfBcb": Bcb, "TfTailFrac": 100 * tail_frac, "TfFwCB": fw_cb,
})
print(f"max |m-M| = {dev:.2e} GeV; E 5-95%: {E_lo:.1f}-{E_hi:.1f}; acceptance {A_all:.3f} (eta only {A_eta:.3f})")
print(f"sigma_m measured {sig_meas:.3f}, predicted {sig_pred:.3f}, half 68% {sig_eff:.3f}")
print(f"FWHM Z: BW {fw_bz:.3f}, G {fw_gz:.3f}, V {fw_vz:.3f}; H: V {fw_vh:.4f}, G {fw_gh:.4f}; CB tail {tail_frac:.3f}")
