"""14_pivot.py -- why Planck quotes A_s and n_s at the pivot k_* = 0.05 / Mpc.

Question: the primordial spectrum P(k) = A_s (k/k_0)^(n_s - 1) can be written about any
pivot k_0. At which pivot are the amplitude ln A_s(k_0) and the tilt n_s uncorrelated for a
temperature power-spectrum measurement, and why does it sit near 0.05 / Mpc?
Computes: lensed TT spectra from CAMB with central differences in (ln A_s, n_s) at the pivots
0.05 and 0.002 / Mpc, and in (H0, ombh2, omch2, tau) (with a Gaussian prior of 0.0073 on tau); the Fisher matrix
F_ij = sum_l (2l+1) fsky/2 dC/dth_i dC/dth_j / (C_l + N_l/B_l^2)^2 for a Planck-like
experiment (7' beam, 33 muK arcmin, fsky = 0.6, 2 <= l <= 2500) and for cosmic-variance-limited
spectra cut at l_max = 500; the correlation of (ln A(k), n_s) as a function of the pivot k by
the linear change of variables ln A(k) = ln A(k_0) + (n_s - 1) ln(k/k_0), checked against the
Fisher matrix computed directly at the second pivot; the decorrelation pivot
k_dec = k_0 exp(-Cov/Var); the comoving distance to last scattering D_*, and l = k D_*.
Writes: figures/ch08/pivot.pdf, results/ch08/14_pivot.tex, data/ch08/pivot_cls.npz
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES, INK2, DATA

import numpy as np
import matplotlib.pyplot as plt
from camb_fiducial import FIDUCIAL

LMAX = 2500
K_PLANCK, K_WMAP = 0.05, 0.002
CACHE = DATA / "ch08" / "pivot_cls.npz"
SIG_TAU = 0.0073                   # Planck 2018 low-l polarisation error on tau, used as a Gaussian prior
STEPS = dict(lnAs=0.02, ns=0.01, H0=0.5, ombh2=0.0003, omch2=0.003, tau=0.01)


def tt(lnAs=None, ns=None, pivot=K_PLANCK, **change):
    """Lensed raw TT C_l (muK^2), l = 0..LMAX, for the fiducial model with some parameters changed.
    lnAs is ln A_s at the given pivot; by default A_s(pivot) of the fiducial power law."""
    import camb
    p = dict(FIDUCIAL, **change)
    p.pop("As"), p.pop("ns")
    n = FIDUCIAL["ns"] if ns is None else ns
    # the fiducial power law, P(k) = A_s (k/0.05)^(n_s-1), has amplitude A_s (pivot/0.05)^(n_s-1) at the pivot
    A = FIDUCIAL["As"] * (pivot / K_PLANCK) ** (FIDUCIAL["ns"] - 1) if lnAs is None else np.exp(lnAs)
    pars = camb.set_params(**p, As=A, ns=n, pivot_scalar=pivot, lmax=LMAX + 500,
                           lens_potential_accuracy=1)
    res = camb.get_results(pars)
    cl = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["total"][: LMAX + 1, 0]
    return cl, res


def derivatives():
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    out = {}
    c0, res = tt()
    out["C0"] = c0
    zstar = res.get_derived_params()["zstar"]
    out["Dstar"] = np.array(res.comoving_radial_distance(zstar))     # Mpc (flat: = D_A comoving)
    for piv, tag in [(K_PLANCK, "p05"), (K_WMAP, "p002")]:
        lnA0 = np.log(FIDUCIAL["As"] * (piv / K_PLANCK) ** (FIDUCIAL["ns"] - 1))
        h = STEPS["lnAs"]
        out[f"dlnAs_{tag}"] = (tt(lnAs=lnA0 + h, pivot=piv)[0] - tt(lnAs=lnA0 - h, pivot=piv)[0]) / (2 * h)
        h = STEPS["ns"]
        out[f"dns_{tag}"] = (tt(lnAs=lnA0, ns=FIDUCIAL["ns"] + h, pivot=piv)[0]
                             - tt(lnAs=lnA0, ns=FIDUCIAL["ns"] - h, pivot=piv)[0]) / (2 * h)
    for name in ["H0", "ombh2", "omch2", "tau"]:
        h = STEPS[name]
        out[f"d{name}"] = (tt(**{name: FIDUCIAL[name] + h})[0] - tt(**{name: FIDUCIAL[name] - h})[0]) / (2 * h)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def fisher(dcl, C, Ntot, lmin=2, lmax=LMAX, fsky=0.6):
    """F_ij = sum_l (2l+1) fsky / 2 * dC_i dC_j / (C + N/B^2)^2  (the C_l Fisher information times the chain rule)."""
    l = np.arange(len(C))
    sel = (l >= lmin) & (l <= lmax)
    w = (2 * l[sel] + 1) * fsky / 2 / (C[sel] + Ntot[sel]) ** 2
    D = np.array([d[sel] for d in dcl])
    return (D * w) @ D.T


def corr_vs_pivot(cov2, k0, ks):
    """Correlation of (ln A(k), n) for pivots ks, given the 2x2 covariance of (ln A(k0), n)."""
    L = np.log(ks / k0)
    vA = cov2[0, 0] + 2 * L * cov2[0, 1] + L**2 * cov2[1, 1]
    cAn = cov2[0, 1] + L * cov2[1, 1]
    return cAn / np.sqrt(vA * cov2[1, 1])


def kdec(cov2, k0):
    return k0 * np.exp(-cov2[0, 1] / cov2[1, 1])


d = derivatives()
C = d["C0"]
ell = np.arange(len(C))
Dstar = float(d["Dstar"])

# Planck-like noise on the sky, N_l / B_l^2
fwhm = np.radians(7.0 / 60)
sb = fwhm / np.sqrt(8 * np.log(2))
Nl = (33.0 * np.radians(1 / 60)) ** 2
Nsky = Nl * np.exp(ell * (ell + 1) * sb**2)
zero = np.zeros_like(C)

others = [d["dH0"], d["dombh2"], d["domch2"], d["dtau"]]
cases = {}
for tag, k0 in [("p05", K_PLANCK), ("p002", K_WMAP)]:
    two = [d[f"dlnAs_{tag}"], d[f"dns_{tag}"]]
    cases[(tag, "cond")] = np.linalg.inv(fisher(two, C, Nsky))
    F6 = fisher(two + others, C, Nsky)
    F6[5, 5] += 1 / SIG_TAU**2          # TT alone cannot separate A_s from tau: large-angle polarisation fixes tau
    cases[(tag, "marg")] = np.linalg.inv(F6)[:2, :2]
    cases[(tag, "cv500")] = np.linalg.inv(fisher(two, C, zero, lmax=500))

rho = lambda c: c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
ks = np.logspace(np.log10(5e-4), np.log10(0.5), 300)

# the analytic shortcut: ln k_dec = information-weighted mean of ln(l / D_*)
sel = (ell >= 2)
wA = np.zeros_like(C)
wA[sel] = (2 * ell[sel] + 1) / 2 * (C[sel] / (C[sel] + Nsky[sel])) ** 2
ldec_approx = np.exp(np.sum(wA[sel] * np.log(ell[sel])) / np.sum(wA[sel]))

nums = {}
c05, c002 = cases[("p05", "cond")], cases[("p002", "cond")]
nums["EightAPivRhoFive"] = f"{rho(c05):.2f}"
nums["EightAPivRhoTwo"] = f"{rho(c002):.2f}"
nums["EightAPivRhoTwoTrans"] = f"{corr_vs_pivot(c05, K_PLANCK, np.array([K_WMAP]))[0]:.2f}"
nums["EightAPivKdec"] = f"{kdec(c05, K_PLANCK):.3f}"
nums["EightAPivKdecFromTwo"] = f"{kdec(c002, K_WMAP):.3f}"
nums["EightAPivLdec"] = f"{kdec(c05, K_PLANCK) * Dstar:.0f}"
nums["EightAPivKdecMarg"] = f"{kdec(cases[('p05', 'marg')], K_PLANCK):.3f}"
nums["EightAPivRhoFiveMarg"] = f"{rho(cases[('p05', 'marg')]):.2f}"
nums["EightAPivRhoTwoMarg"] = f"{rho(cases[('p002', 'marg')]):.2f}"
nums["EightAPivKdecCV"] = f"{kdec(cases[('p05', 'cv500')], K_PLANCK):.3f}"
nums["EightAPivLdecCV"] = f"{kdec(cases[('p05', 'cv500')], K_PLANCK) * Dstar:.0f}"
nums["EightAPivRhoTwoCV"] = f"{rho(cases[('p002', 'cv500')]):.2f}"
nums["EightAPivLdecApprox"] = f"{ldec_approx:.0f}"
nums["EightAPivDstar"] = f"{Dstar / 1000:.2f}"
nums["EightAPivLfive"] = f"{K_PLANCK * Dstar:.0f}"
nums["EightAPivLtwo"] = f"{K_WMAP * Dstar:.0f}"
nums["EightAPivSigNsFive"] = f"{np.sqrt(c05[1, 1]):.4f}"
nums["EightAPivSigAFive"] = f"{np.sqrt(c05[0, 0]):.4f}"
nums["EightAPivSigATwo"] = f"{np.sqrt(c002[0, 0]):.4f}"
save_numbers("ch08", "14_pivot", nums)
for k, v in nums.items():
    print(k, v)

# ---------------- figure ----------------
setup(9.0, 3.4)
fig, (a, b) = plt.subplots(1, 2)
for (tag, lab), c in zip([("cond", r"$(\ln A_s,n_s)$ only"),
                          ("marg", r"4 others marginalised"),
                          ("cv500", r"no noise, $\ell\leq500$")], SERIES):
    cov = cases[("p05", tag)]
    a.semilogx(ks, corr_vs_pivot(cov, K_PLANCK, ks), color=c, label=lab)
    a.plot([kdec(cov, K_PLANCK)], [0], "o", color=c, ms=5)
    if tag != "marg":
        cov2 = cases[("p002", tag)]
        a.plot([K_WMAP], [rho(cov2)], "s", mfc="none", color=c, ms=6)
a.plot([K_PLANCK], [rho(c05)], "s", mfc="none", color=SERIES[0], ms=6)
for kk, txt in [(K_WMAP, "0.002"), (K_PLANCK, "0.05")]:
    a.axvline(kk, color=INK2, lw=0.8, ls=":")
a.axhline(0, color=INK2, lw=0.6)
a.set_xlabel(r"pivot $k_0$ [Mpc$^{-1}$]")
a.set_ylabel(r"correlation of $\ln A(k_0)$ and $n_s$")
a.set_ylim(-1.05, 1.05)
a.legend(fontsize=7, loc="upper left", frameon=True, framealpha=1.0, edgecolor="none")
top = a.secondary_xaxis("top", functions=(lambda k: k * Dstar, lambda l: l / Dstar))
top.set_xlabel(r"$\ell\approx k_0D_*$")

wn = wA / wA.max()
b.plot(ell, wn, color=SERIES[0], label=r"$\mathcal{I}_\ell$")
b.plot(ell, (2 * ell + 1) / (2 * LMAX + 1), color=INK2, lw=0.8, ls="--", label="no noise")
b.axvline(kdec(c05, K_PLANCK) * Dstar, color=SERIES[1], lw=1.0, label=r"$\ell_{\rm dec}$")
b.axvline(K_PLANCK * Dstar, color=INK2, lw=0.8, ls=":")
b.set_xlabel(r"$\ell$"); b.set_ylabel("relative weight")
b.set_xlim(0, LMAX); b.set_ylim(0, 1.05)
b.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.35))
savefig(fig, "ch08", "pivot")
