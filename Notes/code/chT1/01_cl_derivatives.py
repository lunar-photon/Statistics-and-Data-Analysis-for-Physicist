"""How does each LCDM parameter move the CMB spectra?

Question: if we nudge one of the six LCDM parameters by its Planck 2018 1-sigma
error, how much does C_ell change, and at which multipoles? These fractional
responses are (up to a factor delta) the Fisher derivatives dC_ell/dtheta of
Ch. 10, and curves that look alike are the degeneracies seen in the MCMC of Ch. 9.

Computes: lensed TT and EE C_ell from CAMB at the fiducial point and at
theta_i +/- delta_i for each parameter; the symmetric fractional response
    R_i(ell) = [C_ell(theta_i + delta_i) - C_ell(theta_i - delta_i)] / (2 C_ell(fid)),
and the (2l+1)-weighted cosine between the A_s and tau responses.

Writes: data/chT1/cl_derivs.npz (cache), figures/chT1/cl_response.pdf,
figures/chT1/as_tau_degeneracy.pdf, results/chT1/01_cl_derivatives.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, DATA
from camb_fiducial import FIDUCIAL

LMAX = 2500
CACHE = DATA / "chT1" / "cl_derivs.npz"

# Planck 2018 TT,TE,EE+lowE+lensing 68% errors (Planck 2018 VI, Table 1, Plik column).
# A_s is varied through ln(10^10 A_s) = 3.044 +/- 0.014, i.e. dA_s/A_s = 0.014.
STEP = dict(ombh2=0.00015, omch2=0.0012, H0=0.54, tau=0.0073,
            As=0.014 * FIDUCIAL["As"], ns=0.0042)
LABEL = dict(ombh2=r"$\Omega_b h^2$", omch2=r"$\Omega_c h^2$", H0=r"$H_0$",
             tau=r"$\tau$", As=r"$A_s$", ns=r"$n_s$")


def spectra(params):
    """Lensed TT, EE in muK^2 (raw C_ell) for ell = 0..LMAX."""
    import camb
    p = camb.set_params(**params, lmax=LMAX + 300, lens_potential_accuracy=1)
    r = camb.get_results(p)
    tot = r.get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=True)["total"][: LMAX + 1]
    return tot[:, 0], tot[:, 1]


def compute():
    tt0, ee0 = spectra(FIDUCIAL)
    out = dict(tt0=tt0, ee0=ee0)
    for name, d in STEP.items():
        for sgn, tag in ((+1, "p"), (-1, "m")):
            par = dict(FIDUCIAL)
            par[name] = FIDUCIAL[name] + sgn * d
            out[f"tt_{name}_{tag}"], out[f"ee_{name}_{tag}"] = spectra(par)
        print("done", name)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def main():
    z = dict(np.load(CACHE)) if CACHE.exists() else compute()
    ell = np.arange(LMAX + 1)
    sl = slice(2, LMAX + 1)
    R = {}
    with np.errstate(invalid="ignore", divide="ignore"):   # ell = 0, 1 have C_ell = 0
        for spec in ("tt", "ee"):
            for name in STEP:
                R[spec, name] = (z[f"{spec}_{name}_p"] - z[f"{spec}_{name}_m"]) / (2 * z[f"{spec}0"])

    # Figure 1: fractional response of TT and EE to a 1-sigma step in each parameter
    setup()
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)
    for ax, spec, title in zip(axes, ("tt", "ee"), ("TT", "EE")):
        for c, name in zip(SERIES, STEP):
            ax.plot(ell[sl], 100 * R[spec, name][sl], color=c, lw=1.2, label=LABEL[name])
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_xscale("log")
        ax.set_ylabel(rf"$\Delta C_\ell^{{{title}}}/C_\ell^{{{title}}}$ [%]")
        ax.set_ylim(-4, 4)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=6, loc="upper center",
               bbox_to_anchor=(0.55, 1.03))
    axes[1].set_xlabel(r"multipole $\ell$")
    savefig(fig, "chT1", "cl_response")

    # Figure 2: the A_s - tau degeneracy (A_s response against minus the tau response)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for ax, spec, title in zip(axes, ("tt", "ee"), ("TT", "EE")):
        ax.plot(ell[sl], 100 * R[spec, "As"][sl], color=SERIES[4], label=r"$+\delta A_s$")
        ax.plot(ell[sl], -100 * R[spec, "tau"][sl], color=SERIES[3], ls="--", label=r"$-\delta\tau$")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"multipole $\ell$")
    axes[0].set_ylabel(r"$\Delta C_\ell/C_\ell$ [%]")
    axes[0].set_ylim(-1, 2.5)
    axes[0].legend(loc="upper right")
    savefig(fig, "chT1", "as_tau_degeneracy")

    # numbers quoted in the text
    w = 2 * ell + 1
    hi = slice(30, LMAX + 1)
    lo = slice(2, 11)

    def cosine(a, b, s):
        return np.sum(w[s] * a[s] * b[s]) / np.sqrt(np.sum(w[s] * a[s] ** 2) * np.sum(w[s] * b[s] ** 2))

    save_numbers("chT1", "01_cl_derivatives", {
        "TOneLmax": LMAX,
        "TOneCosAsTauTT": f"{cosine(R['tt', 'As'], R['tt', 'tau'], hi):.3f}",
        "TOneCosAsTauEElow": f"{cosine(R['ee', 'As'], R['ee', 'tau'], lo):.2f}",
        "TOneAsRespHigh": f"{100 * np.mean(R['tt', 'As'][hi]):.2f}",
        "TOneTauRespHigh": f"{100 * np.mean(R['tt', 'tau'][hi]):.2f}",
        "TOneTauRespEElow": f"{100 * np.max(R['ee', 'tau'][lo]):.0f}",
        "TOneNsPivotEll": int(ell[sl][np.argmin(np.abs(R['tt', 'ns'][sl][:1500]))]),
        "TOneMaxRespOmc": f"{100 * np.max(np.abs(R['tt', 'omch2'][sl])):.1f}",
        # first acoustic peak of D_ell = ell(ell+1)C_ell/2pi, a sanity check on units and normalisation
        "TOneDpeakEll": int(ell[np.argmax((ell * (ell + 1) * z["tt0"])[:500])]),
        "TOneDpeak": f"{np.max((ell * (ell + 1) * z['tt0'] / (2 * np.pi))[:500]):.0f}",
    })


if __name__ == "__main__":
    main()
