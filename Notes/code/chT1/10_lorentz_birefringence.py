"""Linear and circular birefringence in the Lorentz oscillator model, and the wavelength law of Faraday rotation.

Question: what does a medium with two different refractive indices do to a wave that is
linearly polarized at 45 degrees, when the two allowed polarizations (the eigenvectors of
the dielectric matrix) are (A) the axes x and y, or (B) the two circular helicities?
And how closely does the Faraday angle of a cold magnetized plasma follow the lambda^2 law?

Computes (no randomness):
  * case A: field E = (x + y e^{i delta}) / sqrt 2 after a retardance delta = 0, pi/4, ..., pi;
    the real field traced over one period is an ellipse;
  * case B: field = (e^{-i psi} e_+ e^{i phi_+} + e^{i psi} e_- e^{i phi_-}) / sqrt 2 with
    e_pm = (x pm i y)/sqrt 2 and phi_+ - phi_- = Delta = 0, pi/4, ..., pi; the real field is a line;
  * the exact Lorentz-model indices of a cold plasma with the field along the wave,
    n_pm^2 = 1 - w_p^2 / (w (w pm w_c)), and the rotation per unit length (n_+ - n_-) w / (2c),
    compared with the high-frequency law w_p^2 w_c / (2 c w^2) (units w_p = c = 1, w_c = 0.2);
  * the rotation-measure constant e^3 / (8 pi^2 eps0 m_e^2 c^3) in rad m^-2 per (cm^-3 microgauss pc).

Writes: figures/chT1/lorentz_birefringence.pdf, results/chT1/10_lorentz_birefringence.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as C
from common import setup, savefig, save_numbers, SERIES, theory_line

PHASES = np.pi * np.arange(5) / 4            # retardance (A) or helicity phase difference (B)
PHASE_LABELS = ["0", r"\pi/4", r"\pi/2", r"3\pi/4", r"\pi"]
T = np.linspace(0, 2 * np.pi, 400)           # one period of the wave, omega t


def trace(field):
    """Real electric field (x, y) over one period for complex amplitudes field = (Ex, Ey)."""
    z = np.exp(-1j * T)
    return np.real(field[0] * z), np.real(field[1] * z)


def case_a(delta):
    return np.array([1.0, np.exp(1j * delta)]) / np.sqrt(2)


def case_b(Delta, psi=np.pi / 4):
    ep = np.array([1.0, 1j]) / np.sqrt(2)
    em = np.array([1.0, -1j]) / np.sqrt(2)
    return (np.exp(-1j * psi) * ep * np.exp(1j * Delta) + np.exp(1j * psi) * em) / np.sqrt(2)


def plasma_rotation(w, wc=0.2):
    """Exact rotation per unit length (n_+ - n_-) w / 2, units w_p = c = 1."""
    npl = np.sqrt(1 - 1 / (w * (w + wc)))
    nmi = np.sqrt(1 - 1 / (w * (w - wc)))
    return (npl - nmi) * w / 2


def main():
    setup(6.2, 5.0)
    fig = plt.figure()
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.5], hspace=0.55, wspace=0.15)
    for row, (fn, colour, name, sym) in enumerate([(case_a, SERIES[0], "(a) axes x, y", r"\delta"),
                                                   (case_b, SERIES[1], "(b) two helicities", r"\Delta")]):
        for j, ph in enumerate(PHASES):
            ax = fig.add_subplot(gs[row, j])
            ax.plot([-0.8, 0.8], [-0.8, 0.8], color="0.6", ls=":", lw=0.9)   # starting polarization, 45 deg
            ex, ey = trace(fn(ph))
            ax.plot(ex, ey, color=colour, lw=1.6)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("0.75")
            ax.set_title(rf"${sym}={PHASE_LABELS[j]}$", fontsize=9)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)

    # (c) Faraday angle against wavelength^2: exact Lorentz plasma vs the lambda^2 law
    ax = fig.add_subplot(gs[2, :])
    wref = 10.0
    w = np.linspace(1.5, wref, 400)
    x = (wref / w) ** 2                       # (lambda / lambda_ref)^2
    exact = plasma_rotation(w) / plasma_rotation(wref)
    ax.plot(x, exact, color=SERIES[1], label="Faraday, exact Lorentz plasma")
    theory_line(ax, x, x, label=r"Faraday, $\lambda^2$ law")
    ax.plot(x, np.ones_like(x), color=SERIES[2], label="cosmic birefringence")
    ax.set_xlabel(r"$(\lambda/\lambda_{\rm ref})^2$")
    ax.set_ylabel("angle / angle at $\\lambda_{\\rm ref}$")
    ax.set_title(r"(c) rotation against wavelength squared ($\omega_{\rm ref}=10\,\omega_p$, $\omega_c=0.2\,\omega_p$)",
                 fontsize=9)
    ax.legend(loc="upper left")
    savefig(fig, "chT1", "lorentz_birefringence")

    # numbers for the text
    w3 = 3.0
    dev3 = plasma_rotation(w3) / (0.2 / (2 * w3 ** 2)) - 1      # exact / approx - 1 at w = 3 w_p
    dev10 = plasma_rotation(wref) / (0.2 / (2 * wref ** 2)) - 1
    rm_si = C.e ** 3 / (8 * np.pi ** 2 * C.epsilon_0 * C.m_e ** 2 * C.c ** 3)   # rad m^-2 per (m^-3 T m)
    rm_astro = rm_si * 1e6 * 1e-10 * C.parsec                     # per (cm^-3 microgauss pc)
    save_numbers("chT1", "10_lorentz_birefringence", {
        "TOneCFarDevThree": round(100 * dev3, 1),
        "TOneCFarDevTen": round(100 * dev10, 2),
        "TOneCRMconst": round(rm_astro, 3),
    })
    print(f"deviation from lambda^2 law at w=3 wp: {100*dev3:.2f}%, at w=10 wp: {100*dev10:.3f}%")
    print(f"RM constant {rm_astro:.4f} rad m^-2 per (cm^-3 muG pc)")


if __name__ == "__main__":
    main()
