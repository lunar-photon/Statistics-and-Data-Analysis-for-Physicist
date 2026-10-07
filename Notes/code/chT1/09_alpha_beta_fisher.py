"""How Galactic dust breaks the degeneracy between birefringence beta and a miscalibration alpha.

Question: the CMB alone fixes only the sum alpha + beta. If the same map also contains dust,
which is rotated by alpha but not by beta, how well can alpha and beta be told apart from the
EB spectrum of one frequency channel?

Model (a Fisher forecast, so no random numbers): one channel at frequency nu, a fraction
f_sky = 0.7 of the sky, white noise 50 muK arcmin and a 5 arcmin beam, l = 51..1500;
the observed EB spectrum is
    C_EB(alpha, beta) = 1/2 sin(4 alpha) (C_EE - C_BB)_dust + 1/2 sin(4 alpha + 4 beta) (C_EE - C_BB)_CMB,
with the dust EE from the Planck 2018 power-law fit over 71% of the sky (D_l = 315 muK^2 at
l = 80 and 353 GHz, D_l proportional to l^-0.42, BB/EE = 0.53), scaled in frequency by a
modified black body (beta_d = 1.53, T_d = 19.6 K).  Var(C_EB) = C_EE^tot C_BB^tot / ((2l+1) f_sky).
Computes the 2x2 Fisher matrix in (alpha, beta) for the CMB alone and for 143 and 353 GHz with
dust, the errors on alpha, beta and alpha + beta and the correlation coefficient, and draws the 1-sigma ellipses.

Writes: figures/chT1/alpha_beta_fisher.pdf, results/chT1/09_alpha_beta_fisher.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
from camb_fiducial import load_fiducial

FSKY, NOISE, FWHM = 0.7, 50.0, 5.0         # -, muK arcmin, arcmin
LMIN, LMAX = 51, 1500
A_EE353, ALPHA_L, BB_EE = 315.4, -0.42, 0.53
BETA_D, T_D, T_CMB, H_OVER_K = 1.53, 19.6, 2.7255, 0.0479924
ARCMIN = np.pi / (180 * 60)
DEG = np.pi / 180


def sed(nu):
    """Dust amplitude in CMB units, relative to 353 GHz."""
    def f(n):
        x = H_OVER_K * n / T_CMB
        rj = n ** (BETA_D + 1) / np.expm1(H_OVER_K * n / T_D)
        return rj * np.expm1(x) ** 2 / (x**2 * np.exp(x))
    return f(nu) / f(353.0)


def fisher(nu):
    L, EE = load_fiducial("EE")
    BB = load_fiducial("BB")[1]
    ell = np.arange(LMIN, LMAX + 1)
    ee, bb = EE[ell], BB[ell]
    if nu is None:
        de, db = 0 * ee, 0 * bb
    else:
        de = A_EE353 * sed(nu) ** 2 * (ell / 80.0) ** ALPHA_L * 2 * np.pi / (ell * (ell + 1))
        db = BB_EE * de
    nl = (NOISE * ARCMIN) ** 2 * np.exp(ell * (ell + 1) * (FWHM * ARCMIN) ** 2 / (8 * np.log(2)))
    var = (ee + de + nl) * (bb + db + nl) / ((2 * ell + 1) * FSKY)
    d_alpha = 2 * ((de - db) + (ee - bb))      # dC_EB/dalpha at alpha = beta = 0
    d_beta = 2 * (ee - bb)
    J = np.vstack([d_alpha, d_beta])
    return (J / var) @ J.T * DEG**2            # per degree^2


def ellipse(ax, F, colour, label):
    C = np.linalg.inv(F)
    w, v = np.linalg.eigh(C)
    t = np.linspace(0, 2 * np.pi, 200)
    pts = v @ (np.sqrt(w)[:, None] * np.vstack([np.cos(t), np.sin(t)]))
    ax.plot(pts[0], pts[1], color=colour, label=label)


def main():
    F0 = fisher(None)
    sig_sum = 1 / np.sqrt(F0[0, 0])            # all information is on alpha + beta
    out = {"TOneCSigSum": round(float(sig_sum), 4),
           "TOneCDetZero": float(f"{np.linalg.det(F0) / (F0[0, 0] * F0[1, 1]):.1e}")}
    setup(4.6, 4.0)
    fig, ax = plt.subplots()
    x = np.linspace(-1.5, 1.5, 2)
    ax.fill_between(x, -x - sig_sum, -x + sig_sum, color=SERIES[1], alpha=0.25, lw=0,
                    label=r"CMB alone: $\alpha+\beta$ only")
    for nu, col, name in [(143.0, SERIES[0], "OneFortyThree"), (353.0, SERIES[2], "ThreeFiftyThree")]:
        F = fisher(nu)
        C = np.linalg.inv(F)
        sa, sb = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
        rho = C[0, 1] / (sa * sb)
        ellipse(ax, F, col, r"with dust, %d GHz" % nu)
        out[f"TOneCSigA{name}"] = round(float(sa), 4)
        out[f"TOneCSigB{name}"] = round(float(sb), 4)
        out[f"TOneCRho{name}"] = round(float(rho), 3)
        out[f"TOneCSigSum{name}"] = round(float(np.sqrt(C[0, 0] + C[1, 1] + 2 * C[0, 1])), 3)
        print(nu, "sigma alpha", sa, "sigma beta", sb, "rho", rho)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$\beta$ [deg]")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    savefig(fig, "chT1", "alpha_beta_fisher")
    out["TOneCDustRatioFish"] = round(float((sed(353.0) / sed(143.0)) ** 2), 0)
    save_numbers("chT1", "09_alpha_beta_fisher", out)
    print("sigma(alpha+beta) CMB only:", sig_sum)


if __name__ == "__main__":
    main()
