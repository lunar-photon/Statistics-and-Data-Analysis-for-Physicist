"""E and B patterns on a small flat patch of sky, their mirror images, and a rotated E pattern.

Question: what do pure E and pure B polarization look like as fields of headless sticks,
what does a mirror do to each, and what does turning every stick by the same angle do
to a pure E pattern?

Computes (no randomness): on a flat periodic patch, Q and U from a chosen E(l), B(l) through
    Q(l) + i U(l) = exp(2 i phi_l) [E(l) + i B(l)],
i.e. Q = cos(2 phi_l) E - sin(2 phi_l) B and U = sin(2 phi_l) E + cos(2 phi_l) B; the stick angle
psi = atan2(U, Q) / 2 is measured from the first axis towards the second.
  * a single plane wave of E and of B along the first axis;
  * a localised blob of E and of B (sticks around a point), the mirror image of the B blob
    (theta_2 -> -theta_2, which sends U -> -U), and the E blob with every stick turned by 20 deg.
For the turned E blob it also recomputes E and B from the turned Q, U and checks the rule
E' = E cos 2beta - B sin 2beta, B' = E sin 2beta + B cos 2beta to machine precision.

Writes: figures/chT1/eb_patterns.pdf, results/chT1/05_eb_patterns.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES

N = 64                      # pixels per side
X = (np.arange(N) - N / 2) / N   # coordinates in units of the patch size
XX, YY = np.meshgrid(X, X, indexing="xy")   # XX varies along the first axis (columns)
LX = np.fft.fftfreq(N)[None, :] * np.ones((N, 1))
LY = np.fft.fftfreq(N)[:, None] * np.ones((1, N))
PHI = np.arctan2(LY, LX)   # direction of each wavevector


def qu_from_eb(E, B):
    """Q, U maps from E, B maps on the periodic patch (flat-sky spin-2 rule)."""
    Ek, Bk = np.fft.fft2(E), np.fft.fft2(B)
    Ek[0, 0] = Bk[0, 0] = 0.0        # the uniform mode carries no E/B direction
    c, s = np.cos(2 * PHI), np.sin(2 * PHI)
    Q = np.real(np.fft.ifft2(c * Ek - s * Bk))
    U = np.real(np.fft.ifft2(s * Ek + c * Bk))
    return Q, U


def eb_from_qu(Q, U):
    Qk, Uk = np.fft.fft2(Q), np.fft.fft2(U)
    c, s = np.cos(2 * PHI), np.sin(2 * PHI)
    return np.real(np.fft.ifft2(c * Qk + s * Uk)), np.real(np.fft.ifft2(-s * Qk + c * Uk))


def rotate(Q, U, beta):
    """Turn every stick by beta: Q + iU -> exp(2 i beta)(Q + iU)."""
    c, s = np.cos(2 * beta), np.sin(2 * beta)
    return c * Q - s * U, s * Q + c * U


def sticks(ax, Q, U, title, step=4, colour=SERIES[0], scale=None):
    P = np.hypot(Q, U)
    psi = 0.5 * np.arctan2(U, Q)
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))
    amp = P[sl] / P.max()
    u, v = amp * np.cos(psi[sl]), amp * np.sin(psi[sl])
    ax.quiver(XX[sl], YY[sl], u, v, color=colour, pivot="middle", headwidth=0,
              headlength=0, headaxislength=0, scale=scale or 12, width=0.012)
    ax.set_aspect("equal")
    ax.set_xlim(X[0], X[-1]); ax.set_ylim(X[0], X[-1])
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title, fontsize=9)


def main():
    setup(7.2, 5.0)
    zero = np.zeros((N, N))
    wave = np.cos(2 * np.pi * 3 * XX)                   # three wavelengths along the first axis
    blob = np.exp(-(XX**2 + YY**2) / (2 * 0.09**2))      # a localised lump of E or B

    Qe, Ue = qu_from_eb(wave, zero)
    Qb, Ub = qu_from_eb(zero, wave)
    Qeb, Ueb = qu_from_eb(blob, zero)
    Qbb, Ubb = qu_from_eb(zero, blob)
    # mirror theta_2 -> -theta_2: the field at (x, y) is the old field at (x, -y), and U flips sign
    idx = (-np.arange(N)) % N          # row i holds y_i; y_{N-i} = -y_i on the periodic grid
    Qm, Um = Qbb[idx, :], -Ubb[idx, :]
    beta = np.deg2rad(20.0)
    Qr, Ur = rotate(Qeb, Ueb, beta)

    # checks: mirror of the B blob is minus the B blob; rotation rule for E', B'
    Em, Bm = eb_from_qu(Qm, Um)
    mirror_err = np.max(np.abs(Bm + blob - blob.mean())) / blob.max()
    Er, Br = eb_from_qu(Qr, Ur)
    E0 = blob - blob.mean()
    rot_err = max(np.max(np.abs(Er - np.cos(2 * beta) * E0)), np.max(np.abs(Br - np.sin(2 * beta) * E0))) / blob.max()

    fig, axs = plt.subplots(2, 3)
    sticks(axs[0, 0], Qe, Ue, "(a) E plane wave", step=6, scale=20)   # sparser, shorter: sticks stay apart
    sticks(axs[0, 1], Qb, Ub, "(b) B plane wave", colour=SERIES[1], step=6, scale=20)
    sticks(axs[0, 2], Qeb, Ueb, "(c) E lump")
    sticks(axs[1, 0], Qbb, Ubb, "(d) B lump", colour=SERIES[1])
    sticks(axs[1, 1], Qm, Um, "(e) mirror image of (d)", colour=SERIES[1])
    sticks(axs[1, 2], Qr, Ur, r"(f) (c) with every stick turned by $20^\circ$", colour=SERIES[2])
    for ax in axs[0, :2]:
        ax.annotate("", xy=(0.42, -0.45), xytext=(0.22, -0.45),
                    arrowprops=dict(arrowstyle="->", color="0.3"))
        ax.text(0.32, -0.42, r"$\boldsymbol{\ell}$", ha="center", va="bottom", fontsize=9, color="0.3")
    fig.tight_layout()
    savefig(fig, "chT1", "eb_patterns")
    save_numbers("chT1", "05_eb_patterns", {
        "TOneCMirrorErr": float(f"{mirror_err:.1e}") if mirror_err > 0 else 0.0,
        "TOneCRotErr": float(f"{rot_err:.1e}") if rot_err > 0 else 0.0,
    })
    print(f"mirror check {mirror_err:.2e}, rotation check {rot_err:.2e}")


if __name__ == "__main__":
    main()
