"""Linear maps move covariance matrices as C' = A C A^T; eigenvectors of C decorrelate.

Question: two detector channels give (X1, X2) with covariance C.  We form the sum and the
difference, Y = A X with A = [[1, 1], [1, -1]].  Is the sample covariance of Y equal to
A C A^T?  And does rotating into the eigenbasis of C (A = R^T, rows = eigenvectors) give
uncorrelated variables whose variances are the eigenvalues?
We generate X = L Z with Z ~ N(0, 1) and L the Cholesky factor of C (itself an instance of
C = L 1 L^T).
Writes: figures/ch01/covariance_transform.pdf, results/ch01/18_covariance_transform.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch01", "18_covariance_transform")
setup(7.0, 3.2)

C = np.array([[4.0, 1.8], [1.8, 1.0]])
L = np.linalg.cholesky(C)
N = 200_000
Z = rng.normal(size=(2, N))
X = L @ Z
C_hat = np.cov(X)

A = np.array([[1.0, 1.0], [1.0, -1.0]])
Y = A @ X
CY_pred = A @ C @ A.T
CY_hat = np.cov(Y)

lam, R = np.linalg.eigh(C)          # columns of R are unit eigenvectors, lam ascending
W = R.T @ X                         # coordinates along the eigenvectors
CW_hat = np.cov(W)


def ellipse(Cm, nsig=1.0, n=200):
    lam_, R_ = np.linalg.eigh(Cm)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])
    return R_ @ (nsig * np.sqrt(lam_)[:, None] * circle)


fig, axes = plt.subplots(1, 3)
for ax, data, Cm, title, lab in [
        (axes[0], X, C, "$X$: covariance $C$", ("$x_1$", "$x_2$")),
        (axes[1], Y, CY_pred, "$Y=AX$: $ACA^{T}$", ("$y_1=x_1+x_2$", "$y_2=x_1-x_2$")),
        (axes[2], W, np.diag(lam), "$W=R^{T}X$: diagonal", ("$w_1$", "$w_2$"))]:
    ax.plot(data[0, :2000], data[1, :2000], ".", ms=1.2, color=SERIES[0], alpha=0.4)
    for k in (1, 2):
        e = ellipse(Cm, k)
        ax.plot(e[0], e[1], color=SERIES[1], lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
# eigenvector arrows on the first panel
for j in range(2):
    v = R[:, j] * 2 * np.sqrt(lam[j])
    axes[0].annotate("", xy=v, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=SERIES[2], lw=1.5))
fig.tight_layout()
savefig(fig, "ch01", "covariance_transform")

f = lambda v: f"{v:.3f}"
save_numbers("ch01", "18_covariance_transform", {
    "OneBCovN": "2\\times10^{5}",
    "OneBChataa": f(C_hat[0, 0]), "OneBChatab": f(C_hat[0, 1]), "OneBChatbb": f(C_hat[1, 1]),
    "OneBCYaa": f(CY_hat[0, 0]), "OneBCYab": f(CY_hat[0, 1]), "OneBCYbb": f(CY_hat[1, 1]),
    "OneBCWaa": f(CW_hat[0, 0]), "OneBCWab": f(CW_hat[0, 1]), "OneBCWbb": f(CW_hat[1, 1]),
    "OneBLamA": f(lam[0]), "OneBLamB": f(lam[1]),
    "OneBLaa": f(L[0, 0]), "OneBLba": f(L[1, 0]), "OneBLbb": f(L[1, 1]),
})
