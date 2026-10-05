"""03_fourier_diagonal.py -- Fourier modes diagonalise the covariance of a homogeneous field.

Question: why are the Fourier modes of a statistically homogeneous field uncorrelated?
Computes: on a periodic line of n = 64 cells, the covariance matrix
C_ij = xi(|i - j| periodic) with xi(r) = exp(-r^2 / 2 ell^2), ell = 3.  It is
circulant (each row is the previous row shifted by one).  We
  (a) show C in pixel space (a band, every pixel correlated with its neighbours);
  (b) rotate it to the Fourier basis, F C F^dagger / n, and find it diagonal to
      machine precision, with diagonal = DFT of the first row (Wiener-Khinchin);
  (c) check the same thing statistically: the sample covariance of the FFT
      coefficients of M = 20000 simulated lines is diagonal up to noise ~ 1/sqrt(M).
Writes: figures/ch07/fourier_diagonal.pdf, results/ch07/03_fourier_diagonal.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt

setup(9.6, 3.1)
rng = rng_for("ch07", "03_fourier_diagonal")
n, ell = 64, 3.0
i = np.arange(n)
dist = np.minimum(np.abs(i[:, None] - i[None, :]), n - np.abs(i[:, None] - i[None, :]))
C = np.exp(-dist**2 / (2 * ell**2))               # circulant covariance matrix

F = np.exp(-2j * np.pi * np.outer(i, i) / n)       # DFT matrix: (F x)_m = sum_j x_j e^{-2 pi i jm/n}
Ck = F @ C @ F.conj().T / n                        # covariance of the DFT coefficients / n
lam = np.fft.fft(C[0]).real                        # DFT of the first row = "power spectrum"
off = np.abs(Ck - np.diag(np.diag(Ck))).max()
out = {"OffDiagExact": off,
       "DiagMatch": np.abs(np.diag(Ck).real - lam).max(),
       "EigMatch": np.abs(np.sort(np.linalg.eigvalsh(C)) - np.sort(lam)).max(),
       "MinLam": lam.min()}

# statistical version: simulate with the Cholesky factor, never using Fourier
M = 20000
Lc = np.linalg.cholesky(C + 1e-12 * np.eye(n))
x = rng.standard_normal((M, n)) @ Lc.T
xk = np.fft.fft(x, axis=1)
S = (xk.T @ xk.conj()) / M / n                     # sample <x_k x_k'^*> / n
# correlation coefficients between distinct modes that carry real power; the pair (k, -k)
# is included: <x_k x_{-k}^*> = <x_k x_k> vanishes too (real and imaginary parts independent)
big = np.where(lam > 1e-3 * lam.max())[0]
dS = np.sqrt(np.abs(np.diag(S)))
R = np.abs(S) / np.outer(dS, dS)
mask = np.ones_like(R, bool)
np.fill_diagonal(mask, False)
sub = R[np.ix_(big, big)][mask[np.ix_(big, big)]]
out["NBig"] = big.size
out["OffDiagRMS"] = np.sqrt(np.mean(sub ** 2))
out["NoiseExpect"] = 1 / np.sqrt(M)
out["M"] = M

fig, ax = plt.subplots(1, 3)
ax[0].imshow(C, cmap="Blues", vmin=0, vmax=1)
ax[0].set_title(r"pixel space: $C_{ij}=\xi(|i-j|)$")
ax[0].set_xlabel("$j$"); ax[0].set_ylabel("$i$"); ax[0].grid(False)
im = ax[1].imshow(np.abs(np.fft.fftshift(S)), cmap="Blues", vmin=0, vmax=np.abs(S).max())
ax[1].set_title(r"Fourier space: $|\langle\delta_k\delta_{k'}^*\rangle|/n$ (sim.)")
ax[1].set_xlabel("$k'$ (shifted)"); ax[1].set_ylabel("$k$ (shifted)"); ax[1].grid(False)
kk = np.fft.fftshift(np.fft.fftfreq(n) * 2 * np.pi)
ax[2].plot(kk, np.fft.fftshift(np.diag(S).real), "o", ms=3, color=SERIES[0], label="diagonal (sim.)")
ax[2].plot(kk, np.fft.fftshift(lam), "-", color="k", lw=1.2, label="DFT of $\\xi$")
ax[2].plot(kk, np.sort(np.linalg.eigvalsh(C))[::-1][np.argsort(np.argsort(-np.fft.fftshift(lam)))], "x",
           color=SERIES[1], ms=4, label="eigenvalues of $C$")
ax[2].set_xlabel(r"$k$ (rad/cell)"); ax[2].set_ylabel("variance per mode")
ax[2].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch07", "fourier_diagonal")
save_numbers("ch07", "03_fourier_diagonal", {f"SevA{k}": v for k, v in out.items()})
