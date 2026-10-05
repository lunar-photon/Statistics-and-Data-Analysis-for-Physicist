"""20_leakage_1d.py -- mode coupling in one dimension: a pure wave seen through a window.

Question: a signal made of a single Fourier mode is multiplied by a window that keeps only part
of the domain.  Where does its power go in the Fourier spectrum, and how does a smooth (tapered)
edge change that?
Computes: f(x) = cos(2 pi k0 x / Lbox) on a periodic grid; windows keeping 40 per cent of the
box with sharp edges and with cosine-tapered edges; |FFT(W f)|^2; the fraction of the power
that lands more than 10 modes away from k0.
Writes: figures/ch08/leakage_1d.pdf, results/ch08/20_leakage_1d.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
import lib_masks as lm

setup()
N, K0 = 4096, 40
x = np.arange(N) / N                                  # box length 1, periodic
f = np.cos(2 * np.pi * K0 * x)
d = np.minimum(x - 0.3, 0.7 - x)                      # distance inside the kept interval [0.3, 0.7]
W_sharp = lm.taper(d, 0.0)
W_taper = lm.taper(d, 0.08)                           # cosine taper over 0.08 of the box

nums = {}
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0))
ax = axs[0]
ax.plot(x, W_sharp * f, color=SERIES[0], lw=0.8, label="sharp window")
ax.plot(x, W_taper * f - 2.6, color=SERIES[1], lw=0.8, label="tapered window")
ax.plot(x, W_sharp, color=SERIES[0], lw=1.4, ls="--")
ax.plot(x, W_taper - 2.6, color=SERIES[1], lw=1.4, ls="--")
ax.set_yticks([])
ax.set_xlabel("position $x$ (box length 1)")
ax.set_title("(a) the wave times the window", fontsize=9)
ax = axs[1]
k = np.arange(N // 2)
for W, c, lab, key in [(W_sharp, SERIES[0], "sharp", "Sharp"), (W_taper, SERIES[1], "tapered", "Taper")]:
    p = np.abs(np.fft.rfft(W * f))[: N // 2] ** 2
    p /= p.sum()
    ax.semilogy(k[:201], p[:201], color=c, lw=1.1, label=lab)
    far = p[np.abs(k - K0) > 10].sum()
    nums[f"EightBoneDfar{key}"] = lm.tex_sci(far)
    print(lab, "far fraction", far)
ax.axvline(K0, color="0.6", lw=0.6, ls=":")
ax.set_ylim(1e-12, 1)
ax.set_xlabel("Fourier mode $k$")
ax.set_ylabel("share of the power")
ax.set_title("(b) where the power of $k_0=40$ goes", fontsize=9)
ax.legend(loc="upper right")
savefig(fig, "ch08", "leakage_1d")
nums["EightBoneDkzero"] = K0
save_numbers("ch08", "20_leakage_1d", nums)
