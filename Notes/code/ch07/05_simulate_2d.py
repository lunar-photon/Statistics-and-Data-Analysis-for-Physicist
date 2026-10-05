"""05_simulate_2d.py -- simulate 2-D Gaussian random fields by FFT and check P(k) and xi(r).

Question: does "white noise x sqrt(P(k)) -> inverse FFT" really produce a field
with the power spectrum and correlation function we asked for?
Computes: power-law spectra P(k) = A k^n, n = -1, -2, -3, on a 512^2 periodic box of
unit cells (A chosen so that each box has unit variance), all from the SAME white
noise.  For each: the shell-averaged P_hat(k) of one map against the input P(k)
averaged over the same modes, and the measured xi_hat(r) against its exact box
expectation (1/V) sum_k P(k) e^{ik.r}.  Mean ratio P_hat/P over 20 maps per n, and the
map-to-map standard deviation of xi_hat(0) over the same 20 maps.
Writes: figures/ch07/simulate_2d.pdf, results/ch07/05_simulate_2d.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_fields import gaussian_field, measure_pk, measure_xi, xi_box, kgrid, radial_average

setup(9.0, 6.0)
N, L = 512, 512.0
ns = [-1, -2, -3]
out = {}
fig, ax = plt.subplots(2, 3)
kmag, _ = kgrid(N, L, 2, real=False)
names = ["One", "Two", "Three"]
for j, n in enumerate(ns):
    # amplitude for unit variance in the box: sigma^2 = (1/V) sum_k P(k)
    A = L**2 / np.sum(kmag[kmag > 0] ** n)
    P = lambda k, A=A, n=n: A * k ** n
    f = gaussian_field(P, N, L, d=2, rng=rng_for("ch07", "05_simulate_2d"))   # same noise each n
    ax[0, j].imshow(f, cmap="RdBu_r", origin="lower", vmin=-3, vmax=3)
    ax[0, j].set_title(rf"$P\propto k^{{{n}}}$, $\hat\sigma^2={f.var():.2f}$")
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([]); ax[0, j].grid(False)
    k, ph, nm, pt = measure_pk(f, L, nbins=30, log=True, P=P)
    ax[1, 0].loglog(k, ph, "o", ms=3, color=SERIES[j], label=f"$n={n}$")
    theory_line(ax[1, 0], k, pt, label="input" if j == 0 else None)
    # 20 independent maps: average ratio over all shells, and the map-to-map scatter of xi_hat(0)
    rng = rng_for("ch07", "05_simulate_2d", 10 + j)
    maps20 = gaussian_field(P, N, L, 2, rng, nsim=20)
    rat = [np.mean(measure_pk(g, L, nbins=30, log=True, P=P)[1] / pt) for g in maps20]
    out[f"Ratio{names[j]}"] = np.mean(rat)
    xi0 = np.array([measure_xi(g, L, nbins=120, rmax=120)[1][0] for g in maps20])
    out[f"XiZeroSd{names[j]}"] = xi0.std(ddof=1)
    r, xh = measure_xi(f, L, nbins=120, rmax=120)
    _, xbr = radial_average(xi_box(P, N, L, 2), L, nbins=120, rmax=120)   # exact expectation, same bins
    ax[1, 1 + (j > 0)].plot(r, xh, "-", color=SERIES[j], label=f"$n={n}$ measured")
    theory_line(ax[1, 1 + (j > 0)], r, xbr, label="box expectation" if j != 2 else None)
    out[f"XiZero{names[j]}"] = xh[0]
ax[1, 0].set_xlabel("$k$"); ax[1, 0].set_ylabel(r"$\hat P(k)$"); ax[1, 0].legend(fontsize=7)
for a in ax[1, 1:]:
    a.axhline(0, color="k", lw=0.5); a.set_xlabel("$r$ (cells)"); a.set_ylabel(r"$\hat\xi(r)$"); a.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch07", "simulate_2d")
save_numbers("ch07", "05_simulate_2d", {f"SevA{k}": v for k, v in out.items()})
