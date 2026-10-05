"""04_pk_xi_pairs.py -- three power spectra, their correlation functions, and what the fields look like.

Question: how does the shape of P(k) show up in xi(r) and in the field itself?
Computes, in two dimensions (xi(r) = int k dk/2pi P(k) J0(kr)):
  (a) Gaussian correlation   xi = exp(-r^2/2 ell^2)  <->  P = 2 pi ell^2 exp(-k^2 ell^2/2);
  (b) exponential            xi = exp(-r/ell)         <->  P = 2 pi ell^2 / (1 + k^2 ell^2)^{3/2};
  (c) a band of preferred wavelength: P = A exp(-(k - k0)^2 / 2 w^2), k0 = 0.5, w = 0.05,
      whose xi oscillates (negative lobes) -- a field of stripes and cells.
The Hankel integral is evaluated numerically and compared with the analytic xi of (a), (b).
Writes: figures/ch07/pk_xi_pairs.pdf, results/ch07/04_pk_xi_pairs.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_fields import gaussian_field, xi_from_pk

setup(9.0, 5.6)
rng = rng_for("ch07", "04_pk_xi_pairs")
ell, k0, w = 4.0, 0.5, 0.05
Pa = lambda k: 2 * np.pi * ell**2 * np.exp(-k**2 * ell**2 / 2)
Pb = lambda k: 2 * np.pi * ell**2 / (1 + (k * ell) ** 2) ** 1.5
norm_c = 1 / np.trapezoid(np.linspace(0, 3, 30000) * np.exp(-(np.linspace(0, 3, 30000) - k0) ** 2 / (2 * w**2)) / (2 * np.pi),
                          np.linspace(0, 3, 30000))
Pc = lambda k: norm_c * np.exp(-(k - k0) ** 2 / (2 * w**2))          # normalised to unit variance
specs = [("Gaussian", Pa, lambda r: np.exp(-r**2 / (2 * ell**2))),
         ("exponential", Pb, lambda r: np.exp(-r / ell)),
         ("band at $k_0$", Pc, None)]

r = np.linspace(0, 40, 400)
out = {}
fig = plt.figure()
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05])
N, L = 256, 256.0
for j, (name, P, xa) in enumerate(specs):
    a = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
    f = gaussian_field(P, N, L, d=2, rng=rng_for("ch07", "04_pk_xi_pairs", 1))  # same noise, three colourings
    a.imshow(f[:128, :128], cmap="RdBu_r", origin="lower", vmin=-2.5 * f.std(), vmax=2.5 * f.std())
    a.set_title(f"({'abc'[j]}) {name}"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
axP = fig.add_subplot(gs[1, :3]); axX = fig.add_subplot(gs[1, 3:])
k = np.logspace(-2, 1, 400)
for j, (name, P, xa) in enumerate(specs):
    axP.loglog(k, P(k), color=SERIES[j], label=f"({'abc'[j]})")
    xn = xi_from_pk(P, r, d=2, kmax=10.0, nk=40000)
    axX.plot(r, xn, color=SERIES[j], label=f"({'abc'[j]}) Hankel integral")
    if xa is not None:
        theory_line(axX, r[::12], xa(r[::12]), label="analytic" if j == 0 else None)
        out[f"HankelErr{'AB'[j]}"] = np.abs(xn - xa(r)).max()
xc = xi_from_pk(Pc, r, d=2, kmax=10.0, nk=40000)
imin = np.argmin(xc)
out["BandXiMin"] = xc[imin]
out["BandRMin"] = r[imin]
out["BandFirstZero"] = 2.405 / k0                 # zeros of J0(k0 r): k0 r = 2.405 ...
out["BandRMinJ"] = 3.832 / k0                     # first minimum of J0 at k0 r = 3.832
axP.set_ylim(1e-3, 3e3)
axP.set_xlabel("$k$ (rad/cell)"); axP.set_ylabel("$P(k)$ (cell$^2$)"); axP.legend()
axX.axhline(0, color="k", lw=0.6)
axX.set_xlabel("$r$ (cells)"); axX.set_ylabel(r"$\xi(r)$"); axX.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch07", "pk_xi_pairs")
save_numbers("ch07", "04_pk_xi_pairs", {f"SevA{k}": v for k, v in out.items()})
