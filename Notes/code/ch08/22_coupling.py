"""22_coupling.py -- the mode-coupling matrix M_ll' of each mask, from Wigner 3j symbols.

Question: how does each mask spread the power of one true multipole l' over the measured
pseudo-multipoles l, and how much of it leaks far from l'?
Computes: (l1 l2 l3; 0 0 0)^2 from the closed form in log space, checked against sympy's exact
values; M_ll' for l, l' <= 767 for every mask of 21_masks.py (and Xi_ll' = M_ll'/(2l'+1) built
from the spectrum of W^2, needed for the covariance approximation of 27_covariance.py);
the exact sum rules  sum_l' M_ll' = fsky w2  (white noise) and
sum_l (2l+1) M_ll' = (2l'+1) fsky w2 (power conservation); the far-leakage fraction; the
condition number of the per-multipole system.
Writes: data/ch08/coupling.npz, figures/ch08/coupling_columns.pdf, results/ch08/22_coupling.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
import lib_masks as lm

setup()
L = 767                                   # analysis band: l, l' <= 3 nside - 1
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"
z = np.load(DATA / "masks.npz")
KEYS = ["full", "cap10", "cap10apo", "cap30", "cap70", "band20", "band20apo", "holes"]
nums = {}

# ---------------------------------------------------------------- 1. the 3j symbols, checked
from sympy.physics.wigner import wigner_3j
checks = [(2, 2, 2), (2, 3, 5), (10, 20, 14), (50, 60, 30), (200, 150, 300), (400, 380, 30)]
worst = 0.0
for l1, l2, l3 in checks:
    exact = float(wigner_3j(l1, l2, l3, 0, 0, 0)) ** 2
    ours = lm.three_j_000_sq(l1, l2, l3)
    rel = abs(ours / exact - 1) if exact else abs(ours)
    worst = max(worst, rel)
    print(f"  ({l1},{l2},{l3};0,0,0)^2  sympy {exact:.12e}  ours {ours:.12e}  rel {rel:.1e}")
nums["EightBthreejWorst"] = float(worst) if worst > 0 else "0"
nums["EightBthreejExample"] = f"{lm.three_j_000_sq(2, 2, 2):.6f}"          # = 2/35
lf = lm.log_factorials(4000)
s_orth = sum((2 * l3 + 1) * lm._w3j000_sq(100, 140, l3, lf) for l3 in range(40, 241))
nums["EightBorthSum"] = f"{s_orth:.12f}"

# ---------------------------------------------------------------- 2. coupling matrices
path = DATA / "coupling.npz"
if path.exists():
    cz = np.load(path)
    MM = {k: cz["M_" + k] for k in KEYS}
    XI2 = {k: cz["Xi2_" + k] for k in KEYS}
else:
    MM, XI2 = {}, {}
    t0 = time.time()
    for k in KEYS:
        MM[k] = lm.coupling_matrix(z["wl_" + k], L, L)
        ell = np.arange(L + 1)
        XI2[k] = lm.coupling_matrix(z["wl2_" + k], L, L) / (2 * ell + 1)[None, :]
        print(f"  M for {k}: {time.time() - t0:.1f} s")
    np.savez(path, **{"M_" + k: v for k, v in MM.items()}, **{"Xi2_" + k: v for k, v in XI2.items()})
    nums["EightBcouplingSeconds"] = f"{(time.time() - t0) / len(KEYS):.0f}"

# ---------------------------------------------------------------- 3. sum rules and leakage
ell = np.arange(L + 1)
NAMES = {"full": "Full", "cap10": "CapTen", "cap10apo": "CapTenApo", "cap30": "CapThirty",
         "cap70": "CapSeventy", "band20": "Band", "band20apo": "BandApo", "holes": "Holes"}
LP = 100                                    # the probe multipole l'
for k in KEYS:
    m = z["mask_" + k].astype(float)
    fsky = np.mean(m > 0); w2 = np.mean(m ** 2) / fsky
    M = MM[k]
    row = M[2:401].sum(axis=1) / (fsky * w2)                   # white-noise rule, rows l <= 400
    col = ((2 * ell + 1)[:, None] * M)[:, 2:401].sum(axis=0) / ((2 * ell[2:401] + 1) * fsky * w2)
    resp = (2 * ell + 1) * M[:, LP]
    far = resp[np.abs(ell - LP) > 20].sum() / resp.sum()
    sub = M[2:, 2:]
    cond = np.linalg.cond(sub)
    n = NAMES[k]
    nums[f"EightBrowRule{n}"] = lm.tex_sci(np.max(np.abs(row - 1)))
    nums[f"EightBfarLeak{n}"] = lm.tex_sci(100 * far)
    nums[f"EightBcond{n}"] = lm.tex_sci(cond)
    print(f"{k:10s} row rule max dev {np.max(np.abs(row - 1)):.1e}  col rule {np.max(np.abs(col - 1)):.1e}"
          f"  far leak (|l-100|>20) {100 * far:.3g}%  cond(M[2:,2:]) {cond:.3g}  M_100,100 {M[LP, LP]:.4f}")

# ---------------------------------------------------------------- 4. figure: one column of M
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
for ax, group in zip(axs, [[("cap10", "cap 0.1"), ("cap10apo", "cap 0.1 tapered"), ("cap30", "cap 0.3")],
                           [("band20", "band"), ("band20apo", "band tapered"), ("holes", "holes")]]):
    for (k, lab), c in zip(group, SERIES):
        m = z["mask_" + k].astype(float)
        fsky = np.mean(m > 0); w2 = np.mean(m ** 2) / fsky
        y = (2 * ell + 1) * MM[k][:, LP] / ((2 * LP + 1) * fsky * w2)
        ev = ell[:301:2]                      # same parity as l' = 100 (the band masks vanish on odd l)
        ax.semilogy(ev, np.where(y[ev] > 0, y[ev], np.nan), color=c, lw=1.1, label=lab)
    ax.axvline(LP, color="0.6", lw=0.6, ls=":")
    ax.set_title("(a) caps" if ax is axs[0] else "(b) band cut and holes", fontsize=9)
    ax.set_xlabel(r"measured multipole $\ell$")
    ax.set_ylim(1e-9, 1)
    ax.legend(loc="upper right", fontsize=8)
axs[0].set_ylabel(r"share of the power of $\ell'=100$")
savefig(fig, "ch08", "coupling_columns")
save_numbers("ch08", "22_coupling", nums)
