"""08_smoothing.py -- smoothing a field multiplies its power spectrum by |W(k)|^2.

Question: what does convolving a field with a window (a beam, a pixel, a top-hat
sphere) do to its power spectrum, its variance sigma_R^2 and its gradient variance?
Computes: a 2-D Gaussian field with P(k) = A k^{-1} on 512^2 unit cells (A gives unit
variance), smoothed with a Gaussian window W = exp(-k^2 R^2 / 2) and a disc (2-D
top-hat) W = 2 J1(kR)/(kR) for several R.  Checks P_hat of the smoothed map against
|W|^2 P, sigma_R^2 = <delta_R^2> against (1/V) sum_k |W|^2 P, and the gradient
variance sigma_1^2 = <|grad delta_R|^2> against (1/V) sum_k k^2 |W|^2 P.
Writes: figures/ch07/smoothing.pdf, results/ch07/08_smoothing.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from lib_fields import gaussian_field, measure_pk, smooth, kgrid

setup(9.0, 6.0)
rng = rng_for("ch07", "08_smoothing")
N, L = 512, 512.0
kf, _ = kgrid(N, L, 2, real=False)
A = L**2 / np.sum(1 / kf[kf > 0])
P = lambda k: A / k
Wg = lambda R: (lambda k: np.exp(-k**2 * R**2 / 2))
Wd = lambda R: (lambda k: np.where(k * R > 1e-8, 2 * j1(k * R) / np.where(k * R > 1e-8, k * R, 1.0), 1.0))
f = gaussian_field(P, N, L, 2, rng)
out = {}

def predicted(W, power=0):
    """(1/V) sum_k k^{2 power} |W(k)|^2 P(k) on the box's own modes."""
    pk = np.where(kf > 0, P(np.where(kf > 0, kf, 1.0)), 0.0)
    return np.sum(kf ** (2 * power) * W(kf) ** 2 * pk) / L**2

fig = plt.figure()
gs = fig.add_gridspec(2, 3)
R0 = 4.0
maps = [("raw field", f), (rf"Gaussian, $R={R0:.0f}$", smooth(f, L, Wg(R0))), (rf"disc, $R={R0:.0f}$", smooth(f, L, Wd(R0)))]
for j, (t, m) in enumerate(maps):
    a = fig.add_subplot(gs[0, j])
    a.imshow(m[:200, :200] / m.std(), cmap="RdBu_r", origin="lower", vmin=-3, vmax=3)
    a.set_title(t + rf", $\sigma={m.std():.2f}$"); a.set_xticks([]); a.set_yticks([]); a.grid(False)

aP = fig.add_subplot(gs[1, 0])
k, ph, nm, pt = measure_pk(f, L, nbins=30, log=True, P=P)
aP.loglog(k, ph, "o", ms=2.5, color="0.5", label="raw")
for (name, W), c in zip([("Gaussian", Wg(R0)), ("disc", Wd(R0))], SERIES):
    k, ph, nm, pt = measure_pk(smooth(f, L, W), L, nbins=30, log=True, P=lambda q: W(q) ** 2 * P(q))
    aP.loglog(k, ph, "o", ms=2.5, color=c, label=name)
    theory_line(aP, k, pt, label=r"$|W|^2P$" if c == SERIES[0] else None)
aP.set_ylim(1e-6, 1e2); aP.set_xlabel("$k$"); aP.set_ylabel(r"$\hat P(k)$"); aP.legend(fontsize=7)

aS = fig.add_subplot(gs[1, 1])
Rs = np.array([1, 2, 4, 8, 16, 32])
for (name, Wf), c in zip([("Gaussian", Wg), ("disc", Wd)], SERIES):
    meas = [smooth(f, L, Wf(R)).var() for R in Rs]
    pred = [predicted(Wf(R)) for R in Rs]
    aS.loglog(Rs, meas, "o", color=c, label=f"{name}, measured")
    aS.loglog(Rs, pred, "--", color="k", lw=1.2)
    out[f"SigRFour{name[0].upper()}"] = np.sqrt(meas[2]); out[f"SigRFourTh{name[0].upper()}"] = np.sqrt(pred[2])
aS.set_xlabel("$R$ (cells)"); aS.set_ylabel(r"$\sigma_R^2$"); aS.legend(fontsize=7)

aG = fig.add_subplot(gs[1, 2])
for (name, Wf), c in zip([("Gaussian", Wg)], SERIES):
    ratio_m, ratio_p = [], []
    for R in Rs:
        m = smooth(f, L, Wf(R))
        fk = np.fft.rfftn(m)
        kk, comps = kgrid(N, L, 2)
        grad2 = sum(np.fft.irfftn(1j * c_ * fk, s=[N, N], axes=(0, 1)) ** 2 for c_ in comps)   # |grad m|^2 by spectral derivative
        ratio_m.append(np.sqrt(grad2.mean() / m.var()))
        ratio_p.append(np.sqrt(predicted(Wf(R), 1) / predicted(Wf(R))))
    aG.loglog(Rs, ratio_m, "o", color=c, label="Gaussian, measured")
    aG.loglog(Rs, ratio_p, "--", color="k", lw=1.2, label="predicted")
    out["GradRatioFour"] = ratio_m[2]; out["GradRatioFourTh"] = ratio_p[2]
aG.set_xlabel("$R$ (cells)"); aG.set_ylabel(r"$\sigma_1/\sigma_0$ (1/cell)"); aG.legend(fontsize=7)
out["SigRaw"] = f.std()
fig.tight_layout()
savefig(fig, "ch07", "smoothing")
save_numbers("ch07", "08_smoothing", {f"SevA{k}": v for k, v in out.items()})
