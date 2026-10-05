"""07_sample_variance.py -- one realisation, finite volume: how well can it measure P(k) and the mean?

Question: (a) what is the distribution of a single Fourier mode's power |delta_k|^2?
(b) how large is the scatter of the shell-averaged P_hat(k) between realisations,
and is it sqrt(2 / N_modes)?  (c) does the spatial average over a patch of side s
converge to the ensemble mean as s grows (ergodicity), at the rate predicted by xi?
And what happens for a field that is NOT ergodic (a random constant offset per universe)?
Computes: 2-D fields with xi = exp(-r^2/2 ell^2), ell = 4 cells.
  (a) |delta_k|^2 / (V P(k)) over all modes of 50 maps of 128^2 -> histogram vs e^{-x};
  (b) 500 maps of 128^2, 60 linear k-shells: std(P_hat)/<P_hat> vs N_modes;
  (c) 200 maps of 256^2 tiled into s x s patches: variance of patch means vs the exact
      (1/V) sum_k P(k) |W_s(k)|^2 and the large-patch limit P(0)/s^2; the same with an
      extra offset A ~ N(0, 0.5^2) common to the whole map.
Writes: figures/ch07/sample_variance.pdf, results/ch07/07_sample_variance.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from lib_fields import gaussian_field, measure_pk, kgrid

setup(9.6, 3.2)
ell = 4.0
P = lambda k: 2 * np.pi * ell**2 * np.exp(-k**2 * ell**2 / 2)
out = {}
fig, ax = plt.subplots(1, 3)

# (a) the power in one mode is exponentially distributed
N, L = 128, 128.0
rng = rng_for("ch07", "07_sample_variance", 0)
f = gaussian_field(P, N, L, 2, rng, nsim=50)
fk = np.fft.rfftn(f, axes=(1, 2))                       # D = 1, so delta_k = FFT
kmag, comps = kgrid(N, L, 2)
interior = (comps[1] > 0) & (np.abs(comps[1]) < np.pi) & (kmag < 1.2)   # not self-conjugate, P not tiny
x = (np.abs(fk) ** 2 / (L**2 * P(kmag)))[:, interior].ravel()
ax[0].hist(x, bins=np.linspace(0, 6, 61), density=True, color=SERIES[0], alpha=0.7, label="all modes")
xx = np.linspace(0, 6, 200)
theory_line(ax[0], xx, np.exp(-xx), label=r"$e^{-x}$")
ax[0].set_yscale("log"); ax[0].set_xlabel(r"$x=|\delta_{\mathbf{k}}|^2/VP(k)$"); ax[0].set_ylabel("density")
ax[0].set_title("(a) power in one mode"); ax[0].legend()
out["ModeMean"] = x.mean(); out["ModeVar"] = x.var(); out["ModeCount"] = x.size

# (b) scatter of the shell-averaged estimate
M = 500
rng = rng_for("ch07", "07_sample_variance", 1)
est = []
for c in range(5):
    for g in gaussian_field(P, N, L, 2, rng, nsim=M // 5):
        k, ph, nm, pt = measure_pk(g, L, nbins=60, kmax=1.2, P=P)
        est.append(ph)
est = np.array(est)
frac = est.std(axis=0, ddof=1) / est.mean(axis=0)
ax[1].loglog(nm, frac, "o", ms=3, color=SERIES[1], label=f"{M} maps")
nn = np.logspace(np.log10(nm.min()), np.log10(nm.max()), 50)
theory_line(ax[1], nn, np.sqrt(2 / nn), label=r"$\sqrt{2/N_{\rm modes}}$")
ax[1].set_xlabel(r"$N_{\rm modes}$ in the shell"); ax[1].set_ylabel(r"std$(\hat P)/P$")
ax[1].set_title(r"(b) scatter of $\hat P(k)$"); ax[1].legend()
from matplotlib.ticker import NullFormatter, FixedLocator, ScalarFormatter
ax[1].xaxis.set_minor_formatter(NullFormatter()); ax[1].yaxis.set_minor_formatter(NullFormatter())
ax[1].xaxis.set_major_formatter(ScalarFormatter()); ax[1].yaxis.set_major_formatter(ScalarFormatter())
ax[1].set_xticks([10, 100, 1000]); ax[1].set_yticks([0.2, 0.5, 1.0])
out["FracFirst"] = frac[0]; out["NmFirst"] = nm[0]; out["PredFirst"] = np.sqrt(2 / nm[0])
out["FracTen"] = frac[9]; out["NmTen"] = nm[9]; out["PredTen"] = np.sqrt(2 / nm[9])
out["FracRatio"] = np.mean(frac / np.sqrt(2 / nm))

# (c) ergodicity: variance of the spatial mean over an s x s patch
N2, L2, M2 = 256, 256.0, 200
rng = rng_for("ch07", "07_sample_variance", 2)
sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128])
sig_A = 0.5
v_erg, v_non = np.zeros(sizes.size), np.zeros(sizes.size)
v64_maps = []                                                      # per-map estimate at s = 64
for c in range(4):
    g = gaussian_field(P, N2, L2, 2, rng, nsim=M2 // 4)
    A = sig_A * rng.standard_normal(M2 // 4)[:, None, None]       # one offset per universe
    for j, s in enumerate(sizes):
        m = g.reshape(M2 // 4, N2 // s, s, N2 // s, s).mean(axis=(2, 4))
        v_erg[j] += np.mean(m**2) / 4
        v_non[j] += np.mean((m + A) ** 2) / 4
        if s == 64:
            v64_maps.extend(np.mean(m**2, axis=(1, 2)))
kf, kc = kgrid(N2, L2, 2, real=False)
th = []
for s in sizes:                                                    # exact: (1/V) sum_k P |W_s|^2
    W = np.ones_like(kf)
    for kc_i in kc:
        a = kc_i * s / 2
        W *= np.where(np.abs(kc_i) > 0, np.sin(a) / (s * np.sin(kc_i / 2) + (kc_i == 0)), 1.0)
    pk = np.where(kf > 0, P(np.where(kf > 0, kf, 1.0)), 0.0)
    th.append(np.sum(pk * W**2) / L2**2)
th = np.array(th)
ax[2].loglog(sizes, v_erg, "o", color=SERIES[0], label="Gaussian field")
ax[2].loglog(sizes, v_non, "s", color=SERIES[3], label=r"field $+\,A$, $A\sim\mathcal{N}(0,0.5^2)$")
theory_line(ax[2], sizes, th, label=r"$(1/V)\sum_k P|W_s|^2$")
ax[2].loglog(sizes, P(0) / sizes**2.0, ":", color="k", lw=1.2, label=r"$P(0)/s^2$")
ax[2].axhline(sig_A**2, color=SERIES[3], lw=0.8, ls="--")
ax[2].set_ylim(1e-4, 2)
ax[2].set_xlabel(r"patch side $s$ (cells)"); ax[2].set_ylabel("Var(patch mean)")
ax[2].set_title("(c) spatial average"); ax[2].legend(fontsize=6.5, loc="lower left")
out["VarSixtyFour"] = v_erg[sizes == 64][0]; out["ThSixtyFour"] = th[sizes == 64][0]
out["VarSixtyFourErr"] = np.std(v64_maps, ddof=1) / np.sqrt(len(v64_maps))   # maps are independent
out["AsySixtyFour"] = P(0) / 64**2
out["NonSixtyFour"] = v_non[sizes == 64][0]
out["PZero"] = P(0)
out["SigA"] = sig_A
fig.tight_layout()
savefig(fig, "ch07", "sample_variance")
save_numbers("ch07", "07_sample_variance", {f"SevA{k}": v for k, v in out.items()})
