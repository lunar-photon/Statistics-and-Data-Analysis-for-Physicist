"""08_same_spectrum.py -- two universes with the same power spectrum.

Question: if we keep the amplitude |delta_k| of every Fourier mode of an evolved matter field and
scramble the phases, the power spectrum is unchanged mode by mode. What changes?
Computes: Zel'dovich particles at z = 0 in a periodic box (side 600 Mpc/h, 192^3 particles), their
cloud-in-cell density; a phase-randomised twin with identical |delta_k|; for both, the power
spectrum (identical), the variance in spheres (identical), the one-point distribution of the
density (different), the fraction of cells with delta < -1 (impossible for a real density), and the
number of voids returned by the spherical void finder of lib_web (different).
Writes: figures/ch11/same_spectrum.pdf, results/ch11/08_same_spectrum.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from common import setup, savefig, save_numbers, rng_for, SERIES
from lib_web import BOX_L, BOX_N, zeldovich_box, cic, find_voids, sphere_average

L, N = BOX_L, BOX_N
x, psi, dlin = zeldovich_box(L, N, rng_for("ch11", "za_box"))
rho = cic(x, N, L)                                   # 1 + delta of the Zel'dovich universe
dk = np.fft.rfftn(rho - 1)

# the twin: same |delta_k|, phases taken from the FFT of fresh white noise (Hermitian by construction)
w = np.fft.rfftn(rng_for("ch11", "08_same_spectrum").standard_normal((N, N, N)))
twin = 1 + np.fft.irfftn(np.abs(dk) * np.exp(1j * np.angle(w)), s=(N, N, N), axes=(0, 1, 2))
pk_ratio_dev = float(np.max(np.abs(np.abs(np.fft.rfftn(twin - 1)) ** 2 / np.maximum(np.abs(dk) ** 2, 1e-30) - 1)[1:]))

R = 8.0                                               # sphere radius for the one-point statistics
sm = {"za": sphere_average(rho, R, L) - 1, "tw": sphere_average(twin, R, L) - 1}
stats = {}
for key, d in sm.items():
    v = d.ravel()
    stats[key] = dict(sig=v.std(), skew=np.mean((v - v.mean()) ** 3) / v.std() ** 3,
                      neg=np.mean(v < -1), deep=np.mean(v < -0.6))
radii = np.arange(3.0, 60.0, 1.0)
cz, rz = find_voids(rho, L, radii, rmin=8.0)
ct, rt = find_voids(np.clip(twin, 0, None), L, radii, rmin=8.0)
volfrac = lambda r: float(np.sum(4 / 3 * np.pi * r ** 3) / L ** 3)

# ---- figure: two slices and the one-point distributions
setup(6.8, 2.6)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
sl = slice(N // 2, N // 2 + 3)
for a, f, t in [(axs[0], rho, "Zel'dovich universe"), (axs[1], twin, "same $|\\delta_k|$, random phases")]:
    img = f[:, :, sl].mean(axis=2)
    a.imshow(np.log10(np.clip(img, 0.05, None)).T, origin="lower", extent=[0, L, 0, L], cmap="Blues",
             vmin=-1.3, vmax=1.0)
    a.set_title(t, fontsize=8)
    a.set_xticks([0, 300, 600]); a.set_yticks([0, 300, 600]); a.grid(False)
    a.set_xlabel(r"$x\ [h^{-1}\mathrm{Mpc}]$", fontsize=8)
axs[0].set_ylabel(r"$y\ [h^{-1}\mathrm{Mpc}]$", fontsize=8)
bins = np.linspace(-1.6, 3.0, 93)
for key, col, lab in [("za", SERIES[0], "Zel'dovich"), ("tw", SERIES[1], "random phases")]:
    h, e = np.histogram(sm[key].ravel(), bins=bins, density=True)
    axs[2].semilogy(0.5 * (e[1:] + e[:-1]), h, color=col, label=lab)
axs[2].axvline(-1, color="0.5", ls=":", lw=1)
axs[2].set_xlabel(r"$\delta$ in spheres of $8\,h^{-1}$Mpc", fontsize=8)
axs[2].set_ylabel("density", fontsize=8)
axs[2].set_ylim(1e-4, 5)
axs[2].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch11", "same_spectrum")

save_numbers("ch11", "08_same_spectrum", {
    "SameL": int(L), "SameN": N, "SameR": R, "SamePkDev": pk_ratio_dev,
    "SameSigZA": stats["za"]["sig"], "SameSigTw": stats["tw"]["sig"],
    "SameSkewZA": stats["za"]["skew"], "SameSkewTw": stats["tw"]["skew"],
    "SameNegZA": 100 * stats["za"]["neg"], "SameNegTw": 100 * stats["tw"]["neg"],
    "SameDeepZA": 100 * stats["za"]["deep"], "SameDeepTw": 100 * stats["tw"]["deep"],
    "SameNvZA": len(rz), "SameNvTw": len(rt),
    "SameVfZA": 100 * volfrac(rz), "SameVfTw": 100 * volfrac(rt),
    "SameRmaxZA": float(rz.max()), "SameRmaxTw": float(rt.max()) if len(rt) else 0.0,
})
print(stats, len(rz), len(rt), volfrac(rz), volfrac(rt), pk_ratio_dev)
