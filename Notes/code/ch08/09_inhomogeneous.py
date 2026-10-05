"""09_inhomogeneous.py -- noise whose level changes across the sky.

Question: if the scan piles up observations near the poles, so that the noise variance per pixel
sigma_p^2 changes across the sky, is the noise bias still Omega_pix <sigma^2>, and how much larger
than the uniform-noise value is the scatter of C_hat_l?
Computes: from the simulations of 05_mc_run, the mean and the variance of the noise-only spectrum
and of the signal+noise spectrum for the inhomogeneous pattern, divided by those of uniform noise
with the same mean variance; the exact prediction from the harmonic noise covariance
N_mm = Omega^2 sum_p sigma_p^2 |Y_lm(p)|^2 (diagonal in m for an axisymmetric pattern).
Writes: figures/ch08/inhomogeneous.pdf, results/ch08/09_inhomogeneous.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from common import setup, savefig, save_numbers, SERIES, theory_line
from camb_fiducial import load_fiducial
import lib_cmbsim as cs

setup()
ell_all, cl_all = load_fiducial()
exp = cs.Experiment()
L = exp.lmax
ell = np.arange(L + 1)
nu = 2 * ell + 1
v = cs.variance_pattern(exp.nside, contrast=4.0)
sims = cs.cached("mc_fullsky", lambda: (_ for _ in ()).throw(RuntimeError("run 05_mc_run.py first")))
nsim = sims["auto"].shape[0]
N = exp.nl(L)[0]
S = exp.transfer(L) ** 2 * cl_all[: L + 1]

# ---------------------------------------------------- exact prediction from N_mm
theta_p, _ = hp.pix2ang(exp.nside, np.arange(exp.npix))
th_r, idx, cnt = np.unique(np.round(theta_p, 12), return_inverse=True, return_counts=True)
v_r = np.bincount(idx, weights=v) / cnt                       # v on each ring (axisymmetric)
lt = np.unique(np.r_[np.arange(2, 40, 3), np.arange(40, L + 1, 16), L])
pred_noise, pred_tot, pred_bias = [], [], []
for l in lt:
    m = np.arange(0, l + 1)
    Y2 = np.abs(sph_harm_y(l, m[:, None], th_r[None, :], 0.0)) ** 2      # (m, ring)
    Nmm = exp.omega_pix ** 2 * exp.sigma_pix ** 2 * (Y2 * (cnt * v_r)[None, :]).sum(1)
    Nfull = np.r_[Nmm[:0:-1], Nmm]                             # m = -l..l (|Y_l,-m| = |Y_lm|)
    nbar = Nfull.mean()
    pred_bias.append(nbar / N)
    pred_noise.append(np.sum(Nfull ** 2) / (Nfull.size * nbar ** 2))
    pred_tot.append(np.sum((S[l] + Nfull) ** 2) / (Nfull.size * (S[l] + nbar) ** 2))
pred_bias, pred_noise, pred_tot = map(np.array, (pred_bias, pred_noise, pred_tot))

# ---------------------------------------------------- simulations
ninh, nwh = sims["ninh"].astype(float), sims["noise"].astype(float)
inh, aut = sims["inh"].astype(float), sims["auto"].astype(float)
bias_ratio = ninh.mean(0) / N
vr_noise = ninh.var(0, ddof=1) / nwh.var(0, ddof=1)
vr_tot = inh.var(0, ddof=1) / aut.var(0, ddof=1)
edges = np.r_[np.arange(2, 503, 25), 513]
lc = 0.5 * (edges[:-1] + edges[1:] - 1)

fig = plt.figure(figsize=(7.8, 3.0))
ax0 = fig.add_axes([0.0, 0.08, 0.36, 0.84])
plt.sca(ax0)
hp.mollview(v, hold=True, title=r"noise variance $\sigma_p^2/\langle\sigma^2\rangle$", cmap="Blues",
            cbar=True, notext=True, format="%.2f")
ax1 = fig.add_axes([0.47, 0.17, 0.5, 0.72])
ax1.plot(lc, cs.bin_spectrum(vr_noise, edges), "o", ms=3.5, color=SERIES[1], label="noise only, MC")
theory_line(ax1, lt, pred_noise, label=r"$(2\ell+1)\sum_m N_{mm}^2/(\sum_m N_{mm})^2$")
ax1.plot(lc, cs.bin_spectrum(vr_tot, edges), "s", ms=3.5, color=SERIES[0], label="signal + noise, MC")
ax1.plot(lt, pred_tot, color=SERIES[0], lw=1, label="signal + noise, exact")
ax1.plot(lc, cs.bin_spectrum(bias_ratio, edges), "^", ms=3.5, color=SERIES[2], label=r"noise bias / $\Omega_{\rm pix}\langle\sigma^2\rangle$")
ax1.axhline(1, color="0.5", lw=0.8)
ax1.set_xlabel(r"$\ell$"); ax1.set_ylabel("inhomogeneous / uniform")
ax1.legend(fontsize=6.5, loc="upper right")
ax1.set_ylim(0.9, max(1.5, 1.05 * pred_noise.max()))
savefig(fig, "ch08", "inhomogeneous")

hi = ell >= 100
save_numbers("ch08", "09_inhomogeneous", {
    "EightAInhContrast": "5",
    "EightAInhVmax": f"{v.max():.2f}", "EightAInhVmin": f"{v.min():.2f}",
    "EightAInhVsq": f"{np.mean(v ** 2):.3f}",
    "EightAInhBias": f"{bias_ratio[2:].mean():.4f}",
    "EightAInhBiasErr": f"{ninh[:, 2:].mean(1).std(ddof=1) / np.sqrt(nsim) / N:.4f}",
    "EightAInhVarNoise": f"{vr_noise[hi].mean():.3f}",
    "EightAInhVarNoisePred": f"{pred_noise[lt >= 100].mean():.3f}",
    "EightAInhVarTotLmax": f"{vr_tot[edges[-2]:].mean():.3f}",
    "EightAInhVarTotPredLmax": f"{pred_tot[-1]:.3f}",
    "EightAInhVarTotLow": f"{vr_tot[2:100].mean():.3f}",
})
print("bias ratio", bias_ratio[2:].mean(), "var noise MC", vr_noise[hi].mean(), "pred", pred_noise[lt >= 100].mean(),
      "<v^2>", np.mean(v ** 2), "tot lmax", vr_tot[edges[-2]:].mean(), pred_tot[-1], "pred bias", pred_bias.min(), pred_bias.max())
