"""07_fisher_vs_mcmc.py -- is the CMB Fisher forecast right? Against the chapter-9 chain and 1000 fits.

Question: chapter 9 sampled the posterior of the six LCDM parameters (H0 basis) from one simulated
Planck-like temperature spectrum: one 143 GHz-like channel (7.22', 33 muK arcmin), f_sky = 0.57,
2 <= l <= 2500, and a tau prior of width 0.0086 (run P6, data/ch09/planck_chains.npz).
Our Fisher matrix lives in the Planck basis (omega_b, omega_c, 100 theta_MC, tau, ln A, n_s).
 (1) Transform it to the chain's basis (ln A, n_s, H0, omega_b, omega_c, tau) with F' = J^T F J and
     compare the marginalised errors and correlations with the chain.
 (2) Draw 1000 new spectra from the same experiment, find the maximum-likelihood point of each by
     Fisher scoring (theta <- theta + F^-1 S) with chapter 9's second-order emulator, and compare the
     scatter of the estimates with F^-1; count how many fall inside the Fisher ellipsoids.
Writes: figures/ch10/fisher_mcmc.pdf, results/ch10/07_fisher_vs_mcmc.tex
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES
import lib_fisher10 as L

setup()
rng = rng_for("ch10", "07_fisher_vs_mcmc")
NOTES = pathlib.Path(__file__).resolve().parents[2]
t0 = time.time()
FS, SIG_TAU, LMAX = 0.57, 0.0086, 2500
nums = {}

# ---------------------------------------------------------------- (1) Fisher in the Planck basis, then J^T F J
dC = L.derivatives(1.0, 4)
C = L.fiducial_spectra()
NT, NP = L.noise_143(lmax=L.LMAX)
FP = L.fisher_cmb(dC, C, NT, NP, FS, 2, LMAX, "TT") + L.tau_prior(SIG_TAU)
fidP = L.fiducial_vector()

# chain basis: (ln A, n_s, H0, omega_b, omega_c, tau).  Planck basis: (omega_b, omega_c, 100 theta, tau, ln A, n_s).
# J[a, b] = d(planck_a) / d(chain_b).  Only 100 theta_MC depends on H0, omega_b, omega_c non-trivially.
import camb  # noqa: E402
from camb_fiducial import FIDUCIAL  # noqa: E402


def theta100(H0, ombh2, omch2):
    p = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=FIDUCIAL["mnu"])
    return 100 * camb.get_background(p, no_thermo=False).cosmomc_theta()


H0f, obf, ocf = FIDUCIAL["H0"], FIDUCIAL["ombh2"], FIDUCIAL["omch2"]
dth = np.array([(theta100(H0f + 0.05, obf, ocf) - theta100(H0f - 0.05, obf, ocf)) / 0.1,
                (theta100(H0f, obf + 1e-5, ocf) - theta100(H0f, obf - 1e-5, ocf)) / 2e-5,
                (theta100(H0f, obf, ocf + 1e-4) - theta100(H0f, obf, ocf - 1e-4)) / 2e-4])
J = np.zeros((6, 6))
J[0, 3] = 1            # omega_b  <- omega_b
J[1, 4] = 1            # omega_c  <- omega_c
J[2, 2], J[2, 3], J[2, 4] = dth   # 100 theta <- H0, omega_b, omega_c
J[3, 5] = 1            # tau      <- tau
J[4, 0] = 1            # ln A     <- ln A
J[5, 1] = 1            # n_s      <- n_s
FH = J.T @ FP @ J
CH = np.linalg.inv(FH)
sF = np.sqrt(np.diag(CH))
nums.update({"TenGFmDthHz": f"{dth[0]:.5f}", "TenGFmDthOb": f"{dth[1]:.2f}", "TenGFmDthOc": f"{dth[2]:.3f}"})

z = np.load(NOTES / "data" / "ch09" / "planck_chains.npz")
post = z["post_P6"]
sC = post.std(0, ddof=1)
RC = np.corrcoef(post.T)
RF = CH / np.outer(sF, sF)
F9 = np.load(NOTES / "data" / "ch09" / "fisher_planck_like.npz")
s9 = np.sqrt(np.diag(F9["cov"]))
short = ["As", "Ns", "Hz", "Ob", "Oc", "Tau"]
for k, s in enumerate(short):
    nums[f"TenGFmF{s}"] = f"{sF[k]:.3g}"
    nums[f"TenGFmC{s}"] = f"{sC[k]:.3g}"
    nums[f"TenGFmN{s}"] = f"{s9[k]:.3g}"
    nums[f"TenGFmR{s}"] = f"{sF[k] / sC[k]:.2f}"
nums["TenGFmMaxRhoDiff"] = f"{np.max(np.abs(RF - RC)):.2f}"
nums["TenGFmRhoHzOcF"] = f"{RF[2, 4]:+.2f}"; nums["TenGFmRhoHzOcC"] = f"{RC[2, 4]:+.2f}"
nums["TenGFmRhoAsTauF"] = f"{RF[0, 5]:+.2f}"; nums["TenGFmRhoAsTauC"] = f"{RC[0, 5]:+.2f}"
nums["TenGFmNchain"] = post.shape[0]

# ---------------------------------------------------------------- (2) 1000 maximum-likelihood fits
emu = np.load(NOTES / "data" / "ch09" / "emu.npz")
c0, lnc0, D, H, fidH = emu["c0"], emu["lnc0"], emu["D"], emu["H"], emu["fid"]
l = np.arange(LMAX + 1)
use = l >= 2
nu = (2 * l + 1.0) * FS
N = NT[: LMAX + 1]


def cl_and_grad(th):
    """Second-order emulator of chapter 9: ln C = ln C0 + D d + d^T H d / 2; returns C and dC/dtheta."""
    d = th - fidH
    Hd = np.einsum("ijl,j->il", H, d)
    lnc = lnc0 + d @ D + 0.5 * d @ Hd
    c = np.exp(lnc)
    return c, c[None, :] * (D + Hd)


NSIM = 1000
fits = np.empty((NSIM, 6))
niter = np.zeros(NSIM, int)
for k in range(NSIM):
    chat = (c0 + N) * rng.chisquare(nu) / nu
    tau_obs = fidH[5] + SIG_TAU * rng.standard_normal()
    th = fidH.copy()
    for it in range(30):
        c, g = cl_and_grad(th)
        tot = c + N
        r = (chat - tot) / tot**2
        S = 0.5 * (g[:, use] * (nu * r)[use]).sum(1)              # score of the exact likelihood
        S[5] -= (th[5] - tau_obs) / SIG_TAU**2                    # ... plus the tau prior
        w = 0.5 * nu / tot**2
        Fk = (g[:, use] * w[use]) @ g[:, use].T                   # Fisher matrix at the current point
        Fk[5, 5] += 1 / SIG_TAU**2
        step = np.linalg.solve(Fk, S)
        th = th + step
        if np.max(np.abs(step) / sF) < 1e-4:
            break
    fits[k] = th
    niter[k] = it + 1
sM = fits.std(0, ddof=1)
d = fits - fidH
q6 = np.einsum("ki,ij,kj->k", d, FH, d)
q2 = np.einsum("ki,ij,kj->k", d[:, :2], np.linalg.inv(CH[:2, :2]), d[:, :2])
in2 = np.mean(q2 <= 2.30)
in6 = np.mean(q6 <= stats.chi2.ppf(0.683, 6))
for k, s in enumerate(short):
    nums[f"TenGFmM{s}"] = f"{sM[k]:.3g}"
    nums[f"TenGFmMR{s}"] = f"{sM[k] / sF[k]:.2f}"
    nums[f"TenGFmBias{s}"] = f"{(fits[:, k].mean() - fidH[k]) / sF[k]:+.2f}"
nums.update({"TenGFmNsim": NSIM, "TenGFmIter": f"{np.median(niter):.0f}", "TenGFmInTwo": f"{in2:.3f}",
             "TenGFmInSix": f"{in6:.3f}", "TenGFmChiSix": f"{stats.chi2.ppf(0.683, 6):.2f}",
             "TenGFmErr": f"{100 / np.sqrt(2 * NSIM):.1f}", "TenGFmBinErr": f"{np.sqrt(0.683 * 0.317 / NSIM):.3f}",
             "TenGFmMinutes": f"{(time.time() - t0) / 60:.1f}"})

# ---------------------------------------------------------------- figure: chain, fits and Fisher in two planes
fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.9))
tt = np.linspace(0, 2 * np.pi, 300)
centre_chain = post.mean(0)
for ax, (i, j) in zip(axs, [(0, 5), (2, 4), (0, 1)]):
    sel = rng.choice(post.shape[0], 3000, replace=False)
    ax.scatter(post[sel, i] - centre_chain[i], post[sel, j] - centre_chain[j], s=2, alpha=0.25, lw=0,
               color=SERIES[0], label="ch. 9 chain (centred)")
    ax.scatter(fits[:, i] - fidH[i], fits[:, j] - fidH[j], s=2, alpha=0.5, lw=0, color=SERIES[1],
               label="ML fits (centred on truth)")
    sub = CH[np.ix_([i, j], [i, j])]
    Lc = np.linalg.cholesky(sub)
    for k2, ls in [(2.30, "-"), (6.18, "--")]:
        e = np.sqrt(k2) * Lc @ np.vstack([np.cos(tt), np.sin(tt)])
        ax.plot(e[0], e[1], color="k", ls=ls, lw=1.1)
    lab = [r"$\Delta\ln(10^{10}A_s)$", r"$\Delta n_s$", r"$\Delta H_0$", r"$\Delta\omega_b$",
           r"$\Delta\omega_c$", r"$\Delta\tau$"]
    ax.set_xlabel(lab[i]); ax.set_ylabel(lab[j])
    ax.tick_params(labelsize=7)
    ax.locator_params(nbins=4)
axs[0].legend(fontsize=6.5, loc="upper left", markerscale=3)
fig.tight_layout()
savefig(fig, "ch10", "fisher_mcmc")
save_numbers("ch10", "07_fisher_vs_mcmc", nums)
for k in nums:
    print(k, nums[k])
