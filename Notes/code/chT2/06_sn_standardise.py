"""How much does standardisation shrink the scatter of type Ia supernovae?

Question: raw peak magnitudes of type Ia supernovae scatter by ~0.35 mag about the Hubble
line. If brighter supernovae decline more slowly (stretch x1) and redder ones are fainter
(colour c), a linear correction m_B - M + alpha*x1 - beta*c should remove most of it.
How small does the scatter become, and what does least squares return for alpha and beta
when x1 and c are themselves measured with noise?
Computes:
  * a toy light-curve family (brighter = broader) for the figure,
  * one mock Hubble-flow sample: raw and standardised Hubble residuals,
  * least-squares (alpha, beta, M) and, over many mock samples, the attenuation of beta
    by the noise in c, compared with the reliability ratio Var(c)/[Var(c)+sigma_c^2].
Writes: figures/chT2/sn_standardise.pdf, results/chT2/06_sn_standardise.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

C_KMS = 299792.458
H0, OM = 70.0, 0.3


def mu_of_z(z):
    """Distance modulus in flat LCDM: 5 log10(D_L / 10 pc)."""
    zz = np.linspace(0, z.max() * 1.01, 4000)
    E = np.sqrt(OM * (1 + zz) ** 3 + 1 - OM)
    dc = np.concatenate([[0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zz))])
    dl = (1 + z) * C_KMS / H0 * np.interp(z, zz, dc)          # Mpc
    return 5 * np.log10(dl) + 25


# --- the population (truths chosen to be realistic, not fitted to any survey)
ALPHA, BETA, M0 = 0.15, 3.0, -19.25
SIG_INT, SIG_X1, SIG_C = 0.10, 1.0, 0.10      # intrinsic scatter, population widths
ERR_M, ERR_X1, ERR_C = 0.03, 0.15, 0.03       # measurement errors
SIG_V = 250.0                                 # peculiar velocity [km/s]
N = 400


def mock(rng):
    z = rng.uniform(0.02, 0.10, N)
    x1 = rng.normal(0, SIG_X1, N)
    c = rng.normal(0, SIG_C, N)
    Mtrue = M0 - ALPHA * x1 + BETA * c + rng.normal(0, SIG_INT, N)
    zobs = z + SIG_V / C_KMS * rng.normal(0, 1, N) * (1 + z)    # peculiar velocity
    mB = Mtrue + mu_of_z(z) + rng.normal(0, ERR_M, N)
    return zobs, mB, x1 + rng.normal(0, ERR_X1, N), c + rng.normal(0, ERR_C, N)


def fit(zobs, mB, x1, c):
    """Least squares for y = m_B - mu(z) = M - alpha x1 + beta c."""
    y = mB - mu_of_z(zobs)
    A = np.column_stack([np.ones_like(y), -x1, c])
    th, *_ = np.linalg.lstsq(A, y, rcond=None)
    return th, y, y - A @ th


rng = rng_for("chT2", "06_sn_standardise")
zobs, mB, x1, c = mock(rng)
(Mhat, ahat, bhat), y, res_std = fit(zobs, mB, x1, c)
res_raw = y - y.mean()

# --- many mock samples: attenuation of the slopes by noisy regressors
nmock = 2000
fits = np.array([fit(*mock(rng))[0] for _ in range(nmock)])
lam_c = SIG_C**2 / (SIG_C**2 + ERR_C**2)
lam_x = SIG_X1**2 / (SIG_X1**2 + ERR_X1**2)
pv_mag = 5 / np.log(10) * SIG_V / (C_KMS * 0.06)     # peculiar-velocity scatter at z = 0.06
pred_raw = np.sqrt(ALPHA**2 * SIG_X1**2 + BETA**2 * SIG_C**2 + SIG_INT**2 + ERR_M**2 + pv_mag**2)

# --- figure
setup(7.2, 2.6)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
t = np.linspace(-15, 40, 400)
for x, col in zip([-2, 0, 2], [SERIES[1], SERIES[0], SERIES[2]]):
    s = 1 + 0.1 * x                                  # stretch of the time axis
    tau = np.where(t < 0, 7.0, 10.5) * s
    mpk = M0 - ALPHA * x * 2.5                       # exaggerated x2.5 so the trend is visible
    ax[0].plot(t, mpk + 1.0857 * 0.5 * (t / tau) ** 2 * (t < 15 * s)
               + (t >= 15 * s) * (1.0857 * 0.5 * (15 / 10.5) ** 2 + 0.03 * (t - 15 * s)),
               color=col, label=rf"$x_1={x}$")
ax[0].invert_yaxis()
ax[0].set(xlabel="rest-frame days from peak", ylabel=r"$M_B$ (toy)")
ax[0].legend(fontsize=7, frameon=False)
cc = np.linspace(-0.3, 0.3, 2)
ax[1].scatter(c, y + ahat * x1 - Mhat, s=4, color=SERIES[0], alpha=0.6, lw=0)
theory_line(ax[1], cc, BETA * cc, label=r"true $\beta c$")
ax[1].plot(cc, bhat * cc, color=SERIES[1], lw=1.4, label=r"fitted $\hat\beta c$")
ax[1].set(xlabel=r"measured colour $c$", ylabel=r"$m_B-\mu-\hat M+\hat\alpha x_1$")
ax[1].legend(fontsize=7, frameon=False)
bins = np.linspace(-1.2, 1.2, 49)
ax[2].hist(res_raw, bins=bins, color=SERIES[3], alpha=0.6, label="raw")
ax[2].hist(res_std, bins=bins, color=SERIES[0], alpha=0.7, label="standardised")
ax[2].set(xlabel="Hubble residual [mag]", ylabel="supernovae")
ax[2].legend(fontsize=7, frameon=False)
fig.tight_layout()
savefig(fig, "chT2", "sn_standardise")

save_numbers("chT2", "06_sn_standardise", {
    "TwbN": N, "TwbAlpha": ALPHA, "TwbBeta": BETA, "TwbSigInt": SIG_INT,
    "TwbSigC": SIG_C, "TwbErrC": ERR_C, "TwbSigXone": SIG_X1, "TwbErrXone": ERR_X1,
    "TwbRawSd": f"{res_raw.std(ddof=1):.3f}", "TwbRawPred": f"{pred_raw:.3f}",
    "TwbStdSd": f"{res_std.std(ddof=3):.3f}",
    "TwbStdPred": f"{np.sqrt(SIG_INT**2 + ERR_M**2 + pv_mag**2 + BETA**2 * ERR_C**2 * lam_c + ALPHA**2 * ERR_X1**2 * lam_x):.3f}",
    "TwbAhat": f"{ahat:.3f}", "TwbBhat": f"{bhat:.2f}", "TwbMhat": f"{Mhat:.3f}",
    "TwbNmock": nmock, "TwbBmean": f"{fits[:, 2].mean():.3f}", "TwbBsd": f"{fits[:, 2].std():.3f}",
    "TwbAmean": f"{fits[:, 1].mean():.4f}",
    "TwbAmeanErr": f"{fits[:, 1].std(ddof=1) / np.sqrt(nmock):.4f}",     # Monte Carlo error of that mean
    "TwbLamC": f"{lam_c:.3f}", "TwbBpred": f"{BETA * lam_c:.3f}",
    "TwbLamX": f"{lam_x:.3f}", "TwbApred": f"{ALPHA * lam_x:.4f}",
    "TwbPvMag": f"{pv_mag:.3f}",
})
print(f"raw sd {res_raw.std():.3f} (pred {pred_raw:.3f}), std sd {res_std.std():.3f}; "
      f"beta mean {fits[:,2].mean():.3f} vs pred {BETA*lam_c:.3f}; alpha {fits[:,1].mean():.4f} vs {ALPHA*lam_x:.4f}")
