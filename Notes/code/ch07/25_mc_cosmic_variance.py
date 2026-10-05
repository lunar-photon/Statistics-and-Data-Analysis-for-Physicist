"""25_mc_cosmic_variance.py -- a Monte Carlo check of cosmic variance on the full sky.

Question: if we simulate many full skies from the CAMB fiducial C_l and measure C_hat_l on
each, do we recover (i) the scaled chi^2_{2l+1} distribution, (ii) the variance
2 C_l^2/(2l+1), with a Monte Carlo error that shrinks like 1/sqrt(N_sim) at the predicted
rate sqrt(2/(N-1) + 12/((2l+1) N)), and (iii) a diagonal covariance between multipoles,
with off-diagonal noise of size 1/sqrt(N_sim)?
Computes: N_SIM full-sky C_hat_l for l = 2..300 from a_lm drawn one by one (lib_harmonic);
extra cheap skies at l = 2 and 10 for the convergence study; a cross-check against
healpy.synalm + healpy.alm2cl.
Writes: figures/ch07/mc_histograms.pdf, figures/ch07/mc_variance.pdf, figures/ch07/mc_correlation.pdf,
        results/ch07/25_mc_cosmic_variance.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES, DIV
from camb_fiducial import load_fiducial
import lib_harmonic as lh

setup()
ell, cl = load_fiducial()
N_SIM, LMAX = 20000, 300
ELLS = np.arange(2, LMAX + 1)
rng = rng_for("ch07", "25_mc_cosmic_variance", stream=0)

# ---------------------------------------------------------------- the simulation
chat = lh.chat_sims(cl, ELLS, N_SIM, rng)          # shape (N_SIM, 299): one row per simulated sky
y = chat / cl[ELLS]                                 # C_hat_l / C_l, mean 1, variance 2/(2l+1)
nu = 2 * ELLS + 1

# ---------------------------------------------------------------- (1) histograms vs scaled chi^2
SHOW = [2, 10, 50, 300]
fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.0))
ks = {}
for axx, l in zip(axs.ravel(), SHOW):
    j = l - 2
    n = 2 * l + 1
    ks[l] = stats.kstest(n * y[:, j], stats.chi2(n).cdf).pvalue
    lo, hi = (0.0 if l < 20 else np.quantile(y[:, j], 0.0005)), np.quantile(y[:, j], 0.999)
    axx.hist(y[:, j], bins=70, range=(lo, hi), density=True, color=SERIES[0], alpha=0.55,
             label=rf"{N_SIM} skies")
    xx = np.linspace(max(lo, 1e-4), hi, 600)
    theory_line(axx, xx, n * stats.chi2.pdf(n * xx, n), label=r"scaled $\chi^2_{2\ell+1}$")
    axx.plot(xx, stats.norm.pdf(xx, 1, np.sqrt(2 / n)), color="0.45", ls=":", lw=1.2,
             label="Gaussian, same mean and variance")
    axx.set_title(rf"$\ell={l}$:  KS $p={ks[l]:.2f}$")
    axx.set_xlabel(r"$\widehat C_\ell/C_\ell$")
axs[0, 0].legend(loc="upper right", fontsize=7)
fig.tight_layout()
savefig(fig, "ch07", "mc_histograms")

# ---------------------------------------------------------------- (2) convergence of the MC variance
# extra skies at l = 2 and l = 10 only (5 and 21 Gaussian numbers per sky): cheap, many batches
rng_b = rng_for("ch07", "25_mc_cosmic_variance", stream=1)
N_BIG = 1_000_000
ybig = {l: lh.chat_sims(cl, [l], N_BIG, rng_b)[:, 0] / cl[l] for l in (2, 10)}


def frac_err_theory(N, n):
    """Fractional sd of the sample variance S^2 (ddof=1) of N draws of C_hat/C, with nu = n."""
    return np.sqrt(2.0 / (N - 1) + 12.0 / (n * N))


Ns = np.unique(np.round(np.geomspace(10, 10000, 16)).astype(int))
rms_lo = {}
for l in (2, 10):
    n = 2 * l + 1
    out = []
    for N in Ns:
        b = ybig[l][: (N_BIG // N) * N].reshape(-1, N)           # independent batches of N skies
        s2 = b.var(axis=1, ddof=1) / (2.0 / n)                      # MC variance / cosmic variance
        out.append(np.sqrt(np.mean((s2 - 1) ** 2)))
    rms_lo[l] = np.array(out)
# high l: pool l = 100..300 (independent multipoles) from the main run
hi_idx = ELLS >= 100
Ns_hi = Ns[Ns <= N_SIM // 10]
rms_hi = []
for N in Ns_hi:
    b = y[: (N_SIM // N) * N, hi_idx].reshape(-1, N, hi_idx.sum())
    s2 = b.var(axis=1, ddof=1) / (2.0 / nu[hi_idx])
    rms_hi.append(np.sqrt(np.mean((s2 - 1) ** 2)))
rms_hi = np.array(rms_hi)

# running estimate from the main run, as one would watch it while simulating
Nrun = np.unique(np.round(np.geomspace(10, N_SIM, 60)).astype(int))
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5))
for k, l in enumerate([2, 10, 100]):
    j = l - 2
    n = 2 * l + 1
    run = np.array([y[:N, j].var(ddof=1) / (2.0 / n) for N in Nrun])
    ax[0].plot(Nrun, run, color=SERIES[k], label=rf"$\ell={l}$")
    ax[0].fill_between(Nrun, 1 - frac_err_theory(Nrun, n), 1 + frac_err_theory(Nrun, n),
                       color=SERIES[k], alpha=0.12, lw=0)
theory_line(ax[0], Nrun, np.ones_like(Nrun, dtype=float), label=r"$2C_\ell^2/(2\ell+1)$")
ax[0].set_xscale("log")
ax[0].set_ylim(0.0, 2.0)
ax[0].set_xlabel(r"number of simulated skies $N$")
ax[0].set_ylabel(r"MC variance of $\widehat C_\ell$ / $[2C_\ell^2/(2\ell+1)]$")
ax[0].set_title("(a) running estimate, with predicted $\\pm1\\sigma$ bands")
ax[0].legend(loc="upper right", fontsize=8)

ax[1].loglog(Ns, rms_lo[2], "o", color=SERIES[0], label=r"$\ell=2$")
ax[1].loglog(Ns, rms_lo[10], "s", color=SERIES[1], label=r"$\ell=10$")
ax[1].loglog(Ns_hi, rms_hi, "^", color=SERIES[2], label=r"$100\leq\ell\leq300$")
NN = np.geomspace(8, 12000, 100)
theory_line(ax[1], NN, frac_err_theory(NN, 5), label=r"$\sqrt{2/(N-1)+12/[(2\ell+1)N]}$")
theory_line(ax[1], NN, frac_err_theory(NN, 21), label=None)
ax[1].loglog(NN, np.sqrt(2 / (NN - 1)), color="0.45", ls=":", lw=1.2, label=r"$\sqrt{2/(N-1)}$")
ax[1].set_xlabel(r"number of simulated skies $N$")
ax[1].set_ylabel("rms fractional error of the MC variance")
ax[1].set_title("(b) the error shrinks as $N^{-1/2}$")
ax[1].legend(loc="lower left", fontsize=7)
fig.tight_layout()
savefig(fig, "ch07", "mc_variance")

# slope of the log-log line (least squares) for the pooled high-l points
slope_hi = np.polyfit(np.log(Ns_hi), np.log(rms_hi), 1)[0]
slope_two = np.polyfit(np.log(Ns), np.log(rms_lo[2]), 1)[0]

# ---------------------------------------------------------------- (3) covariance between multipoles
LC = np.arange(2, 42)                               # l = 2..41
jc = LC - 2


def corr_offdiag(N):
    R = np.corrcoef(y[:N, jc], rowvar=False)
    off = R[~np.eye(len(LC), dtype=bool)]
    return R, off

R500, off500 = corr_offdiag(500)
Rall, offall = corr_offdiag(N_SIM)
Nc = np.unique(np.round(np.geomspace(50, N_SIM, 14)).astype(int))
sd_off = np.array([corr_offdiag(N)[1].std() for N in Nc])
# all multipoles l=2..300 with the full run: the largest off-diagonal element
Rbig = np.corrcoef(y, rowvar=False)
off_big = Rbig[~np.eye(len(ELLS), dtype=bool)]

fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.6))
im = ax[0].imshow(R500, cmap=DIV, vmin=-0.25, vmax=0.25, origin="lower",
                  extent=[LC[0] - 0.5, LC[-1] + 0.5, LC[0] - 0.5, LC[-1] + 0.5])
ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel(r"$\ell'$")
ax[0].set_title(r"(a) MC correlation of $\widehat C_\ell,\widehat C_{\ell'}$, $N=500$")
ax[0].grid(False)
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
ax[1].loglog(Nc, sd_off, "o", color=SERIES[0], label="sd of off-diagonal elements")
theory_line(ax[1], Nc, 1 / np.sqrt(Nc), label=r"$1/\sqrt{N}$")
ax[1].set_xlabel(r"number of simulated skies $N$")
ax[1].set_ylabel("spread of the off-diagonal correlations")
ax[1].set_title("(b) noise around the identity")
ax[1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
savefig(fig, "ch07", "mc_correlation")

# ---------------------------------------------------------------- (4) cross-check with healpy
N_HP, L_HP = 3000, 50
np.random.seed(int(rng_for("ch07", "25_hp", 0).integers(2**31)))   # healpy draws from numpy's global stream
c_hp = np.array([hp.alm2cl(hp.synalm(cl[: L_HP + 1], lmax=L_HP, new=True)) for _ in range(N_HP)])
y_hp = c_hp[:, 2:] / cl[2: L_HP + 1]
ks2 = {l: stats.ks_2samp(y_hp[:, l - 2], y[:, l - 2]).pvalue for l in (2, 10, 50)}
var_hp = {l: f"{y_hp[:, l - 2].var(ddof=1) / (2 / (2 * l + 1)):.3f}" for l in (2, 10, 50)}

# ---------------------------------------------------------------- numbers for the text
vr = {l: f"{y[:, l - 2].var(ddof=1) / (2 / (2 * l + 1)):.3f}" for l in SHOW}
mean_dev = np.max(np.abs(y.mean(axis=0) - 1) / np.sqrt(2 / nu / N_SIM))
save_numbers("ch07", "25_mc_cosmic_variance", {
    "SBNsim": f"{N_SIM:,}".replace(",", "{,}"),
    "SBNbig": r"10^6",
    "SBksTwo": ks[2], "SBksTen": ks[10], "SBksFifty": ks[50], "SBksThree": ks[300],
    "SBvrTwo": vr[2], "SBvrTen": vr[10], "SBvrFifty": vr[50], "SBvrThree": vr[300],
    "SBerrTwo": frac_err_theory(N_SIM, 5), "SBerrThree": frac_err_theory(N_SIM, 601),
    "SBmeanDevMax": mean_dev,
    "SBslopeHi": slope_hi, "SBslopeTwo": slope_two,
    "SBcoefTwo": np.sqrt(2 + 12 / 5),
    "SBoffSdFiveH": off500.std(), "SBoffInvFiveH": 1 / np.sqrt(500),
    "SBoffSdAll": offall.std(), "SBoffInvAll": 1 / np.sqrt(N_SIM),
    "SBoffMaxBig": np.max(np.abs(off_big)), "SBnOffBig": off_big.size // 2,
    "SBknoxErr": f"{100 * 0.5 * np.sqrt(2 / 99):.1f}",
    "SBhpN": N_HP,
    "SBhpKsTwo": ks2[2], "SBhpKsTen": ks2[10], "SBhpKsFifty": ks2[50],
    "SBhpVrTwo": var_hp[2], "SBhpVrFifty": var_hp[50],
})
