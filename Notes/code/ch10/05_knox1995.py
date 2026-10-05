"""05_knox1995.py -- Knox (1995) redone: a Fisher forecast against 100 simulated experiments.

Question: Knox (1995, astro-ph/9504054) asked how well a full-sky satellite would measure the
inflationary observables Q_S (scalar quadrupole), n_S (scalar tilt) and r = Q_T^2/Q_S^2. He drew
C_l^est from its chi^2_{2l+1} distribution 100 times per experiment, maximised the exact likelihood
each time and quoted the standard deviation of the maxima (his Table 1, 'numerical' columns).
A Fisher matrix computed from the model alone should predict those numbers. Does it?

Model (Knox's): standard CDM, h = 0.5, Omega_b h^2 = 0.0125, Omega = 1, no Lambda, no reionisation;
n_S = 0.94, n_T = -0.04 (fixed), r = 0.28, Q_S = 20 muK; spectra from CAMB, unlensed.
C_l = (4 pi / 5) Q_S^2 [ s_l(n_S) + r t_l ], s_l = C_l^S / C_2^S and t_l = C_l^T / C_2^T (quadrupole-normalised).
Noise: w^-1 exp(l^2 sigma_b^2), w^-1/2 = 15 or 30 muK deg, FWHM = 20', 30', 40'; full sky, 2 <= l <= 1500.

Computes: the 3x3 Fisher matrix for the six experiments of Knox's Table 1; marginalised errors on
n_S and r; the pivot l* at which the amplitude decorrelates from n_S and the error of that
amplitude; and, for the (20', 15 muK deg) experiment, 2000 simulated experiments fitted by maximum
likelihood (our own Monte Carlo, the same procedure as Knox's but 20 times larger).
Writes: figures/ch10/knox_mle.pdf, figures/ch10/knox_table.pdf, data/ch10/knox_spectra.npz,
        results/ch10/05_knox1995.tex
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from common import setup, savefig, save_numbers, rng_for, SERIES

setup()
rng = rng_for("ch10", "05_knox1995")
NOTES = pathlib.Path(__file__).resolve().parents[2]
DATA = NOTES / "data" / "ch10"
DATA.mkdir(parents=True, exist_ok=True)
LMAX = 1500
l = np.arange(LMAX + 1)
QS0, NS0, R0, NT0 = 20.0, 0.94, 0.28, -0.04
NGRID = np.round(np.arange(0.80, 1.0801, 0.02), 2)          # n_S grid for the maximum-likelihood fits
ARC = np.pi / 180 / 60
t_start = time.time()


def camb_scdm(ns):
    """Unlensed scalar and tensor TT (muK^2, raw C_l) for Knox's standard CDM model."""
    import camb
    p = camb.CAMBparams()
    p.set_cosmology(H0=50.0, ombh2=0.0125, omch2=0.25 - 0.0125, mnu=0.0, omk=0.0, tau=None)
    p.Reion.Reionization = False
    p.InitPower.set_params(As=2e-9, ns=ns, r=1.0, nt=NT0)
    p.WantTensors = True
    p.set_for_lmax(LMAX + 300, lens_potential_accuracy=0)
    res = camb.get_results(p)
    cl = res.get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=True)
    return cl["unlensed_scalar"][: LMAX + 1, 0], cl["tensor"][: LMAX + 1, 0]


path = DATA / "knox_spectra.npz"
if path.exists():
    z = np.load(path)
    S_grid, T_shape = z["S_grid"], z["T_shape"]
else:
    S_grid = []
    for ns in NGRID:
        s, t = camb_scdm(ns)
        S_grid.append(s / s[2])
        if abs(ns - NS0) < 1e-9:
            T_shape = t / t[2]
    S_grid = np.array(S_grid)
    np.savez(path, S_grid=S_grid, T_shape=T_shape, ngrid=NGRID)
S_grid[:, :2] = 0.0
T_shape[:2] = 0.0
# ln s_l is smooth in n_S (chapter 9: d^2 ln C / dn^2 is the variance of ln k): a cubic spline per l
spl = interpolate.CubicSpline(NGRID, np.log(np.maximum(S_grid[:, 2:], 1e-300)), axis=0)


def s_of(ns):
    out = np.zeros(LMAX + 1)
    out[2:] = np.exp(spl(ns))
    return out


def ds_dn(ns):
    out = np.zeros(LMAX + 1)
    out[2:] = np.exp(spl(ns)) * spl(ns, 1)
    return out


Q2 = 4 * np.pi / 5


def model(Q, ns, r):
    return Q2 * Q**2 * (s_of(ns) + r * T_shape)


def noise(fwhm_arcmin, w_ukdeg):
    sb = fwhm_arcmin * ARC / np.sqrt(8 * np.log(2))
    winv = (w_ukdeg * np.pi / 180) ** 2               # (muK rad)^2 = muK^2 sr
    return winv * np.exp(l**2 * sb**2)                 # Knox's eq. (4): exp(l^2 sigma_b^2)


C0 = model(QS0, NS0, R0)
dC = np.array([2 * C0 / QS0,                           # d/dQ_S
               Q2 * QS0**2 * ds_dn(NS0),               # d/dn_S
               Q2 * QS0**2 * T_shape])                 # d/dr


def fisher(N):
    c = C0[2:] + N[2:]
    w = (2 * l[2:] + 1) / 2 / c**2
    return (dC[:, 2:] * w) @ dC[:, 2:].T


def summarise(F):
    Cv = np.linalg.inv(F)
    sQ, sn, sr = np.sqrt(np.diag(Cv))
    # pivot: ln Qbar = ln Q + (n - n0)/2 ln(l*/2) is uncorrelated with n when ln(l*/2) = -2 Cov(lnQ, n)/Var(n)
    cov_lnq_n = Cv[0, 1] / QS0
    lpiv = 2 * np.exp(-2 * cov_lnq_n / Cv[1, 1])
    var_lnqbar = Cv[0, 0] / QS0**2 - cov_lnq_n**2 / Cv[1, 1]
    return dict(sQ=sQ / QS0, sn=sn, sr=sr, lpiv=lpiv, sQbar=np.sqrt(var_lnqbar))


# Knox's Table 1: fwhm, w^-1/2, l_C, dQbar/Qbar (analytic, numerical), dn (an, num), dr (an, num)
KNOX = [(20, 15, 260, 0.0025, 0.0027, 0.012, 0.016, 0.10, 0.11),
        (20, 30, 210, 0.0039, 0.0041, 0.016, 0.029, 0.14, 0.18),
        (30, 15, 233, 0.0028, 0.0028, 0.013, 0.022, 0.12, 0.14),
        (30, 30, 191, 0.0043, 0.0047, 0.017, 0.044, 0.19, 0.23),
        (40, 15, 207, 0.0032, 0.0035, 0.015, 0.034, 0.16, 0.19),
        (40, 30, 168, 0.0050, 0.0050, 0.019, 0.058, 0.22, 0.42)]
nums = {}
rows = []
words = ["One", "Two", "Three", "Four", "Five", "Six"]
for k, (fw, w, lc, qa, qn, na, nn, ra, rn) in enumerate(KNOX):
    sm = summarise(fisher(noise(fw, w)))
    rows.append(sm)
    W = words[k]
    nums.update({f"TenEKn{W}Sn": f"{sm['sn']:.3f}", f"TenEKn{W}Sr": f"{sm['sr']:.2f}",
                 f"TenEKn{W}Sq": f"{sm['sQbar']:.4f}", f"TenEKn{W}Piv": f"{sm['lpiv']:.0f}",
                 f"TenEKn{W}SQ": f"{sm['sQ']:.3f}"})

# ---------------------------------------------------------------- our Monte Carlo of maximum-likelihood fits
FW, W = 20, 15
N = noise(FW, W)
nu = 2 * l + 1.0
use = slice(2, LMAX + 1)
NSIM = 2000


def m2lnL(p, chat):
    Q, ns, r = p
    if not (NGRID[0] < ns < NGRID[-1]) or Q <= 0:
        return 1e30
    c = model(Q, ns, r)[use] + N[use]
    return np.sum(nu[use] * (chat[use] / c + np.log(c)))


fits = np.empty((NSIM, 3))
Ctrue = C0 + N
for k in range(NSIM):
    chat = np.zeros(LMAX + 1)
    chat[use] = Ctrue[use] * rng.chisquare(nu[use]) / nu[use]
    out = optimize.minimize(m2lnL, x0=[QS0, NS0, R0], args=(chat,), method="Nelder-Mead",
                            options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    fits[k] = out.x
mc = np.cov(fits.T)
msQ, msn, msr = np.sqrt(np.diag(mc))
cov_lnq_n = mc[0, 1] / QS0
mc_lpiv = 2 * np.exp(-2 * cov_lnq_n / mc[1, 1])
mc_sQbar = np.sqrt(mc[0, 0] / QS0**2 - cov_lnq_n**2 / mc[1, 1])
F1 = fisher(N)
rho_F = np.linalg.inv(F1)[0, 1] / np.sqrt(np.linalg.inv(F1)[0, 0] * np.linalg.inv(F1)[1, 1])
nums.update({"TenEKnMCSn": f"{msn:.3f}", "TenEKnMCSr": f"{msr:.2f}", "TenEKnMCSq": f"{mc_sQbar:.4f}",
             "TenEKnMCPiv": f"{mc_lpiv:.0f}", "TenEKnMCSQ": f"{msQ / QS0:.3f}",
             "TenEKnMCRho": f"{mc[0, 1] / (msQ * msn):+.2f}", "TenEKnFRho": f"{rho_F:+.2f}",
             "TenEKnMCMeanN": f"{fits[:, 1].mean():.4f}", "TenEKnMCMeanR": f"{fits[:, 2].mean():.3f}",
             "TenEKnMCMeanQ": f"{fits[:, 0].mean():.3f}",
             "TenEKnTSratio": f"{T_shape[55] / s_of(NS0)[55]:.2f}",
             "TenEKnNsim": NSIM, "TenEKnLmax": LMAX, "TenEKnErr": f"{100 / np.sqrt(2 * NSIM):.1f}",
             "TenEKnMinutes": f"{(time.time() - t_start) / 60:.1f}"})

# ---------------------------------------------------------------- figures
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4))
Cv = np.linalg.inv(F1)
t = np.linspace(0, 2 * np.pi, 300)
for ax, (i, j), lab in [(axs[0], (1, 0), (r"$\hat n_S$", r"$\hat Q_S$ [$\mu$K]")),
                        (axs[1], (1, 2), (r"$\hat n_S$", r"$\hat r$"))]:
    ax.scatter(fits[:, i], fits[:, j], s=3, alpha=0.4, color=SERIES[0], lw=0, label=f"{NSIM} ML fits")
    sub = Cv[np.ix_([i, j], [i, j])]
    Lc = np.linalg.cholesky(sub)
    ctr = np.array([[QS0, NS0, R0][i], [QS0, NS0, R0][j]])   # parameter order of the fit: (Q_S, n_S, r)
    for k2, ls in [(2.30, "-"), (6.18, "--")]:
        e = ctr[:, None] + np.sqrt(k2) * Lc @ np.vstack([np.cos(t), np.sin(t)])
        ax.plot(e[0], e[1], color="k", ls=ls, lw=1.2)
    ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
axs[0].legend(fontsize=7.5, loc="upper left")
axs[0].set_title(r"20$'$ beam, $w^{-1/2}=15\,\mu$K deg: fits and Fisher", fontsize=8.5)
savefig(fig, "ch10", "knox_mle")

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
xk = np.arange(6)
labels = [f"{fw}'/{w}" for fw, w, *_ in KNOX]
axs[0].plot(xk, [r[6] for r in KNOX], "o", color=SERIES[1], label="Knox, 100 simulations")
axs[0].plot(xk, [r[5] for r in KNOX], "^", color=SERIES[3], label="Knox, analytic estimate")
axs[0].plot(xk, [r["sn"] for r in rows], "s", mfc="none", color="k", label="Fisher (ours)")
axs[0].errorbar([0], [msn], yerr=[msn / np.sqrt(2 * NSIM)], fmt="D", color=SERIES[0], ms=4,
                label=f"{NSIM} simulations (ours)")
axs[0].set_ylabel(r"$\sigma(n_S)$")
axs[1].plot(xk, [r[8] for r in KNOX], "o", color=SERIES[1])
axs[1].plot(xk, [r[7] for r in KNOX], "^", color=SERIES[3])
axs[1].plot(xk, [r["sr"] for r in rows], "s", mfc="none", color="k")
axs[1].errorbar([0], [msr], yerr=[msr / np.sqrt(2 * NSIM)], fmt="D", color=SERIES[0], ms=4)
axs[1].set_ylabel(r"$\sigma(r)$")
for ax in axs:
    ax.set_xticks(xk); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlabel(r"FWHM / $w^{-1/2}$ [$\mu$K deg]")
axs[0].legend(fontsize=6.5, loc="upper left")
savefig(fig, "ch10", "knox_table")
save_numbers("ch10", "05_knox1995", nums)
print(nums)
for r, k in zip(rows, KNOX):
    print(k[:2], {a: round(float(b), 4) for a, b in r.items()}, "Knox num", k[4], k[6], k[8])
