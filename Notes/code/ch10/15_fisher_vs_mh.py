"""15_fisher_vs_mh.py -- when does the gravitational-wave Fisher ellipse describe the posterior?

Question: for the benchmark IMRI of 12_ppe_scan.py (1e3 + 1.4 Msun, last 5 yr, LISA), a power-law dephasing
dPsi = A (f/f_lo)^{...} = A (u/u_lo)^b with the phase shift A at the start frequency f_lo and the exponent b both
free, and reduced parameters theta = (ln Mc, t_c, A, b) (eta held at its true value, the phase phi_c marginalised
analytically), how do the Metropolis--Hastings posterior (sampler of chapter 9) and the Fisher ellipse compare as
the signal-to-noise ratio grows?  The data are noise-free (d = h(theta_true)), the median experiment.
  log L(theta) = -rho^2 + ln I0(rho^2 |Z(theta)|),  Z = int w(f) exp(i[Psi(theta) - Psi_true]) df,
  w = 4|h|^2/(S_n rho^2) normalised to one: the phase phi_c has been integrated out with a flat prior.
Truth: b = -4, A chosen so that A = 3 sigma_A at rho = 10; then rho = 10, 20, 40, 80 at fixed A.
Also: Vallisneri's mismatch |log r| on the Fisher 1 sigma surface for each rho, and a one-sided prior (A <= 0, the
sign of a dissipative effect) on a vacuum signal: the posterior is a half-Gaussian, not the Fisher ellipse.
Writes: figures/ch10/mh_banana.pdf, mh_widths.pdf, results/ch10/15_fisher_vs_mh.tex, data/ch10/mh_chains.npz
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch09"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0e
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA
import lib_gw10 as L
from lib_mcmc import metropolis

setup()
rng = rng_for("ch10", "15_fisher_vs_mh")
t0 = time.time()
m1, m2 = 1e3, 1.4
Mc, eta = L.chirp_mass(m1, m2), L.sym_ratio(m1, m2)
fisco = L.f_isco(m1 + m2)
f_lo = L.f_of_tau(L.tau_of_f(fisco, Mc) + 5 * L.YEAR, Mc)
f = L.logfgrid(f_lo, 1.0, 3000)
Sn = L.Sn_robson(f, "4yr")
w = L.amp_avg(f, Mc, 1.0) ** 2 / Sn
w /= L.integ(f, w)                          # normalised weight; the SNR enters only as rho^2 below
dlf = np.diff(np.log(f))
fw = f * w
u = L.u_of(f, Mc)
B_TRUE = -4.0
lnMc0 = np.log(Mc)


def psi(th):
    lnMc, tc, A, b = th
    M_ = np.exp(lnMc)
    return (L.psi_pn(f, M_, eta, tc=tc, phic=0.0) + A * (L.u_of(f, M_) / L.u_of(f_lo, M_)) ** b)


def Z(th, psi_true):
    y = fw * np.exp(1j * (psi(th) - psi_true))
    return np.sum(0.5 * (y[1:] + y[:-1]) * dlf)


def loglike(th, psi_true, rho):
    x = rho**2 * np.abs(Z(th, psi_true))
    return np.log(i0e(x)) + x - rho**2


def fisher4(th, rho):
    """Fisher matrix of (ln Mc, tc, A, b) with phi_c marginalised (Schur complement of the 5 x 5 matrix)."""
    lnMc, tc, A, b = th
    M_ = np.exp(lnMc)
    dM, _ = L.dpsi_pn(f, M_, eta)
    g = (u / L.u_of(f_lo, M_)) ** b
    rows = np.array([dM, 2 * np.pi * f, g, A * np.log(u / L.u_of(f_lo, M_)) * g, -np.ones_like(f)])
    F = rho**2 * np.array([[L.integ(f, w * ri * rj) for rj in rows] for ri in rows])
    return F[:4, :4] - np.outer(F[:4, 4], F[4, :4]) / F[4, 4]


# the true A: 3 sigma at rho = 10
th_probe = np.array([lnMc0, 0.0, 1.0, B_TRUE])
sigA10 = np.sqrt(L.inv_scaled(fisher4(th_probe, 10.0))[2, 2])
A_TRUE = -3 * sigA10                         # friction-like sign: the sweep is faster, dPsi < 0
TH = np.array([lnMc0, 0.0, A_TRUE, B_TRUE])
psi_true = psi(TH)
nums = dict(TenBmhAtrue=abs(A_TRUE), TenBmhSigATen=sigA10, TenBmhflo=1e3 * f_lo)


def logpost(th, rho, lo=(-7.5, -2.0)):
    if not (lo[0] < th[3] < lo[1]) or abs(th[2]) > 60:
        return -np.inf
    return loglike(th, psi_true, rho)


RHOS = [10.0, 20.0, 40.0, 80.0]
chains, fishers = {}, {}
NSTEP = 40000
for rho in RHOS:
    F = fisher4(TH, rho)
    C = L.inv_scaled(F)
    fishers[rho] = C
    cov = C.copy()
    x0 = TH.copy()
    for it in range(3):                      # short tuning runs: proposal covariance from the previous chain
        ch, lp, acc = metropolis(lambda x: logpost(x, rho), x0, 6000, 2.38 / 2, rng, cov=cov)
        burn = ch[2000:]
        cov = np.cov(burn.T) + 1e-12 * np.diag(np.diag(C))
        x0 = burn[-1]
    ch, lp, acc = metropolis(lambda x: logpost(x, rho), x0, NSTEP, 2.38 / 2, rng, cov=cov)
    ch = ch[NSTEP // 10:]
    chains[rho] = ch
    tag = {10.0: "Ten", 20.0: "Twenty", 40.0: "Forty", 80.0: "Eighty"}[rho]
    sF = np.sqrt(np.diag(C))
    sM = ch.std(axis=0)
    q16, q84 = np.percentile(ch, [15.865, 84.135], axis=0)
    for i, nm in enumerate(("Mc", "Tc", "A", "B")):
        nums[f"TenBmhF{nm}{tag}"] = sF[i]
        nums[f"TenBmhM{nm}{tag}"] = sM[i]
        nums[f"TenBmhH{nm}{tag}"] = 0.5 * (q84[i] - q16[i])
        nums[f"TenBmhR{nm}{tag}"] = sM[i] / sF[i]
    nums[f"TenBmhAcc{tag}"] = acc
    # fraction of samples inside the Fisher 68.3% ellipse in the (A, b) plane
    Cab = C[2:, 2:]
    dx = ch[:, 2:] - TH[2:]
    q = np.einsum("ni,ij,nj->n", dx, np.linalg.inv(Cab), dx)
    nums[f"TenBmhIn{tag}"] = 100 * np.mean(q < 2.30)
    nums[f"TenBmhAzero{tag}"] = 100 * np.mean(ch[:, 2] > 0)
    # Vallisneri's mismatch on the 1 sigma surface (4 parameters + phi_c handled as a profiled phase)
    d4 = np.sqrt(np.diag(C))
    lam, V = np.linalg.eigh(C / np.outer(d4, d4))
    Lc = (V * np.sqrt(np.clip(lam, 0, None))) * d4[:, None]
    lr = []
    for _ in range(200):
        z = rng.standard_normal(4)
        th = TH + Lc @ (z / np.linalg.norm(z))
        # exact minus linearised log-likelihood drop; Fisher predicts a drop of 1/2 on the 1 sigma surface
        lr.append(abs(-(loglike(th, psi_true, rho) - loglike(TH, psi_true, rho)) - 0.5))
    nums[f"TenBmhLogr{tag}"] = np.median(lr)
    print(rho, "acc", acc, "MH/Fisher", np.round(sM / sF, 3), "logr", np.median(lr), f"{time.time() - t0:.0f}s")

# one-sided prior on a vacuum signal: A <= 0, b fixed, rho = 13 (the IMRI at 76 Mpc)
rho_v = 13.25
TH0 = np.array([lnMc0, 0.0, 0.0, B_TRUE])
psi0 = psi(TH0)


def logpost_v(x):
    th = np.array([x[0], x[1], x[2], B_TRUE])
    if th[2] > 0:
        return -np.inf
    xx = rho_v**2 * np.abs(Z(th, psi0))
    return np.log(i0e(xx)) + xx - rho_v**2


C3 = L.inv_scaled(fisher4(np.array([lnMc0, 0.0, 0.0, B_TRUE]), rho_v)[:3, :3])   # b known: drop it before inverting
chv, _, accv = metropolis(logpost_v, np.array([lnMc0, 0.0, -0.1]), 40000, 2.38 / np.sqrt(3), rng, cov=C3)
chv = chv[4000:]
sAv = np.sqrt(C3[2, 2])
nums.update(TenBmhVacSig=sAv, TenBmhVacUpper=-np.percentile(chv[:, 2], 5), TenBmhVacUpperRatio=-np.percentile(chv[:, 2], 5) / sAv,
            TenBmhVacMean=-chv[:, 2].mean())
nums["TenBmhRuntime"] = time.time() - t0
save_numbers("ch10", "15_fisher_vs_mh", L.tidy(nums))
np.savez(DATA / "ch10" / "mh_chains.npz", **{f"rho{int(r)}": chains[r][::5] for r in RHOS}, vac=chv[::5])
print(nums)


# ---------------------------------------------------------------- figures
def ellipse(C2, centre, k=2.30, n=200):
    lam, V = np.linalg.eigh(C2)
    t = np.linspace(0, 2 * np.pi, n)
    xy = V @ (np.sqrt(k * lam)[:, None] * np.vstack([np.cos(t), np.sin(t)]))
    return centre[0] + xy[0], centre[1] + xy[1]


fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
for col, rho in enumerate((10.0, 40.0)):
    ch, C = chains[rho], fishers[rho]
    for row, (i, j, lab) in enumerate(((2, 3, (r"$A=\delta\Psi(f_{\rm lo})$ [rad]", "$b$")),
                                       (0, 3, (r"$\ln\mathcal{M}-\ln\mathcal{M}_{\rm true}$", "$b$")))):
        ax = axes[row, col]
        xs = ch[:, i] - (lnMc0 if i == 0 else 0)
        ax.plot(xs[::4], ch[::4, j], ".", ms=1.2, color=SERIES[0], alpha=0.4, rasterized=True)
        cx = TH[i] - (lnMc0 if i == 0 else 0)
        for k, ls in ((2.30, "-"), (6.18, "--")):
            ex, ey = ellipse(C[np.ix_([i, j], [i, j])], (cx, TH[j]), k)
            ax.plot(ex, ey, color="k", ls=ls, lw=1.1)
        ax.plot(cx, TH[j], "+", color=SERIES[1], ms=10, mew=2)
        ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
        ax.set_title(rf"$\rho={rho:.0f}$", fontsize=9)
fig.tight_layout()
savefig(fig, "ch10", "mh_banana")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
rr = np.array(RHOS)
for k, (nm, lab) in enumerate((("A", "$A$"), ("B", "$b$"), ("Mc", r"$\ln\mathcal{M}$"), ("Tc", "$t_c$"))):
    tag = lambda r: {10.0: "Ten", 20.0: "Twenty", 40.0: "Forty", 80.0: "Eighty"}[r]
    ax[0].plot(rr, [nums[f"TenBmhR{nm}{tag(r)}"] for r in rr], "o-", color=SERIES[k], label=lab)
ax[0].axhline(1, color="k", lw=0.8, ls=":")
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel(r"$\sigma_{\rm MH}/\sigma_{\rm Fisher}$")
ax[0].legend(fontsize=8)
ax[1].loglog(rr, [nums[f"TenBmhLogr{tag(r)}"] for r in rr], "o-", color=SERIES[0], label=r"median $|\log r|$ on the 1$\sigma$ surface")
ax[1].axhline(0.1, color="k", lw=0.8, ls=":")
ax[1].set_xlabel(r"$\rho$"); ax[1].legend(fontsize=8)
for a_ in ax:
    a_.set_xticks(RHOS); a_.set_xticklabels([f"{r:.0f}" for r in RHOS])
    a_.xaxis.set_minor_formatter(plt.NullFormatter()); a_.yaxis.set_minor_formatter(plt.NullFormatter())
fig.tight_layout()
savefig(fig, "ch10", "mh_widths")
