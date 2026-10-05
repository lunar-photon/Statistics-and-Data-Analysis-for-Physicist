"""13_marginal.py -- nuisance parameters: marginalise (integrate out) or profile (maximise out)?

Question: the posterior depends on a parameter we want and one we do not.  How do we get
the posterior of the one we want, and when do integrating and maximising disagree?

Computes:
  (a) Sivia's signal peak on a flat background, Poisson counts: joint posterior for the
      amplitude A and background B on a grid for two set-ups (15 bins over a wide range,
      7 bins over a narrow range); marginal posterior of A against the conditional
      posterior of A given B = 2; the A-B correlation coefficient;
  (b) a weak bump of unknown position: y_i = A exp(-(x_i - mu)^2 / 2w^2) + noise.  The
      marginal posterior of A (integrate over mu) against the profile (maximise over mu):
      the marginal is pulled towards A = 0 by the prior volume in mu.

Writes: figures/ch05/marg_signal_bkg.pdf, figures/ch05/marg_volume.pdf,
        results/ch05/13_marginal.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch05", "13_marginal")
setup()

# ---------------------------------------------------------------- (a) signal + background
A_T, B_T, W = 1.0, 2.0, 2.12                            # FWHM = 5
A = np.linspace(0, 3, 301); B = np.linspace(0, 3.5, 351)
dA, dB = A[1] - A[0], B[1] - B[0]
AA, BB = np.meshgrid(A, B, indexing="ij")


def analyse(xk, n0):
    D_true = n0 * (A_T * np.exp(-xk**2 / (2 * W**2)) + B_T)
    Nk = rng.poisson(D_true)
    Dk = n0 * (AA[..., None] * np.exp(-xk**2 / (2 * W**2)) + BB[..., None])
    L = np.sum(Nk * np.log(np.maximum(Dk, 1e-300)) - Dk, axis=-1)   # Sivia (3.8), flat prior A,B >= 0
    P = np.exp(L - L.max()); P /= P.sum() * dA * dB
    margA = P.sum(axis=1) * dB
    jB = np.argmin(np.abs(B - B_T))
    condA = P[:, jB] / (P[:, jB].sum() * dA)
    mA = np.sum(A * margA) * dA
    margB = P.sum(axis=0) * dA
    mB = np.sum(B * margB) * dB
    cov = np.sum((AA - mA) * (BB - mB) * P) * dA * dB
    sA = np.sqrt(np.sum((A - mA) ** 2 * margA) * dA)
    sB = np.sqrt(np.sum((B - mB) ** 2 * margB) * dB)
    sAc = np.sqrt(np.sum((A - np.sum(A * condA) * dA) ** 2 * condA) * dA)
    return dict(xk=xk, Nk=Nk, P=P, margA=margA, condA=condA, rho=cov / (sA * sB), sA=sA, sAc=sAc)


setups = [analyse(np.linspace(-7, 7, 15), 100 / 3), analyse(np.linspace(-3, 3, 7), 100 / 3)]
fig, ax = plt.subplots(2, 3, figsize=(7.8, 4.4))
for row, r in zip(ax, setups):
    row[0].step(r["xk"], r["Nk"], where="mid", color=SERIES[0])
    row[0].set_xlim(-7.5, 7.5); row[0].set_ylim(0, None)
    row[0].set_xlabel("measurement variable $x$"); row[0].set_ylabel("counts $N_k$")
    lev = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) * r["P"].max()
    row[1].contour(A, B, r["P"].T, levels=lev, colors=[SERIES[0]], linewidths=0.8)
    row[1].plot(A_T, B_T, "+", color=INK2)
    row[1].set_xlabel("amplitude $A$"); row[1].set_ylabel("background $B$")
    row[1].set_title(rf"correlation $\rho={r['rho']:.2f}$", fontsize=8)
    row[2].plot(A, r["margA"], color=SERIES[0], label="marginal")
    row[2].plot(A, r["condA"], color=SERIES[1], ls="--", label="given $B=2$")
    row[2].set_xlabel("amplitude $A$"); row[2].set_yticks([]); row[2].legend(fontsize=6)
fig.tight_layout()
savefig(fig, "ch05", "marg_signal_bkg")

# ---------------------------------------------------------------- (b) volume effect
xs = np.linspace(0, 10, 60)
A2_T, MU_T, W2, SIG = 1.2, 5.0, 0.4, 1.0
y = A2_T * np.exp(-(xs - MU_T) ** 2 / (2 * W2**2)) + SIG * rng.standard_normal(xs.size)
A2 = np.linspace(0, 4, 401); mu = np.linspace(0, 10, 1001)
dA2, dmu = A2[1] - A2[0], mu[1] - mu[0]
tmpl = np.exp(-(xs[None, :] - mu[:, None]) ** 2 / (2 * W2**2))         # (n_mu, n_x)
# chi^2(A, mu) = sum (y - A t)^2 = yy - 2 A (y.t) + A^2 (t.t)
yt, tt, yy = tmpl @ y, np.sum(tmpl**2, axis=1), y @ y
chi2 = yy - 2 * A2[:, None] * yt[None, :] + A2[:, None] ** 2 * tt[None, :]
L2 = -chi2 / (2 * SIG**2)
P2 = np.exp(L2 - L2.max()); P2 /= P2.sum() * dA2 * dmu
marg = P2.sum(axis=1) * dmu
prof = np.exp(L2.max(axis=1) - L2.max()); prof /= prof.sum() * dA2
null_prob = None
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.7))
ax[0].plot(xs, y, ".", color=SERIES[0], ms=3)
ax[0].plot(xs, A2_T * np.exp(-(xs - MU_T) ** 2 / (2 * W2**2)), color="k", ls="--", lw=1)
ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$"); ax[0].set_title("data and true bump", fontsize=8)
lev = np.array([0.05, 0.2, 0.5, 0.8]) * P2.max()
ax[1].contour(mu, A2, P2, levels=lev, colors=[SERIES[0]], linewidths=0.8)
ax[1].set_xlabel(r"position $\mu$"); ax[1].set_ylabel("amplitude $A$")
ax[1].set_title("joint posterior", fontsize=8)
ax[2].plot(A2, marg, color=SERIES[0], label=r"marginal $\int d\mu$")
ax[2].plot(A2, prof, color=SERIES[1], ls="--", label=r"profile $\max_\mu$")
ax[2].axvline(A2_T, color=INK2, ls=":", lw=0.8)
ax[2].set_xlabel("amplitude $A$"); ax[2].set_yticks([]); ax[2].legend(fontsize=6)
fig.tight_layout()
savefig(fig, "ch05", "marg_volume")

margmode, profmode = A2[np.argmax(marg)], A2[np.argmax(prof)]
margmean = np.sum(A2 * marg) * dA2
p_small_marg = np.sum(marg[A2 < 0.5]) * dA2
p_small_prof = np.sum(prof[A2 < 0.5]) * dA2

# the same analysis on many fresh data sets: is the pull of the marginal towards zero typical?
rng_loop = rng_for("ch05", "13_marginal_loop")
R_VOL = 500
truth = A2_T * np.exp(-(xs - MU_T) ** 2 / (2 * W2**2))
ps_marg, ps_prof, mode0 = np.empty(R_VOL), np.empty(R_VOL), np.empty(R_VOL, bool)
for r in range(R_VOL):
    yr = truth + SIG * rng_loop.standard_normal(xs.size)
    Lr = -(yr @ yr - 2 * A2[:, None] * (tmpl @ yr)[None, :] + A2[:, None] ** 2 * tt[None, :]) / (2 * SIG**2)
    Pr = np.exp(Lr - Lr.max())
    mr = Pr.sum(axis=1); mr /= mr.sum() * dA2
    pr = Pr.max(axis=1); pr /= pr.sum() * dA2
    ps_marg[r], ps_prof[r] = np.sum(mr[A2 < 0.5]) * dA2, np.sum(pr[A2 < 0.5]) * dA2
    mode0[r] = np.argmax(mr) == 0
s1, s2 = setups
save_numbers("ch05", "13_marginal", {
    "FiveBMaRhoWide": s1["rho"], "FiveBMaRhoNarrow": s2["rho"],
    "FiveBMaSigWide": s1["sA"], "FiveBMaSigCondWide": s1["sAc"],
    "FiveBMaSigNarrow": s2["sA"], "FiveBMaSigCondNarrow": s2["sAc"],
    "FiveBVolMargMode": margmode, "FiveBVolProfMode": profmode, "FiveBVolMargMean": margmean,
    "FiveBVolPsmallMarg": p_small_marg, "FiveBVolPsmallProf": p_small_prof,
    "FiveBVolAtrue": A2_T,
    "FiveBVolLoopR": R_VOL,
    "FiveBVolLoopPsmallMarg": f"{ps_marg.mean():.2f}",
    "FiveBVolLoopPsmallProf": f"{ps_prof.mean():.2f}",
    "FiveBVolLoopErr": f"{max(ps_marg.std(), ps_prof.std()) / np.sqrt(R_VOL):.2f}",
    "FiveBVolLoopModeZero": int(mode0.sum()),
})
