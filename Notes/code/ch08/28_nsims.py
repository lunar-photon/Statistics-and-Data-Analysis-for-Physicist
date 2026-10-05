"""28_nsims.py -- how many simulations a covariance needs: noise, the Hartlap factor, Taylor's rule.

Question: a covariance matrix estimated from N simulations is itself a random matrix.  How noisy
are its elements, why is its inverse biased high (Hartlap, Simon & Schneider 2007), and how many
simulations make a parameter error trustworthy to a given precision (Taylor, Joachimi &
Kitching 2013)?
Computes:
  (1) element noise: from the 2000 MASTER estimates of the 0.3 cap (p = 30 bins below l = 600),
      subsets of N skies; rms fractional error of the diagonal against sqrt(2/(N-1)) and rms of
      the off-diagonal correlations against 1/sqrt(N);
  (2) the Hartlap experiment (their Fig. 1): n = 60 Gaussian vectors, p1 = 240 bins rebinned to
      p = 240/j, three population covariances (their eqs 18-20), 2000 repetitions; the ratio
      tr(Sigma^-1) / tr(C^-1) with and without the factor (n-p-2)/(n-1);
  (3) the same on the CMB: subsets of n skies, p = 30 MASTER bins of the 0.3 cap, against the
      inverse of the covariance from all 2000 skies (itself Hartlap-corrected);
  (4) Taylor's rule: the amplitude A of the spectrum (D_b = A D_b^fid) fitted with a precision
      matrix from N_S Gaussian draws; scatter of sigma_A^2 over 2000 repetitions against
      sqrt(2/(N_S - N_D - 4)) (their eq. 55).
Writes: figures/ch08/cov_noise.pdf, figures/ch08/hartlap.pdf, results/ch08/28_nsims.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
import lib_masks as lm

setup()
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"
ms = np.load(DATA / "master.npz")
rep = ms["edges"][1:] <= 601
D = ms["master_cap30"][:, rep].astype(float)          # (2000, p)
NALL, P = D.shape
rng = rng_for("ch08", "28_nsims")
nums = {"EightBpcov": P, "EightBNall": NALL}

Ctrue = np.cov(D, rowvar=False)
sd = np.sqrt(np.diag(Ctrue))

# ---------------------------------------------------------------- (1) element noise
Ns = np.array([25, 50, 100, 200, 400, 1000])
diag_err, off_err = [], []
for n in Ns:
    de, oe = [], []
    for _ in range(200):
        idx = rng.choice(NALL, n, replace=False)
        C = np.cov(D[idx], rowvar=False)
        de.append(np.diag(C) / np.diag(Ctrue) - 1)
        R = C / np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))
        R0 = Ctrue / np.outer(sd, sd)
        oe.append((R - R0)[np.triu_indices(P, 1)])
    diag_err.append(np.sqrt(np.mean(np.square(de))))
    off_err.append(np.sqrt(np.mean(np.square(oe))))
diag_err, off_err = np.array(diag_err), np.array(off_err)
for n, a, b in zip(Ns, diag_err, off_err):
    print(f"N={n:5d}: rms diag error {a:.3f} (sqrt(2/(N-1)) = {np.sqrt(2/(n-1)):.3f}), "
          f"rms off-diag corr error {b:.3f} (1/sqrt(N) = {1/np.sqrt(n):.3f})")
nums["EightBdiagErrHundred"] = f"{diag_err[2]:.3f}"
nums["EightBoffErrHundred"] = f"{off_err[2]:.3f}"

# ---------------------------------------------------------------- (2) the Hartlap experiment
p1, n_h, REP = 240, 60, 2000
ii = np.arange(p1)
models = {
    "diag, const": np.eye(p1),
    "diag, linear": np.diag(1 - ii / (1 + p1)),
    "non-diagonal": 1.0 / (1 + 0.05 * np.abs(ii[:, None] - ii[None, :])),
}
js = [j for j in range(4, p1 // 2 + 1) if p1 % j == 0]              # p = p1/j, keep p < n - 2
ps = np.array([p1 // j for j in js])
keep = ps < n_h - 2
ps = ps[keep]
js = [j for j, k in zip(js, keep) if k]
hart = {}
for name, S1 in models.items():
    Lc = np.linalg.cholesky(S1)
    ratio_raw = np.zeros(len(ps))
    ratio_cor = np.zeros(len(ps))
    for r in range(REP):
        d1 = rng.standard_normal((n_h, p1)) @ Lc.T                    # n vectors of length p1
        for t, j in enumerate(js):
            d = d1.reshape(n_h, p1 // j, j).mean(axis=2)            # rebin: average j neighbours
            C = np.cov(d, rowvar=False)
            tri = np.trace(np.linalg.inv(C))
            ratio_raw[t] += tri
    for t, j in enumerate(js):
        Sj = S1.reshape(p1 // j, j, p1 // j, j).mean(axis=(1, 3))   # rebinned population covariance
        tr_true = np.trace(np.linalg.inv(Sj))
        mean_tri = ratio_raw[t] / REP
        ratio_cor[t] = tr_true / (mean_tri * (n_h - ps[t] - 2) / (n_h - 1))
        ratio_raw[t] = tr_true / mean_tri
    hart[name] = (ratio_raw.copy(), ratio_cor.copy())
    print(name, "p:", ps, "raw:", np.round(ratio_raw, 3), "corrected:", np.round(ratio_cor, 3))
nums["EightBhartlapN"] = n_h
nums["EightBhartlapPone"] = p1
nums["EightBhartlapRep"] = REP
t40 = int(np.argmin(np.abs(ps - 40)))
nums["EightBhartlapPforty"] = int(ps[t40])
nums["EightBhartlapRawForty"] = f"{hart['diag, const'][0][t40]:.3f}"
nums["EightBhartlapPredForty"] = f"{(n_h - ps[t40] - 2) / (n_h - 1):.3f}"
nums["EightBhartlapCorWorst"] = f"{max(np.max(np.abs(v[1] - 1)) for v in hart.values()):.3f}"

# ---------------------------------------------------------------- (3) the same on the CMB simulations
Ctrue_inv = np.linalg.inv(Ctrue) * (NALL - P - 2) / (NALL - 1)        # unbiased precision from 2000 skies
tr_true = np.trace(Ctrue_inv)
n_cmb = np.array([40, 50, 60, 80, 120, 200, 400])
cmb_raw = []
for n in n_cmb:
    tr = []
    for _ in range(300):
        idx = rng.choice(NALL, n, replace=False)
        tr.append(np.trace(np.linalg.inv(np.cov(D[idx], rowvar=False))))
    cmb_raw.append(tr_true / np.mean(tr))
cmb_raw = np.array(cmb_raw)
print("CMB: n", n_cmb, "tr ratio", np.round(cmb_raw, 3), "Hartlap", np.round((n_cmb - P - 2) / (n_cmb - 1), 3))
nums["EightBcmbRatioSixty"] = f"{cmb_raw[2]:.2f}"
nums["EightBcmbHartSixty"] = f"{(60 - P - 2) / 59:.2f}"
nums["EightBhartlapAll"] = f"{(NALL - P - 2) / (NALL - 1):.3f}"

# ---------------------------------------------------------------- (4) Taylor's rule for a parameter error
t = ms["Dbin_true"][rep]                                    # dD_b/dA at A = 1
Lt = np.linalg.cholesky(Ctrue)
sigA2_true = 1.0 / (t @ np.linalg.solve(Ctrue, t))
NSs = np.array([40, 50, 70, 100, 150, 250, 500, 1000])
tay = []
for ns in NSs:
    s2 = np.empty(2000)
    for r in range(2000):
        d = rng.standard_normal((ns, P)) @ Lt.T
        Psi = np.linalg.inv(np.cov(d, rowvar=False)) * (ns - P - 2) / (ns - 1)   # unbiased precision
        s2[r] = 1.0 / (t @ Psi @ t)
    tay.append((s2.mean() / sigA2_true, s2.std() / s2.mean()))
    print(f"N_S={ns:5d}: <sigma_A^2>/true = {tay[-1][0]:.3f}, frac scatter {tay[-1][1]:.3f}, "
          f"Taylor sqrt(2/(N_S-N_D-4)) = {np.sqrt(2/(ns-P-4)):.3f}")
tay = np.array(tay)
nums["EightBtaylorBiasForty"] = f"{100 * (tay[0, 0] - 1):.0f}"
nums["EightBtaylorScatForty"] = f"{tay[0, 1]:.2f}"
nums["EightBtaylorPredForty"] = f"{np.sqrt(2 / (40 - P - 4)):.2f}"
nums["EightBsigAtrue"] = f"{np.sqrt(sigA2_true):.4f}"
i100 = int(np.where(NSs == 100)[0][0])
nums["EightBtaylorHundred"] = f"{tay[i100, 1]:.3f}"
nums["EightBtaylorPredHundred"] = f"{np.sqrt(2 / (100 - P - 4)):.3f}"
nums["EightBtaylorFivePct"] = f"{2 / 0.1 ** 2 + P + 4:.0f}"     # NS for 10% on the variance = 5% on the error

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(6.0, 3.3))
ax.loglog(Ns, diag_err, "o", color=SERIES[0], label="diagonal: rms fractional error")
fpc = np.sqrt(1 - Ns / NALL)                        # subsets share skies with the 2000-sky reference
theory_line(ax, Ns, np.sqrt(2 / (Ns - 1)) * fpc, label=r"$\sqrt{2/(N-1)}\,(1-N/N_{\rm all})^{1/2}$")
ax.loglog(Ns, off_err, "s", color=SERIES[1], label="off-diagonal correlations: rms error")
ax.loglog(Ns, fpc / np.sqrt(Ns), color=SERIES[1], ls=":", lw=1.2, label=r"$N^{-1/2}\,(1-N/N_{\rm all})^{1/2}$")
ax.set_xlabel("number of simulations $N$")
ax.set_ylabel("error of the estimated covariance")
ax.legend(fontsize=8)
savefig(fig, "ch08", "cov_noise")

fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.3))
ax = axs[0]
mk = ["^", "s", "o"]
for (name, (raw, cor)), c, m in zip(hart.items(), SERIES, mk):
    ax.plot(ps / n_h, raw, m, color=c, ms=4, mfc="none", label=f"$\\hat C^{{-1}}$, {name}")
    ax.plot(ps / n_h, cor, m, color=c, ms=4, label=f"debiased, {name}")
g = np.linspace(0, 1, 50)
theory_line(ax, g, 1 - g * n_h / (n_h - 1) - 1 / (n_h - 1), label=r"$(n-p-2)/(n-1)$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.15)
ax.set_xlabel("$p/n$")
ax.set_ylabel(r"tr$\,\Sigma^{-1}\,/\,$tr$\,\langle\hat\Psi\rangle$")
ax.set_title(f"(a) Hartlap et al.'s test, $n={n_h}$", fontsize=9)
ax.legend(fontsize=6.5, loc="lower left")
ax = axs[1]
ax.plot(n_cmb, cmb_raw, "o", color=SERIES[0], label=f"CMB, $p={P}$ MASTER bins")
nn = np.linspace(P + 3, 420, 200)
theory_line(ax, nn, (nn - P - 2) / (nn - 1), label=r"$(n-p-2)/(n-1)$")
ax2 = ax.twinx()
ax2.plot(NSs, tay[:, 1], "s", color=SERIES[1], label=r"scatter of $\sigma_A^2$")
nt = np.linspace(P + 6, 1000, 300)
ax2.plot(nt, np.sqrt(2 / (nt - P - 4)), color=SERIES[1], ls=":", lw=1.2, label=r"$\sqrt{2/(N_S-N_D-4)}$")
ax2.set_ylim(0, 1.2)
ax2.set_ylabel(r"fractional scatter of $\sigma_A^2$", color=SERIES[1])
ax2.grid(False)
ax.set_xscale("log")
ax.set_ylim(0, 1.15)
ax.set_xlabel("number of simulations")
ax.set_ylabel("trace ratio")
ax.set_title("(b) the CMB covariance: bias and noise", fontsize=9)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="center right")
savefig(fig, "ch08", "hartlap")
save_numbers("ch08", "28_nsims", nums)
