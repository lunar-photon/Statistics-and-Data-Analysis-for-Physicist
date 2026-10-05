"""17_bump_scan.py -- the look-elsewhere effect: scanning a background-only spectrum for a bump.

Question: we do not know the mass of the particle, so we test every mass in 105-155 GeV and
report the biggest excess.  If the background-only hypothesis is true, how often does the
biggest excess anywhere reach a given local significance?  What are the local and global
p-values of the largest bump in one background-only pseudo-experiment, and the trials factor?
Does the Gross-Vitells upcrossing formula, calibrated on 100 toys, reproduce the global
p-value found with 20 000 toys?

Computes: binned Poisson spectrum (60 bins of 1 GeV, known exponential background), signal
template Gaussian with sigma_m = 1.5 GeV; for each hypothesised mass m on a 0.25 GeV grid,
the one-sided likelihood-ratio statistic q0(m) (mu >= 0, Newton iterations vectorised over
toys and masses); the maximum over m for 20 000 background-only toys; the upcrossings of a
reference level c0 = 0.5; local vs global p-values; trials factor versus local significance.

Writes: figures/ch04/scan_spectrum.pdf, figures/ch04/scan_tail.pdf,
        results/ch04/17_bump_scan.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "17_bump_scan")
setup()

# ---------- the model ----------
edges = np.arange(100.0, 161.0, 1.0)                  # 60 bins of 1 GeV
centres = 0.5 * (edges[1:] + edges[:-1])
B_TOT, SLOPE = 3000.0, 30.0
cdf_b = lambda x: 1 - np.exp(-(x - edges[0]) / SLOPE)
b = B_TOT * np.diff(cdf_b(edges)) / cdf_b(edges[-1])  # expected background per bin (known)
SIG_M = 1.5
masses = np.arange(105.0, 155.01, 0.25)                # scan grid
S = np.diff(stats.norm.cdf(edges[None, :], masses[:, None], SIG_M), axis=1)   # (M, bins)
S /= S.sum(1, keepdims=True)                           # unit signal: mu = expected signal events

def q0_scan(n):
    """n: (T, bins) counts.  Returns q0 (T, M) and mu_hat (T, M), one-sided (mu >= 0)."""
    n = n[:, None, :].astype(float)                    # (T, 1, bins)
    s = S[None, :, :]                                  # (1, M, bins)
    bb = b[None, None, :]
    g0 = (n * s / bb).sum(-1) - 1.0                    # d lnL / d mu at mu = 0 (sum s = 1)
    mu = np.maximum((s * (n - bb) / bb).sum(-1) / (s * s / bb).sum(-1), 0.0)   # linear start
    for _ in range(12):                                # Newton on a concave function
        lam = bb + mu[..., None] * s
        g = (n * s / lam).sum(-1) - 1.0
        h = -(n * s * s / lam**2).sum(-1)
        mu = np.maximum(mu - g / h, 0.0)
    mu = np.where(g0 > 0, mu, 0.0)
    lam = bb + mu[..., None] * s
    lnr = (n * np.log(lam / bb)).sum(-1) - mu          # lnL(mu) - lnL(0)
    return np.maximum(2 * lnr, 0.0), mu

def upcrossings(q, c0):
    above = q > c0
    return np.sum(~above[:, :-1] & above[:, 1:], axis=1)

# ---------- the background-only ensemble ----------
T, CH = 20_000, 500
qmax = np.empty(T); mhat = np.empty(T); nup = np.empty(T, int)
C0 = 0.5
i125 = np.argmin(np.abs(masses - 125.0))
q125 = np.empty(T)
for k in range(0, T, CH):
    n = rng.poisson(b, size=(CH, b.size))
    q, _ = q0_scan(n)
    qmax[k:k + CH] = q.max(1)
    mhat[k:k + CH] = masses[q.argmax(1)]
    nup[k:k + CH] = upcrossings(q, C0)
    q125[k:k + CH] = q[:, i125]

# ---------- one "observed" background-only data set with a ~3 sigma local excess ----------
rng_obs = rng_for("ch04", "17_bump_scan", stream=1)
cand = rng_obs.poisson(b, size=(100, b.size))
qc, muc = q0_scan(cand)
zc = np.sqrt(qc.max(1))
j = int(np.argmin(np.abs(zc - 3.1)))                   # the one closest to a 3.1 sigma local excess
n_obs = cand[j]
q_obs_m, mu_obs_m = qc[j], muc[j]
im = int(q_obs_m.argmax())
q_obs, m_obs, mu_obs = q_obs_m[im], masses[im], mu_obs_m[im]
z_loc = np.sqrt(q_obs)
p_loc = stats.norm.sf(z_loc)                            # one-sided: P(q0 > c) = Phi(-sqrt c)
p_glob = np.mean(qmax >= q_obs)
p_glob_err = np.sqrt(p_glob * (1 - p_glob) / T)
z_glob = stats.norm.isf(p_glob)
trials = p_glob / p_loc

# ---------- Gross-Vitells: <N(c0)> from the first 100 toys ----------
N0_100 = nup[:100].mean()
N0_all = nup.mean()
def gv(c, N0):
    return stats.norm.sf(np.sqrt(c)) + N0 * np.exp(-(c - C0) / 2)
p_gv = gv(q_obs, N0_100)
N1 = N0_all * np.exp(C0 / 2)                            # <N(c)> = N1 exp(-c/2)

# local check at a fixed mass: P(q0 > c) = 1/2 P(chi2_1 > c)
frac_zero_125 = np.mean(q125 == 0)
p125_4 = np.mean(q125 > 4.0)

# ---------- figures ----------
fig, axs = plt.subplots(2, 1, figsize=(6.2, 4.8), sharex=True,
                        gridspec_kw=dict(height_ratios=[1.6, 1]))
ax = axs[0]
ax.errorbar(centres, n_obs, yerr=np.sqrt(n_obs), fmt="o", ms=2.5, color="0.2", lw=0.8,
            label="one background-only pseudo-experiment")
ax.step(edges, np.r_[b, b[-1]], where="post", color="k", ls="--", lw=1.0, label="known background")
ax.plot(centres, b + mu_obs * S[im], color=SERIES[1], lw=1.3,
        label=rf"best bump: $m={m_obs:.2f}$ GeV, $\hat\mu={mu_obs:.0f}$")
ax.set_ylabel("events / GeV")
ax.legend(fontsize=7.5, loc="upper right")
ax = axs[1]
zloc_m = np.sqrt(q_obs_m)
ax.plot(masses, zloc_m, color=SERIES[0])
ax.axhline(np.sqrt(C0), color="k", ls=":", lw=0.9)
upc = np.nonzero((q_obs_m[:-1] <= C0) & (q_obs_m[1:] > C0))[0] + 1
ax.plot(masses[upc], zloc_m[upc], "o", color=SERIES[1], ms=3.5)
ax.set_xlabel(r"hypothesised mass $m$ (GeV)")
ax.set_ylabel(r"local $Z(m)=\sqrt{q_0(m)}$")
ax.set_xlim(100, 160)
fig.tight_layout()
savefig(fig, "ch04", "scan_spectrum")

cgrid = np.linspace(0.0, 16.0, 161)
p_mc = np.array([np.mean(qmax > c) for c in cgrid])
p_localc = stats.norm.sf(np.sqrt(cgrid))
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axs[0]
ax.semilogy(cgrid, p_mc, color=SERIES[0], label=f"max over masses, {T} toys")
ax.semilogy(cgrid, p_localc, color=SERIES[1], ls="--", label=r"one fixed mass: $\Phi(-\sqrt{c})$")
ax.semilogy(cgrid[cgrid >= 1], gv(cgrid[cgrid >= 1], N0_100), color="k", ls=":",
            label="Gross-Vitells, 100 toys")
ax.axvline(q_obs, color="0.5", lw=0.8)
ax.set_xlabel(r"level $c$"); ax.set_ylabel(r"$P(q>c)$ under background only")
ax.set_ylim(1e-5, 1.2); ax.legend(fontsize=7, loc="lower left")
ax = axs[1]
zz = np.sqrt(cgrid[1:])
ok = p_mc[1:] * T >= 20
ax.plot(zz[ok], (p_mc[1:] / p_localc[1:])[ok], color=SERIES[0], label="toys")
ax.plot(zz, gv(cgrid[1:], N0_100) / p_localc[1:], color="k", ls=":", label="Gross-Vitells")
ax.plot(zz, 1 + np.sqrt(2 * np.pi) * N1 * zz, color=SERIES[2], ls="-.",
        label=r"$1+\sqrt{2\pi}\,\mathcal{N}_1 Z$")
ax.set_xlabel(r"local significance $Z_{\rm loc}$"); ax.set_ylabel("trials factor")
ax.set_xlim(0.5, 4.0); ax.set_ylim(0, 1.05 * (1 + np.sqrt(2 * np.pi) * N1 * 4.0))
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
savefig(fig, "ch04", "scan_tail")

def sci(v):
    m, e = f"{v:.2e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"

p_gv_five = gv(25.0, N0_100)                 # a local 5 sigma excess, global p by Gross-Vitells
save_numbers("ch04", "17_bump_scan", {
    "FourBScPgvFive": sci(p_gv_five), "FourBScZgvFive": f"{stats.norm.isf(p_gv_five):.2f}",
    "FourBScPlocFive": sci(stats.norm.sf(5.0)),
    "FourBScBtot": f"{B_TOT:.0f}", "FourBScSlope": f"{SLOPE:.0f}", "FourBScSigM": f"{SIG_M}",
    "FourBScNmass": masses.size, "FourBScToys": T,
    "FourBScMobs": f"{m_obs:.2f}", "FourBScMuObs": f"{mu_obs:.0f}", "FourBScQobs": f"{q_obs:.2f}",
    "FourBScZloc": f"{z_loc:.2f}", "FourBScPloc": sci(p_loc),
    "FourBScPglob": f"{p_glob:.4f}", "FourBScPglobErr": f"{p_glob_err:.4f}",
    "FourBScZglob": f"{z_glob:.2f}", "FourBScTrials": f"{trials:.1f}",
    "FourBScNzeroHundred": f"{N0_100:.2f}", "FourBScNzeroAll": f"{N0_all:.3f}",
    "FourBScNone": f"{N1:.2f}", "FourBScPgv": f"{p_gv:.4f}",
    "FourBScFracZero": f"{frac_zero_125:.3f}", "FourBScPfourMC": f"{p125_4:.4f}",
    "FourBScPfourTh": f"{stats.norm.sf(2.0):.4f}",
    "FourBScPthreeGlob": f"{np.mean(qmax > 9.0):.3f}",
    "FourBScTrialsThreeAsym": f"{1 + np.sqrt(2 * np.pi) * N1 * 3.0:.0f}",
})
print(f"observed: m={m_obs:.2f} mu={mu_obs:.1f} q={q_obs:.3f} Zloc={z_loc:.3f} ploc={p_loc:.3e}")
print(f"global p={p_glob:.4f}+-{p_glob_err:.4f} Zglob={z_glob:.3f} trials={trials:.2f}")
print(f"<N(c0)> 100 toys {N0_100:.3f}, all {N0_all:.3f}; N1={N1:.3f}; GV p={p_gv:.4f}")
print(f"fixed mass: frac q=0 {frac_zero_125:.4f}, P(q>4) {p125_4:.4f} vs {stats.norm.sf(2):.4f}")
print(f"P(max q > 9) = {np.mean(qmax > 9):.4f}")
