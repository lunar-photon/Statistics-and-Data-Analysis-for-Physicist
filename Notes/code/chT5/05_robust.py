"""05_robust.py -- outliers: influence, breakdown, and the price of protection.

Question: n readings of one level mu with Gaussian noise of unit width, a fraction eps of which
are wild (either symmetric, from a Gaussian ten times wider, or one-sided, like cosmic-ray hits on
a detector, which only ever add signal). How do the mean, the median, Huber's M-estimator
(k = 1.345, scale from the median absolute deviation) and the 10% trimmed mean compare?
(i) How much does one added reading at x move each estimate (sensitivity curve, against the
influence function derived in the text)? (ii) How many readings can be sent to infinity before
the estimate follows them (breakdown)? (iii) What is the mean squared error against eps?
(iv) What efficiency does Huber's k = 1.345 cost at exactly Gaussian noise?

Computes: sensitivity curves for a stylised sample (the 20 Gaussian quantiles); estimates as m of the 20
          readings are replaced by 1000; n * MSE against eps (n = 50) for both kinds of outlier;
          Huber's asymptotic efficiency at the Gaussian, analytic and simulated.
Writes:   figures/chT5/influence.pdf, figures/chT5/robust_mse.pdf, results/chT5/05_robust.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from lib_t5a import huber_location, trimmed_mean

rng = rng_for("chT5", "05_robust")
setup()
K = 1.345
est = {
    "mean": lambda x: np.atleast_2d(x).mean(1),
    "median": lambda x: np.median(np.atleast_2d(x), 1),
    "Huber": lambda x: huber_location(x, K),
    "trimmed 10%": lambda x: trimmed_mean(x, 0.1),
}
cols = dict(zip(est, SERIES))

# ---------- Huber's efficiency at the Gaussian (known scale) ----------
Ppsi = 2 * stats.norm.cdf(K) - 1                                 # E psi' = P(|Z| < k)
Epsi2 = Ppsi - 2 * K * stats.norm.pdf(K) + 2 * K**2 * stats.norm.sf(K)
eff_huber = Ppsi**2 / Epsi2
eps_lf = 1 / (1 + 1 / (2 * stats.norm.pdf(K) / K - 2 * stats.norm.sf(K)))   # eps for which k is minimax
assert abs(eff_huber - 0.95) < 0.002

# ---------- (i) sensitivity curves, (ii) breakdown ----------
n0 = 20
base = stats.norm.ppf((np.arange(n0) + 0.5) / n0)                # a 'typical' sample: the n0 Gaussian quantiles
xs = np.linspace(-6, 6, 241)
sc = {}
for name, T in est.items():
    t0 = T(base)[0]
    aug = np.hstack([np.tile(base, (len(xs), 1)), xs[:, None]])
    sc[name] = (n0 + 1) * (T(aug) - t0)
mbad = np.arange(0, n0 // 2 + 1)
brk = {name: [] for name in est}
for m in mbad:
    x = base.copy(); x[:m] = 1000.0
    for name, T in est.items():
        brk[name].append(T(x)[0])

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1))
for name in est:
    axs[0].plot(xs, sc[name], color=cols[name], label=name)
theory_line(axs[0], xs, np.clip(xs, -K, K) / Ppsi, label=r"Huber: $\psi_k(x)/P(|Z|<k)$")
axs[0].set_ylim(-4, 4); axs[0].set_xlabel(r"value $x$ of the added reading")
axs[0].set_ylabel(r"$(n+1)\,[T(\ldots,x)-T(\ldots)]$"); axs[0].legend(fontsize=7, loc="upper left")
for name in est:
    axs[1].plot(mbad, np.abs(np.array(brk[name])) + 1e-3, "o-", color=cols[name], ms=3, label=name)
axs[1].set_yscale("log"); axs[1].set_xlabel(r"readings replaced by $1000$ (of $%d$)" % n0)
axs[1].set_ylabel("|estimate|")
fig.tight_layout()
savefig(fig, "chT5", "influence")

first_break = {name: int(mbad[np.argmax(np.abs(np.array(b)) > 10)]) if np.any(np.abs(np.array(b)) > 10)
               else n0 // 2 + 1 for name, b in brk.items()}

# ---------- (iii) mean squared error against the outlier fraction ----------
n, M = 50, 20_000
epss = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2])
mse = {"sym": {k: [] for k in est}, "one": {k: [] for k in est}}
for kind in ("sym", "one"):
    for e in epss:
        bad = rng.random((M, n)) < e
        core = rng.standard_normal((M, n))
        wild = (10 * rng.standard_normal((M, n)) if kind == "sym"
                else np.abs(10 * rng.standard_normal((M, n))))     # cosmic rays only add
        x = np.where(bad, wild, core)
        for name, T in est.items():
            mse[kind][name].append(n * np.mean(T(x)**2))
mse = {kind: {k: np.array(v) for k, v in d.items()} for kind, d in mse.items()}

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1))
for ax, kind, title in [(axs[0], "sym", "symmetric wild readings"), (axs[1], "one", "one-sided (cosmic rays)")]:
    for name in est:
        ax.plot(epss, mse[kind][name], "o-", color=cols[name], ms=3, label=name)
    ax.set_yscale("log"); ax.set_xlabel(r"fraction of wild readings $\epsilon$"); ax.set_title(title)
axs[0].set_ylabel(r"$n\times$ mean squared error ($n=%d$)" % n); axs[0].legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "chT5", "robust_mse")

# ---------- Casella & Berger Example 10.2.5: eight Gaussian readings and three outliers ----------
cb = np.array([-1.28, -0.96, -0.46, -0.44, -0.26, -0.21, -0.063, 0.39, 3, 6, 9])
cb_k = [1, 2, 3, 4, 5, 6, 8, 10]
cb_hub = [huber_location(cb, kk, scale=1.0)[0] for kk in cb_k]       # their rho has no scale
print("CB table 10.2.3:", dict(zip(cb_k, np.round(cb_hub, 2))))

g = {k: mse["sym"][k][0] for k in est}                           # eps = 0: pure Gaussian
i5 = int(np.argmin(np.abs(epss - 0.05))); i10 = int(np.argmin(np.abs(epss - 0.1)))
print("eff Huber", eff_huber, "eps_lf", eps_lf, "breaks", first_break)
for kind in mse:
    print(kind, {k: np.round(v, 2) for k, v in mse[kind].items()})
save_numbers("chT5", "05_robust", {
    "TFiveARobK": K, "TFiveARobEff": f"{eff_huber:.3f}", "TFiveARobEpsLF": f"{eps_lf:.3f}",
    "TFiveARobPpsi": f"{Ppsi:.4f}", "TFiveARobEpsiTwo": f"{Epsi2:.4f}",
    "TFiveARobNzero": n0, "TFiveARobN": n, "TFiveARobM": M,
    "TFiveARobBreakMean": first_break["mean"], "TFiveARobBreakTrim": first_break["trimmed 10%"],
    "TFiveARobBreakMed": first_break["median"], "TFiveARobBreakHub": first_break["Huber"],
    "TFiveARobGMean": f"{g['mean']:.3f}", "TFiveARobGMed": f"{g['median']:.3f}",
    "TFiveARobGHub": f"{g['Huber']:.3f}", "TFiveARobGTrim": f"{g['trimmed 10%']:.3f}",
    "TFiveARobSymTenMean": f"{mse['sym']['mean'][i10]:.2f}", "TFiveARobSymTenMed": f"{mse['sym']['median'][i10]:.2f}",
    "TFiveARobSymTenHub": f"{mse['sym']['Huber'][i10]:.2f}", "TFiveARobSymTenTrim": f"{mse['sym']['trimmed 10%'][i10]:.2f}",
    "TFiveARobOneFiveMean": f"{mse['one']['mean'][i5]:.2f}", "TFiveARobOneFiveMed": f"{mse['one']['median'][i5]:.2f}",
    "TFiveARobOneFiveHub": f"{mse['one']['Huber'][i5]:.2f}", "TFiveARobOneFiveTrim": f"{mse['one']['trimmed 10%'][i5]:.2f}",
    "TFiveARobOneTwentyMed": f"{mse['one']['median'][-1]:.2f}", "TFiveARobOneTwentyHub": f"{mse['one']['Huber'][-1]:.2f}",
    "TFiveARobOneTwentyMean": f"{mse['one']['mean'][-1]:.1f}",
    "TFiveARobCBMean": f"{cb.mean():.2f}", "TFiveARobCBMed": f"{np.median(cb):.2f}",
    **{f"TFiveARobCBk{w}": f"{v:.2f}" for w, v in zip(["One", "Two", "Three", "Four", "Five", "Six",
                                                       "Eight", "Ten"], cb_hub)},
})
