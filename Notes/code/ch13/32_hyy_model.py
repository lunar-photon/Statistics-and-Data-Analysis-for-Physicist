"""32_hyy_model.py -- the signal line shape and the choice of background function.

Question: what shape does a 125 GeV Higgs boson leave in the measured diphoton mass, and which
smooth function can describe the background without inventing or hiding a signal?
Computes: (1) weighted unbinned fits of a Crystal Ball and of a Gaussian to the simulated signal
(all production modes together, and separately in the two categories); the full width at half
maximum; the total signal yield before the window cut. (2) Background-only fits of six smooth
families to the 76 394 selected data events (0.25 GeV bins), with the deviance per degree of
freedom in 1 GeV bins. (3) The spurious-signal test: two flexible "truth" shapes are fitted to
the whole data spectrum together with a 125 GeV signal, and their background part is kept; for every candidate family, a signal+background
fit to the expected (Asimov) spectrum of each truth, at every m_H from 110 to 150 GeV, gives the
fitted signal S_spur that the background function invents; it is compared with the statistical
uncertainty sigma_S of the fitted signal. A family passes if |S_spur| < 0.2 sigma_S everywhere
(the ATLAS 2012 criterion).
Writes: data/ch13/hyy_model.npz, figures/ch13/hyy_signal_shape.pdf, figures/ch13/hyy_spurious.pdf,
results/ch13/32_hyy_model.tex
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.stats import chi2
from common import setup, savefig, save_numbers, SERIES, DATA
from lib_hyy import (EDGES, CENT, U, MLO, MHI, cb_cdf, cb_pdf, gauss_cdf, fit_cb, fit_gauss,
                     SignalShape, BKG, bkg_shape)

SIGS = ["ggH", "VBF", "WH", "ZH"]
CANDS = ["Exp1", "Pow", "Exp2", "Bern3", "Exp3", "Bern4", "Exp4", "Bern5"]
TRUTHS = ["Bern5", "Exp4"]
NAMES = {"Exp1": "ExpOne", "Exp2": "ExpTwo", "Exp3": "ExpThree", "Pow": "Pow", "Bern3": "BernThree",
         "Bern4": "BernFour", "Exp4": "ExpFour", "Bern5": "BernFive"}             # LaTeX macro names may contain letters only
MSCAN = np.arange(110.0, 150.01, 1.0)


def load():
    d = np.load(DATA / "ch13" / "hyy_events.npz")
    m = np.concatenate([d[f"data_{p}"] for p in "ABCD"])
    c = np.concatenate([d[f"cat_{p}"] for p in "ABCD"])
    ms = np.concatenate([d[f"{s}_m"] for s in SIGS])
    ws = np.concatenate([d[f"{s}_w"] for s in SIGS])
    cs = np.concatenate([d[f"{s}_cat"] for s in SIGS])
    return m, c, ms, ws, cs


def fwhm(p):
    x = np.linspace(115, 135, 20001)
    f = cb_pdf(x, *p)
    above = x[f >= f.max() / 2]
    return above[-1] - above[0]


def bfit(n, name, mask=None):
    """Background-only Poisson fit of family `name`; mask selects the bins used."""
    mask = np.ones(len(n), bool) if mask is None else mask
    k = BKG[name][0]

    def f(p):
        g = bkg_shape(name, p[1:])
        g = g[mask] / g[mask].sum()
        nu = p[0] * g
        return 2 * np.sum(nu - n[mask] * np.log(nu)) if np.all(nu > 0) else 1e30
    p0 = np.array([n[mask].sum()] + list(BKG[name][2]))
    r = minimize(f, p0, method="Nelder-Mead", options=dict(maxiter=40000, maxfev=40000, xatol=1e-7, fatol=1e-7))
    r = minimize(f, r.x, method="Nelder-Mead", options=dict(maxiter=40000, maxfev=40000, xatol=1e-8, fatol=1e-8))
    return r.x


def deviance_1gev(n, nu):
    """Poisson deviance in 1 GeV bins (four 0.25 GeV bins merged)."""
    N, V = n.reshape(-1, 4).sum(1), nu.reshape(-1, 4).sum(1)
    return 2 * np.sum(V - N + np.where(N > 0, N * np.log(np.where(N > 0, N, 1) / V), 0.0)), len(N)


def sb_fit_asimov(nA, name, sig, mH):
    """Signal+background fit (stat only) to an Asimov spectrum: S_hat and its sigma_S."""
    tmpl = sig.bins(mH) / sig.s_tot                     # unit-yield signal shape
    k = BKG[name][0]

    def nu_of(p):
        return p[0] * tmpl + p[1] * bkg_shape(name, p[2:])

    def f(p):
        nu = nu_of(p)
        return 2 * np.sum(nu - nA * np.log(nu)) if np.all(nu > 0) else 1e30
    pb = bfit(nA, name)
    r = minimize(f, np.r_[0.0, pb], method="Nelder-Mead",
                 options=dict(maxiter=40000, maxfev=40000, xatol=1e-7, fatol=1e-9))
    p = r.x
    # Fisher matrix J^T diag(1/nu) J with a numerical Jacobian, then sigma of the signal yield
    nu = nu_of(p)
    J = []
    for j in range(len(p)):
        h = 1e-4 * max(1.0, abs(p[j]))
        dp = np.zeros(len(p))
        dp[j] = h
        J.append((nu_of(p + dp) - nu_of(p - dp)) / (2 * h))
    J = np.array(J)
    V = np.linalg.inv((J / nu) @ J.T)
    win = tmpl.sum()                                     # fraction of a unit signal inside the window
    return p[0] * win, np.sqrt(V[0, 0]) * win, p


def main():
    setup()
    m, c, ms, ws, cs = load()
    n, _ = np.histogram(m, EDGES)
    nums = {}

    # ---------------- (1) signal line shape
    pcb = fit_cb(ms, ws)
    pg = fit_gauss(ms, ws)
    s_win = ws.sum()
    s_tot = s_win / (cb_cdf(MHI, *pcb) - cb_cdf(MLO, *pcb))
    sig = SignalShape(pcb[0] - 125.0, pcb[1], pcb[2], pcb[3], s_tot)
    cat_pars = []
    for sel in (cs, ~cs):
        p = fit_cb(ms[sel], ws[sel])
        cat_pars.append(np.r_[p, ws[sel].sum() / (cb_cdf(MHI, *p) - cb_cdf(MLO, *p))])
    tail_frac = cb_cdf(pcb[0] - 2 * pcb[1], *pcb)          # fraction more than 2 sigma below the peak
    gauss_tail = 0.5 * (1 + __import__("math").erf(-2 / np.sqrt(2)))
    nums.update(HyyCBmean=round(pcb[0], 2), HyyCBsig=round(pcb[1], 3), HyyCBalpha=round(pcb[2], 2),
                HyyCBn=round(pcb[3], 1), HyyGmean=round(pg[0], 2), HyyGsig=round(pg[1], 2),
                HyyFWHM=round(fwhm(pcb), 2), HyyFWHMratio=round(fwhm(pcb) / pcb[1], 2),
                HyyStot=round(s_tot, 1), HyyTailFrac=round(100 * tail_frac, 1),
                HyyGaussTail=round(100 * gauss_tail, 1),
                HyyCBsigCat=round(cat_pars[0][1], 3), HyyCBsigRest=round(cat_pars[1][1], 3))
    print("CB:", pcb, "Gauss:", pg, "FWHM", fwhm(pcb), "s_tot", s_tot)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    e = np.arange(110, 140.01, 0.5)
    h, _ = np.histogram(ms, e, weights=ws)
    ax.errorbar(0.5 * (e[1:] + e[:-1]), h / 0.5, fmt="o", ms=2.5, color="k", label="simulated signal")
    x = np.linspace(110, 140, 600)
    ax.plot(x, s_win * cb_pdf(x, *pcb) / (cb_cdf(MHI, *pcb) - cb_cdf(MLO, *pcb)), color=SERIES[0],
            label=f"Crystal Ball, $\\sigma={pcb[1]:.2f}$ GeV")
    gpdf = np.exp(-0.5 * ((x - pg[0]) / pg[1]) ** 2) / (np.sqrt(2 * np.pi) * pg[1])
    ax.plot(x, s_win * gpdf, color=SERIES[1], ls="--", label=f"Gaussian, $\\sigma={pg[1]:.2f}$ GeV")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 200)
    ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    ax.set_ylabel("expected events / GeV")
    ax.legend(loc="upper left")
    savefig(fig, "ch13", "hyy_signal_shape")

    # ---------------- (2) background families on the data
    fits = {}
    for name in CANDS:
        p = bfit(n, name)
        fits[name] = p
        dev, nb = deviance_1gev(n, p[0] * bkg_shape(name, p[1:]))
        ndf = nb - 1 - BKG[name][0]
        nums[f"HyyDev{NAMES[name]}"] = round(dev, 1)
        nums[f"HyyNdf{NAMES[name]}"] = ndf
        nums[f"HyyPfit{NAMES[name]}"] = round(chi2.sf(dev, ndf), 3)
        print(name, p, dev, ndf, chi2.sf(dev, ndf))

    # ---------------- (3) spurious signal against two truths: background part of an s+b fit to the data
    res = {}
    for t in TRUTHS:
        _, _, pt = sb_fit_asimov(n.astype(float), t, sig, 125.0)
        nA = pt[1] * bkg_shape(t, pt[2:])                 # the truth: fitted background, no Higgs
        nums[f"HyyTruthS{NAMES[t]}"] = round(pt[0] * (sig.bins(125.0).sum() / sig.s_tot), 0)
        for name in CANDS:
            if name == t:
                continue
            out = np.array([sb_fit_asimov(nA, name, sig, mh)[:2] for mh in MSCAN])
            res[(t, name)] = out
            print(t, name, np.abs(out[:, 0]).max(), (np.abs(out[:, 0]) / out[:, 1]).max())
    worst = {name: max((np.abs(res[(t, name)][:, 0]) / res[(t, name)][:, 1]).max()
                       for t in TRUTHS if (t, name) in res) for name in CANDS}
    sspur = {name: max(np.abs(res[(t, name)][:, 0]).max() for t in TRUTHS if (t, name) in res)
             for name in CANDS}
    sigS = np.mean([res[(t, n2)][:, 1].mean() for (t, n2) in res])
    for name in CANDS:
        nums[f"HyySpur{NAMES[name]}"] = round(sspur[name], 1)
        nums[f"HyySpurRel{NAMES[name]}"] = round(worst[name], 2)
    passing = [nm for nm in CANDS if worst[nm] < 0.2 and nums[f"HyyPfit{NAMES[nm]}"] > 0.01]
    chosen = (min(passing, key=lambda nm: (BKG[nm][0], worst[nm])) if passing
              else min(CANDS, key=lambda nm: worst[nm]))
    nums["HyySigmaS"] = round(sigS, 0)
    nums["HyyChosen"] = NAMES[chosen]
    nums["HyySspurChosen"] = round(sspur[chosen], 1)
    print("passing:", passing, "chosen:", chosen)

    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, t in zip(axs, TRUTHS):
        for i, name in enumerate(CANDS):
            if (t, name) in res:
                r = res[(t, name)]
                ax.plot(MSCAN, r[:, 0] / r[:, 1], color=SERIES[i], label=name)
        ax.axhspan(-0.2, 0.2, color="0.85", zorder=0)
        ax.set_title(f"truth: {t} (background part of an s+b fit)", fontsize=8)
        ax.set_xlabel(r"$m_H$ [GeV]")
    axs[0].set_ylabel(r"$S_{\rm spur}/\sigma_S$")
    axs[0].set_ylim(-1.5, 1.5)
    hs = {}
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hs.setdefault(l, h)
    fig.legend([hs[k] for k in CANDS if k in hs], [k for k in CANDS if k in hs], fontsize=7,
               ncol=len(hs), loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    savefig(fig, "ch13", "hyy_spurious")

    np.savez(DATA / "ch13" / "hyy_model.npz", pcb=pcb, s_tot=s_tot, chosen=chosen,
             s_spur=sspur[chosen], cat_pars=np.array(cat_pars), bfit=fits[chosen],
             **{f"fit_{k}": v for k, v in fits.items()})
    save_numbers("ch13", "32_hyy_model", nums)


if __name__ == "__main__":
    main()
