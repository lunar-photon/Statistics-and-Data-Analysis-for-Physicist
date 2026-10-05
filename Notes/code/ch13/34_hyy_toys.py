"""34_hyy_toys.py -- pseudo-experiments for the diphoton search: are the asymptotic formulas right?

Question: with 76 000 events and a six-parameter background, do q0 and q~_mu follow the
half-chi^2 laws of Cowan et al. (2011); how often does a background-only spectrum produce, somewhere
between 110 and 150 GeV, an excess as large as the one in the data (the global p-value); does the
Gross-Vitells upcrossing formula, calibrated on 100 toys, reproduce it; and how far is the
asymptotic CLs limit from the limit computed with toys?
Computes: background-only spectra drawn from the fitted background (statistical fluctuations;
the background shape is refitted in every toy, the constrained nuisances are held fixed), each
scanned in m_H exactly like the data with the Newton fitter lib_hyy.scan_q0; signal-plus-background
spectra at 125 GeV; q~_mu at 140 GeV under mu and under 0 on a grid of mu, and the toy CLs limit.
Writes: data/ch13/hyy_toys.npz, figures/ch13/hyy_toys_q0.pdf, figures/ch13/hyy_toys_qtilde.pdf,
results/ch13/34_hyy_toys.tex
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA, theory_line
from lib_hyy import EDGES, SignalShape, bern_basis, lin_fit, scan_q0

MSCAN = np.arange(110.0, 150.01, 0.5)
N_BKG, N_SB, N_LIM = 4000, 2000, 1000      # toys: background-only scans, s+b at 125, per mu at 140 GeV
BATCH = 250
C0 = 0.5                                    # reference level for counting upcrossings
M_LIM = 140.0
MU_GRID = np.round(np.arange(0.15, 1.51, 0.05), 2)


def load():
    d = np.load(DATA / "ch13" / "hyy_events.npz")
    m = np.concatenate([d[f"data_{p}"] for p in "ABCD"])
    mod = np.load(DATA / "ch13" / "hyy_model.npz")
    pcb, s_tot = mod["pcb"], float(mod["s_tot"])
    deg = int(str(mod["chosen"]).replace("Bern", ""))
    return np.histogram(m, EDGES)[0].astype(float), SignalShape(pcb[0] - 125.0, *pcb[1:], s_tot), deg


def upcrossings(q, c):
    """Number of times each row of q (toys x masses) crosses the level c upwards."""
    above = q > c
    return np.sum(~above[:, :-1] & above[:, 1:], 1)


def qtilde_batch(n, s, B, mu):
    """q~_mu (Cowan et al. 2011, eq. 16) for many spectra, background linear in B, signal mu * s."""
    T = n.shape[0]
    A = np.concatenate([s[:, None], B], 1)
    cf, lf = lin_fit(n, A)                                   # mu free
    cm, lm = lin_fit(n, B, offset=mu * s)                    # mu fixed
    cz, lz = lin_fit(n, B)                                   # mu = 0
    muh = cf[:, 0]
    q = np.where(muh < 0, 2 * (lz - lm), 2 * (lf - lm))
    return np.where(muh > mu, 0.0, np.maximum(q, 0.0)), muh


def sf_q0(q, mu_over_sigma):
    """P(q0 >= q | mu'), asymptotic: Phi(mu'/sigma - sqrt q) for q > 0."""
    return norm.cdf(mu_over_sigma - np.sqrt(q))


def sf_qt(q, r, under_mu):
    """P(q~_mu >= q) for r = mu/sigma, asymptotic (Cowan et al. 2011, eq. 65), under mu or under 0."""
    q = np.asarray(q, float)
    if under_mu:
        return np.where(q <= r ** 2, norm.sf(np.sqrt(q)), norm.sf((q + r ** 2) / (2 * r)))
    return np.where(q <= r ** 2, norm.cdf(r - np.sqrt(q)), norm.sf((q - r ** 2) / (2 * r)))


def main():
    setup()
    n, sig, deg = load()
    B = bern_basis(deg)
    tmpl = np.array([sig.bins(m) for m in MSCAN])            # expected Higgs counts per bin, mu = 1
    cb, _ = lin_fit(n[None], B)
    nu_b = B @ cb[0]
    nums = {}

    # ---------------- the data, scanned with the same (statistical-only) fitter
    q_dat, mu_dat, _ = scan_q0(n[None], tmpl, B)
    q_obs = q_dat[0].max()
    nums.update(HyyToyQobs=round(q_obs, 1), HyyToyZobs=round(np.sqrt(q_obs), 2),
                HyyToyMobs=MSCAN[np.argmax(q_dat[0])])

    # ---------------- background-only scans
    rng = rng_for("ch13", "34_hyy_toys", 0)
    qmax, nup, nup1, q125 = [], [], [], []
    j125 = np.argmin(np.abs(MSCAN - 125.0))
    for b in range(N_BKG // BATCH):
        nt = rng.poisson(nu_b, size=(BATCH, len(nu_b))).astype(float)
        q, _, _ = scan_q0(nt, tmpl, B, nit=15)
        qmax.append(q.max(1))
        nup.append(upcrossings(q, C0))
        nup1.append(upcrossings(q, 1.0))
        q125.append(q[:, j125])
        print(f"background batch {b}: {len(np.concatenate(qmax))} toys")
    qmax, nup, nup1, q125 = map(np.concatenate, (qmax, nup, nup1, q125))
    p_glob_toy = np.mean(qmax >= q_obs)
    n_exceed = int(np.sum(qmax >= q_obs))
    EN100, ENall = nup[:100].mean(), nup.mean()
    gv = lambda c, EN=EN100, c0=C0: norm.sf(np.sqrt(c)) + EN * np.exp(-(c - c0) / 2)
    p_loc_obs = norm.sf(np.sqrt(q_obs))
    p_gv_obs = gv(q_obs)
    # same formula applied to the full-likelihood local significance of 33_hyy_search, if available
    try:
        srch = np.load(DATA / "ch13" / "hyy_search.npz")
        q_full = float(srch["q_full"].max())
    except FileNotFoundError:
        q_full = q_obs
    N1 = EN100 * np.exp(C0 / 2)
    nums.update(HyyToyNbkg=N_BKG, HyyToyFracZero=round(np.mean(q125 == 0), 3),
                HyyToyPfour=round(np.mean(q125 > 4), 4), HyyToyPfourTh=round(norm.sf(2), 4),
                HyyToyPnine=round(np.mean(q125 > 9), 4), HyyToyPnineTh=round(norm.sf(3), 4),
                HyyToyNexceed=n_exceed, HyyToyPglob=round(p_glob_toy, 4),
                HyyToyPglobErr=round(np.sqrt(max(p_glob_toy, 1 / N_BKG) * (1 - p_glob_toy) / N_BKG), 4),
                HyyENhundred=round(EN100, 2), HyyENall=round(ENall, 2), HyyENoneAll=round(nup1.mean(), 2),
                HyyNone=round(N1, 2), HyyPlocObs=p_loc_obs, HyyPgvObs=p_gv_obs,
                HyyZgvObs=round(norm.isf(p_gv_obs), 2),
                HyyTrialsObs=round(p_gv_obs / p_loc_obs, 1),
                HyyTrialsAsym=round(1 + np.sqrt(2 * np.pi) * N1 * np.sqrt(q_obs), 1),
                HyyPgvFull=gv(q_full), HyyZgvFull=round(norm.isf(gv(q_full)), 2),
                HyyZlocFull=round(np.sqrt(q_full), 2),
                HyyPthreeGlobToy=round(np.mean(qmax >= 9), 3), HyyPthreeGlobGV=round(gv(9.0), 3))
    print("q_obs", q_obs, "toys exceeding", n_exceed, "E[N] 100/all", EN100, ENall, "GV", p_gv_obs)

    # ---------------- signal-plus-background at 125 GeV: is the Asimov value the median?
    s125 = tmpl[j125]
    A1 = np.concatenate([s125[:, None], B], 1)
    nA = nu_b + s125
    cA0, lA0 = lin_fit(nA[None], B)
    cA1, lA1 = lin_fit(nA[None], A1)
    qA = 2 * (lA1[0] - lA0[0])
    rng2 = rng_for("ch13", "34_hyy_toys", 1)
    qsb = []
    for b in range(N_SB // BATCH):
        nt = rng2.poisson(nA, size=(BATCH, len(nA))).astype(float)
        c0, l0 = lin_fit(nt, B)
        c1, l1 = lin_fit(nt, A1)
        qsb.append(np.where(c1[:, 0] > 0, np.maximum(2 * (l1 - l0), 0), 0))
    qsb = np.concatenate(qsb)
    nums.update(HyyToyZAsb=round(np.sqrt(qA), 2), HyyToyZmedSb=round(np.sqrt(np.median(qsb)), 2),
                HyyToyZsbLo=round(np.sqrt(np.quantile(qsb, 0.16)), 2),
                HyyToyZsbHi=round(np.sqrt(np.quantile(qsb, 0.84)), 2),
                HyyToyPsbFive=round(np.mean(qsb >= 25), 3))

    # ---------------- q~_mu at 140 GeV: toys against the asymptotic laws, and the CLs limit
    jl = np.argmin(np.abs(MSCAN - M_LIM))
    s140 = tmpl[jl]
    rng3 = rng_for("ch13", "34_hyy_toys", 2)
    qt_obs = np.array([qtilde_batch(n[None], s140, B, u)[0][0] for u in MU_GRID])
    clsb_t, clb_t, cls_a, sigA = [], [], [], []
    keep = {}
    for u, qo in zip(MU_GRID, qt_obs):
        q_mu = qtilde_batch(rng3.poisson(nu_b + u * s140, size=(N_LIM, len(nu_b))).astype(float), s140, B, u)[0]
        q_0 = qtilde_batch(rng3.poisson(nu_b, size=(N_LIM, len(nu_b))).astype(float), s140, B, u)[0]
        qa = qtilde_batch(nu_b[None], s140, B, u)[0][0]                 # Asimov, background only
        sA = u / np.sqrt(qa)
        sigA.append(sA)
        clsb_t.append(np.mean(q_mu >= qo))
        clb_t.append(np.mean(q_0 >= qo))
        cls_a.append(sf_qt(qo, u / sA, True) / sf_qt(qo, u / sA, False))
        if abs(u - 0.5) < 1e-9:
            keep = dict(q_mu=q_mu, q_0=q_0, sA=sA, mu=u, qo=qo)
    clsb_t, clb_t, cls_a, sigA = map(np.array, (clsb_t, clb_t, cls_a, sigA))
    cls_t = clsb_t / np.maximum(clb_t, 1e-9)

    def cross(y):
        i = np.flatnonzero(y < 0.05)
        if len(i) == 0 or i[0] == 0:
            return np.nan
        i = i[0]
        return np.interp(0.05, [y[i], y[i - 1]], [MU_GRID[i], MU_GRID[i - 1]])
    lim_toy, lim_asym = cross(cls_t), cross(cls_a)
    nums.update(HyyLimToyOneFourty=round(lim_toy, 2), HyyLimAsymOneFourty=round(lim_asym, 2),
                HyyNlimToys=N_LIM, HyySigAstatOneFourty=round(sigA[np.argmin(np.abs(MU_GRID - 0.5))], 3),
                HyyQtMuTest=keep["mu"])
    print("limit at 140: toys", lim_toy, "asymptotic", lim_asym)

    # ---------------- figures
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    x = np.linspace(0.01, 30, 400)
    for qq, lab, col, r in ((q125, "background only", SERIES[0], 0.0),
                            (qsb, r"SM Higgs, $\mu=1$", SERIES[1], np.sqrt(qA))):
        xs = np.sort(qq)
        a1.step(xs, 1 - np.arange(len(xs)) / len(xs), where="post", color=col, label=lab)
        a1.plot(x, sf_q0(x, r), color="k", ls="--", lw=0.9)
    a1.set_yscale("log")
    a1.set_ylim(1e-3, 1.1)
    a1.set_xlim(0, 30)
    a1.set_xlabel(r"$q_0$ at $m_H=125$ GeV")
    a1.set_ylabel(r"$P(q_0\geq x)$")
    a1.legend(fontsize=7, loc="lower left")
    cs = np.linspace(0, 30, 301)
    a2.plot(cs, [np.mean(qmax >= c) for c in cs], color=SERIES[0], label=f"toys, largest excess ({N_BKG})")
    a2.plot(cs, norm.sf(np.sqrt(cs)), color=SERIES[1], ls="--", label="one fixed mass")
    a2.plot(cs, np.minimum(gv(cs), 1), color="k", ls=":", label="Gross--Vitells, 100 toys")
    a2.axvline(q_obs, color="0.4", lw=0.8)
    a2.set_yscale("log")
    a2.set_ylim(1e-7, 1.5)
    a2.set_xlabel(r"level $c$")
    a2.set_ylabel(r"$P(q_{\max}>c)$")
    a2.legend(fontsize=7, loc="lower left")
    savefig(fig, "ch13", "hyy_toys_q0")

    fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    r = keep["mu"] / keep["sA"]
    xq = np.linspace(0, 12, 300)
    for qq, lab, col, um in ((keep["q_mu"], rf"under $\mu={keep['mu']}$", SERIES[0], True),
                             (keep["q_0"], "under background only", SERIES[1], False)):
        xs = np.sort(qq)
        b1.step(xs, 1 - np.arange(len(xs)) / len(xs), where="post", color=col, label=lab)
        b1.plot(xq, sf_qt(xq, r, um), color="k", ls="--", lw=0.9)
    b1.axvline(keep["qo"], color="0.4", lw=0.8)
    b1.set_yscale("log")
    b1.set_ylim(1e-3, 1.1)
    b1.set_xlabel(rf"$\tilde q_\mu$ at {M_LIM:.0f} GeV")
    b1.set_ylabel(r"$P(\tilde q_\mu\geq x)$")
    b1.legend(fontsize=7, loc="lower left")
    b2.plot(MU_GRID, cls_t, "o", ms=3, color=SERIES[0], label=f"toys ({N_LIM} per point)")
    b2.plot(MU_GRID, cls_a, color="k", ls="--", label="asymptotic")
    b2.axhline(0.05, color=SERIES[7], lw=0.8)
    b2.set_yscale("log")
    b2.set_ylim(1e-3, 1.1)
    b2.set_xlabel(r"$\mu$")
    b2.set_ylabel(r"CL$_s$ at 140 GeV")
    b2.legend(fontsize=7)
    savefig(fig, "ch13", "hyy_toys_qtilde")

    np.savez(DATA / "ch13" / "hyy_toys.npz", qmax=qmax, nup=nup, q125=q125, qsb=qsb, MU_GRID=MU_GRID,
             cls_t=cls_t, cls_a=cls_a, q_dat=q_dat[0])
    save_numbers("ch13", "34_hyy_toys", nums)
    print(nums)


if __name__ == "__main__":
    main()
