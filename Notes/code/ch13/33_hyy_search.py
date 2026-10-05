"""33_hyy_search.py -- the Higgs search on the 13 TeV diphoton spectrum: discovery, measurement, limits.

Question: is there a resonance in the measured diphoton masses; if so, how strong is it compared
with the Standard Model Higgs boson and where is it; and where can a Standard Model Higgs boson be
excluded?
Computes, with the binned likelihood of lib_hyy.Search (degree-5 Bernstein background, Crystal Ball
signal, four constrained nuisance parameters: yield, resolution, energy scale, spurious signal):
(1) the background-only fit and the q0 scan from 110 to 150 GeV with and without the nuisance
parameters, the local p-value and significance, and the expected significance for mu = 1 from
Asimov data sets; (2) the profile likelihood of mu at 125 GeV (total and statistical) and of the
mass m_H; (3) the 95% CLs upper limits on mu from the asymptotic formulas, with the expected
median and the 1 and 2 sigma bands; (4) the expected and observed significance at 125 GeV when the
events are split into two categories (central unconverted photon pairs and the rest), and the
leading-order expected significance sum_i s_i^2/b_i against the full Asimov value.
Writes: data/ch13/hyy_search.npz, figures/ch13/hyy_fit.pdf, hyy_p0.pdf, hyy_mu_mass.pdf,
hyy_brazil.pdf, hyy_categories.pdf, results/ch13/33_hyy_search.tex
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
from common import setup, savefig, save_numbers, SERIES, DATA, theory_line
from lib_hyy import EDGES, CENT, BW, SignalShape, Search, bern_basis, lin_fit, cb_cdf

MSCAN = np.arange(110.0, 150.01, 0.5)
MLIM = np.arange(110.0, 150.01, 1.0)
ALPHA = 0.05


def load():
    d = np.load(DATA / "ch13" / "hyy_events.npz")
    m = np.concatenate([d[f"data_{p}"] for p in "ABCD"])
    c = np.concatenate([d[f"cat_{p}"] for p in "ABCD"])
    mod = np.load(DATA / "ch13" / "hyy_model.npz")
    pcb, s_tot = mod["pcb"], float(mod["s_tot"])
    sig = SignalShape(pcb[0] - 125.0, pcb[1], pcb[2], pcb[3], s_tot)
    deg = int(str(mod["chosen"]).replace("Bern", ""))
    return m, c, sig, deg, float(mod["s_spur"]), mod["cat_pars"]


def interval(x, y, level=1.0):
    """Points where the curve y(x) (minimum 0) crosses `level`, by linear interpolation."""
    i0 = np.argmin(y)
    lo = np.interp(level, y[:i0 + 1][::-1], x[:i0 + 1][::-1])
    hi = np.interp(level, y[i0:], x[i0:])
    return x[i0], lo, hi


def ranges(x, step=1.0):
    """Contiguous runs of a sorted grid, written as 'a--b, c--d' (GeV)."""
    if len(x) == 0:
        return "none"
    out, a = [], x[0]
    for u, v in zip(x[:-1], x[1:]):
        if v - u > step + 1e-6:
            out.append((a, u))
            a = v
    out.append((a, x[-1]))
    return ", ".join(f"{a:.0f}" if a == b else f"{a:.0f}--{b:.0f}" for a, b in out)


def main():
    setup()
    m, cat, sig, deg, s_spur, cat_pars = load()
    n = np.histogram(m, EDGES)[0].astype(float)
    S = Search(sig, deg=deg, s_spur=s_spur)
    nums = {}

    # ---------------- (1) background-only fit, q0 scan, expected q0 from Asimov data
    pb, fb = S.fit(n, 125.0, mu=0.0)
    b_hat = S.B @ pb[1:1 + S.k]                                  # fitted background per bin
    q_full, q_stat, mu_full = [], [], []
    for mh in MSCAN:
        q, p1, _ = S.q0(n, mh)
        qs, _, _ = S.q0(n, mh, fix_theta=True)
        q_full.append(q)
        q_stat.append(qs)
        mu_full.append(p1[0])
    q_full, q_stat, mu_full = map(np.array, (q_full, q_stat, mu_full))
    q_exp = np.array([S.q0(b_hat + sig.bins(mh), mh)[0] for mh in MSCAN])
    j = np.argmax(q_full)
    m_max, z_max = MSCAN[j], np.sqrt(q_full[j])
    j125 = np.argmin(np.abs(MSCAN - 125.0))
    nums.update(HyyMmax=m_max, HyyZloc=round(z_max, 2), HyyPloc=norm.sf(z_max),
                HyyZlocStat=round(np.sqrt(q_stat.max()), 2), HyyMmaxStat=MSCAN[np.argmax(q_stat)],
                HyyZoneTwoFive=round(np.sqrt(q_full[j125]), 2),
                HyyZexpOneTwoFive=round(np.sqrt(q_exp[j125]), 2),
                HyyMuMax=round(mu_full[j], 2), HyyQmax=round(q_full[j], 1),
                HyyBhatTot=int(round(b_hat.sum())))
    far = np.abs(MSCAN - m_max) > 8.0                              # the largest excess elsewhere
    j2 = np.flatnonzero(far)[np.argmax(q_full[far])]
    nums.update(HyyMsecond=MSCAN[j2], HyyZsecond=round(np.sqrt(q_full[j2]), 2),
                HyyMuSecond=round(mu_full[j2], 2))
    print("q0 max", q_full[j], "at", m_max, "Z", z_max, "stat-only Z", np.sqrt(q_stat.max()),
          "expected Z(125)", np.sqrt(q_exp[j125]))

    # leading-order expected significance: q0_A ~ sum s_i^2 / b_i, and its Gaussian-peak form
    s125 = sig.bins(125.0)
    beta = np.interp(125.0, CENT, b_hat) / BW                       # background density, events/GeV
    nums.update(HyyZsumsb=round(np.sqrt(np.sum(s125 ** 2 / b_hat)), 2),
                HyyBeta=round(beta, 0),
                HyyZgaussForm=round(s125.sum() / np.sqrt(2 * np.sqrt(np.pi) * sig.sig * beta), 2),
                HyySwin=round(s125.sum(), 1))

    # the price of a flexible background: expected Z with the background known, and profiled with
    # Bernstein polynomials of increasing degree; and the Fisher-matrix factor 1 - rho^2
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    # Each degree is tested on a background it can describe exactly (its own best fit to the
    # fitted background), so that the comparison measures the loss of information only and not
    # a mismodelling bias (that bias is the spurious signal of 32_hyy_model).
    for d in range(1, 7):
        Bd = bern_basis(d)
        cd, _ = lin_fit(b_hat[None], Bd)
        b_d = Bd @ cd[0]                                                # background of degree d
        nA = b_d + s125
        c0, l0 = lin_fit(nA[None], Bd)
        c1, l1 = lin_fit(nA[None], np.concatenate([s125[:, None], Bd], 1))
        A = np.concatenate([s125[:, None], Bd], 1)
        F = (A / b_d[:, None]).T @ A                                    # Fisher matrix at mu = 0
        keep_frac = 1.0 / (F[0, 0] * np.linalg.inv(F)[0, 0])           # 1 - rho^2
        nums[f"HyyZAdeg{words[d]}"] = round(np.sqrt(2 * (l1[0] - l0[0])), 2)
        nums[f"HyyKeepDeg{words[d]}"] = round(keep_frac, 2)
    nums["HyyZAknown"] = round(np.sqrt(2 * np.sum((b_hat + s125) * np.log1p(s125 / b_hat) - s125)), 2)

    # ---------------- (2) profile likelihood of mu at 125 GeV and of m_H
    mus = np.linspace(0.0, 2.5, 51)
    free_t = S.fit(n, 125.0)
    free_s = S.fit(n, 125.0, fix_theta=True)
    t_tot = np.array([S.fit(n, 125.0, mu=u, p0=free_t[0])[1] - free_t[1] for u in mus])
    t_stat = np.array([S.fit(n, 125.0, mu=u, fix_theta=True, p0=free_s[0])[1] - free_s[1] for u in mus])
    fine = np.linspace(0, 2.5, 2001)
    mu_hat, lo_t, hi_t = interval(fine, np.interp(fine, mus, t_tot))
    _, lo_s, hi_s = interval(fine, np.interp(fine, mus, t_stat))
    sig_tot, sig_stat = 0.5 * (hi_t - lo_t), 0.5 * (hi_s - lo_s)
    nums.update(HyyMuHat=round(free_t[0][0], 2), HyyMuLo=round(free_t[0][0] - lo_t, 2),
                HyyMuHi=round(hi_t - free_t[0][0], 2), HyyMuStatLo=round(free_s[0][0] - lo_s, 2),
                HyyMuStatHi=round(hi_s - free_s[0][0], 2), HyySigTot=round(sig_tot, 2),
                HyySigStat=round(sig_stat, 2),
                HyySigSyst=round(np.sqrt(max(sig_tot ** 2 - sig_stat ** 2, 0.0)), 2),
                HyyThY=round(free_t[0][1 + S.k], 2), HyyThSig=round(free_t[0][2 + S.k], 2),
                HyyThM=round(free_t[0][3 + S.k], 2), HyyThSS=round(free_t[0][4 + S.k], 2))
    print("mu-hat", free_t[0][0], lo_t, hi_t, "stat", lo_s, hi_s)

    mhs = np.arange(120.0, 130.01, 0.1)
    fm_t = np.array([S.fit(n, mh)[1] for mh in mhs])
    fm_s = np.array([S.fit(n, mh, fix_theta=True)[1] for mh in mhs])
    fm_t -= fm_t.min()
    fm_s -= fm_s.min()
    mf = np.linspace(120, 130, 4001)
    m_hat, mlo_t, mhi_t = interval(mf, np.interp(mf, mhs, fm_t))
    m_hat_s, mlo_s, mhi_s = interval(mf, np.interp(mf, mhs, fm_s))
    nums.update(HyyMhat=round(m_hat, 1), HyyMhatLo=round(m_hat - mlo_t, 1), HyyMhatHi=round(mhi_t - m_hat, 1),
                HyyMhatStat=round(m_hat_s, 1), HyyMhatStatLo=round(m_hat_s - mlo_s, 1),
                HyyMhatStatHi=round(mhi_s - m_hat_s, 1))
    print("m-hat", m_hat, mlo_t, mhi_t, "stat", m_hat_s, mlo_s, mhi_s)

    # ---------------- (3) CLs limits, asymptotic, with the expected band
    def sigma_A(mh, mu_grid=(0.25, 0.5, 1.0, 2.0)):
        """sigma of mu-hat from the background-only Asimov data set: sigma^2 = mu^2 / q_mu,A."""
        nA = b_hat
        p1, f1 = S.fit(nA, mh)
        out = []
        for u in mu_grid:
            fu = S.fit(nA, mh, mu=u, p0=p1)[1]
            out.append(u / np.sqrt(max(fu - f1, 1e-12)))
        return np.array(mu_grid), np.array(out)

    def cls(qt, mu, sA):
        """CLs = CL_{s+b}/CL_b from the asymptotic distribution of q~_mu (Cowan et al. 2011, eq. 65)."""
        r = mu / sA
        if qt <= r ** 2:
            clsb, clb = norm.sf(np.sqrt(qt)), norm.cdf(r - np.sqrt(qt))
        else:
            clsb = norm.sf((qt + r ** 2) / (2 * r))
            clb = norm.sf((qt - r ** 2) / (2 * r))
        return clsb / clb

    obs, band, sAs = [], [], []
    for mh in MLIM:
        ug, sg = sigma_A(mh)
        sA = lambda u: np.interp(u, ug, sg)
        sAs.append(sg[2])
        free = S.fit(n, mh)
        f = lambda u: cls(S.qtilde(n, mh, u, free=free), u, sA(u)) - ALPHA
        obs.append(brentq(f, max(free[0][0], 0.0) + 1e-3, 20.0, xtol=1e-3))
        row = []
        for N in (-2, -1, 0, 1, 2):
            g = lambda u: u - sA(u) * (N + norm.ppf(1 - ALPHA * norm.cdf(N)))
            row.append(brentq(g, 1e-3, 20.0, xtol=1e-4))
        band.append(row)
        print(f"m={mh}: obs {obs[-1]:.3f} band {np.round(row, 3)} sigmaA {sg}")
    obs, band, sAs = np.array(obs), np.array(band), np.array(sAs)
    excl = MLIM[obs < 1.0]
    k125 = np.argmin(np.abs(MLIM - 125))
    k140 = np.argmin(np.abs(MLIM - 140))
    nums.update(HyyLimObsOneTwoFive=round(obs[k125], 2), HyyLimExpOneTwoFive=round(band[k125, 2], 2),
                HyyLimExpMin=round(band[:, 2].min(), 2), HyyLimExpMax=round(band[:, 2].max(), 2),
                HyyNexcl=len(excl), HyyNmlim=len(MLIM),
                HyyExclRanges=ranges(excl),
                HyyLimObsOneFourty=round(obs[k140], 2), HyyLimExpOneFourty=round(band[k140, 2], 2),
                HyySigmaAOneFourty=round(sAs[k140], 3), HyySigmaAOneTwoFive=round(sAs[k125], 3),
                HyyBandCoefMed=round(band[k140, 2] / sAs[k140], 2))

    # ---------------- (4) two categories at 125 GeV (statistical part only, linear fits)
    ncat = [np.histogram(m[cat], EDGES)[0].astype(float), np.histogram(m[~cat], EDGES)[0].astype(float)]
    scat = [cp[4] * np.diff(cb_cdf(EDGES, *cp[:4])) for cp in cat_pars]   # expected events per bin, mu = 1
    Bm = bern_basis(deg)
    nb, k = len(CENT), Bm.shape[1]

    def joint(ns, ss):
        """q0 and mu-hat for one common mu over several independent spectra (block-diagonal model)."""
        nn = np.concatenate(ns)[None]
        A0 = np.zeros((len(ns) * nb, len(ns) * k))
        for i in range(len(ns)):
            A0[i * nb:(i + 1) * nb, i * k:(i + 1) * k] = Bm
        A1 = np.concatenate([np.concatenate(ss)[:, None], A0], 1)
        c0, l0 = lin_fit(nn, A0)
        c1, l1 = lin_fit(nn, A1, c0=np.concatenate([[[0.0]], c0], 1))
        return max(2 * (l1[0] - l0[0]), 0.0) * (c1[0, 0] > 0), c1[0, 0]

    s_incl = sig.bins(125.0)
    bcat = []
    for ni in ncat:
        c0, _ = lin_fit(ni[None], Bm)
        bcat.append(Bm @ c0[0])
    zA1 = np.sqrt(joint([b_hat + s_incl], [s_incl])[0])
    zA2 = np.sqrt(joint([bcat[0] + scat[0], bcat[1] + scat[1]], scat)[0])
    zO1, mO1 = joint([n], [s_incl])
    zO2, mO2 = joint(ncat, scat)
    nums.update(HyyZAincl=round(zA1, 2), HyyZAcat=round(zA2, 2), HyyZOincl=round(np.sqrt(zO1), 2),
                HyyZOcat=round(np.sqrt(zO2), 2), HyyMuOincl=round(mO1, 2), HyyMuOcat=round(mO2, 2),
                HyySbCat=round(scat[0].sum() / np.interp(125, CENT, bcat[0]) * BW, 3),
                HyySbRest=round(scat[1].sum() / np.interp(125, CENT, bcat[1]) * BW, 3),
                HyyNcat=int(ncat[0].sum()), HyyNrest=int(ncat[1].sum()),
                HyyScat=round(scat[0].sum(), 1), HyySrest=round(scat[1].sum(), 1))
    print("categories: Asimov", zA1, zA2, "observed", np.sqrt(zO1), np.sqrt(zO2), mO1, mO2)

    # ---------------- figures
    # (a) the fit
    pbest, _ = S.fit(n, m_max)
    nu_sb = S.nu(pbest, m_max)
    nb_best = S.B @ pbest[1:1 + S.k] + pbest[4 + S.k] * S.s_spur * sig.bins(m_max) / sig.s_tot
    N4, B4, SB4 = (x.reshape(-1, 4).sum(1) for x in (n, b_hat, nu_sb))
    Bb4 = nb_best.reshape(-1, 4).sum(1)
    c4 = CENT.reshape(-1, 4).mean(1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 4.8), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.2, 1.4], hspace=0.08))
    a1.errorbar(c4, N4, yerr=np.sqrt(N4), fmt="o", ms=2.2, color="k", lw=0.7, label="data")
    a1.plot(c4, B4, color=SERIES[0], ls="--", label="background-only fit")
    a1.plot(c4, SB4, color=SERIES[1], label=f"signal + background, $m_H={m_max:.1f}$ GeV")
    a1.set_ylabel("events / GeV")
    a1.legend()
    a2.errorbar(c4, N4 - Bb4, yerr=np.sqrt(N4), fmt="o", ms=2.2, color="k", lw=0.7)
    a2.plot(c4, SB4 - Bb4, color=SERIES[1], label=r"fitted signal")
    a2.plot(c4, sig.bins(125.0).reshape(-1, 4).sum(1), color=SERIES[2], ls=":",
            label=r"SM Higgs, $\mu=1$")
    a2.axhline(0, color="0.5", lw=0.6)
    a2.set_ylabel("data $-$ background")
    a2.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    a2.legend(fontsize=7, loc="upper right")
    savefig(fig, "ch13", "hyy_fit")

    # (b) local p0
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.semilogy(MSCAN, norm.sf(np.sqrt(q_full)), color=SERIES[0], label="observed")
    ax.semilogy(MSCAN, norm.sf(np.sqrt(q_stat)), color=SERIES[0], ls=":", lw=1,
                label="observed, nuisances fixed")
    ax.semilogy(MSCAN, norm.sf(np.sqrt(q_exp)), color=SERIES[1], ls="--", label=r"expected for $\mu=1$")
    for z in range(1, 6):
        ax.axhline(norm.sf(z), color="0.6", lw=0.6, ls=":")
        ax.text(150.4, norm.sf(z), f"{z}$\\sigma$", va="center", fontsize=7, color="0.4")
    ax.set_ylim(1e-7, 1.0)
    ax.set_xlabel(r"$m_H$ [GeV]")
    ax.set_ylabel(r"local $p_0$")
    ax.legend(loc="lower left", fontsize=7)
    savefig(fig, "ch13", "hyy_p0")

    # (c) profile likelihoods
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    b1.plot(mus, t_tot, color=SERIES[0], label="all nuisances profiled")
    b1.plot(mus, t_stat, color=SERIES[1], ls="--", label="nuisances fixed")
    b1.axhline(1, color="0.5", lw=0.6, ls=":")
    b1.set_ylim(0, 6)
    b1.set_xlabel(r"$\mu$ at $m_H=125$ GeV")
    b1.set_ylabel(r"$-2\ln\lambda$")
    b1.legend(fontsize=7)
    b2.plot(mhs, fm_t, color=SERIES[0], label="all nuisances profiled")
    b2.plot(mhs, fm_s, color=SERIES[1], ls="--", label="nuisances fixed")
    b2.axhline(1, color="0.5", lw=0.6, ls=":")
    b2.set_ylim(0, 6)
    b2.set_xlabel(r"$m_H$ [GeV]")
    savefig(fig, "ch13", "hyy_mu_mass")

    # (d) Brazil band
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.fill_between(MLIM, band[:, 0], band[:, 4], color="#f2d21b", label=r"expected $\pm2\sigma$")
    ax.fill_between(MLIM, band[:, 1], band[:, 3], color="#3cb043", label=r"expected $\pm1\sigma$")
    ax.plot(MLIM, band[:, 2], color="k", ls="--", lw=1, label="expected median")
    ax.plot(MLIM, obs, color="k", lw=1.4, label="observed")
    ax.axhline(1.0, color=SERIES[7], lw=0.9)
    ax.set_xlabel(r"$m_H$ [GeV]")
    ax.set_ylabel(r"95% CL$_s$ limit on $\mu$")
    ax.set_ylim(0, max(3.0, obs.max() * 1.1))
    ax.legend(fontsize=7, loc="upper right")
    savefig(fig, "ch13", "hyy_brazil")

    # (e) two categories: data minus background with the expected signal
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), sharex=True)
    titles = ["central, unconverted", "all other pairs"]
    for ax, ni, bi, si, t in zip(axs, ncat, bcat, scat, titles):
        Ni, Bi, Si = (x.reshape(-1, 4).sum(1) for x in (ni, bi, si))
        ax.errorbar(c4, Ni - Bi, yerr=np.sqrt(Ni), fmt="o", ms=2, color="k", lw=0.6)
        ax.plot(c4, Si, color=SERIES[2], label=r"SM Higgs, $\mu=1$")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_title(f"{t}: {int(ni.sum())} events", fontsize=8)
        ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    axs[0].set_ylabel("data $-$ background / GeV")
    axs[1].legend(fontsize=7)
    savefig(fig, "ch13", "hyy_categories")

    np.savez(DATA / "ch13" / "hyy_search.npz", MSCAN=MSCAN, q_full=q_full, q_stat=q_stat, q_exp=q_exp,
             mu_full=mu_full, b_hat=b_hat, pb=pb, MLIM=MLIM, obs=obs, band=band, sA=sAs,
             mus=mus, t_tot=t_tot, t_stat=t_stat, mhs=mhs, fm_t=fm_t, fm_s=fm_s)
    save_numbers("ch13", "33_hyy_search", nums)
    print(nums)


if __name__ == "__main__":
    main()
