"""18_biref_mc.py -- simulated skies with a known rotation: does the estimator of beta reach the forecast?

Question: on a masked, noisy, beamed sphere, the E and B pseudo-spectra are no longer the clean
spectra of the forecast: the mask moves E power into B.  Is the estimator of beta still unbiased,
does its scatter reach the Fisher error, and do the quoted error bars cover the truth 68% of the
time?  What changes when the covariance ignores the leakage?

Simulation (HEALPix nside 256, a_lm to l = 767, analysis 51 <= l <= 600 in bins of 25):
  * Gaussian T, E, B skies with the lensed fiducial spectra (EB = TB = 0), a 7.22' Gaussian beam
    and the pixel window; the E part (T, E) and the B part (B) are mapped separately, so that the
    leakage of E into B can be read off directly;
  * white noise, two depths: Planck-like (52.3 muK' in Q, U; 25.5 muK' in T) and deep (3 muK' in
    Q, U; 3/sqrt2 in T); one unit-noise map per sky, scaled (the transforms are linear);
  * two Galactic-like masks keeping |b| > 17.5 deg: a sharp one, and one with a 5-deg cosine taper;
  * the rotation beta_in = 0.35 deg is applied to the pseudo-a_lm of the masked signal (it commutes
    with the mask, checked below on one sky to round-off), then noise is added.
Estimator: linearised template fit e_b = 2 beta D_b with e = EB, D = EE - BB (and t = TB, G = TE
for the joint fit), all pseudo-spectra divided by w2 = <W^2>; weights from three covariances:
  (a) model:   Var(EB_b) = E~ B~ / nu_b with the full-sky signal + noise, i.e. leakage ignored;
  (b) data:    the same formula with the sky's own measured pseudo-spectra (leakage included);
  (c) sims:    the covariance of the bandpowers over the other half of the skies.
Records bias, scatter, mean quoted error and coverage over N_SIM skies, with Monte Carlo errors.
Writes: data/ch10/biref_mc.npz, figures/ch10/biref_leakage.pdf, figures/ch10/biref_mc.pdf,
        results/ch10/18_biref_mc.tex
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch08"))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line, DATA, SEED_SALT
import lib_biref10 as LB
from lib_masks import band

N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else 1000   # a smaller number only for a quick test
NSIDE, LSIM, LAN = 256, 767, 610
EDGES = np.arange(51, 602, 25)              # 22 bins [51,76), ..., [576,601)
NB = len(EDGES) - 1
FWHM = 7.22
BETA_IN = 0.35 * LB.DEG
SETUPS = {"pl": (25.5, 52.3), "deep": (3.0 / np.sqrt(2), 3.0)}     # (Delta_T, Delta_P) muK arcmin
MASKS = ["apo", "sharp"]
CACHE = DATA / "ch10" / (f"biref_mc_salt{SEED_SALT}.npz" if SEED_SALT else "biref_mc.npz")   # a seed test never reuses the book's skies

cl = LB.fiducial(LSIM)
bl = hp.gauss_beam(FWHM * LB.ARCMIN, lmax=LSIM) * hp.pixwin(NSIDE, lmax=LSIM, pol=False)
pix_arcmin = np.sqrt(4 * np.pi / hp.nside2npix(NSIDE)) / LB.ARCMIN
W = {"apo": band(NSIDE, 17.5, apo_deg=5.0), "sharp": band(NSIDE, 17.5)}
w2 = {k: np.mean(m ** 2) for k, m in W.items()}
w4 = {k: np.mean(m ** 4) for k, m in W.items()}
fsky = {k: w2[k] ** 2 / w4[k] for k in W}
ell = np.arange(LAN + 1)


def binned(c):
    return np.array([c[EDGES[i]:EDGES[i + 1]].mean() for i in range(NB)])


nu_full = np.array([(2 * ell[EDGES[i]:EDGES[i + 1]] + 1).sum() for i in range(NB)])


def pseudo(maps, mask):
    return hp.map2alm([mask * m for m in maps], lmax=LAN, iter=0, pol=True)


def spectra(T, E, B):
    c = {"EE": hp.alm2cl(E), "BB": hp.alm2cl(B), "EB": hp.alm2cl(E, B), "TE": hp.alm2cl(T, E),
         "TB": hp.alm2cl(T, B), "TT": hp.alm2cl(T)}
    return np.array([binned(c[k]) for k in ("TT", "EE", "BB", "TE", "EB", "TB")])


def simulate():
    rng = rng_for("ch10", "18_biref_mc")
    npix = hp.nside2npix(NSIDE)
    # spectra[s, setup, mask, (TT,EE,BB,TE,EB,TB), bin]; eb0 = EB with no rotation; leak = BB of E part
    S = np.zeros((N_SIM, 2, 2, 6, NB)); EB0 = np.zeros((N_SIM, 2, 2, NB)); LEAK = np.zeros((N_SIM, 2, NB))
    t0 = time.time()
    for s in range(N_SIM):
        aT, aE, aB = LB.corr_alm(cl, LSIM, rng)
        aT, aE, aB = (hp.almxfl(a, bl) for a in (aT, aE, aB))
        z = np.zeros_like(aT)
        mE = hp.alm2map([aT, aE, z], NSIDE, lmax=LSIM, pol=True)
        mB = hp.alm2map([z, z, aB], NSIDE, lmax=LSIM, pol=True)
        nmap = rng.standard_normal((3, npix)) / pix_arcmin          # 1 muK arcmin of white noise
        for k, mk in enumerate(MASKS):
            tE, eE, bE = pseudo(mE, W[mk])
            _, eB, bB = pseudo(mB, W[mk])
            tN, eN, bN = pseudo(nmap, W[mk])
            LEAK[s, k] = binned(hp.alm2cl(bE))
            Es, Bs = LB.rotate(eE + eB, bE + bB, BETA_IN)
            for j, (dT, dP) in enumerate(SETUPS.values()):
                T = tE + dT * tN
                S[s, j, k] = spectra(T, Es + dP * eN, Bs + dP * bN)
                EB0[s, j, k] = binned(hp.alm2cl(eE + eB + dP * eN, bE + bB + dP * bN))
        if s % 100 == 0:
            print(f"sky {s}  {time.time() - t0:.0f} s", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, S=S, EB0=EB0, LEAK=LEAK, n=N_SIM)
    return S, EB0, LEAK


if CACHE.exists() and int(np.load(CACHE)["n"]) == N_SIM:
    z = np.load(CACHE); S, EB0, LEAK = z["S"], z["EB0"], z["LEAK"]
else:
    S, EB0, LEAK = simulate()

# ---------------------------------------------------------------- commutation check on one sky
rng1 = rng_for("ch10", "18_biref_mc", stream=99)
aT, aE, aB = (hp.almxfl(a, bl) for a in LB.corr_alm(cl, LSIM, rng1))
Tm, Qm, Um = hp.alm2map([aT, aE, aB], NSIDE, lmax=LSIM, pol=True)
c2, s2 = np.cos(2 * BETA_IN), np.sin(2 * BETA_IN)
_, e1, b1 = pseudo([Tm, c2 * Qm - s2 * Um, s2 * Qm + c2 * Um], W["sharp"])     # rotate the sky, then mask
_, e0, b0 = pseudo([Tm, Qm, Um], W["sharp"])
e2, b2 = LB.rotate(e0, b0, BETA_IN)                                               # mask, then rotate
commute = max(np.abs(e1 - e2).max(), np.abs(b1 - b2).max()) / np.abs(e0).max()

# ---------------------------------------------------------------- the estimator
nums = {"TenCNsim": N_SIM, "TenCNside": NSIDE, "TenCNbins": NB,
        "TenCFskyApo": round(fsky["apo"], 3), "TenCFskySharp": round(fsky["sharp"], 3),
        "TenCCommute": r"%s\times10^{%s}" % tuple(f"{commute:.0e}".split("e"))}
EBIN = binned(cl["EE"][: LAN + 1] * bl[: LAN + 1] ** 2)
BBIN = binned(cl["BB"][: LAN + 1] * bl[: LAN + 1] ** 2)
TBIN = binned(cl["TT"][: LAN + 1] * bl[: LAN + 1] ** 2)
XBIN = binned(cl["TE"][: LAN + 1] * bl[: LAN + 1] ** 2)


def fit(y, x, V):
    """GLS for y = 2 beta x per bin; V[b] is the 1x1 or 2x2 covariance of y at bin b (no bin-bin
    correlation) or a full matrix over the stacked vector when V.ndim == 2."""
    if V.ndim == 2:
        Vi = np.linalg.inv(V)
        F = x @ Vi @ x
        return (x @ Vi @ y) / (2 * F), 1 / (2 * np.sqrt(F))
    if y.ndim == 1:
        F = np.sum(x * x / V)
        return np.sum(x * y / V) / (2 * F), 1 / (2 * np.sqrt(F))
    Vi = np.linalg.inv(V)                                  # (NB, 2, 2)
    F = np.einsum("bi,bij,bj->", x, Vi, x)
    return np.einsum("bi,bij,bj->", x, Vi, y) / (2 * F), 1 / (2 * np.sqrt(F))


def analyse(j, k, which):
    sp = S[:, j, k] / w2[MASKS[k]]                             # (N, 6, NB) full-sky-normalised
    TT, EE, BB, TE, EB, TB = (sp[:, i] for i in range(6))
    dT, dP = list(SETUPS.values())[j]
    nw = lambda d: (d * LB.ARCMIN) ** 2
    nub = nu_full * fsky[MASKS[k]]
    mE, mB, mT, mX = EBIN + nw(dP), BBIN + nw(dP), TBIN + nw(dT), XBIN
    out = {}
    if which == "EB":
        y, x = EB, EE - BB
        Va = mE * mB / nub
        Vb = None
    else:
        y = np.stack([EB, TB], -1); x = np.stack([EE - BB, TE], -1)
        Va = np.zeros((NB, 2, 2)); Va[:, 0, 0] = mE * mB / nub; Va[:, 1, 1] = mT * mB / nub
        Va[:, 0, 1] = Va[:, 1, 0] = mX * mB / nub
    res = {"a": [], "b": [], "c": []}
    halves = [np.arange(N_SIM) % 2 == 0, np.arange(N_SIM) % 2 == 1]
    covs = []
    for h in halves:                                           # covariance from the other half
        yy = y[~h].reshape(np.sum(~h), -1)
        n_, p_ = yy.shape
        covs.append(np.cov(yy, rowvar=False) * (n_ - 1) / (n_ - p_ - 2))   # Hartlap: unbiased inverse
    for s in range(N_SIM):
        res["a"].append(fit(y[s], x[s], Va))
        if which == "EB":
            Vb = EE[s] * BB[s] / nub
        else:
            Vb = np.zeros((NB, 2, 2)); Vb[:, 0, 0] = EE[s] * BB[s] / nub; Vb[:, 1, 1] = TT[s] * BB[s] / nub
            Vb[:, 0, 1] = Vb[:, 1, 0] = TE[s] * BB[s] / nub
        res["b"].append(fit(y[s], x[s], Vb))
        Vc = covs[0] if halves[0][s] else covs[1]
        res["c"].append(fit(y[s].reshape(-1), x[s].reshape(-1), Vc))
    for key in res:
        r = np.array(res[key]) / LB.DEG
        out[key] = r
    return out


def fisher_sigma(j, k, which, extra_B=None):
    dT, dP = list(SETUPS.values())[j]
    NT = (dT * LB.ARCMIN) ** 2 / bl[: LAN + 1] ** 2 * np.ones(LAN + 1)
    NP = (dP * LB.ARCMIN) ** 2 / bl[: LAN + 1] ** 2 * np.ones(LAN + 1)
    c = {kk: v[: LAN + 1].copy() for kk, v in cl.items()}
    if extra_B is not None:
        c["BB"] = c["BB"] + extra_B
    return LB.sigma_beta(c, NT, NP, EDGES[0], EDGES[-1] - 1, fsky[MASKS[k]], which) / LB.DEG


tag = {"pl": "Pl", "deep": "Deep"}
tagm = {"apo": "Apo", "sharp": "Sharp"}
table = {}
for j, su in enumerate(SETUPS):
    for k, mk in enumerate(MASKS):
        # leakage as extra B power (per l, from the binned mean, beam removed) for the Fisher comparison
        leak_b = LEAK[:, k].mean(0) / w2[mk]
        leak_l = np.zeros(LAN + 1)
        for i in range(NB):
            leak_l[EDGES[i]:EDGES[i + 1]] = leak_b[i] / np.mean(bl[EDGES[i]:EDGES[i + 1]] ** 2)
        for which in ("EB", "joint"):
            r = analyse(j, k, which)
            sF = fisher_sigma(j, k, which)
            sFL = fisher_sigma(j, k, which, extra_B=leak_l)
            key = tag[su] + tagm[mk] + ("EB" if which == "EB" else "J")
            bc = r["c"][:, 0]
            sd = bc.std(ddof=1)
            row = dict(fisher=sF, fisherleak=sFL, sd=sd, sderr=sd / np.sqrt(2 * (N_SIM - 1)),
                       bias=bc.mean() - 0.35, biaserr=sd / np.sqrt(N_SIM))
            for v in "abc":
                b_, s_ = r[v][:, 0], r[v][:, 1]
                cov = np.mean(np.abs(b_ - 0.35) < s_)
                row[f"cov{v}"] = cov
                row[f"sig{v}"] = s_.mean()
                row[f"sd{v}"] = b_.std(ddof=1)
            table[key] = row
            print(key, {kk: round(float(vv), 4) for kk, vv in row.items()})
            nums[f"TenC{key}Fisher"] = f"{sF:.3g}"
            nums[f"TenC{key}FisherLeak"] = f"{sFL:.3g}"
            nums[f"TenC{key}SD"] = f"{sd:.3g}"
            nums[f"TenC{key}SDerr"] = f"{row['sderr']:.1g}"
            nums[f"TenC{key}Ratio"] = round(float(sd / sF), 3)
            nums[f"TenC{key}Bias"] = f"{row['bias']:.2g}"
            nums[f"TenC{key}BiasErr"] = f"{row['biaserr']:.1g}"
            for v, nm in zip("abc", ("Model", "Data", "Sims")):
                nums[f"TenC{key}Cov{nm}"] = round(100 * row[f"cov{v}"], 1)
                nums[f"TenC{key}Sig{nm}"] = f"{row[f'sig{v}']:.3g}"
                nums[f"TenC{key}SD{nm}"] = f"{row[f'sd{v}']:.3g}"
nums["TenCCovErr"] = round(100 * np.sqrt(0.683 * 0.317 / N_SIM), 1)
# how far the bias and the coverage sit from their targets, in Monte Carlo errors (all set-ups share skies)
bsig = {k: abs(r["bias"]) / r["biaserr"] for k, r in table.items()}
nums["TenCBiasSigMax"] = f"{max(bsig.values()):.1f}"
nums["TenCBiasFracMax"] = f"{max(abs(r['bias']) / r['sd'] for r in table.values()):.2f}"
nums["TenCBiasResol"] = f"{3 / np.sqrt(N_SIM):.2f}"                 # 3 MC errors, in units of the scatter
for k in ("DeepApoJ", "DeepSharpJ"):
    nums[f"TenC{k}CovModelSig"] = f"{(68.27 - 100 * table[k]['cova']) / (100 * np.sqrt(0.683 * 0.317 / N_SIM)):.1f}"
    nums[f"TenC{k}SigModelGap"] = f"{(table[k]['sda'] - table[k]['siga']) / (table[k]['sda'] / np.sqrt(2 * (N_SIM - 1))):.0f}"

# chance EB with no rotation: mean over skies against its own Monte Carlo error
for j, su in enumerate(SETUPS):
    for k, mk in enumerate(MASKS):
        e = EB0[:, j, k] / w2[mk]
        chi2 = np.sum((e.mean(0) / (e.std(0, ddof=1) / np.sqrt(N_SIM))) ** 2)
        nums[f"TenCChiEB{tag[su]}{tagm[mk]}"] = round(float(chi2), 1)
# leakage size relative to lensing B, per mask, in the lowest bins and at l ~ 300
LKM = {}
for k, mk in enumerate(MASKS):
    lk = LEAK[:, k].mean(0) / w2[mk]
    nums[f"TenCLeakLow{tagm[mk]}"] = f"{lk[0] / BBIN[0]:.2g}"
    i300 = np.searchsorted(EDGES, 300) - 1
    nums[f"TenCLeakMid{tagm[mk]}"] = f"{lk[i300] / BBIN[i300]:.2g}"
    nums[f"TenCLeakEEpct{tagm[mk]}"] = f"{100 * lk[0] / EBIN[0]:.2g}"
    nums[f"TenCLeakEEpctMid{tagm[mk]}"] = f"{100 * lk[i300] / EBIN[i300]:.2g}"
    LKM[mk] = lk
nums["TenCLeakRatioLow"] = f"{LKM['apo'][0] / LKM['sharp'][0]:.2f}"          # taper / sharp, lowest bin
nums["TenCLeakRatioMid"] = f"{LKM['apo'][i300] / LKM['sharp'][i300]:.2f}"    # taper / sharp, l ~ 300
nums["TenCLeakLcLow"] = int(0.5 * (EDGES[0] + EDGES[1]))
nums["TenCLeakLcMid"] = int(0.5 * (EDGES[i300] + EDGES[i300 + 1]))

# ---------------------------------------------------------------- figures
lc = 0.5 * (EDGES[1:] + EDGES[:-1])
dl = lc * (lc + 1) / (2 * np.pi)
setup(7.2, 3.2)
fig, ax = plt.subplots(1, 2)
for k, mk in enumerate(MASKS):
    lk = LEAK[:, k] / w2[mk]
    ax[0].errorbar(lc, dl * lk.mean(0), dl * lk.std(0, ddof=1) / np.sqrt(N_SIM), fmt="o", ms=3,
                   color=SERIES[k], label="leak, " + (r"5$^\circ$ taper" if mk == "apo" else "sharp edge"))
ax[0].plot(lc, dl * BBIN, color=SERIES[2], label="lensing $BB$")
ax[0].plot(lc, dl * EBIN, color=SERIES[3], label="$EE$")
for j, (su, (dT, dP)) in enumerate(SETUPS.items()):
    ax[0].plot(lc, dl * (dP * LB.ARCMIN) ** 2 * np.ones(NB), ls="--", color="0.4" if j == 0 else "0.7",
               label=f"noise, {'Planck-like' if su == 'pl' else 'deep'}")
ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
ax[0].set_ylim(1e-4, 1e4); ax[0].legend(fontsize=6, ncol=2, loc="upper center", handlelength=1.4, columnspacing=0.8)  # legend above the curves
# EB of the rotated skies against the prediction 1/2 sin 4beta (EE - BB)
for j, su in enumerate(SETUPS):
    k = 0
    eb = S[:, j, k, 4] / w2["apo"]
    ax[1].errorbar(lc + 4 * j, dl * eb.mean(0), dl * eb.std(0, ddof=1) / np.sqrt(N_SIM), fmt="o", ms=3,
                   color=SERIES[j], label=f"mean $EB$, {'Planck-like' if su == 'pl' else 'deep'}, taper")
theory_line(ax[1], lc, dl * 0.5 * np.sin(4 * BETA_IN) * (EBIN - BBIN), label=r"$\frac{1}{2}\sin4\beta\,(EE-BB)$")
ax[1].set_xlabel(r"$\ell$"); ax[1].set_ylabel(r"$D^{EB}_\ell$ [$\mu$K$^2$]"); ax[1].set_ylim(-0.005, 0.28); ax[1].legend(fontsize=6.5, loc="upper left")  # room above the peak
fig.subplots_adjust(wspace=0.35)
savefig(fig, "ch10", "biref_leakage")

setup(7.2, 3.0)
fig, ax = plt.subplots(1, 2)
for j, su in enumerate(SETUPS):
    r = analyse(j, 0, "joint")
    sF = fisher_sigma(j, 0, "joint")
    z = (r["c"][:, 0] - 0.35) / sF
    ax[0].hist(z, bins=40, density=True, histtype="step", color=SERIES[j],
               label=f"{'Planck-like' if su == 'pl' else 'deep'}, taper")
x = np.linspace(-4.5, 4.5, 200)
theory_line(ax[0], x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), label="$\\mathcal{N}(0,1)$")
ax[0].set_xlabel(r"$(\hat\beta-\beta_{\rm in})/\sigma_{\rm Fisher}$"); ax[0].legend(fontsize=6.5)
labels, xs = [], 0
for j, su in enumerate(SETUPS):
    for k, mk in enumerate(MASKS):
        row = table[tag[su] + tagm[mk] + "J"]
        for v, col in zip("abc", SERIES[:3]):
            ax[1].bar(xs + "abc".index(v) * 0.25, 100 * row[f"cov{v}"], 0.25, color=col,
                      label=["model cov.", "measured spectra", "simulations"]["abc".index(v)] if xs == 0 else None)
        labels.append(f"{'Pl' if su == 'pl' else 'deep'}\n{'taper' if mk == 'apo' else 'sharp'}")
        xs += 1
ax[1].axhline(68.3, color="k", ls="--", lw=1)
ax[1].set_xticks(np.arange(xs) + 0.25); ax[1].set_xticklabels(labels, fontsize=7)
ax[1].set_ylabel("coverage of $\\pm1\\sigma$ [%]"); ax[1].set_ylim(0, 95); ax[1].legend(fontsize=5.5, loc="upper center", ncol=3, handlelength=1.0, columnspacing=0.8)  # legend above the bars
savefig(fig, "ch10", "biref_mc")
save_numbers("ch10", "18_biref_mc", nums)
for k_, v_ in nums.items():
    print(k_, v_)
