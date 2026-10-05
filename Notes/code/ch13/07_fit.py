"""07_fit.py -- A_s and n_s from our own Planck bandpowers, with our own sampler.

Question: with the other four LCDM parameters fixed at the Planck 2018 best fit, what values of
ln(10^10 A_s) and n_s does the real sky prefer, from our SMICA bandpowers (30 <= l < 1020), our
simulated covariance and our Metropolis sampler?  How does that compare with (a) the same fit
run on Planck's own binned spectrum over the same multipoles and over 30 <= l < 2490, and
(b) Planck 2018 VI, Table 2 (TT+lowE, all six parameters free)?  And do the low and high
multipoles agree with each other as well as they do in the simulations?
Model: <D_b> = sum_l F_bl C_l(theta), F the bandpower windows of 03_bandpowers.py,
C_l(theta) the second-order emulator of chapter 9 (ln C_l Taylor-expanded about the fiducial).
Likelihood: Gaussian in the bandpowers, covariance from 04_sims.py, Hartlap-corrected.
Prior: flat, 2.8 < ln(10^10 A_s) < 3.3, 0.85 < n_s < 1.10.
Writes: data/ch13/chains.npz, figures/ch13/fit.pdf, figures/ch13/split.pdf, results/ch13/07_fit.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch09"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES
from camb_fiducial import FIDUCIAL
import lib_planck as lp
import lib_mcmc as mc
import lib_cmbemu as ce

setup()
rng = rng_for("ch13", "07_fit")
LC = 1199
t0 = time.time()
z = np.load(lp.DATA / "bandpowers.npz")
REP = z["rep"]
sims = [np.load(f) for f in sorted(lp.DATA.glob("sims_*.npz"))]
X = np.concatenate([s["main_cross"] for s in sims])[:, REP]
n, p = X.shape
D = z["main_cross"][REP]
F = z["main_F"][REP]
lb = z["main_leff"][REP]
C = np.cov(X, rowvar=False)
Psi = (n - p - 2) / (n - 1) * np.linalg.inv(C)
emu = ce.Emulator(order=2, names=["lnAs", "ns"], lmax=2500)
FID = emu.fid.copy()
TAU = FIDUCIAL["tau"]
nums = {"TwelveAFidLnAs": f"{FID[0]:.4f}", "TwelveAFidNs": f"{FID[1]:.4f}", "TwelveATau": f"{TAU}"}
LO, HI = np.array([2.8, 0.85]), np.array([3.3, 1.10])


def make_logpost(Fmat, data, prec):
    def logpost(th):
        if np.any(th < LO) or np.any(th > HI):
            return -np.inf
        r = data - Fmat @ emu.cl(th)[: Fmat.shape[1]]
        return -0.5 * r @ prec @ r
    return logpost


def fisher(Fmat, prec, th=FID, h=np.array([0.01, 0.005])):
    J = np.array([(Fmat @ emu.cl(th + h[i] * np.eye(2)[i])[: Fmat.shape[1]]
                   - Fmat @ emu.cl(th - h[i] * np.eye(2)[i])[: Fmat.shape[1]]) / (2 * h[i]) for i in range(2)]).T
    return J, J.T @ prec @ J


def ml(Fmat, data, prec, iters=3):
    """Gauss-Newton maximum of the Gaussian likelihood (exact for a linear model)."""
    th = FID.copy()
    for _ in range(iters):
        J, Fi = fisher(Fmat, prec, th)
        th = th + np.linalg.solve(Fi, J.T @ prec @ (data - Fmat @ emu.cl(th)[: Fmat.shape[1]]))
    return th


def run(logpost, cov, nchain=4, nstep=20000):
    starts = FID + rng.multivariate_normal(np.zeros(2), 9 * cov, nchain)
    chains, accs = [], []
    for s in starts:
        ch, _, acc = mc.metropolis(logpost, s, nstep, 2.38 / np.sqrt(2), rng, cov=cov)
        chains.append(ch[nstep // 5:])                        # drop the first 20% as burn-in
        accs.append(acc)
    chains = np.array(chains)
    rhat = max(mc.gelman_rubin(chains[:, :, i], split=True) for i in range(2))
    return chains.reshape(-1, 2), float(np.mean(accs)), rhat, chains


def summary(sm):
    m, s = sm.mean(0), sm.std(0, ddof=1)
    return m, s, float(np.corrcoef(sm.T)[0, 1])


# ---------------------------------------------------------------- (1) our bandpowers
J, Fi = fisher(F, Psi)
covF = np.linalg.inv(Fi)
samp, acc, rhat, chains = run(make_logpost(F, D, Psi), covF)
m, s, rho = summary(samp)
tau = np.mean([mc.tau_int(chains[c, :, i])[0] for c in range(chains.shape[0]) for i in range(2)])
AsE = np.exp(samp[:, 0]) * 1e-10 * np.exp(-2 * TAU) * 1e9
nums.update({"TwelveAFitA": f"{m[0]:.4f}", "TwelveAFitSA": f"{s[0]:.4f}", "TwelveAFitN": f"{m[1]:.4f}",
             "TwelveAFitSN": f"{s[1]:.4f}", "TwelveAFitRho": f"{rho:+.2f}", "TwelveAFitAcc": f"{acc:.2f}",
             "TwelveAFitRhat": f"{rhat:.4f}", "TwelveAFitTau": f"{tau:.0f}",
             "TwelveAFitAsE": f"{AsE.mean():.4f}", "TwelveAFitSAsE": f"{AsE.std(ddof=1):.4f}",
             "TwelveAFisherSA": f"{np.sqrt(covF[0, 0]):.4f}", "TwelveAFisherSN": f"{np.sqrt(covF[1, 1]):.4f}"})
th_ml = ml(F, D, Psi)
r = D - F @ emu.cl(th_ml)[:LC + 1]
chi_ml = float(r @ Psi @ r)
nums.update({"TwelveAChiMl": f"{chi_ml:.1f}", "TwelveAPteMl": f"{stats.chi2.sf(chi_ml, p - 2):.2f}"})

# emcee cross-check (the library way)
try:
    import emcee
    lp_fun = make_logpost(F, D, Psi)
    nw = 16
    p0 = FID + 0.5 * rng.multivariate_normal(np.zeros(2), covF, nw)
    sampler = emcee.EnsembleSampler(nw, 2, lp_fun)
    sampler.run_mcmc(p0, 3000, progress=False)
    es = sampler.get_chain(discard=600, flat=True)
    me, se, rhoe = summary(es)
    nums.update({"TwelveAEmA": f"{me[0]:.4f}", "TwelveAEmSA": f"{se[0]:.4f}", "TwelveAEmN": f"{me[1]:.4f}",
                 "TwelveAEmSN": f"{se[1]:.4f}"})
except ImportError:
    es = None

# ---------------------------------------------------------------- (2) the same fit on Planck's bandpowers
l_pl, D_pl, s_pl, D_bf = lp.load_planck_binned()
edges_pl = lp.planck_edges(30, 2508, 30)
P_pl, _, _ = lp.planck_binning(edges_pl, 2508)
res_pl = {}
for tag, nb in (("Same", p), ("Full", 82)):                  # 82 bins: l < 2490 (emulator stops at 2500)
    Fp = P_pl[:nb, :2501]
    prec = np.diag(1 / s_pl[:nb] ** 2)
    _, Fi_p = fisher(Fp, prec)
    sm, acc_p, rh_p, _ = run(make_logpost(Fp, D_pl[:nb], prec), np.linalg.inv(Fi_p), nstep=12000)
    mp, sp, rp = summary(sm)
    res_pl[tag] = sm
    nums.update({f"TwelveAPl{tag}A": f"{mp[0]:.4f}", f"TwelveAPl{tag}SA": f"{sp[0]:.4f}",
                 f"TwelveAPl{tag}N": f"{mp[1]:.4f}", f"TwelveAPl{tag}SN": f"{sp[1]:.4f}",
                 f"TwelveAPl{tag}Rho": f"{rp:+.2f}", f"TwelveAPl{tag}Rhat": f"{rh_p:.4f}"})
nums["TwelveAPlFullLmax"] = int(edges_pl[82] - 1)
nums.update({"TwelveAPubA": "3.040", "TwelveAPubSA": "0.016", "TwelveAPubN": "0.9626", "TwelveAPubSN": "0.0057",
             "TwelveAPubAsE": "1.884", "TwelveAPubSAsE": "0.014"})

# ---------------------------------------------------------------- (3) low l against high l
SPLIT = 510
lo = lb < SPLIT
hi = ~lo


def split_fit(data, Xs):
    out = []
    for sel in (lo, hi):
        Cs = np.cov(Xs[:, sel], rowvar=False)
        ps = sel.sum()
        prec = (n - ps - 2) / (n - 1) * np.linalg.inv(Cs)
        out.append(ml(F[sel], data[sel], prec, iters=2))
    return out[0] - out[1], out


delta_d, (th_lo, th_hi) = split_fit(D, X)
delta_s = np.array([split_fit(x, X)[0] for x in X])
Cd = np.cov(delta_s, rowvar=False)
chi_split = float(delta_d @ np.linalg.solve(Cd, delta_d))
chi_split_s = np.einsum("ij,ij->i", delta_s, np.linalg.solve(Cd, delta_s.T).T)
nums.update({"TwelveASplit": SPLIT, "TwelveALoA": f"{th_lo[0]:.3f}", "TwelveALoN": f"{th_lo[1]:.3f}",
             "TwelveAHiA": f"{th_hi[0]:.3f}", "TwelveAHiN": f"{th_hi[1]:.3f}",
             "TwelveADeltaA": f"{delta_d[0]:+.3f}", "TwelveADeltaN": f"{delta_d[1]:+.3f}",
             "TwelveADeltaSA": f"{np.sqrt(Cd[0, 0]):.3f}", "TwelveADeltaSN": f"{np.sqrt(Cd[1, 1]):.3f}",
             "TwelveASplitChi": f"{chi_split:.2f}", "TwelveASplitPte": f"{np.mean(chi_split_s >= chi_split):.2f}",
             "TwelveASplitPteChi": f"{stats.chi2.sf(chi_split, 2):.2f}"})
np.savez(lp.DATA / "chains.npz", ours=samp, planck_same=res_pl["Same"], planck_full=res_pl["Full"],
         delta_sims=delta_s, delta_data=delta_d)
nums["TwelveAFitSeconds"] = f"{time.time() - t0:.0f}"
print(nums)

# ---------------------------------------------------------------- figures
import lib_mcmc9c as m9
fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.0))
names = [r"$\ln(10^{10}A_s)$", r"$n_s$"]
sets = [samp, res_pl["Same"], res_pl["Full"]]
cols = [SERIES[0], SERIES[1], SERIES[2]]
labs = [r"ours, SMICA $30\leq\ell<1020$", r"Planck bandpowers, $30\leq\ell<1020$", r"Planck bandpowers, $30\leq\ell<2490$"]
rng_ = [(3.00, 3.09), (0.925, 1.005)]
for sm, c, lab in zip(sets, cols, labs):
    m9.hist1d(axs[0, 0], sm[:, 0], c, ranges=rng_[0], label=lab)
    m9.hist1d(axs[1, 1], sm[:, 1], c, ranges=rng_[1])
    m9.contour2d(axs[1, 0], sm[:, 0], sm[:, 1], c, ranges=rng_)
for ax, (mu, sd, i) in zip([axs[0, 0], axs[1, 1]], [(3.040, 0.016, 0), (0.9626, 0.0057, 1)]):
    m9.gauss_1d(ax, mu, sd, rng_[i])
axs[0, 0].plot([], [], "k--", lw=1.1, label="Planck 2018 VI, TT+lowE (6 free)")
axs[1, 0].plot(FID[0], FID[1], "k+", ms=8)
axs[0, 1].axis("off")
h, l = axs[0, 0].get_legend_handles_labels()
axs[0, 1].legend(h, l, loc="center", fontsize=7.5)
axs[1, 0].set_xlabel(names[0])
axs[1, 0].set_ylabel(names[1])
axs[1, 1].set_xlabel(names[1])
axs[0, 0].set_xlim(rng_[0])
axs[1, 1].set_xlim(rng_[1])
fig.tight_layout()
savefig(fig, "ch13", "fit")

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax = axs[0]
ax.plot(delta_s[:, 0], delta_s[:, 1], ".", ms=2, color="0.6", label="simulations")
ax.plot(delta_d[0], delta_d[1], "*", ms=11, color=SERIES[1], label="SMICA")
ax.set_xlabel(r"$\Delta\ln(10^{10}A_s)$ (low $-$ high)")
ax.set_ylabel(r"$\Delta n_s$")
ax.legend(fontsize=7.5)
ax = axs[1]
ax.hist(chi_split_s, bins=40, density=True, color="0.75")
xx = np.linspace(0, 14, 200)
ax.plot(xx, stats.chi2.pdf(xx, 2), "k--", lw=1)
ax.axvline(chi_split, color=SERIES[1], lw=1.5)
ax.set_xlabel(r"$\chi^2$ of the shift")
ax.set_ylabel("density")
fig.tight_layout()
savefig(fig, "ch13", "split")
save_numbers("ch13", "07_fit", nums)
