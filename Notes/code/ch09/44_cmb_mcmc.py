"""44_cmb_mcmc.py -- (ln 10^10 A_s, n_s) from one simulated sky: full sky, noisy, masked.

Question: one sky is drawn from the fiducial LCDM spectrum and observed three ways with the toy
instrument of chapter 8 (nside 256, 30' beam, 200 muK arcmin, l <= 601).  What does the biased
random walker say about the amplitude and the tilt of the primordial spectrum, how do the
posteriors change from (a) a noiseless full sky to (b) the noisy full sky to (c) the noisy sky
behind the apodised Galactic band of chapter 8 (MASTER bandpowers, Gaussian likelihood with the
covariance of 2000 simulations), and do they agree with the Fisher forecast?
Also: on the full noiseless sky, do the Gaussian approximations L_Q and L_f of section 9c
shift n_s?  (500 skies drawn from the exact chi^2 distribution, maximum of each likelihood.)
Model C_l(theta): lib_cmbemu first-order Taylor series of ln C_l in (ln 10^10 A_s, n_s), others
fixed at the fiducial (validated in 43_emulator.py).
Writes: figures/ch09/cmb_data.pdf, figures/ch09/cmb_triangle.pdf, figures/ch09/cmb_diag.pdf,
        figures/ch09/cmb_gauss_bias.pdf, data/ch09/cmb_chains.npz, results/ch09/44_cmb_mcmc.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch08"))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import optimize
from common import setup, savefig, save_numbers, rng_for, SERIES
import lib_mcmc9c as mc
import lib_cmbemu as ce
import lib_cmbsim as cs
import lib_masks as lm

setup()
rng = rng_for("ch09", "44_cmb_mcmc")
nums = {}
t_start = time.time()
DATA8 = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"
L = 767                                   # synthesis / MASTER band limit of chapter 8
LMAX = 601                                # last multipole analysed (end of the 30th bin of 20)
exp = cs.Experiment()
emu = ce.Emulator(order=1, names=["lnAs", "ns"], lmax=L)
TRUTH = emu.fid.copy()
ell = np.arange(L + 1)
cl_true = emu.cl(TRUTH)                   # = CAMB fiducial (to 3e-6 for l <= 600)
T2 = exp.transfer(L) ** 2
Nw = exp.nl(L)[0]
use = (ell >= 2) & (ell <= LMAX)
nu = 2 * ell + 1.0

# ---------------------------------------------------------------- the data: one sky, three views
a = cs.synalm(cl_true, L, rng)
chat_sky = hp.alm2cl(a)                                        # (a) the sky itself, full, no noise
d_map = cs.observe_signal(a, exp) + cs.noise_map(exp, rng)     # what the instrument records
chat_obs = hp.alm2cl(hp.map2alm(d_map, lmax=L, iter=3))        # (b) full sky, beam + noise
zm = np.load(DATA8 / "masks.npz")
W = zm["mask_band20apo"].astype(float)
fsky, wi = lm.mask_moments(W)
M = np.load(DATA8 / "coupling.npz")["M_band20apo"]
edges = lm.bin_edges(2, L, 20)
P, Q = lm.binning_operators(edges, L)
K = lm.master_matrix(M, T2, P, Q)
Kinv = np.linalg.inv(K)
pcl = hp.alm2cl(hp.map2alm(W * d_map, lmax=L, iter=0))       # same quadrature as chapter 8
Db_data = Kinv @ (P @ (pcl - Nw * fsky * wi[2]))               # (c) MASTER bandpowers
zM = np.load(DATA8 / "master.npz")
rep = edges[1:] <= LMAX + 1
Fwin = zM["Fwin_band20apo"][rep]
sims = zM["master_band20apo"][:, rep].astype(float)
NS, NB = sims.shape
Csim = np.cov(sims.T)
hartlap = (NS - NB - 2) / (NS - 1)
Cinv = hartlap * np.linalg.inv(Csim)
Db = Db_data[rep]
lb = zM["lb"][rep]
nums.update({"NineCToyLmax": LMAX, "NineCToyNbins": NB, "NineCToyNsims": NS,
             "NineCToyHartlap": f"{hartlap:.4f}", "NineCToyFsky": f"{fsky:.3f}"})


# ---------------------------------------------------------------- the three likelihoods (+ priors)
PRIOR = np.array([[2.5, 3.6], [0.8, 1.1]])


def inprior(th):
    return np.all((th > PRIOR[:, 0]) & (th < PRIOR[:, 1]))


def lp_full(th):
    if not inprior(th):
        return -np.inf
    c = emu.cl(th)
    return -0.5 * np.sum(nu[use] * (chat_sky[use] / c[use] + np.log(c[use])))


def lp_noisy(th):
    if not inprior(th):
        return -np.inf
    c = T2 * emu.cl(th) + Nw                          # what the instrument's spectrum should be
    return -0.5 * np.sum(nu[use] * (chat_obs[use] / c[use] + np.log(c[use])))


def lp_mask(th):
    if not inprior(th):
        return -np.inf
    r = Db - Fwin @ emu.cl(th)
    return -0.5 * r @ Cinv @ r


def lp_Q(th, chat=None):
    """Quadratic approximation L_Q (HL eq. 23) on the noiseless full sky."""
    chat = chat_sky if chat is None else chat
    if not inprior(th):
        return -np.inf
    c = emu.cl(th)
    return -0.5 * np.sum(nu[use] / 2 * ((chat[use] - c[use]) / c[use]) ** 2)


def lp_f(th, chat=None):
    """Gaussian with the fiducial spectrum in the denominator, L_f (HL eq. 24)."""
    chat = chat_sky if chat is None else chat
    if not inprior(th):
        return -np.inf
    c = emu.cl(th)
    return -0.5 * np.sum(nu[use] / 2 * ((chat[use] - c[use]) / cl_true[use]) ** 2)


# ---------------------------------------------------------------- Fisher forecasts for the three views
dC = emu.D * cl_true[None, :]                                    # dC_l / d theta_i
F_full = (dC[:, use] * (nu[use] / 2 / cl_true[use] ** 2)) @ dC[:, use].T
ctot = cl_true + Nw / T2
F_noisy = (dC[:, use] * (nu[use] / 2 / ctot[use] ** 2)) @ dC[:, use].T
J = Fwin @ dC.T
F_mask = J.T @ Cinv @ J / hartlap                                # Fisher uses the true inverse
covs = {"full": np.linalg.inv(F_full), "noisy": np.linalg.inv(F_noisy), "mask": np.linalg.inv(F_mask)}

# ---------------------------------------------------------------- the chains
NCH, NST, BURN = 4, 20000, 2000
runs = {}
for name, lp in (("full", lp_full), ("noisy", lp_noisy), ("mask", lp_mask), ("Q", lp_Q), ("f", lp_f)):
    cov_prop = covs["full" if name in ("Q", "f") else name]
    starts = TRUTH + 6 * np.sqrt(np.diag(cov_prop)) * rng.uniform(-1, 1, (NCH, 2))
    t0 = time.time()
    ch, lpv, acc = mc.run_chains(lp, starts, cov_prop, NST, rng)
    secs = time.time() - t0
    post = ch[:, BURN:].reshape(-1, 2)
    m, s, R = mc.summary(post)
    runs[name] = dict(ch=ch, post=post, m=m, s=s, rho=R[0, 1], acc=acc.mean(), rhat=mc.rhat(ch[:, BURN:]),
                      tau=[mc.tau_int(ch[0, BURN:, k]) for k in range(2)], secs=secs)
    T = name.capitalize() if name not in ("Q", "f") else {"Q": "Quad", "f": "Fid"}[name]
    nums.update({f"NineCToy{T}A": f"{m[0]:.4f}", f"NineCToy{T}SA": f"{s[0]:.4f}",
                 f"NineCToy{T}N": f"{m[1]:.4f}", f"NineCToy{T}SN": f"{s[1]:.4f}",
                 f"NineCToy{T}Rho": f"{R[0, 1]:.2f}", f"NineCToy{T}Acc": f"{acc.mean():.2f}",
                 f"NineCToy{T}Rhat": f"{runs[name]['rhat'].max():.4f}",
                 f"NineCToy{T}Tau": f"{max(runs[name]['tau']):.1f}",
                 f"NineCToy{T}ESS": f"{sum(mc.ess(ch[c, BURN:, 1]) for c in range(NCH)):.0f}",
                 f"NineCToy{T}Sec": f"{secs:.0f}"})
    if name in covs:
        Cf = covs[name]
        nums.update({f"NineCToy{T}FSA": f"{np.sqrt(Cf[0, 0]):.4f}", f"NineCToy{T}FSN": f"{np.sqrt(Cf[1, 1]):.4f}",
                     f"NineCToy{T}FRho": f"{Cf[0, 1] / np.sqrt(Cf[0, 0] * Cf[1, 1]):.2f}"})
for name in ("full", "noisy", "mask", "Q", "f"):
    r = runs[name]
    T = name.capitalize() if name not in ("Q", "f") else {"Q": "Quad", "f": "Fid"}[name]
    nums[f"NineCToy{T}PullN"] = f"{(r['m'][1] - TRUTH[1]) / r['s'][1]:+.1f}"
    nums[f"NineCToy{T}PullA"] = f"{(r['m'][0] - TRUTH[0]) / r['s'][0]:+.1f}"
nums.update({"NineCToyTruthA": f"{TRUTH[0]:.4f}", "NineCToyTruthN": f"{TRUTH[1]:.4f}",
             "NineCToyNch": NCH, "NineCToyNst": NST, "NineCToyBurn": BURN,
             "NineCToyShiftQ": f"{(runs['Q']['m'][1] - runs['full']['m'][1]) / runs['full']['s'][1]:+.2f}",
             "NineCToyShiftF": f"{(runs['f']['m'][1] - runs['full']['m'][1]) / runs['full']['s'][1]:+.2f}"})

# library cross-checks on case (c): emcee, and getdist for the marginal statistics
import emcee
nw = 16
sampler = emcee.EnsembleSampler(nw, 2, lp_mask)
sampler.random_state = np.random.RandomState(rng.integers(2 ** 31)).get_state()
sampler.run_mcmc(TRUTH + 1e-3 * rng.standard_normal((nw, 2)), 5000, progress=False)
em = sampler.get_chain(discard=1000, flat=True)
nums.update({"NineCToyEmA": f"{em[:, 0].mean():.4f}", "NineCToyEmSA": f"{em[:, 0].std():.4f}",
             "NineCToyEmN": f"{em[:, 1].mean():.4f}", "NineCToyEmSN": f"{em[:, 1].std():.4f}"})
try:
    from getdist import MCSamples
    g = MCSamples(samples=runs["mask"]["post"], names=["lnA", "ns"], settings={"smooth_scale_2D": 0.3})
    ms = g.getMargeStats()
    lim = ms.parWithName("ns").limits[0]
    nums.update({"NineCToyGdNlo": f"{lim.lower:.4f}", "NineCToyGdNhi": f"{lim.upper:.4f}"})
    q = np.percentile(runs["mask"]["post"][:, 1], [16, 84])
    nums.update({"NineCToyOurNlo": f"{q[0]:.4f}", "NineCToyOurNhi": f"{q[1]:.4f}"})
except Exception as err:                       # getdist missing: the text then quotes our numbers only
    print("getdist cross-check skipped:", err)

# ---------------------------------------------------------------- ensemble: do L_Q and L_f shift n_s?
NREP = 500
r2 = rng_for("ch09", "44_cmb_mcmc", stream=1)
best = {k: np.empty((NREP, 2)) for k in ("exact", "Q", "f")}
for i in range(NREP):
    ch_i = np.zeros(L + 1)
    ch_i[use] = cl_true[use] * r2.chisquare(nu[use]) / nu[use]    # a full-sky noiseless C_hat, exact law
    fns = {"exact": lambda th: -np.sum(nu[use] * (ch_i[use] / emu.cl(th)[use] + np.log(emu.cl(th)[use]))) / 2,
           "Q": lambda th: lp_Q(th, ch_i), "f": lambda th: lp_f(th, ch_i)}
    for k, f in fns.items():
        best[k][i] = optimize.minimize(lambda th: -f(th), TRUTH, method="Nelder-Mead",
                                       options=dict(xatol=1e-6, fatol=1e-8)).x
sN = np.sqrt(covs["full"][1, 1])
sA = np.sqrt(covs["full"][0, 0])
dQ = (best["Q"][:, 1] - best["exact"][:, 1]) / sN
dF = (best["f"][:, 1] - best["exact"][:, 1]) / sN
nums.update({"NineCToyEnsN": NREP,
             "NineCToyEnsQmean": f"{dQ.mean():+.2f}", "NineCToyEnsQsd": f"{dQ.std():.2f}",
             "NineCToyEnsFmean": f"{dF.mean():+.3f}", "NineCToyEnsFsd": f"{dF.std():.3f}",
             "NineCToyEnsQAmean": f"{((best['Q'][:, 0] - best['exact'][:, 0]) / sA).mean():+.2f}",
             "NineCToyEnsExactBias": f"{((best['exact'][:, 1] - TRUTH[1]) / sN).mean():+.2f}",
             "NineCToyEnsExactSd": f"{((best['exact'][:, 1] - TRUTH[1]) / sN).std():.2f}"})
np.savez(pathlib.Path(__file__).resolve().parents[2] / "data" / "ch09" / "cmb_chains.npz",
         **{f"post_{k}": v["post"] for k, v in runs.items()}, truth=TRUTH)
nums["NineCToyMinutes"] = f"{(time.time() - t_start) / 60:.1f}"

# ---------------------------------------------------------------- figure 1: the data
dl = lambda c: ell * (ell + 1) * c / (2 * np.pi)
fig, ax = plt.subplots(figsize=(7.4, 3.5))
ax.plot(ell[use], dl(chat_sky)[use], ".", ms=1.6, color=SERIES[0], label="(a) the sky, full, no noise")
cdec = (chat_obs - Nw) / T2
ax.plot(ell[use], dl(cdec)[use], ".", ms=1.6, color=SERIES[1], alpha=0.6,
        label="(b) full sky with beam and noise (debiased, deconvolved)")
ax.errorbar(lb, Db, yerr=np.sqrt(np.diag(Csim)), fmt="s", ms=3.5, color=SERIES[2], lw=1,
            label="(c) behind the Galactic band: MASTER bandpowers")
ax.plot(ell[use], dl(cl_true)[use], color="k", lw=1.1, label="fiducial $D_\\ell$")
ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$ ($\mu$K$^2$)")
ax.set_ylim(0, 9000); ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
savefig(fig, "ch09", "cmb_data")

# ---------------------------------------------------------------- figure 2: the triangle plot
sets = [runs["full"]["post"], runs["noisy"]["post"], runs["mask"]["post"]]
gauss = [(TRUTH, covs[k], c, None) for k, c in (("full", SERIES[0]), ("noisy", SERIES[1]), ("mask", SERIES[2]))]
fig, axs = mc.triangle(sets, [r"$\ln(10^{10}A_s)$", r"$n_s$"], [SERIES[0], SERIES[1], SERIES[2]],
                       labels=["(a) full sky, no noise", "(b) full sky, noisy", "(c) masked, noisy"],
                       truths=TRUTH, size=2.6, filled=[True, False, False])
for k, (m_, C_, c_, _) in enumerate(gauss):
    sub = C_
    mc.gauss_ellipse(axs[1, 0], m_, sub, color=c_, ls=":", lw=0.9)
axs[0, 0].plot([], [], color="0.3", ls=":", lw=0.9, label="Fisher (centred on truth)")
axs[0, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.05, 1.0))
savefig(fig, "ch09", "cmb_triangle")

# ---------------------------------------------------------------- figure 3: diagnostics, case (c)
ch = runs["mask"]["ch"]
fig, axs = plt.subplots(2, 2, figsize=(9.0, 5.2))
for c in range(NCH):
    axs[0, 0].plot(ch[c, :, 1], lw=0.4, color=SERIES[c])
axs[0, 0].axvline(BURN, color="0.4", ls="--", lw=0.8)
axs[0, 0].set_xlabel("step"); axs[0, 0].set_ylabel("$n_s$"); axs[0, 0].set_title("(a) four chains", fontsize=9)
axs[0, 0].set_xlim(0, 6000)
lag = np.arange(201)
for c in range(NCH):
    axs[0, 1].plot(lag, mc.acf(ch[c, BURN:, 1], 200), lw=0.9, color=SERIES[c])
axs[0, 1].axhline(0, color="0.5", lw=0.6)
axs[0, 1].set_xlabel("lag"); axs[0, 1].set_ylabel(r"$\rho(k)$ of $n_s$"); axs[0, 1].set_title("(b) autocorrelation", fontsize=9)
ns_ = np.unique(np.geomspace(100, NST, 40).astype(int))
rh = [mc.rhat(ch[:, :n])[1] for n in ns_]
axs[1, 0].semilogx(ns_, rh, "o-", ms=3, color=SERIES[0])
axs[1, 0].axhline(1.01, color="0.4", ls=":", lw=0.8)
axs[1, 0].set_xlabel("steps used (second half kept)"); axs[1, 0].set_ylabel(r"$\hat R$ for $n_s$")
axs[1, 0].set_title("(c) Gelman--Rubin shrink factor", fontsize=9)
for c in range(NCH):
    mc.hist1d(axs[1, 1], ch[c, BURN:, 1], SERIES[c], bins=40)
axs[1, 1].set_xlabel("$n_s$"); axs[1, 1].set_title("(d) marginal of each chain", fontsize=9)
fig.tight_layout()
savefig(fig, "ch09", "cmb_diag")

# ---------------------------------------------------------------- figure 4: the Gaussian approximations
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.2))
mc.triangle  # (kept for symmetry with the text)
ax = axs[0]
mc.contour2d(ax, runs["full"]["post"][:, 0], runs["full"]["post"][:, 1], SERIES[0], label="exact")
mc.contour2d(ax, runs["Q"]["post"][:, 0], runs["Q"]["post"][:, 1], SERIES[1], filled=False, label=r"$\mathcal{L}_Q$")
mc.contour2d(ax, runs["f"]["post"][:, 0], runs["f"]["post"][:, 1], SERIES[2], filled=False, ls="--",
             label=r"$\mathcal{L}_f$")
ax.plot(*TRUTH, "x", color=SERIES[7], ms=7, mew=1.5)
ax.set_xlabel(r"$\ln(10^{10}A_s)$"); ax.set_ylabel(r"$n_s$"); ax.legend(fontsize=7)
ax.set_title("(a) one sky, three likelihoods", fontsize=9)
ax = axs[1]
ax.hist(dQ, bins=40, color=SERIES[1], alpha=0.7, label=r"$\mathcal{L}_Q$ $-$ exact")
ax.hist(dF, bins=40, color=SERIES[2], alpha=0.7, label=r"$\mathcal{L}_f$ $-$ exact")
ax.set_xlabel(r"shift of the best-fit $n_s$ (in units of $\sigma_{n_s}$)"); ax.legend(fontsize=7)
ax.set_title(f"(b) {NREP} skies", fontsize=9)
fig.tight_layout()
savefig(fig, "ch09", "cmb_gauss_bias")
save_numbers("ch09", "44_cmb_mcmc", nums)
print(nums)
