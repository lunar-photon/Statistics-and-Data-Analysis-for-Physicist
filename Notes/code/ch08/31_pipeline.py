"""31_pipeline.py -- a pseudo-C_l analysis run the way a real one is run, at laptop scale.

Question: put every piece of the cut-sky analysis together on one "observed" sky.  The data are
two half-mission maps (same sky, independent noise); the mask is a Planck-like Galactic cut
plus apodised point-source holes.  What is the spectrum, what are its error bars and
correlations, does the data pass a null test, and what amplitude A of the fiducial spectrum
does it prefer?
Steps (each the laptop version of a step of a real pipeline):
  1. mask: apodised Galactic band (|b| < 20 deg, 5 deg taper) times 400 holes of 1 deg radius
     with a 1 deg taper; its moments, its spectrum and M_ll' from 3j symbols;
  2. data: one sky with a seed kept apart from the simulations, observed by the toy instrument
     of chapter 8 (nside 256, 30 arcmin beam, 200 muK arcmin), split into two halves with noise
     sqrt(2) larger each;
  3. estimator: the MASTER cross-spectrum of the two halves (no noise bias to subtract), in
     bins of 20, reported for 22 <= l < 602;
  4. covariance: N_SIM simulations of the whole chain, Hartlap-corrected inverse;
  5. null test: the half-difference map (d1 - d2)/2 contains no sky; its spectrum minus the mean
     of the simulated null spectra, chi^2 against the simulated scatter, probability to exceed;
     repeated after adding a small striping systematic to the second half (a random offset
     of 3 muK rms on every ring of constant latitude), which the test must catch;
  6. likelihood: the amplitude A in D_b = A D_b^fid, Gaussian likelihood, its maximum and
     width; the same estimator run on every simulation to check bias and coverage.
Writes: data/ch08/pipeline.npz, figures/ch08/pipeline.pdf, results/ch08/31_pipeline.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES
from camb_fiducial import load_fiducial
import lib_cmbsim as cs
import lib_masks as lm

setup()
L, N_SIM = 767, 400
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ch08"
exp = cs.Experiment()
NSIDE = exp.nside
_, cl = load_fiducial()
cl = cl[: L + 1]
T2 = exp.transfer(L) ** 2
ell = np.arange(L + 1)
nums = {"EightBpipeNsim": N_SIM}

# ---------------------------------------------------------------- 1. the mask
rng_mask = rng_for("ch08", "31_pipeline_mask")
W = lm.band(NSIDE, 20.0, apo_deg=5.0) * lm.holes(NSIDE, lm.hole_centres(400, rng_mask), 1.0, apo_deg=1.0)
fsky, wi = lm.mask_moments(W)
path = DATA / "pipeline.npz"
if path.exists():
    zc = np.load(path)
    M = zc["M"]
else:
    M = lm.coupling_matrix(lm.window_spectrum(W, 2 * L + 1), L, L)
edges = lm.bin_edges(2, L, 20)
P, Q = lm.binning_operators(edges, L)
lb = 0.5 * (edges[:-1] + edges[1:] - 1)
rep = (edges[:-1] >= 22) & (edges[1:] <= 610)              # report 22 <= l < 602
Kinv = np.linalg.inv(lm.master_matrix(M, T2, P, Q))
Dfid = P @ cl
nums.update({"EightBpipeFsky": f"{fsky:.3f}", "EightBpipeWtwo": f"{wi[2]:.3f}",
             "EightBpipeNbins": int(rep.sum()), "EightBpipeLlo": int(edges[:-1][rep][0]),
             "EightBpipeLhi": int(edges[1:][rep][-1] - 1)})

# toy systematic: an unremoved offset on every scan ring of the second half (residual striping),
# drawn once, SIG_SYS muK rms
SIG_SYS = 3.0
theta, _ = hp.pix2ang(NSIDE, np.arange(exp.npix))
_, ring = np.unique(theta, return_inverse=True)
STRIPE = SIG_SYS * rng_for("ch08", "31_pipeline_stripes").standard_normal(ring.max() + 1)[ring]
nums["EightBpipeSigSys"] = f"{SIG_SYS:.0f}"


def observe(rng, systematic=False):
    """Two half-mission maps of one sky: same signal, independent noise of variance 2 sigma^2."""
    a = cs.synalm(cl, L, rng)
    s = hp.alm2map(hp.almxfl(a, exp.transfer(L)), NSIDE, lmax=L)
    d1 = s + cs.noise_map(exp, rng, scale=np.sqrt(2.0))
    d2 = s + cs.noise_map(exp, rng, scale=np.sqrt(2.0))
    if systematic:
        d2 = d2 + STRIPE
    return d1, d2


def bandpowers(d1, d2):
    """MASTER cross-spectrum of the halves, and the MASTER spectrum of the half-difference."""
    a1 = hp.map2alm(W * d1, lmax=L, iter=0)
    a2 = hp.map2alm(W * d2, lmax=L, iter=0)
    cross = hp.alm2cl(a1, a2)
    null = hp.alm2cl(0.5 * (a1 - a2))
    return Kinv @ (P @ cross), Kinv @ (P @ null)


# ---------------------------------------------------------------- 2-4. simulations and the data
t0 = time.time()
if path.exists():
    sim_x, sim_n = zc["sim_x"], zc["sim_n"]
else:
    rng = rng_for("ch08", "31_pipeline_sims")
    sim_x = np.empty((N_SIM, len(lb)))
    sim_n = np.empty((N_SIM, len(lb)))
    for i in range(N_SIM):
        sim_x[i], sim_n[i] = bandpowers(*observe(rng))
    np.savez(path, M=M, sim_x=sim_x, sim_n=sim_n)
nums["EightBpipeSeconds"] = f"{time.time() - t0:.0f}"
rng_data = rng_for("ch08", "31_pipeline_DATA")              # the "observed" sky: its own seed
d1, d2 = observe(rng_data)
data_x, data_n = bandpowers(d1, d2)
_, data_n_sys = bandpowers(d1, d2 + STRIPE)

X = sim_x[:, rep]
C = np.cov(X, rowvar=False)
p = X.shape[1]
hart = (N_SIM - p - 2) / (N_SIM - 1)
Psi = hart * np.linalg.inv(C)                               # Hartlap-corrected precision matrix
sig = np.sqrt(np.diag(C))
nums["EightBpipeHartlap"] = f"{hart:.3f}"
nums["EightBpipeP"] = p

# ---------------------------------------------------------------- 5. null test
Nn = sim_n[:, rep]
Cn = np.cov(Nn, rowvar=False)
Psin = (N_SIM - p - 2) / (N_SIM - 1) * np.linalg.inv(Cn)
mn = Nn.mean(0)


def chi2_null(v):
    r = v[rep] - mn
    return float(r @ Psin @ r)


chi_sims = np.array([chi2_null(v) for v in sim_n])          # the null distribution, from the sims
chi_data = chi2_null(data_n)
chi_sys = chi2_null(data_n_sys)
pte = np.mean(chi_sims >= chi_data)
pte_sys = np.mean(chi_sims >= chi_sys)
nums.update({"EightBpipeChiData": f"{chi_data:.1f}", "EightBpipePTE": f"{pte:.2f}",
             "EightBpipeChiSys": f"{chi_sys:.0f}", "EightBpipePTESys": f"{max(pte_sys, 1 / N_SIM):.3f}",
             "EightBpipeChiSysLess": "<" if pte_sys == 0 else "=",
             # the simulations enter their own covariance, which pulls their chi^2 low:
             "EightBpipeChiSimMean": f"{chi_sims.mean():.1f}", "EightBpipeChiSimStd": f"{chi_sims.std(ddof=1):.1f}"})
print(f"null test: chi2 = {chi_data:.1f} for p = {p}, PTE = {pte:.2f}; with stripes chi2 = {chi_sys:.0f}, PTE = {pte_sys:.3f}")

# what this one data set with stripes shows bin by bin
z_sys = (data_n_sys[rep] - mn) / np.sqrt(np.diag(Cn))
n_high = int(np.sum(z_sys > 0))
p_high = stats.binom.sf(n_high - 1, p, 0.5)                  # P(at least n_high of p bins high) under noise
nums.update({"EightBpipeSysMaxPull": f"{z_sys.max():.1f}", "EightBpipeSysNhigh": n_high,
             "EightBpipeSysPhigh": f"{max(p_high, 1e-3):.3f}",
             "EightBpipeSysPhighRel": "<" if p_high < 1e-3 else "=",
             "EightBpipeSysNsig": f"{(chi_sys - p) / np.sqrt(2 * p):.1f}"})
# the same test on fresh data sets, each with its own sky, noise and stripes: is the failure typical?
N_REP = 20
chi_rep = []
for k in range(N_REP):
    rk = rng_for("ch08", "31_pipeline_DATA_rep", stream=k)
    stripe_k = SIG_SYS * rng_for("ch08", "31_pipeline_stripes_rep", stream=k).standard_normal(ring.max() + 1)[ring]
    e1, e2 = observe(rk)
    chi_rep.append(chi2_null(bandpowers(e1, e2 + stripe_k)[1]))
chi_rep = np.array(chi_rep)
nums.update({"EightBpipeNrep": N_REP, "EightBpipeChiSysMed": f"{np.median(chi_rep):.0f}",
             "EightBpipeChiSysMin": f"{chi_rep.min():.0f}",
             "EightBpipeNsigSysMin": f"{(chi_rep.min() - p) / np.sqrt(2 * p):.1f}"})
print(f"stripes on {N_REP} fresh data sets: chi2 median {np.median(chi_rep):.0f}, min {chi_rep.min():.0f}")

# ---------------------------------------------------------------- 6. amplitude
t = Dfid[rep]


def amplitude(v):
    """Maximum of the Gaussian likelihood in A for D_b = A t, and its width (Fisher)."""
    F = t @ Psi @ t
    return float(t @ Psi @ v[rep] / F), float(1 / np.sqrt(F))


A_data, sA = amplitude(data_x)
A_sims = np.array([amplitude(v)[0] for v in sim_x])
cover = np.mean(np.abs(A_sims - 1) <= sA)
chi_fit = float((data_x[rep] - A_data * t) @ Psi @ (data_x[rep] - A_data * t))
nums.update({"EightBpipeA": f"{A_data:.3f}", "EightBpipesA": f"{sA:.3f}",
             "EightBpipeAsimMean": f"{A_sims.mean():.4f}", "EightBpipeAsimStd": f"{A_sims.std(ddof=1):.4f}",
             "EightBpipeCover": f"{100 * cover:.0f}", "EightBpipeChiFit": f"{chi_fit:.1f}",
             "EightBpipePTEFit": f"{stats.chi2.sf(chi_fit, p - 1):.2f}"})
print(f"A = {A_data:.4f} +- {sA:.4f}; sims: mean {A_sims.mean():.4f} std {A_sims.std(ddof=1):.4f};"
      f" 1-sigma coverage {cover:.3f}; chi2 of the fit {chi_fit:.1f} for {p-1} dof")

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axs[0]
ax.plot(ell[2:], (ell * (ell + 1) * cl / (2 * np.pi))[2:], color="k", lw=0.8, label="fiducial $D_\\ell$")
ax.errorbar(lb[rep], data_x[rep], yerr=sig, fmt="o", ms=2.5, color=SERIES[0], lw=0.9,
            label="half-mission cross, MASTER")
ax.set_xlim(0, 620)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
ax.set_title(f"(a) the 'data', $\\hat A={A_data:.3f}\\pm{sA:.3f}$", fontsize=9)
ax.legend(fontsize=7.5)
ax = axs[1]
sn = np.sqrt(np.diag(Cn))                                   # simulated scatter of each null bin
ax.fill_between(lb[rep], -2, 2, color="0.88", lw=0, label=r"$\pm2\sigma$ of the simulations")
ax.plot(lb[rep], (data_n[rep] - mn) / sn, "o-", ms=2.5, lw=0.8, color=SERIES[2],
        label=f"null, $\\chi^2={chi_data:.0f}$")
ax.plot(lb[rep], (data_n_sys[rep] - mn) / sn, "s-", ms=2.5, lw=0.8, color=SERIES[3],
        label=f"null with stripes, $\\chi^2={chi_sys:.0f}$")
ax.axhline(0, color="0.4", lw=0.6)
ax.set_xlim(0, 620)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"null bandpower / its scatter")
ax.set_title(f"(b) null test, $p={p}$ bins", fontsize=9)
ax.set_ylim(-3.2, 5.6)
ax.legend(fontsize=6.5, loc="upper center", ncol=2, frameon=False)
fig.tight_layout()
savefig(fig, "ch08", "pipeline")
save_numbers("ch08", "31_pipeline", nums)
