"""04_sims.py -- simulated Planck skies, observed and analysed exactly like the real one.

Question: how much would the bandpowers of 03_bandpowers.py scatter if the sky were a Gaussian
realisation of the Planck best-fit spectrum, seen through the same beam, pixels, noise and
windows?  The answer, sky by sky, is what the covariance, the null tests and the checks use.
The noise model (approximate, built from the data):
  d = (hm1 - hm2)/2 contains no sky; its local variance v(p) (d^2 smoothed over 2 deg FWHM,
  mean 1 on the kept sky) describes how the noise varies over the sky; the spectrum N^g_l of
  d / sqrt(v), divided by <W^2> and averaged over 31 multipoles, describes how it varies with
  scale.  A half-mission noise map is sqrt(2) sqrt(v) g with g Gaussian, spectrum N^g_l.
Each simulated sky: a_lm from the fiducial C_l, times B_l p_l, a map at nside 512; two
half-mission maps s + n1, s + n2; for each of the five windows the MASTER cross-spectrum and
the MASTER spectrum of the half-difference, in the bins of 03_bandpowers.py.
Resumable: chunks of CHUNK skies go to data/ch13/sims_<k>.npz; run until N_SIM are done
(each call stops after ~8 minutes).
Writes: data/ch13/noise_model.npz, data/ch13/sims_*.npz, figures/ch13/noise.pdf,
        results/ch13/04_sims.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch08"))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES
from camb_fiducial import load_fiducial
import lib_planck as lp
import lib_cmbsim as cs

setup()
NSIDE, LC, LS = 512, 1199, 3 * 512 - 1
N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else 400
CHUNK, BUDGET = 50, 900.0
WINDOWS = ("main", "binary", "cut30", "north", "south")
t0 = time.time()
z = np.load(lp.DATA / "bandpowers.npz")
EDGES = z["edges"]
maps = lp.load_maps(NSIDE)
masks = lp.load_masks(NSIDE)
T = (maps["beam"] * maps["pixwin"])[: LS + 1]
_, clf = load_fiducial()
clf = clf[: LS + 1]

# ---------------------------------------------------------------- the noise model
path = lp.DATA / "noise_model.npz"
if not path.exists():
    keep = masks["binary"] > 0
    W = masks["main"].astype(float)
    d = 0.5 * (maps["hm1"] - maps["hm2"])
    v = hp.smoothing(d ** 2, fwhm=2.0 * lp.DEG, iter=0)
    v = np.where(keep, v / v[keep].mean(), 1.0)
    v = np.clip(v, 0.05, None)
    g = d / np.sqrt(v)
    pcl = hp.anafast(W * g, lmax=LS, iter=0) / np.mean(W ** 2)
    ng = np.convolve(pcl, np.ones(31) / 31, mode="same")
    ng[:31] = pcl[:31]                                      # no smoothing where the window is cut
    ng[-15:] = ng[-16]
    ng[:2] = 0.0
    nd = hp.anafast(W * d, lmax=LS, iter=0) / np.mean(W ** 2)  # the half-difference itself
    np.savez(path, v=v.astype(np.float32), ng=ng, nd=nd)
nm = np.load(path)
v, ng = nm["v"].astype(float), nm["ng"]
sqv = np.sqrt(v)

# figure: the noise level against the sky, and where the noise is larger
ell = np.arange(LS + 1)
fac = ell * (ell + 1) / (2 * np.pi)
fig = plt.figure(figsize=(8.0, 3.1))
ax = fig.add_axes([0.07, 0.17, 0.4, 0.75])
ax.semilogy(ell[2:LC], (fac * clf)[2:LC], color="k", lw=0.8, label=r"sky $D_\ell$ (fiducial)")
ax.semilogy(ell[2:LC], (fac * clf * T ** 2)[2:LC], color="k", lw=0.8, ls="--", label=r"sky $\times\,T_\ell^2$")
ax.semilogy(ell[2:LC], (fac * nm["nd"])[2:LC], color=SERIES[1], lw=0.7, label="noise of the full map")
ax.semilogy(ell[2:LC], (fac * 2 * nm["nd"])[2:LC], color=SERIES[3], lw=0.7, label="noise of one half")
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
ax.set_xlim(0, LC)
ax.set_ylim(1e-1, 1e4)
ax.legend(fontsize=7)
hp.mollview(np.where(masks["binary"] > 0, v, hp.UNSEEN), fig=fig.number, sub=(1, 2, 2), min=0, max=3,
            title=r"(b) relative noise variance $v(p)$", cmap="viridis", notext=True, cbar=True)
savefig(fig, "ch13", "noise")
i_eq = int(np.argmax((fac * 2 * nm["nd"])[30:LC] > (fac * clf * T ** 2)[30:LC])) + 30
nums = {"TwelveANoiseLeq": i_eq if i_eq > 30 else ">1199",
        "TwelveANoiseRatioK": f"{100 * nm['nd'][1000] / (clf[1000] * T[1000] ** 2):.0f}",
        "TwelveAVmax": f"{v[masks['binary'] > 0].max():.1f}", "TwelveAVmin": f"{v[masks['binary'] > 0].min():.2f}",
        "TwelveANsim": N_SIM}

# ---------------------------------------------------------------- the simulations
Ws = {k: masks[k].astype(np.float64) for k in WINDOWS}
T2 = T[: LC + 1] ** 2
msts = {k: lp.Master(lp.coupling(Ws[k], LC, name=k), T2, EDGES) for k in WINDOWS}


def one_sky(rng):
    a = cs.synalm(clf, LS, rng)                             # the sky
    s = hp.alm2map(hp.almxfl(a, T), NSIDE, lmax=LS)          # as seen through beam and pixels
    halves = []
    for _ in range(2):
        g = hp.alm2map(cs.synalm(ng, LS, rng), NSIDE, lmax=LS)
        halves.append(s + np.sqrt(2.0) * sqv * g)
    res = {}
    for k in WINDOWS:
        a1, a2 = lp.pseudo_cl(Ws[k], halves, LC)
        res[f"{k}_cross"] = msts[k](hp.alm2cl(a1, a2))
        res[f"{k}_null"] = msts[k](hp.alm2cl(0.5 * (a1 - a2)))
    return res


for c in range(N_SIM // CHUNK):
    out = lp.DATA / f"sims_{c:02d}.npz"
    if out.exists():
        continue
    if time.time() - t0 > BUDGET:
        print("time budget used; run again to continue")
        break
    rng = rng_for("ch13", "04_sims", stream=c)
    tc = time.time()
    rows = [one_sky(rng) for _ in range(CHUNK)]
    np.save(lp.DATA / "sims_seconds_per_sky.npy", (time.time() - tc) / CHUNK)
    np.savez(out, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"chunk {c}: {CHUNK} skies, {time.time() - t0:.0f} s", flush=True)

done = len(list(lp.DATA.glob("sims_*.npz")))
nums["TwelveANsimDone"] = done * CHUNK
sp = lp.DATA / "sims_seconds_per_sky.npy"
nums["TwelveASecPerSky"] = f"{float(np.load(sp)):.1f}" if sp.exists() else "?"
save_numbers("ch13", "04_sims", nums)
print(f"{done * CHUNK} of {N_SIM} skies done")
