"""43_pol_sims.py -- simulated polarized skies, observed and analysed like the SMICA half-missions.

Question: if the sky were a Gaussian realisation of the Planck best-fit spectra (TT, EE, TE and the
lensing BB; no EB, no TB), seen through SMICA's beam, pixels, noise and our windows, how would
the cross-spectra of 42_pol_spectra.py scatter?  And what happens to them if the polarization
plane of the sky is rotated by a known angle?
One simulated sky:
  * T, E, B a_lm from the fiducial spectra (correlated T and E), times the transfer functions;
    the E part and the B part of the polarization are synthesised separately (spin-2 transforms
    are linear, so their pseudo-a_lm simply add) to separate E-to-B leakage from lensing B;
  * two half-mission noise maps: T from the 13a noise model, Q/U from pol_noise_model.npz;
  * pseudo-a_lm of signal and noise separately; a rotation of the sky by beta is applied to the
    signal pseudo-a_lm only (a uniform rotation commutes with the window and the beam, so this is
    exactly the rotated sky observed through them);
  * stored (binned): cross-spectra for beta in BETAS; at beta = 0 also through the stricter window; the same with the lensing B removed;
    the half-difference spectra; noise-free E-only and B-only pseudo-spectra (leakage, lensing).
Resumable: chunks of CHUNK skies in data/ch13/polsims_<k>.npz; each call stops after BUDGET s.
Writes: data/ch13/polsims_*.npz, results/ch13/43_pol_sims.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch08"))
import numpy as np
import healpy as hp
from common import save_numbers, rng_for
from camb_fiducial import load_fiducial
import lib_planck as lp
import lib_biref as lb
import lib_cmbsim as cs

N, L = lb.NSIDE, lb.LS
N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else 600
CHUNK = 25
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1500.0
BETAS = np.array([0.0, 0.3, 1.0]) * lb.DEG
N_BINARY = 100                      # skies that also get the binary-window leakage spectra
t0 = time.time()

pol = lb.load_pol()
win = np.load(lb.DATA / f"pol_windows_n{N}.npz")
WP, WB, WT, WC = (win[k].astype(np.float64) for k in ("main", "binary", "T", "cut"))
TP = (pol["beam_P"] * pol["pixwin_P"])[: L + 1]
tm = np.load(lp.PLANCK / f"smica_transfer_n{N}.npz")
TT_ = (tm["beam"] * tm["pixwin"])[: L + 1]
cls = {k: load_fiducial(k)[1][: L + 1] for k in ("TT", "EE", "BB", "TE")}
nmT = np.load(lb.DATA / "noise_model.npz")              # 13a temperature noise model
sqvT, ngT = np.sqrt(nmT["v"].astype(float)), nmT["ng"]
nmP = np.load(lb.DATA / "pol_noise_model.npz")
sqvP, ngE, ngB = np.sqrt(nmP["v"].astype(float)), nmP["ngEE"], nmP["ngBB"]
zero = np.zeros(hp.Alm.getsize(L), complex)


def spin_synth(aE, aB):
    return hp.alm2map_spin([aE, aB], N, 2, L)


def spin_anal(W, q, u):
    return hp.map2alm_spin([W * q, W * u], 2, lmax=L)


def one_sky(rng, with_binary):
    aT, aE, aB = lb.correlated_alm(cls, L, rng)
    tsky = hp.alm2map(hp.almxfl(aT, TT_), N, lmax=L)
    qE, uE = spin_synth(hp.almxfl(aE, TP), zero)
    qB, uB = spin_synth(zero, hp.almxfl(aB, TP))
    sT = hp.map2alm(WT * tsky, lmax=L, iter=0)
    sE = spin_anal(WP, qE, uE)                  # pseudo (E, B) of the E part: B here is leakage
    sB = spin_anal(WP, qB, uB)                  # pseudo (E, B) of the lensing B part
    cE, cB = spin_anal(WC, qE + qB, uE + uB)      # the stricter window, beta = 0
    cEl = spin_anal(WC, qE, uE)
    noise, cnoise = [], []
    for _ in range(2):
        nt = sqvT * np.sqrt(2.0) * hp.alm2map(cs.synalm(ngT, L, rng), N, lmax=L)
        nq, nu = spin_synth(cs.synalm(ngE, L, rng), cs.synalm(ngB, L, rng))
        nq, nu = np.sqrt(2.0) * sqvP * nq, np.sqrt(2.0) * sqvP * nu
        nE, nB = spin_anal(WP, nq, nu)
        noise.append((hp.map2alm(WT * nt, lmax=L, iter=0), nE, nB))
        cnoise.append(spin_anal(WC, nq, nu))
    out = {}
    sigE, sigB = sE[0] + sB[0], sE[1] + sB[1]
    for i, beta in enumerate(BETAS):
        rE, rB = lb.rotate(sigE, sigB, beta)
        h = [(sT + n[0], rE + n[1], rB + n[2]) for n in noise]
        s = lb.cross_spectra(*h)
        for k, v in s.items():
            out[f"b{i}_{k}"] = v
    h = [(sT + n[0], sE[0] + n[1], sE[1] + n[2]) for n in noise]     # no lensing B in the sky
    for k, v in lb.cross_spectra(*h).items():
        out[f"nolens_{k}"] = v
    h = [(sT + n[0], cE + m[0], cB + m[1]) for n, m in zip(noise, cnoise)]
    for k, v in lb.cross_spectra(*h).items():
        out[f"cut_{k}"] = v
    out["cutE_BB"] = lb.binned(hp.alm2cl(cEl[1]))                   # leakage, stricter window
    out["cutB_BB"] = lb.binned(hp.alm2cl(cB - cEl[1]))              # lensing B, stricter window
    out["null_EE"] = lb.binned(hp.alm2cl(0.5 * (noise[0][1] - noise[1][1])))
    out["null_BB"] = lb.binned(hp.alm2cl(0.5 * (noise[0][2] - noise[1][2])))
    out["sigE_EE"] = lb.binned(hp.alm2cl(sE[0]))
    out["sigE_BB"] = lb.binned(hp.alm2cl(sE[1]))                    # E-to-B leakage
    out["sigB_BB"] = lb.binned(hp.alm2cl(sB[1]))                    # lensing B through the window
    out["noise_BB"] = lb.binned(hp.alm2cl(noise[0][2]))             # noise of one half
    if with_binary:
        e, b = spin_anal(WB, qE, uE)
        out["binE_EE"], out["binE_BB"] = lb.binned(hp.alm2cl(e)), lb.binned(hp.alm2cl(b))
    else:
        out["binE_EE"] = out["binE_BB"] = np.full(lb.NB, np.nan)
    return out


for c in range(N_SIM // CHUNK):
    path = lb.DATA / f"polsims_{c:02d}.npz"
    if path.exists():
        continue
    if time.time() - t0 > BUDGET:
        print("time budget used; run again to continue")
        break
    rng = rng_for("ch13", "43_pol_sims", stream=c)
    tc = time.time()
    rows = [one_sky(rng, c * CHUNK + j < N_BINARY) for j in range(CHUNK)]
    np.save(lb.DATA / "polsims_seconds_per_sky.npy", (time.time() - tc) / CHUNK)
    np.savez(path, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"chunk {c}: {CHUNK} skies, {time.time() - t0:.0f} s", flush=True)

done = len(list(lb.DATA.glob("polsims_*.npz")))
sp = float(np.load(lb.DATA / "polsims_seconds_per_sky.npy"))
save_numbers("ch13", "43_pol_sims", {"ThirteenENsimDone": done * CHUNK, "ThirteenESecPerSky": f"{sp:.1f}",
                                      "ThirteenENsimTarget": N_SIM, "ThirteenENbinary": N_BINARY})
print(f"{done * CHUNK} of {N_SIM} skies done")
