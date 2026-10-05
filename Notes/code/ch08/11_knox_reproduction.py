"""11_knox_reproduction.py -- Knox (1995): the error bar of a full-sky, beamed, noisy map, redone with maps.

Question: do Knox's numbers and his error formula
    (Delta C_l)^2 = 2/(2l+1) (C_l + w^{-1} e^{l^2 sigma_b^2})^2          (Knox 1995, eq. 4)
survive when we actually make maps -- with pixels, a pixel window and healpy transforms -- rather
than drawing C_l^est from a chi^2 as Knox did ("we never create a map")?
Computes:
  * the weight per solid angle w of his satellite (sigma_pix = 22 muK in 20'x20' pixels), the rms
    of the sky through a 20' beam and the signal-to-noise per pixel, for our Planck 2018 fiducial
    spectrum (Knox used a standard-CDM model with h = 0.5, so the rms differs);
  * his Fig. 2: Delta C_l / C_l for FWHM = 20' and 40' at w^{-1/2} = 15 and 30 muK deg (analytic);
  * a map-level Monte Carlo of the two w^{-1/2} = 15 muK deg experiments at nside 512, l <= 1000
    (N_SIM skies each), whose scatter is compared with his formula;
  * the same scatter from Knox's own shortcut (direct chi^2 draws), which takes milliseconds.
Laptop scale: nside 512 and N_SIM = 300 per experiment, about 4 minutes in total; cached in
data/ch08/mc_knox.npz.
Writes: figures/ch08/knox_fig2.pdf, results/ch08/11_knox_reproduction.tex, data/ch08/mc_knox.npz
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from camb_fiducial import load_fiducial
import lib_cmbsim as cs

setup()
ell_all, cl_all = load_fiducial()
DEG = np.pi / 180.0
N_SIM = 300
LMAX = 1000                       # Knox's Fig. 2 runs to l ~ 1000
NSIDE = 512                       # 6.9' pixels: "a few times smaller than the beam" (Knox, footnote 2)

# ------------------------------------------------------------ Knox's numbers for his satellite
sig_pix, pix_arcmin = 22.0, 20.0
w_inv = sig_pix ** 2 * (pix_arcmin / 60.0) ** 2          # muK^2 deg^2:  w^{-1} = sigma^2 Omega_pix
w_inv_sqrt = np.sqrt(w_inv)                              # Knox quotes 7.5 muK deg
b20 = cs.gaussian_beam(20.0, 3000)
ell = np.arange(3001)
rms20 = np.sqrt(np.sum(((2 * ell + 1) / (4 * np.pi) * cl_all[:3001] * b20 ** 2)[2:]))   # Knox eq. 3
snr_pix = rms20 / sig_pix


# ------------------------------------------------------------ Knox eq. (4)-(5)
def knox_frac(fwhm, w_inv_sqrt_uKdeg, lmax=LMAX):
    """Delta C_l / C_l from Knox eq. (5), with his e^{l^2 sigma_b^2} (no pixel window)."""
    l = np.arange(lmax + 1)
    sb = fwhm * cs.ARCMIN / np.sqrt(8 * np.log(2))
    winv = (w_inv_sqrt_uKdeg * DEG) ** 2                  # muK^2 sr
    out = np.sqrt(2 / (2 * l + 1)) * (1 + winv * np.exp(l ** 2 * sb ** 2) / cl_all[: lmax + 1].clip(1e-30))
    out[:2] = np.nan
    return out


EXPS = [(20.0, 15.0), (20.0, 30.0), (40.0, 15.0), (40.0, 30.0)]
curves = {e: knox_frac(*e) for e in EXPS}


# ------------------------------------------------------------ map-level Monte Carlo
def make():
    out = {}
    t0 = time.time()
    for fwhm in (20.0, 40.0):
        exp = cs.Experiment(nside=NSIDE, fwhm_arcmin=fwhm, depth_uK_arcmin=15.0 * 60.0, lmax=LMAX)
        rng = rng_for("ch08", "11_knox_reproduction", stream=int(fwhm))
        ch = np.empty((N_SIM, LMAX + 1), dtype=np.float32)
        for i in range(N_SIM):
            a = cs.synalm(cl_all[: exp.lmax_sim + 1], exp.lmax_sim, rng)
            d = cs.observe_signal(a, exp) + cs.noise_map(exp, rng)
            ch[i] = cs.debias(cs.cross_cl(cs.map2alm(d, LMAX, iter=1)), exp)
        out[f"chat{int(fwhm)}"] = ch
    out["seconds"] = np.array(time.time() - t0)
    return out


sims = cs.cached("mc_knox", make)
secs = float(sims["seconds"])

l = np.arange(LMAX + 1)
cl = cl_all[: LMAX + 1]
edges = np.r_[np.arange(10, LMAX - 9, 30), LMAX + 1]
lc = 0.5 * (edges[:-1] + edges[1:] - 1)
res = {}
for fwhm in (20.0, 40.0):
    exp = cs.Experiment(nside=NSIDE, fwhm_arcmin=fwhm, depth_uK_arcmin=900.0, lmax=LMAX)
    ch = sims[f"chat{int(fwhm)}"].astype(float)
    sd_mc = ch.std(0, ddof=1)
    sd_ours = np.sqrt(cs.var_fullsky(cl_all, exp))          # with the pixel window divided out too
    sd_knox = knox_frac(fwhm, 15.0) * cl                    # Knox's formula as printed
    good = (l >= 2) & (sd_knox / cl < 5)                    # where the estimate still means something
    res[fwhm] = dict(sd_mc=sd_mc, sd_ours=sd_ours, sd_knox=sd_knox, good=good,
                     r_ours=float(np.mean((sd_mc / sd_ours)[l >= 2])),
                     r_knox=float(np.median((sd_mc / sd_knox)[good])),
                     r_knox_hi=float((sd_mc / sd_knox)[good][-50:].mean()),
                     lgood=int(l[good][-1]))
    res[fwhm]["mean_bias"] = float(np.mean(ch[:, 2:200].mean(0) / cl[2:200] - 1))

# every l at which the 20' beam beats the 40' beam at fixed w (Knox's claim)
beats = np.all(curves[(20.0, 15.0)][2:] <= curves[(40.0, 15.0)][2:]) and \
        np.all(curves[(20.0, 30.0)][2:] <= curves[(40.0, 30.0)][2:])

# Knox's shortcut: draw C_est directly from the scaled chi^2 and time it
exp20 = cs.Experiment(nside=NSIDE, fwhm_arcmin=20.0, depth_uK_arcmin=900.0, lmax=LMAX)
rng = rng_for("ch08", "11_knox_reproduction", stream=99)
t0 = time.time()
nu = 2 * l[2:] + 1
v_obs = exp20.transfer(LMAX)[2:] ** 2 * cl[2:] + exp20.nl(LMAX)[2:]
draws = (v_obs * rng.chisquare(nu, size=(N_SIM, nu.size)) / nu - exp20.nl(LMAX)[2:]) / exp20.transfer(LMAX)[2:] ** 2
t_short = time.time() - t0
r_short = float(np.mean(draws.std(0, ddof=1) / np.sqrt(cs.var_fullsky(cl_all, exp20))[2:]))

# ------------------------------------------------------------ figure: Knox Fig. 2, redone
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
sty = {(20.0, 15.0): (SERIES[0], "-"), (20.0, 30.0): (SERIES[0], "-."),
       (40.0, 15.0): (SERIES[1], "-"), (40.0, 30.0): (SERIES[1], "-.")}
for e in EXPS:
    c, ls = sty[e]
    axs[0].loglog(l[2:], curves[e][2:], color=c, ls=ls, lw=1.3,
                  label=rf"FWHM ${e[0]:.0f}'$, $w^{{-1/2}}={e[1]:.0f}\,\mu$K$\,$deg")
axs[0].loglog(l[2:], np.sqrt(2 / (2 * l[2:] + 1)), color="0.55", lw=0.9, ls=":", label="cosmic variance")
axs[0].set_ylim(1e-2, 10); axs[0].set_xlim(2, LMAX)
axs[0].set_xlabel(r"$\ell$"); axs[0].set_ylabel(r"$\Delta C_\ell/C_\ell$")
axs[0].legend(fontsize=6.5, loc="upper left")
for fwhm, c, mk in [(20.0, SERIES[0], "o"), (40.0, SERIES[1], "s")]:
    r = res[fwhm]
    rb = cs.bin_spectrum(r["sd_mc"] / cl.clip(1e-30), edges)
    keep = cs.bin_spectrum(r["sd_knox"] / cl, edges) < 5
    axs[1].semilogy(lc[keep], rb[keep], mk, ms=3.2, color=c, label=rf"maps, FWHM ${fwhm:.0f}'$")
    g = r["good"]
    theory_line(axs[1], l[g], (r["sd_knox"] / cl)[g], label="Knox eq. (5)" if fwhm == 20.0 else None)
    axs[1].semilogy(l[g], (r["sd_ours"] / cl)[g], color=c, lw=0.9, alpha=0.8,
                    label="with pixel window" if fwhm == 20.0 else None)
axs[1].set_xlabel(r"$\ell$"); axs[1].set_ylabel(r"$\sigma(\widehat C_\ell)/C_\ell$")
axs[1].set_xlim(0, LMAX)
axs[1].legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "ch08", "knox_fig2")

save_numbers("ch08", "11_knox_reproduction", {
    "EightAKnoxWinv": f"{w_inv_sqrt:.2f}",
    "EightAKnoxRms": f"{rms20:.0f}",
    "EightAKnoxSnrPix": f"{snr_pix:.1f}",
    "EightAKnoxNsim": N_SIM,
    "EightAKnoxNside": NSIDE,
    "EightAKnoxLmax": LMAX,
    "EightAKnoxMinutes": f"{secs / 60:.1f}",
    "EightAKnoxROursTwenty": f"{res[20.0]['r_ours']:.3f}",
    "EightAKnoxROursForty": f"{res[40.0]['r_ours']:.3f}",
    "EightAKnoxRKnoxTwenty": f"{res[20.0]['r_knox']:.3f}",
    "EightAKnoxRKnoxForty": f"{res[40.0]['r_knox']:.3f}",
    "EightAKnoxRHiTwenty": f"{res[20.0]['r_knox_hi']:.2f}",
    "EightAKnoxLgoodTwenty": res[20.0]["lgood"],
    "EightAKnoxLgoodForty": res[40.0]["lgood"],
    "EightAKnoxMCerr": f"{100 / np.sqrt(2 * (N_SIM - 1)):.1f}",
    "EightAKnoxBeats": "yes" if beats else "no",
    "EightAKnoxShortMs": f"{1000 * t_short:.0f}",
    "EightAKnoxShortRatio": f"{r_short:.3f}",
    "EightAKnoxBiasLow": f"{100 * res[20.0]['mean_bias']:.2f}",
})
print("w^-1/2", w_inv_sqrt, "rms20", rms20, "snr", snr_pix, "min", secs / 60, "beats", beats)
print({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in res.items()})
print("shortcut ms", 1000 * t_short, "ratio", r_short)
