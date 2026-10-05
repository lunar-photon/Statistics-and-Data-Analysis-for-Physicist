"""46_importance.py -- importance sampling: correcting a chain, and adding data without rerunning it.

Question 1: the six-parameter chain of 45_planck_like.py used the emulator, not CAMB.  If we
reweight its samples by L_CAMB / L_emu (Lewis & Bridle 2002, appendix B), do the means and
widths move, and how many effective samples survive?  This is how a research pipeline checks a
fast approximation on the posterior itself.
Question 2: a new measurement of H0 arrives.  Reweighting the same chain by the new likelihood
should give the combined posterior without a new run -- if the new data are compatible with the
old posterior.  Compare with a fresh chain, for an external H0 = 67.0 +- 1.0 (compatible) and
H0 = 73.04 +- 1.04 (the SH0ES value, Riess et al. 2022: in tension).
Writes: figures/ch09/importance.pdf, results/ch09/46_importance.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES
import lib_mcmc9c as mc
import lib_cmbemu as ce

setup()
rng = rng_for("ch09", "46_importance")
nums = {}
t_start = time.time()
NOTES = pathlib.Path(__file__).resolve().parents[2]
z = np.load(NOTES / "data" / "ch09" / "planck_chains.npz")
post = z["post_P6"]
chat, tau_obs = z["chat"], float(z["tau_obs"])
FSKY, SIG_TAU, LMAX = 0.57, 0.0086, 2500
l = np.arange(LMAX + 1)
use = l >= 2
nu = (2 * l + 1.0) * FSKY
Nl = ce.planck_like_noise(LMAX)
emu = ce.Emulator(order=2)


def m2l_spec(c):
    c = c[use] + Nl[use]
    return np.sum(nu[use] * (chat[use] / c + np.log(c)))


# ---------------------------------------------------------------- 1. emulator -> CAMB
NIS = 400
idx = rng.choice(len(post), NIS, replace=False)
S = post[idx]
dchi = np.empty(NIS)
t0 = time.time()
for i, th in enumerate(S):
    dchi[i] = m2l_spec(ce.camb_tt(th)) - m2l_spec(emu.cl(th))      # -2 ln (L_CAMB / L_emu)
secs = time.time() - t0
w = np.exp(-0.5 * (dchi - dchi.min()))
w /= w.sum()
ess = 1.0 / np.sum(w ** 2)
m0, s0 = S.mean(0), S.std(0)
mw = (w[:, None] * S).sum(0)
shift = (mw - m0) / s0
nums.update({"NineCIsN": NIS, "NineCIsSec": f"{secs:.0f}", "NineCIsPerCall": f"{secs / NIS:.2f}",
             "NineCIsESS": f"{ess:.0f}", "NineCIsDchiSd": f"{dchi.std():.3f}",
             "NineCIsDchiMax": f"{np.abs(dchi - dchi.mean()).max():.2f}",
             "NineCIsMaxShift": f"{np.abs(shift).max():.2f}",
             "NineCIsHoursFull": f"{post.shape[0] * secs / NIS / 3600:.0f}"})

# ---------------------------------------------------------------- 2. new data on H0: reweight vs rerun
iH = ce.NAMES.index("H0")
lo = ce.FID - np.array([0.5, 0.1, 10, 0.003, 0.02, 0.06]); lo[5] = 0.01
hi = ce.FID + np.array([0.5, 0.1, 10, 0.003, 0.02, 0.06])


def lp_base(th):
    if np.any(th < lo) or np.any(th > hi):
        return -np.inf
    return -0.5 * m2l_spec(emu.cl(th)) - 0.5 * ((th[5] - tau_obs) / SIG_TAU) ** 2


cov_post = np.cov(post.T)
out = {}
for tag, (H, sH) in (("Ok", (67.0, 1.0)), ("Tens", (73.04, 1.04))):
    wH = np.exp(-0.5 * ((post[:, iH] - H) / sH) ** 2)
    wH /= wH.sum()
    ess_H = 1.0 / np.sum(wH ** 2)
    mw_H = (wH[:, None] * post).sum(0)
    sw_H = np.sqrt((wH[:, None] * (post - mw_H) ** 2).sum(0))
    # how noisy are these reweighted numbers?  Reweight K blocks of the chain (each chain cut in
    # halves; blocks are much longer than tau) separately and use their scatter: the error of the
    # whole-chain value is about the block scatter / sqrt(K).  Kish's N_eff ignores the chain's
    # autocorrelation, so it cannot give this error by itself.
    K = 8
    bm, bs, bf = [], [], []
    for blk in np.array_split(post, K):
        wb = np.exp(-0.5 * ((blk[:, iH] - H) / sH) ** 2)
        wb /= wb.sum()
        mb = (wb * blk[:, iH]).sum()
        bm.append(mb)
        bs.append(np.sqrt((wb * (blk[:, iH] - mb) ** 2).sum()))
        bf.append(100.0 / np.sum(wb ** 2) / len(blk))
    nums.update({f"NineCIs{tag}HWErr": f"{np.std(bm, ddof=1) / np.sqrt(K):.2g}",
                 f"NineCIs{tag}SHWErr": f"{np.std(bs, ddof=1) / np.sqrt(K):.2g}",
                 f"NineCIs{tag}FracErr": f"{np.std(bf, ddof=1) / np.sqrt(K):.1f}"})
    nums["NineCIsBlocks"] = K
    lp_new = lambda th, H=H, sH=sH: lp_base(th) - 0.5 * ((th[iH] - H) / sH) ** 2
    starts = mw_H + 2 * sw_H * rng.uniform(-1, 1, (4, 6))
    ch, _, acc = mc.run_chains(lp_new, starts, cov_post * 0.6, 30000, rng)
    pn = ch[:, 3000:].reshape(-1, 6)
    out[tag] = dict(wH=wH, pn=pn, H=H, sH=sH)
    nums.update({f"NineCIs{tag}ESS": f"{ess_H:.0f}", f"NineCIs{tag}Frac": f"{100 * ess_H / len(post):.1f}",
                 f"NineCIs{tag}HW": f"{mw_H[iH]:.2f}", f"NineCIs{tag}HR": f"{pn[:, iH].mean():.2f}",
                 f"NineCIs{tag}SHW": f"{sw_H[iH]:.2f}", f"NineCIs{tag}SHR": f"{pn[:, iH].std():.2f}",
                 f"NineCIs{tag}CW": f"{mw_H[4]:.4f}", f"NineCIs{tag}CR": f"{pn[:, 4].mean():.4f}",
                 f"NineCIs{tag}NW": f"{mw_H[1]:.4f}", f"NineCIs{tag}NR": f"{pn[:, 1].mean():.4f}",
                 f"NineCIs{tag}Hext": f"{H:.2f}", f"NineCIs{tag}sHext": f"{sH:.2f}",
                 f"NineCIs{tag}Tension": f"{abs(H - post[:, iH].mean()) / np.hypot(sH, post[:, iH].std()):.1f}"})
nums["NineCIsNpost"] = len(post)
nums["NineCIsHzero"] = f"{post[:, iH].mean():.2f}"
nums["NineCIsSHzero"] = f"{post[:, iH].std():.2f}"
nums["NineCIsMinutes"] = f"{(time.time() - t_start) / 60:.1f}"

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.2))
ax = axs[0]
ax.hist(dchi - dchi.mean(), bins=30, color=SERIES[0])
ax.set_xlabel(r"$-2\ln(\mathcal{L}_{\rm CAMB}/\mathcal{L}_{\rm emu})$ (mean removed)")
ax.set_ylabel("samples"); ax.set_title(f"(a) {NIS} samples recomputed with CAMB", fontsize=9)
for k, tag in enumerate(("Ok", "Tens")):
    ax = axs[k + 1]
    o = out[tag]
    H = post[:, iH]
    rng_h = (60, 78)
    h0, e = np.histogram(H, bins=60, range=rng_h, density=True)
    ax.stairs(h0, e, color="0.5", label="CMB chain")
    hw, e = np.histogram(H, bins=60, range=rng_h, weights=o["wH"], density=True)
    ax.stairs(hw, e, color=SERIES[1], lw=1.4, label="reweighted")
    hr, e = np.histogram(o["pn"][:, iH], bins=60, range=rng_h, density=True)
    ax.stairs(hr, e, color=SERIES[2], ls="--", lw=1.2, label="new chain")
    x = np.linspace(*rng_h, 300)
    ax.plot(x, np.exp(-0.5 * ((x - o["H"]) / o["sH"]) ** 2) / (np.sqrt(2 * np.pi) * o["sH"]), color="k", ls=":",
            lw=1, label="new measurement")
    ax.set_xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)"); ax.legend(fontsize=6.5)
    ax.set_title(("(b) compatible" if tag == "Ok" else "(c) in tension") +
                 f": ESS {nums[f'NineCIs{tag}ESS']} of {len(post)}", fontsize=9)
fig.tight_layout()
savefig(fig, "ch09", "importance")
save_numbers("ch09", "46_importance", nums)
print(nums)
