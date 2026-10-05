"""43_emulator.py -- a model C_l(theta) fast enough for a Markov chain, validated against CAMB.

Question: a CAMB call takes about a second; a chain needs 10^5 likelihood calls.  Can a Taylor
expansion of ln C_l around the fiducial model replace CAMB inside the chain, and how far from
the fiducial can we trust it?  Which expansion: C_l to first order, ln C_l to first order, or
ln C_l to second order?
Computes:
  * derivatives d lnC_l / d theta_i and the second derivatives for the six LCDM parameters
    (ln 10^10 A_s, n_s, H0, omega_b, omega_c, tau) by central differences (lib_cmbemu.build);
  * the Fisher matrix of a Planck-like temperature spectrum (143-GHz-like channel: 7.22' beam,
    33 muK arcmin, f_sky = 0.57, 2 <= l <= 2500), with a Gaussian prior sigma(tau) = 0.0086, to
    set the natural size of a step, sigma_i;
  * the error of each emulator against direct CAMB calls: along each parameter axis at
    +-1..5 sigma_i, and at 40 random points drawn from the Fisher Gaussian with widths doubled;
    measured as Delta chi^2 = sum_l (2l+1) f_sky / 2 [(C_emu - C_camb)/(C + N)]^2, the change of
    -2 ln L the emulator error alone would produce.
Writes: data/ch09/emu.npz, figures/ch09/emu_derivs.pdf, figures/ch09/emu_check.pdf,
        results/ch09/43_emulator.tex
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES
import lib_cmbemu as ce

setup()
rng = rng_for("ch09", "43_emulator")
nums = {}
t0 = time.time()
z = ce.build()
nums["NineCEmuBuildCalls"] = 1 + 2 * 6 + 4 * 15
l = np.arange(ce.LMAX + 1)
sel = l >= 2
FSKY = 0.57
SIG_TAU = 0.0086
N = ce.planck_like_noise()
C0 = z["c0"]
w = (2 * l + 1) * FSKY / 2 / (C0 + N) ** 2            # Fisher weight per multipole

# Fisher matrix for the six parameters, chain rule dC = C dlnC
dC = z["D"] * C0[None, :]
F = (dC[:, sel] * w[sel]) @ dC[:, sel].T
F[5, 5] += 1 / SIG_TAU ** 2
cov = np.linalg.inv(F)
sig = np.sqrt(np.diag(cov))
F2 = F[:2, :2] - 0 * F[:2, :2]
cov2 = np.linalg.inv(F[:2, :2] - np.diag([0, 0]))      # (lnAs, ns) alone, others fixed
for i, n in enumerate(["LnAs", "Ns", "Hzero", "Omb", "Omc", "Tau"]):
    nums[f"NineCEmuSig{n}"] = f"{sig[i]:.4g}"
nums.update({"NineCEmuSigLnAsTwo": f"{np.sqrt(cov2[0, 0]):.4f}", "NineCEmuSigNsTwo": f"{np.sqrt(cov2[1, 1]):.4f}",
             "NineCEmuRhoTwo": f"{cov2[0, 1] / np.sqrt(cov2[0, 0] * cov2[1, 1]):.2f}",
             "NineCEmuRhoSix": f"{cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.2f}"})

emus = {"linC": ce.Emulator(order=1, linear_in_C=True), "lin": ce.Emulator(order=1),
        "quad": ce.Emulator(order=2)}


def dchi2(c_emu, c_true):
    return float(np.sum(w[sel] * (c_emu[sel] - c_true[sel]) ** 2))


# ---- 1. axis cuts
KS = np.array([-5, -3, -1, 1, 3, 5])
cut = {k: np.zeros((6, KS.size)) for k in emus}
for i in range(6):
    for j, k in enumerate(KS):
        th = ce.FID.copy(); th[i] += k * sig[i]
        ct = ce.camb_tt(th)
        for name, em in emus.items():
            cut[name][i, j] = dchi2(em.cl(th), ct)
# ---- 2. random points at twice the Fisher width
NR = 40
pts = rng.multivariate_normal(ce.FID, 4 * cov, NR)
dist = np.sqrt(np.einsum("ni,ij,nj->n", pts - ce.FID, F, pts - ce.FID))   # Mahalanobis distance
rnd = {k: np.zeros(NR) for k in emus}
frac_err = []
for n_, th in enumerate(pts):
    ct = ce.camb_tt(th)
    for name, em in emus.items():
        rnd[name][n_] = dchi2(em.cl(th), ct)
    if n_ < 4:
        sd = np.sqrt(2 / ((2 * l[sel] + 1) * FSKY)) * (ct[sel] + N[sel])   # error of one C_l
        frac_err.append(((emus["quad"].cl(th)[sel] - ct[sel]) / sd, (emus["lin"].cl(th)[sel] - ct[sel]) / sd))
for name, tag in (("linC", "LinC"), ("lin", "Lin"), ("quad", "Quad")):
    nums[f"NineCEmuMed{tag}"] = float(f"{np.median(rnd[name]):.2e}")
    nums[f"NineCEmuMax{tag}"] = float(f"{np.max(rnd[name]):.2e}")
    nums[f"NineCEmuCutFive{tag}"] = float(f"{cut[name][:, [0, -1]].max():.2e}")
    nums[f"NineCEmuCutOne{tag}"] = float(f"{cut[name][:, [2, 3]].max():.2e}")
# the two-parameter case (lnAs, ns only)
e2 = ce.Emulator(order=1, names=["lnAs", "ns"])
worst2 = 0.0
for a in (-5, 5):
    for b in (-5, 5):
        th = ce.FID.copy(); th[0] += a * np.sqrt(cov2[0, 0]); th[1] += b * np.sqrt(cov2[1, 1])
        worst2 = max(worst2, dchi2(e2.cl(th[:2]), ce.camb_tt(th)))
nums["NineCEmuTwoWorst"] = float(f"{worst2:.2e}")
nums["NineCEmuNrand"] = NR
nums["NineCEmuMaxDist"] = f"{dist.max():.1f}"
nums["NineCEmuFsky"] = FSKY
nums["NineCEmuSigTauPrior"] = SIG_TAU
# second derivative wrt lnAs: how far from exactly C proportional to A_s (lensing)
nums["NineCEmuHAA"] = float(f"{np.max(np.abs(z['H'][0, 0][30:2000])):.1e}")
# second derivative wrt ns compared with (d lnC/d ns)^2 at l = 1000
nums["NineCEmuHnn"] = f"{z['H'][1, 1][1000]:.2f}"
nums["NineCEmuDnThousand"] = f"{z['D'][1][1000]:.2f}"
nums["NineCEmuDnTen"] = f"{z['D'][1][10]:.2f}"
nums["NineCEmuDtauHigh"] = f"{np.mean(z['D'][5][500:2000]):.2f}"
nums["NineCEmuMinutes"] = f"{(time.time() - t0) / 60:.1f}"

# ---- t_emu vs t_camb
em = emus["quad"]
tt = time.time()
for _ in range(2000):
    em.cl(ce.FID)
t_emu = (time.time() - tt) / 2000
tt = time.time(); ce.camb_tt(); t_camb = time.time() - tt
nums.update({"NineCEmuTemu": f"{t_emu * 1e6:.0f}", "NineCEmuTcamb": f"{t_camb:.1f}",
             "NineCEmuSpeedup": f"{t_camb / t_emu:.0f}"})
np.savez(ce.CACHE.parent / "fisher_planck_like.npz", F=F, cov=cov, fsky=FSKY, sig_tau=SIG_TAU)

# ---- figure 1: what each parameter does to ln C_l (per 1 sigma)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i in range(6):
    ax.plot(l[2:], z["D"][i][2:] * sig[i], color=SERIES[i], lw=1.1, label=ce.LABELS[i])
ax.set_xscale("log"); ax.set_xlim(2, 2500)
ax.axhline(0, color="0.5", lw=0.6)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"$\sigma_i\,\partial\ln C_\ell/\partial\theta_i$")
ax.set_ylim(-0.045, 0.045)
ax.legend(fontsize=7, ncol=4, loc="lower left")
fig.tight_layout()
savefig(fig, "ch09", "emu_derivs")

# ---- figure 2: validation
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
ax = axs[0]
for k, (eq, el) in enumerate(frac_err[:3]):
    ax.plot(l[sel], el, color=SERIES[k], lw=0.6, ls="--")
    ax.plot(l[sel], eq, color=SERIES[k], lw=0.9)
ax.plot([], [], color="k", lw=0.9, label=r"2nd order in $\ln C$")
ax.plot([], [], color="k", lw=0.6, ls="--", label=r"1st order in $\ln C$")
ax.set_xscale("log"); ax.set_xlim(2, 2500)
ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$(C_\ell^{\rm emu}-C_\ell^{\rm CAMB})/\sigma_\ell$")
ax.set_title("(a) three random test models", fontsize=9); ax.legend(fontsize=7)
ax = axs[1]
for name, col, lab in (("linC", SERIES[1], r"1st order in $C$"), ("lin", SERIES[0], r"1st order in $\ln C$"),
                       ("quad", SERIES[2], r"2nd order in $\ln C$")):
    ax.semilogy(dist, rnd[name], "o", ms=3, color=col, label=lab)
ax.axhline(0.1, color="0.4", ls=":", lw=0.8)
ax.set_xlabel(r"distance from the fiducial (in $\sigma$, Fisher metric)")
ax.set_ylabel(r"$\Delta\chi^2$ from the emulator error")
ax.set_title("(b) 40 random models vs CAMB", fontsize=9); ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch09", "emu_check")
save_numbers("ch09", "43_emulator", nums)
print(nums)
print("cuts (rows: params, cols: -5..5 sigma)\n", {k: np.round(v, 3) for k, v in cut.items()})
