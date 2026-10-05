"""04_derivatives.py -- numerical derivatives for a Fisher matrix: choosing the step, testing stability.

Question: the Fisher matrix needs dC_l/dtheta_i, and a Boltzmann code gives them only by finite
differences. Too large a step and the Taylor series is not linear any more (truncation error);
too small and the code's own numerical noise is divided by a tiny number (noise error).
Where is the sweet spot, and how do we check that a Fisher forecast does not depend on the step?

Computes:
 (a) a test function with a known derivative, f(x) = exp(x) sin(3x) at x = 0.7, with and without a
     simulated "code noise" of relative size 1e-6: the error of the 2-point and 4-point central
     differences against the step h;
 (b) CAMB: lensed TT, EE, TE at theta_fid +- m h_i for m = 1/64, 1/16, 1/4, 1/2, 1, 2, 4 (84 CAMB calls, cached in
     data/ch10/spec/), derivatives by the 2-point rule at every m and the 4-point rule at m <= 2, and
     the marginalised errors of a Planck-like T+E forecast for each choice.
Writes: figures/ch10/deriv_test.pdf, figures/ch10/deriv_camb.pdf, data/ch10/derivs_*.npz,
        results/ch10/04_derivatives.tex
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
import lib_fisher10 as L

setup()
rng = rng_for("ch10", "04_derivatives")
nums = {}

# ---------------------------------------------------------------- (a) a test function
x0 = 0.7
f = lambda x: np.exp(x) * np.sin(3 * x)
fp = np.exp(x0) * (np.sin(3 * x0) + 3 * np.cos(3 * x0))
EPS = 1e-6                                       # relative "code noise"
hs = np.logspace(-9, -0.3, 60)


def noisy(x):
    return f(x) * (1 + EPS * rng.standard_normal(np.shape(x)))


def d2(fun, h):
    return (fun(x0 + h) - fun(x0 - h)) / (2 * h)


def d4(fun, h):
    return (-fun(x0 + 2 * h) + 8 * fun(x0 + h) - 8 * fun(x0 - h) + fun(x0 - 2 * h)) / (12 * h)


NREAL = 200                                      # noise realisations: the error of a noisy rule is its rms
err = {k: np.array([abs(fn(g, h) - fp) / abs(fp) for h in hs]) for k, fn, g in [("2", d2, f), ("4", d4, f)]}
for k, fn in [("2n", d2), ("4n", d4)]:
    err[k] = np.array([np.sqrt(np.mean([(fn(noisy, h) - fp) ** 2 for _ in range(NREAL)])) / abs(fp) for h in hs])
h_opt2 = hs[np.argmin(err["2n"])]
h_opt4 = hs[np.argmin(err["4n"])]
# the prediction: error ~ eps |f| / h + h^2 |f'''| / 6, minimised at h = (3 eps |f| / |f'''|)^(1/3)
f3 = abs(np.exp(x0) * (np.sin(3 * x0) * (1 - 27) + np.cos(3 * x0) * (9 - 27)))   # d3/dx3 [e^x sin 3x]
h_pred = (3 * EPS * abs(f(x0)) / f3) ** (1 / 3)
h_opt4s = f"{h_opt4:.1e}".replace("e-0", r"\times10^{-") + "}"
nums.update({"TenDDeEps": r"10^{-6}", "TenDDeHoptFour": h_opt4s, "TenDDeNreal": NREAL, "TenDDeHoptTwo": f"{h_opt2:.1e}".replace("e-0", r"\times10^{-") + "}",
             "TenDDeHpred": f"{h_pred:.1e}".replace("e-0", r"\times10^{-") + "}",
             "TenDDeBestTwo": f"{err['2n'].min():.0e}".replace("e-0", r"\times10^{-") + "}",
             "TenDDeBestFour": f"{err['4n'].min():.0e}".replace("e-0", r"\times10^{-") + "}",
             "TenDDeRelPred": f"{(EPS * abs(f(x0)) / h_pred + h_pred**2 * f3 / 6) / abs(fp):.0e}".replace("e-0", r"\times10^{-") + "}"})

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.loglog(hs, err["2"], color=SERIES[0], lw=1.0, ls=":", label="2-point, exact $f$")
ax.loglog(hs, err["4"], color=SERIES[1], lw=1.0, ls=":", label="4-point, exact $f$")
ax.loglog(hs, err["2n"], color=SERIES[0], label=r"2-point, $f$ with $10^{-6}$ noise (rms)")
ax.loglog(hs, err["4n"], color=SERIES[1], label=r"4-point, $f$ with $10^{-6}$ noise (rms)")
theory_line(ax, hs, EPS * abs(f(x0)) / hs / abs(fp), label=r"noise $\epsilon|f|/h$")
ax.axvline(h_pred, color="k", lw=0.8, ls="-.")
ax.set_xlabel("step $h$")
ax.set_ylabel("relative error of $f'$")
ax.set_ylim(1e-14, 10)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
savefig(fig, "ch10", "deriv_test")

# ---------------------------------------------------------------- (b) CAMB
SPEC = L.DATA / "spec"
SPEC.mkdir(parents=True, exist_ok=True)
fid = L.fiducial_vector()
MULTS = [1 / 64, 1 / 16, 0.25, 0.5, 1.0, 2.0, 4.0]
t0 = time.time()
ncalls = 0


def spectra_at(i, m):
    """Spectra at fid + m STEPS[i] e_i, cached (m may be negative)."""
    global ncalls
    path = SPEC / (f"p{i}_{m:+.6f}".replace(".", "p") + ".npz")
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    e = np.zeros(6); e[i] = m * L.STEPS[i]
    c = L.camb_spectra(fid + e)
    ncalls += 1
    out = {s: c[s] for s in ("TT", "EE", "TE")}
    np.savez(path, **out)
    return out


def derivs(m, stencil):
    out = {s: np.zeros((6, L.LMAX + 1)) for s in ("TT", "EE", "TE")}
    for i in range(6):
        h = m * L.STEPS[i]
        p1, m1 = spectra_at(i, m), spectra_at(i, -m)
        if stencil == 4:
            p2, m2 = spectra_at(i, 2 * m), spectra_at(i, -2 * m)
        for s in out:
            if stencil == 2:
                out[s][i] = (p1[s] - m1[s]) / (2 * h)
            else:
                out[s][i] = (-p2[s] + 8 * p1[s] - 8 * m1[s] + m2[s]) / (12 * h)
    tag = f"{m:g}".replace(".", "p")
    if m in (0.25, 0.5, 1.0, 2.0, 4.0):
        np.savez(L.DATA / f"derivs_{tag}_{stencil}.npz", **out)     # the cache lib_fisher10.derivatives reads
    return out


C = L.fiducial_spectra()
NT, NP = L.noise_planck_like()
FSKY = 0.57


def forecast(dC):
    """Planck-like: TT at 2 <= l <= 29, T+E at 30 <= l <= 2000, TT at 2001 <= l <= 2500, plus a tau prior."""
    F = (L.fisher_cmb(dC, C, NT, NP, FSKY, 2, 29, "TT") + L.fisher_cmb(dC, C, NT, NP, FSKY, 30, 2000)
         + L.fisher_cmb(dC, C, NT, NP, FSKY, 2001, 2500, "TT") + L.tau_prior())
    return L.marg(F)


res2 = np.array([forecast(derivs(m, 2)) for m in MULTS])
res4 = np.array([forecast(derivs(m, 4)) for m in MULTS[:-1]])
IREF = MULTS.index(1.0)
ref = res4[IREF]                                    # 4-point rule at the nominal steps
nums["TenDDeCalls"] = 12 * len(MULTS)
nums["TenDDeMinutes"] = f"{(time.time() - t0) / 60:.1f}"
dev2 = np.abs(res2 / ref - 1).max(1)
dev4 = np.abs(res4 / ref - 1).max(1)
k = {m: MULTS.index(m) for m in MULTS}
pc = lambda v: f"{100 * v:.2f}"
nums.update({"TenDDeDevTwoNom": pc(dev2[k[1.0]]), "TenDDeDevTwoQuarter": pc(dev2[k[0.25]]),
             "TenDDeDevTwoFour": pc(dev2[k[4.0]]), "TenDDeDevFourHalf": pc(dev4[k[0.5]]),
             "TenDDeDevFourQuarter": pc(dev4[k[0.25]]), "TenDDeDevFourTwo": pc(dev4[k[2.0]]),
             "TenDDeDevTwoSixteenth": pc(dev2[k[1 / 16]]), "TenDDeDevTwoSixtyfourth": pc(dev2[k[1 / 64]]),
             "TenDDeDevFourSixtyfourth": pc(dev4[k[1 / 64]])})
worst = np.argmax(np.abs(res2[k[1 / 64]] / ref - 1))
nums["TenDDeWorstParSmall"] = L.TEXNAMES[worst]
for k, nm in enumerate(["Ob", "Oc", "Th", "Tau", "As", "Ns"]):
    nums[f"TenDDeStep{nm}"] = f"{L.STEPS[k]:g}"

fig, ax = plt.subplots(figsize=(5.6, 3.5))
for i in range(6):
    ax.plot(MULTS, res2[:, i] / ref[i], "o-", color=SERIES[i], ms=3.5, lw=1.2, label=L.LABELS[i])
    ax.plot(MULTS[:-1], res4[:, i] / ref[i], "s--", color=SERIES[i], ms=3, lw=0.8)
ax.axhline(1, color="k", lw=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks(MULTS); ax.set_xticklabels(["1/64", "1/16", "1/4", "1/2", "1", "2", "4"])
ax.set_xlabel(r"step multiplier $m$ (step $= m\,h_i$)")
ax.set_ylabel(r"$\sigma_i(m)\,/\,\sigma_i$(4-point, $m=1$)")
ax.legend(fontsize=7, ncol=3, loc="lower right")
ax.set_title("solid: 2-point rule; dashed: 4-point rule", fontsize=9)
savefig(fig, "ch10", "deriv_camb")
save_numbers("ch10", "04_derivatives", nums)
print(nums)
print("res2", res2, "\nres4", res4)
