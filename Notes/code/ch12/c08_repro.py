"""c08_repro.py -- two published claims about topology and f_NL, redone on our maps.

Question 1 (Cole & Shiu 2018, their Table 1): with a Gaussian likelihood and a simulated
covariance, how does the error on the non-Gaussianity amplitude from the Betti curves
(beta0, beta1, both) compare with the one from persistence-diagram histograms (PD0, PD1, both)?
Question 2 (Biagetti, Cole & Shiu 2021, their eqs. 4.1-4.3): how often does a non-Gaussian map
stand out from Gaussian ones (a) with the diagonal 'anomaly' distance
D = sqrt( (1/N) sum_i (d_i - mu_i)^2 / sigma_i^2 ) at the 95th percentile, and (b) after
projecting on a template T = mean(S[eps_big] - S[0]) built from simulations with the same
seeds (D_template = (S - S_fid) . T / sigma) at the 97.5th percentile?  And the power spectrum?
Computes, from the cache of c02_mc_fnl.py (256 x 256 maps, eps = f_NL sigma_g):
  * Fisher sigma(eps) for b0, b1, b0+b1, PD0, PD1, PD0+PD1, and for Betti curves divided by
    their maximum (Cole & Shiu's normalisation);
  * detection rates of eps = 0.05 and 0.025 maps (seeds 250-499) for both of Biagetti's
    statistics and for b0, b1, PD0, PD1 and the power spectrum, with the template from eps = 0.1
    (seeds 0-249) and the null distribution from the 500 'check' maps.
Writes: figures/ch12/c_repro.pdf, results/ch12/c08_repro.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, INK2, DATA

z = np.load(DATA / "ch12" / "c_fnl_mc.npz")
eg = list(z["eps_grid"])
i0 = eg.index(0.0)


def vec(prefix, keys, norm=False):
    parts = []
    for k in keys:
        a = z[prefix + k].astype(float)
        if norm:                                           # divide each curve by its own maximum
            a = a / np.maximum(a.max(-1, keepdims=True), 1)
        parts.append(a)
    return np.concatenate(parts, axis=-1)


def sigma_fisher(keys, norm=False):
    N = vec("null_", keys, norm)
    ok = N.std(0) > 1e-9
    N = N[:, ok]
    n, p = N.shape
    Ci = np.linalg.inv(np.atleast_2d(np.cov(N.T))) * (n - p - 2) / (n - 1)
    G = vec("grid_", keys, norm)[..., ok]
    d = ((G[:, eg.index(0.05)] - G[:, eg.index(-0.05)]) / 0.1).mean(0)
    return 1 / np.sqrt(d @ Ci @ d)


# ---------------------------------------------------------------- 1. Cole & Shiu, Table 1
rows = [("b0", ["b0"]), ("b1", ["b1"]), ("b0+b1", ["b0", "b1"]),
        ("PD0", ["pd0"]), ("PD1", ["pd1"]), ("PD0+PD1", ["pd0", "pd1"])]
ours = {name: sigma_fisher(keys) for name, keys in rows}
ours_norm = {name: sigma_fisher(keys, norm=True) for name, keys in rows[:3]}
theirs = {"b0": 67.4, "b1": 66.1, "b0+b1": 60.6, "PD0": 39.1, "PD1": 37.4, "PD0+PD1": 35.8}


# ---------------------------------------------------------------- 2. Biagetti et al., anomaly and template
dseeds, tseeds = np.arange(250), np.arange(250, 500)


def rates(keys, eps):
    N = vec("null_", keys)
    ok = N.std(0) > 1e-9
    mu, sd = N[:, ok].mean(0), N[:, ok].std(0)
    ref = vec("check_", keys)[:, ok]
    obs = vec("grid_", keys)[tseeds, eg.index(eps)][:, ok]
    D = lambda X: np.sqrt(np.mean(((X - mu) / sd) ** 2, axis=-1))  # noqa: E731  their eq. 4.1
    a = (D(obs) > np.quantile(D(ref), 0.95)).mean()
    G = vec("grid_", keys)[dseeds][..., ok]
    T = (G[:, eg.index(0.1)] - G[:, i0]).mean(0)            # their eq. 4.2: same seeds, big eps
    proj_ref = (ref - mu) @ T
    s = proj_ref.std()
    Dt = lambda X: (X - mu) @ T / s  # noqa: E731                    their eq. 4.3
    t = (Dt(obs) > np.quantile(Dt(ref), 0.975)).mean()
    return a, t


stats_list = [("b0", ["b0"]), ("b1", ["b1"]), ("PD0", ["pd0"]), ("PD1", ["pd1"]), ("P(k)", ["pk"])]
R = {(name, e): rates(keys, e) for name, keys in stats_list for e in (0.025, 0.05)}

setup(8.6, 3.0)
fig, ax = plt.subplots(1, 2)
names = [r[0] for r in rows]
x = np.arange(len(names))
th = np.array([theirs[n] for n in names])
ou = np.array([ours[n] for n in names])
ax[0].bar(x - 0.18, th / th[2], 0.36, color=SERIES[0], label="Cole & Shiu, Table 1")
ax[0].bar(x + 0.18, ou / ou[2], 0.36, color=SERIES[1], label="ours (Fisher, $256^2$ maps)")
ax[0].set_xticks(x)
ax[0].set_xticklabels([n.replace("b", r"$\beta_").replace("+$", "$+") + ("$" if n.startswith("b") and not n.endswith("1") else "") for n in names],
                      fontsize=7)
ax[0].set_xticklabels([r"$\beta_0$", r"$\beta_1$", r"$\beta_0{+}\beta_1$", "PD0", "PD1", "PD0+PD1"], fontsize=7)
ax[0].axhline(1, color=INK2, lw=0.6)
ax[0].set_ylabel(r"error / error of $\beta_0{+}\beta_1$")
ax[0].set_ylim(0, 2.6)
ax[0].legend(fontsize=7, loc="upper left")
lab = [n for n, _ in stats_list]
x = np.arange(len(lab))
for j, (e, c) in enumerate(((0.05, SERIES[2]), (0.025, SERIES[3]))):
    ax[1].bar(x - 0.3 + 0.3 * j, [100 * R[(n, e)][0] for n in lab], 0.15, color=c, alpha=0.5,
              label=rf"anomaly $D$, $\epsilon={e}$")
    ax[1].bar(x - 0.15 + 0.3 * j, [100 * R[(n, e)][1] for n in lab], 0.15, color=c,
              label=rf"template, $\epsilon={e}$")
ax[1].set_xticks(x)
ax[1].set_xticklabels([r"$\beta_0$" if n == "b0" else r"$\beta_1$" if n == "b1" else n for n in lab], fontsize=7)
ax[1].set_ylabel("detection rate (%)")
ax[1].set_ylim(0, 25)
ax[1].legend(fontsize=6.5, ncol=2, loc="upper right")
fig.tight_layout()
savefig(fig, "ch12", "c_repro")

num = {}
for name, key in (("b0", "BZero"), ("b1", "BOne"), ("b0+b1", "BBoth"), ("PD0", "PZero"), ("PD1", "POne"),
                  ("PD0+PD1", "PBoth")):
    num["Cr" + key] = f"{ours[name]:.4f}"
    num["CrRel" + key] = f"{ours[name] / ours['b0+b1']:.2f}"
    num["CrTh" + key] = f"{theirs[name] / theirs['b0+b1']:.2f}"
for name, key in (("b0", "BZero"), ("b1", "BOne"), ("b0+b1", "BBoth")):
    num["CrNorm" + key] = f"{ours_norm[name]:.4f}"
for (name, e), (a, t) in R.items():
    k = {"b0": "BZero", "b1": "BOne", "PD0": "PZero", "PD1": "POne", "P(k)": "Pk"}[name]
    s = "Five" if e == 0.05 else "Two"
    num[f"CrAn{k}{s}"] = f"{100 * a:.0f}"
    num[f"CrTp{k}{s}"] = f"{100 * t:.0f}"
num["CrShiftBZero"] = f"{0.05 / ours['b0']:.1f}"                   # score shift of beta0 at eps = 0.05, in sd
num["CrPdOverNorm"] = f"{ours['PD0+PD1'] / ours_norm['b0+b1']:.2f}"
save_numbers("ch12", "c08_repro", num)
print("ours", ours, "norm", ours_norm)
for k, v in R.items():
    print(k, np.round(v, 3))
