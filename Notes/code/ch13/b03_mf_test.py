"""b03_mf_test.py -- the Minkowski-functional test of the real sky: data against Gaussian simulations.

Question: at each of the four resolutions, are the area, boundary length and Euler characteristic
of the SMICA map's excursion sets (12 thresholds each, 36 numbers) what Gaussian skies with the
best-fit spectrum, processed identically, would produce?  How large is the chi^2 of the data, and
how often does a Gaussian sky give a chi^2 at least that large?

Computes: the statistics of the data map (lib_skytopo.sky_stats, the same code as the simulations);
          the mean and covariance of the 36 numbers over the simulations; the Hartlap-corrected chi^2
          of the data and of every simulation against the others; the rank p-value and its Monte
          Carlo error; the largest normalised deviation; sigma_0 and sigma_1/sigma_0 of data and skies.
Writes:   data/ch13/b_data_stats.npz, figures/ch13/b_mf_curves.pdf, figures/ch13/b_mf_resid.pdf,
          figures/ch13/b_chi2.pdf, figures/ch13/b_corr.pdf, results/ch13/b03_mf_test.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2dist
from common import setup, savefig, save_numbers, SERIES, DATA
from lib_skytopo import mesh, neighbours, sky_stats, NU12

setup()
SCALES = [(64, 160), (128, 80), (256, 40), (512, 20)]
WORD = {64: "SixFour", 128: "OneTwoEight", 256: "TwoFiveSix", 512: "FiveOneTwo"}
PLANCK_SMICA_P = {64: 90.0, 128: 85.4, 256: 93.9, 512: 80.7}     # Planck 2018 VII, Table 5, SMICA, per cent
KEYS = ("v0", "v1", "v2", "chi", "b0", "b1", "b0c")


def load_sims(nside):
    files = sorted((DATA / "ch13").glob(f"b_sims_{nside}_*.npz"))
    zs = [np.load(f) for f in files]
    out = {k: np.concatenate([z[k] for z in zs]) for k in KEYS + ("ratio", "s0")}
    return out


def data_stats():
    path = DATA / "ch13" / "b_data_stats.npz"
    if path.exists():
        return np.load(path, allow_pickle=True)["stats"].item()
    prep = np.load(DATA / "ch13" / "b_prepared.npz")
    stats = {}
    for nside, _ in SCALES:
        obs = prep[f"obs_{nside}"]
        s = sky_stats(prep[f"alm_{nside}"], nside, 3 * nside - 1, obs, mesh(nside), neighbours(nside),
                      neighbours(nside, edge_only=True), keep_pairs=True)
        stats[nside] = s
    np.savez(path, stats=np.array(stats, dtype=object))
    return stats


def chi2_test(y, Y):
    """Hartlap-corrected chi^2 of y against the simulations Y, and of each simulation against the rest."""
    n, p = Y.shape
    C = np.cov(Y, rowvar=False)
    d = y - Y.mean(0)
    c_data = (n - p - 2) / (n - 1) * d @ np.linalg.solve(C, d)
    c_sims = np.zeros(n)
    for i in range(n):
        rest = np.delete(Y, i, axis=0)
        Ci = np.cov(rest, rowvar=False)
        di = Y[i] - rest.mean(0)
        c_sims[i] = (n - 1 - p - 2) / (n - 2) * di @ np.linalg.solve(Ci, di)
    return c_data, c_sims


D = data_stats()
nums, store = {}, {}
for nside, fwhm in SCALES:
    S = load_sims(nside)
    w = WORD[nside]
    Y = np.concatenate([S["v0"], S["v1"], S["v2"]], axis=1)
    y = np.concatenate([D[nside]["v0"], D[nside]["v1"], D[nside]["v2"]])
    n, p = Y.shape
    c_data, c_sims = chi2_test(y, Y)
    pval = np.mean(c_sims >= c_data)
    dev = (y - Y.mean(0)) / Y.std(0, ddof=1)
    store[nside] = dict(S=S, Y=Y, y=y, c_data=c_data, c_sims=c_sims, dev=dev)
    nums.update({f"MfN{w}": n, f"MfHart{w}": round((n - p - 2) / (n - 1), 3),
                 f"MfChi{w}": round(c_data, 1), f"MfChiN{w}": round(c_data / p, 2),
                 f"MfP{w}": round(100 * pval, 1), f"MfPerr{w}": round(100 * np.sqrt(max(pval * (1 - pval), 1 / n) / n), 1),
                 f"MfPchi{w}": round(100 * chi2dist.sf(c_data, p), 1),
                 f"MfSimMean{w}": round(c_sims.mean() / p, 2),
                 f"MfMaxDev{w}": round(float(np.max(np.abs(dev))), 1),
                 f"MfPlanckP{w}": PLANCK_SMICA_P[nside],
                 f"SzeroData{w}": round(float(D[nside]["s0"]), 1), f"SzeroSim{w}": round(float(S["s0"].mean()), 1),
                 f"SzeroSd{w}": round(float(S["s0"].std(ddof=1)), 1),
                 f"SzeroRank{w}": round(100 * float(np.mean(S["s0"] <= D[nside]["s0"])), 1),
                 f"RatioData{w}": round(float(D[nside]["ratio"]), 1), f"RatioSim{w}": round(float(S["ratio"].mean()), 1),
                 f"RatioSd{w}": round(float(S["ratio"].std(ddof=1)), 1)})
    # the same test with thresholds of each kind alone (12 numbers each)
    for k, j in (("Area", 0), ("Length", 1), ("Genus", 2)):
        cd, cs = chi2_test(y[12 * j:12 * j + 12], Y[:, 12 * j:12 * j + 12])
        nums[f"MfP{k}{w}"] = round(100 * float(np.mean(cs >= cd)), 1)
nums["MfNdof"] = 36
# the smallest of the twelve single-functional probabilities, and the chance that none of twelve
# independent uniform p-values is that small
p_single = [nums[f"MfP{k}{WORD[ns]}"] / 100 for ns in store for k in ("Area", "Length", "Genus")]
p_min = min(p_single)
nums.update({"MfPsingleMin": f"{p_min:.3f}", "MfNoneBelow": f"{(1 - p_min) ** len(p_single):.2f}"})
# a worked example: one threshold, one functional (genus at nu = 0.27 at N_side = 256)
i = int(np.argmin(np.abs(NU12 - 0.273)))
S, y = store[256]["S"], D[256]
nums.update({"ExChiData": int(y["chi"][i]), "ExChiMean": round(float(S["chi"][:, i].mean()), 1),
             "ExChiSd": round(float(S["chi"][:, i].std(ddof=1)), 1),
             "ExBzeroData": int(y["b0"][i]), "ExBoneData": int(y["b1"][i]),
             "ExNu": round(float(NU12[i]), 2)})
save_numbers("ch13", "b03_mf_test", nums)
for k, v in nums.items():
    print(k, v)

# ---------------------------------------------------------------- figure: curves at N_side 256
names = ("Area $v_0$", "Boundary length $v_1$", "Euler characteristic $v_2$")
fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))
st = store[256]
for j, ax in enumerate(axs):
    Yk = st["Y"][:, 12 * j:12 * j + 12]
    lo, hi = np.percentile(Yk, [0.5, 99.5], axis=0)
    ax.fill_between(NU12, lo, hi, color="0.82", lw=0, label="central 99% of Gaussian skies")
    ax.plot(NU12, Yk.mean(0), color="k", ls="--", lw=1, label="mean of Gaussian skies")
    ax.plot(NU12, st["y"][12 * j:12 * j + 12], "o-", ms=3.5, color=SERIES[0], lw=1.2, label="SMICA map")
    ax.set_title(names[j], fontsize=9)
    ax.set_xlabel(r"threshold $\nu$")
axs[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
savefig(fig, "ch13", "b_mf_curves")

# ---------------------------------------------------------------- figure: normalised deviations, all scales
fig, axs = plt.subplots(4, 3, figsize=(9, 7.2), sharex=True, sharey=True)
for r, (nside, fwhm) in enumerate(SCALES):
    st = store[nside]
    Z = (st["Y"] - st["Y"].mean(0)) / st["Y"].std(0, ddof=1)
    for j in range(3):
        ax = axs[r, j]
        lo, hi = np.percentile(Z[:, 12 * j:12 * j + 12], [0.5, 99.5], axis=0)
        ax.fill_between(NU12, lo, hi, color="0.85", lw=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.plot(NU12, st["dev"][12 * j:12 * j + 12], "o-", ms=3, color=SERIES[0], lw=1)
        if r == 0:
            ax.set_title(names[j], fontsize=9)
        if j == 0:
            ax.set_ylabel(fr"$N_{{\rm side}}={nside}$" + "\n" + r"$\Delta v/\sigma$", fontsize=8)
        if r == 3:
            ax.set_xlabel(r"$\nu$")
axs[0, 0].set_ylim(-4, 4)
fig.tight_layout()
savefig(fig, "ch13", "b_mf_resid")

# ---------------------------------------------------------------- figure: chi^2 distributions
fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.7), sharey=True)
g = np.linspace(0.2, 2.5, 300)
for ax, (nside, fwhm) in zip(axs, SCALES):
    st = store[nside]
    n = len(st["c_sims"])
    bins = np.linspace(0.2, 2.5, 24)
    ax.hist(st["c_sims"] / 36, bins=bins, color=SERIES[0], alpha=0.45, density=True, label="Gaussian skies")
    ax.plot(g, chi2dist.pdf(g * 36, 36) * 36, color="k", ls="--", lw=1, label=r"$\chi^2_{36}$")
    ax.axvline(st["c_data"] / 36, color=SERIES[1], lw=2, label="SMICA map")
    ax.set_title(fr"$N_{{\rm side}}={nside}$, ${fwhm}'$", fontsize=9)
    ax.set_xlabel(r"$\chi^2/N_{\rm dof}$")
axs[0].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch13", "b_chi2")

# ---------------------------------------------------------------- figure: correlation matrix at N_side 256
fig, ax = plt.subplots(figsize=(4.6, 4.0))
R = np.corrcoef(store[256]["Y"], rowvar=False)
im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
for b in (11.5, 23.5):
    ax.axhline(b, color="k", lw=0.6)
    ax.axvline(b, color="k", lw=0.6)
ax.set_xticks([5.5, 17.5, 29.5])
ax.set_xticklabels([r"$v_0$", r"$v_1$", r"$v_2$"])
ax.set_yticks([5.5, 17.5, 29.5])
ax.set_yticklabels([r"$v_0$", r"$v_1$", r"$v_2$"])
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046, label="correlation")
savefig(fig, "ch13", "b_corr")
