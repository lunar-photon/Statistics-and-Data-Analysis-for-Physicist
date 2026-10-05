"""04_gkf_plane.py -- Monte Carlo test of the Gaussian kinematic formula in the plane.

Question: does the mean of the area fraction, the boundary length and chi of the excursion
sets of simulated 2-D Gaussian fields match the analytic expectations
    area = Psi(nu),  length/A = (sqrt(lam)/2) e^{-nu^2/2},  chi/A = lam nu e^{-nu^2/2} / (2 pi)^{3/2},
and does chi depend on the power spectrum only through lam = <|grad phi|^2>/(2 sigma^2)?
Also: what do the edges of a finite square add, and what does one map's chi scatter like?

Computes: 400 periodic maps (512^2) for each of three spectra with the same lam; the three
          Minkowski functionals for the first spectrum; chi of the same maps treated as
          squares with edges; the sampling distribution of chi at nu = 1.
Writes:   figures/ch12/gkf_plane.pdf, figures/ch12/gkf_lambda.pdf, figures/ch12/gkf_edges.pdf,
          results/ch12/04_gkf_plane.tex, data/ch12/gkf_plane.npz
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES, DATA
from lib_fields import gaussian_field
from lib_topo import euler_cubical, mfs_flat, gkf_plane, gkf_square, lam_from_pk

setup()
rng = rng_for("ch12", "04_gkf_plane")
N, NSIM = 512, 400
nus = np.round(np.linspace(-3, 3, 25), 3)
# three spectra k^n exp(-k^2 R^2) with lam = (n + 2) / (4 R^2) all equal to 1/72 (R = 6 for n = 0)
SPECS = {"n0": (0, 6.0), "nm1": (-1, np.sqrt(18.0)), "n2": (2, np.sqrt(72.0))}

out = DATA / "ch12" / "gkf_plane.npz"
if out.exists():
    z = dict(np.load(out))
else:
    z = {}
    for key, (n, R) in SPECS.items():
        P = (lambda n, R: (lambda k: k ** n * np.exp(-(k * R) ** 2)))(n, R)
        z[f"lam_{key}"] = lam_from_pk(P, N, N)
        chi = np.zeros((NSIM, len(nus)))
        area = np.zeros_like(chi)
        length = np.zeros_like(chi)
        chi_sq = np.zeros_like(chi)
        for b in range(NSIM // 50):                         # 50 maps at a time keeps memory low
            fs = gaussian_field(P, N, N, 2, rng, nsim=50)
            for i, f in enumerate(fs):
                s = b * 50 + i
                if key == "n0":
                    area[s], length[s], c = mfs_flat(f, nus)
                    chi[s] = c * N * N
                    u = (f - f.mean()) / f.std()
                    chi_sq[s] = [euler_cubical(u > nu, 8, periodic=False) for nu in nus]
                else:
                    u = (f - f.mean()) / f.std()
                    chi[s] = [euler_cubical(u > nu, 8, periodic=True) for nu in nus]
            print(key, "maps done:", (b + 1) * 50, flush=True)
        z[f"chi_{key}"] = chi
        if key == "n0":
            z.update(area=area, length=length, chi_sq=chi_sq)
        z[f"example_{key}"] = fs[0][:128, :128]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **z)

A = N * N
lam = 1 / 72
nums = {"GkfN": N, "GkfNsim": NSIM, "GkfLamCont": round(lam, 5)}
TAG = {"n0": "NZero", "nm1": "NMOne", "n2": "NTwo"}
for key in SPECS:
    nums[f"GkfLam{TAG[key]}"] = round(float(z[f"lam_{key}"]), 5)
lam0 = float(z["lam_n0"])
th_area, th_len, th_chi = gkf_plane(nus, lam0)
c = z["chi_n0"]
m, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(NSIM)
pull = (m - th_chi * A) / se
print("chi: max |pull| =", np.abs(pull).max())
i1 = int(np.where(nus == 1.0)[0][0])
i0 = int(np.where(nus == 0.0)[0][0])
im1 = int(np.where(nus == -1.0)[0][0])
nums.update({
    "GkfChiOneTh": round(th_chi[i1] * A, 1), "GkfChiOneMC": round(m[i1], 1), "GkfChiOneSe": round(se[i1], 1),
    "GkfChiOneSd": round(c[:, i1].std(ddof=1), 1),
    "GkfChiMoneTh": round(th_chi[im1] * A, 1), "GkfChiMoneMC": round(m[im1], 1),
    "GkfChiZeroMC": round(m[i0], 2), "GkfChiZeroSe": round(se[i0], 2),
    "GkfMaxPull": round(float(np.abs(pull).max()), 2),
    "GkfAreaOneTh": round(th_area[i1], 4), "GkfAreaOneMC": round(z["area"][:, i1].mean(), 4),
    "GkfLenZeroTh": round(th_len[i0] * 1, 4), "GkfLenZeroMC": round(z["length"][:, i0].mean(), 4),
    "GkfLenZeroRel": round(100 * (z["length"][:, i0].mean() / th_len[i0] - 1), 2),
})
for key in ("nm1", "n2"):
    cc = z[f"chi_{key}"]
    tag = TAG[key]
    nums[f"Gkf{tag}ChiOne"] = round(cc[:, i1].mean(), 1)
    nums[f"Gkf{tag}ChiOneSd"] = round(cc[:, i1].std(ddof=1), 1)
    nums[f"Gkf{tag}ThOne"] = round(gkf_plane(1.0, float(z[f'lam_{key}']))[2] * A, 1)
# edges: the same maps cut as an N x N square
csq = z["chi_sq"]
th_sq = gkf_square(nus, lam0, N)
nums.update({"GkfSqOneTh": round(th_sq[i1], 1), "GkfSqOneMC": round(csq[:, i1].mean(), 1),
             "GkfSqZeroTh": round(th_sq[i0], 1), "GkfSqZeroMC": round(csq[:, i0].mean(), 1),
             "GkfSqZeroSe": round(csq[:, i0].std(ddof=1) / np.sqrt(NSIM), 1),
             "GkfEdgeTermZero": round(2 * N * np.sqrt(lam0) / (2 * np.pi), 1)})
save_numbers("ch12", "04_gkf_plane", nums)
for k, v in nums.items():
    print(k, v)

# ---------------------------------------------------------------- figure 1: the three Minkowski functionals
nn = np.linspace(-3.2, 3.2, 300)
ta, tl, tc = gkf_plane(nn, lam0)
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
for ax, data, th, lab in [(axs[0], z["area"], ta, r"area fraction $v_0$"),
                          (axs[1], z["length"], tl, r"boundary length per area $[\mathrm{pix}^{-1}]$"),
                          (axs[2], z["chi_n0"] / A * 1e3, tc * 1e3, r"$\chi$ per area $[10^{-3}\,\mathrm{pix}^{-2}]$")]:
    ax.errorbar(nus, data.mean(0), yerr=data.std(0, ddof=1), fmt="o", ms=3, color=SERIES[0],
                ecolor=SERIES[0], elinewidth=0.8, label=f"mean of {NSIM} maps (bar: one map)")
    theory_line(ax, nn, th, label="Gaussian kinematic formula")
    ax.set_xlabel(r"$\nu$")
    ax.set_title(lab, fontsize=9)
axs[0].legend(fontsize=7, loc="lower left")
savefig(fig, "ch12", "gkf_plane")

# ---------------------------------------------------------------- figure 2: three spectra, one lambda
fig = plt.figure(figsize=(8.0, 4.2))
gsp = fig.add_gridspec(3, 2, width_ratios=[1, 3.0])
names = {"n0": r"$n=0$", "nm1": r"$n=-1$", "n2": r"$n=2$"}
axm = [fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[1, 0]), fig.add_subplot(gsp[2, 0])]
for ax, key in zip(axm, SPECS):
    e = z[f"example_{key}"]
    ax.imshow(e, cmap="RdBu_r", origin="lower")
    ax.set_title(names[key], fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax = fig.add_subplot(gsp[:, 1])
for col, key in zip(SERIES, SPECS):
    ax.plot(nus, z[f"chi_{key}"].mean(0), "o", ms=3.5, color=col, label=f"{names[key]}, mean of {NSIM}")
theory_line(ax, nn, tc * A, label=r"formula with $\lambda=1/72$")
ax.set_xlabel(r"$\nu$")
ax.set_ylabel(r"mean $\chi$ per $512^2$ map")
ax.legend(fontsize=7.5)
savefig(fig, "ch12", "gkf_lambda")

# ---------------------------------------------------------------- figure 3: edges, and the scatter of one map
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.2))
ax = axs[0]
ax.plot(nus, z["chi_n0"].mean(0), "o", ms=3, color=SERIES[0], label="torus (no edges)")
ax.plot(nus, csq.mean(0), "s", ms=3, color=SERIES[1], label="square (4 edges, 4 corners)")
theory_line(ax, nn, tc * A, label="area term only")
ax.plot(nn, gkf_square(nn, lam0, N), color=SERIES[1], ls="--", lw=1.2, label="area + edge + corner terms")
ax.set_xlabel(r"$\nu$")
ax.set_ylabel(r"mean $\chi$")
ax.legend(fontsize=7, loc="lower right")
ax = axs[1]
x = c[:, i1]
ax.hist(x, bins=25, density=True, color=SERIES[0], alpha=0.6, label=rf"$\chi$ at $\nu=1$, {NSIM} maps")
g = np.linspace(x.min() - 5, x.max() + 5, 200)
ax.plot(g, np.exp(-(g - x.mean()) ** 2 / (2 * x.var())) / np.sqrt(2 * np.pi * x.var()), color="k", lw=1,
        label="Gaussian, same mean and variance")
ax.axvline(th_chi[i1] * A, color="k", ls="--", lw=1)
ax.set_xlabel(r"$\chi$ of one $512^2$ map at $\nu=1$")
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.legend(fontsize=7, loc="upper right")
savefig(fig, "ch12", "gkf_edges")
