"""How does the error on H0 build up along a three-rung distance ladder?

Question: geometric distances to a few anchor galaxies calibrate the Cepheid period-luminosity
(Leavitt) law; Cepheids in nearby supernova hosts calibrate the supernova absolute magnitude
M_B; supernovae in the Hubble flow then measure 5 log10 H0 - M_B. Written as one linear model
y = A theta + noise and solved by generalised least squares, which rung dominates the error
on H0, and does the covariance (A^T C^-1 A)^-1 describe the scatter over repeated ladders?
Computes:
  * one mock ladder (SH0ES-like sizes, round numbers) and its GLS solution,
  * sigma(H0)/H0 against the number of calibrator hosts, with the contribution of each rung,
  * a Monte Carlo of repeated ladders: scatter and coverage of the GLS error bar.
Writes: figures/chT2/distance_ladder.pdf, results/chT2/07_distance_ladder.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES

C_KMS = 299792.458
KAPPA = np.log(10) / 5
TRUE = dict(H0=73.0, MW=-5.90, bW=-3.30, MB=-19.25, q0=-0.55)
CFG = dict(Nanc=3, sig_geo=0.03, Nceph_anc=100, Nceph_cal=30, sig_ceph=0.25,
           Ncal=40, sig_sn=0.13, Nhf=280, sig_hf=0.14)


def build(rng, cfg, truth=TRUE):
    """Return y, A, sigma (diagonal) for one mock ladder.

    theta = [mu_anc (Nanc), mu_cal (Ncal), M_W, b_W, M_B, a = 5 log10 H0]."""
    Na, Nc = cfg["Nanc"], cfg["Ncal"]
    npar = Na + Nc + 4
    iMW, ibW, iMB, ia = Na + Nc, Na + Nc + 1, Na + Nc + 2, Na + Nc + 3
    mu_anc = rng.uniform(18.5, 29.5, Na)            # LMC-like to NGC 4258-like
    mu_cal = rng.uniform(31.0, 33.0, Nc)            # 15-40 Mpc
    rows, y, sig = [], [], []

    def add(coeffs, value, s):
        r = np.zeros(npar)
        for j, v in coeffs:
            r[j] = v
        rows.append(r); y.append(value); sig.append(s)

    # rung 1: geometric distance moduli of the anchors
    for i in range(Na):
        add([(i, 1)], mu_anc[i] + rng.normal(0, cfg["sig_geo"]), cfg["sig_geo"])
    # rung 2: Cepheids, m = mu + M_W + b_W (log P - 1)
    for i, (mu, n) in enumerate([(m, cfg["Nceph_anc"]) for m in mu_anc] +
                                [(m, cfg["Nceph_cal"]) for m in mu_cal]):
        lp = rng.uniform(0.6, 1.8, n) - 1.0
        m = mu + truth["MW"] + truth["bW"] * lp + rng.normal(0, cfg["sig_ceph"], n)
        for k in range(n):
            add([(i, 1), (iMW, 1), (ibW, lp[k])], m[k], cfg["sig_ceph"])
    # rung 3a: one supernova in each calibrator host, m = mu + M_B
    for i in range(Nc):
        add([(Na + i, 1), (iMB, 1)], mu_cal[i] + truth["MB"] + rng.normal(0, cfg["sig_sn"]), cfg["sig_sn"])
    # rung 3b: Hubble-flow supernovae, m = M_B + 5 log10(cz [1 + (1-q0) z/2]) + 25 - a
    z = rng.uniform(0.023, 0.15, cfg["Nhf"])
    x = 5 * np.log10(C_KMS * z * (1 + 0.5 * (1 - truth["q0"]) * z)) + 25
    a_true = 5 * np.log10(truth["H0"])
    m = truth["MB"] + x - a_true + rng.normal(0, cfg["sig_hf"], cfg["Nhf"])
    for k in range(cfg["Nhf"]):
        add([(iMB, 1), (ia, -1)], m[k] - x[k], cfg["sig_hf"])
    return np.array(y), np.array(rows), np.array(sig), ia


def gls(y, A, sig):
    W = 1 / sig**2
    F = A.T @ (A * W[:, None])                       # Fisher matrix A^T C^-1 A
    cov = np.linalg.inv(F)
    theta = cov @ (A.T @ (W * y))
    return theta, cov


def sigma_h0(cfg, rng):
    y, A, s, ia = build(rng, cfg)
    th, cov = gls(y, A, s)
    return KAPPA * np.sqrt(cov[ia, ia]), th, cov, ia


rng = rng_for("chT2", "07_distance_ladder")
frac, th, cov, ia = sigma_h0(CFG, rng)
H0hat = 10 ** (th[ia] / 5)

# --- error budget: switch off one rung's noise at a time (set its sigma tiny)
ncal_grid = np.array([5, 10, 20, 40, 80, 160])
budget = {}
for label, change in [("total", {}), ("perfect anchors", dict(sig_geo=1e-4)),
                      ("perfect Cepheids", dict(sig_ceph=1e-3)),
                      ("perfect calibrator SNe", dict(sig_sn=1e-4)),
                      ("perfect Hubble flow", dict(sig_hf=1e-4))]:
    budget[label] = np.array([sigma_h0({**CFG, **change, "Ncal": n}, rng)[0] for n in ncal_grid])
contrib = {k: np.sqrt(np.maximum(budget["total"] ** 2 - v ** 2, 0)) for k, v in budget.items() if k != "total"}

# --- Monte Carlo over repeated ladders at the default sizes
nmc = 400
est, err = [], []
for _ in range(nmc):
    f, t, _, _ = sigma_h0(CFG, rng)
    est.append(10 ** (t[ia] / 5)); err.append(f * 10 ** (t[ia] / 5))
est, err = np.array(est), np.array(err)
cover = np.mean(np.abs(est - TRUE["H0"]) < err)

# --- analytic approximation of sigma_a^2 rung by rung
c = CFG
approx = {
    "anc": c["sig_geo"] ** 2 / c["Nanc"],
    "ceph": c["sig_ceph"] ** 2 / (c["Nanc"] * c["Nceph_anc"]) + c["sig_ceph"] ** 2 / (c["Ncal"] * c["Nceph_cal"]),
    "sn": c["sig_sn"] ** 2 / c["Ncal"],
    "hf": c["sig_hf"] ** 2 / c["Nhf"],
}
approx_frac = KAPPA * np.sqrt(sum(approx.values()))

# --- figure
setup(7.0, 2.9)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
r2 = np.random.default_rng(1)
for i, (mu, lab, col) in enumerate([(18.48, "anchor (LMC-like)", SERIES[0]),
                                    (29.40, "anchor (maser host)", SERIES[2]),
                                    (32.0, "SN host", SERIES[1])]):
    lp = r2.uniform(0.6, 1.8, 40)
    mm = mu + TRUE["MW"] + TRUE["bW"] * (lp - 1) + r2.normal(0, CFG["sig_ceph"], 40)
    ax[0].scatter(lp, mm, s=5, color=col, label=lab)
    xx = np.array([0.55, 1.85])
    ax[0].plot(xx, mu + TRUE["MW"] + TRUE["bW"] * (xx - 1), color=col, lw=1)
ax[0].invert_yaxis()
ax[0].set(xlabel=r"$\log_{10}P$ [days]", ylabel=r"Cepheid magnitude $m_W$")
ax[0].legend(fontsize=7, frameon=False, loc="center right")
ax[1].plot(ncal_grid, 100 * budget["total"], "o-", color="k", label="total (GLS)")
for (k, v), col in zip(contrib.items(), SERIES):
    ax[1].plot(ncal_grid, 100 * v, "s--", ms=3, color=col, label=k.replace("perfect ", ""))
ax[1].axvline(CFG["Ncal"], color="0.6", lw=0.8, ls=":")
ax[1].set(xscale="log", yscale="log", xlabel=r"number of calibrator hosts $N_{\rm cal}$",
          ylabel=r"$\sigma_{H_0}/H_0$ [%]")
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax[1].yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}" if v in (0.4, 0.6, 2.0) else ""))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax[1].legend(fontsize=6.5, frameon=False, loc="upper right")
fig.tight_layout()
savefig(fig, "chT2", "distance_ladder")

i40 = list(ncal_grid).index(40)
save_numbers("chT2", "07_distance_ladder", {
    "TwlNanc": CFG["Nanc"], "TwlSigGeo": CFG["sig_geo"], "TwlNcephAnc": CFG["Nceph_anc"],
    "TwlNcephCal": CFG["Nceph_cal"], "TwlSigCeph": CFG["sig_ceph"], "TwlNcal": CFG["Ncal"],
    "TwlSigSn": CFG["sig_sn"], "TwlNhf": CFG["Nhf"], "TwlSigHf": CFG["sig_hf"],
    "TwlHtrue": TRUE["H0"], "TwlHhat": f"{H0hat:.2f}", "TwlHerr": f"{frac * H0hat:.2f}",
    "TwlFrac": f"{100 * frac:.2f}", "TwlFracApprox": f"{100 * approx_frac:.2f}",
    "TwlNpar": len(th),
    "TwlCanc": f"{100 * contrib['perfect anchors'][i40]:.2f}",
    "TwlCceph": f"{100 * contrib['perfect Cepheids'][i40]:.2f}",
    "TwlCsn": f"{100 * contrib['perfect calibrator SNe'][i40]:.2f}",
    "TwlChf": f"{100 * contrib['perfect Hubble flow'][i40]:.2f}",
    "TwlFracFive": f"{100 * budget['total'][0]:.2f}", "TwlFracOneSixty": f"{100 * budget['total'][-1]:.2f}",
    "TwlNmc": nmc, "TwlMcMean": f"{est.mean():.2f}", "TwlMcSd": f"{est.std(ddof=1):.2f}",
    "TwlMcErr": f"{err.mean():.2f}", "TwlCover": f"{100 * cover:.0f}\\%",
})
print(f"H0 = {H0hat:.2f} +- {frac*H0hat:.2f} ({100*frac:.2f}%, approx {100*approx_frac:.2f}%); "
      f"MC mean {est.mean():.2f} sd {est.std():.2f} vs err {err.mean():.2f}; cover {cover:.2f}")
print({k: np.round(100 * v, 2) for k, v in contrib.items()})
