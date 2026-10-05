"""25_how_many_bumps.py -- the posterior probability of the number of bumps.

Question: how many Gaussian bumps (of known width) sit on the straight line?
Each number M = 0, 1, 2 is a model with its own parameters: the line (a, b),
and an amplitude A_j in [0, A_max] and position mu_j in [0, 10] per bump.  With
equal prior probabilities for M, p(M | data) is proportional to the evidence.

For each (A_1, mu_1, A_2, mu_2) the line is integrated out exactly, as in
lib_bump; the remaining 2M-dimensional integral is a grid sum.  The chi^2
after projecting out the line is a quadratic form in the amplitudes, so the
grid costs only multiplications.

Computes ln Z_M and p(M | data) for the one-bump data of 22_line_or_bump.py and
for data with two bumps; also BIC and AIC for comparison.

Writes: figures/ch05/how_many_bumps.pdf, results/ch05/25_how_many_bumps.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import lib_bump as lb

setup()
TWO = dict(A1=2.5, mu1=4.0, A2=2.5, mu2=5.4)


def two_bump_data(rng):
    t = lb.TRUE
    mean = t["a"] + t["b"] * lb.X + TWO["A1"] * lb.bump(lb.X, TWO["mu1"]) + TWO["A2"] * lb.bump(lb.X, TWO["mu2"])
    return mean + lb.SIGMA * rng.standard_normal(lb.N_DATA)


def ln_B_two(y, nA=81, nmu=161):
    """ln Z(2 bumps) - ln Z(line), and the smallest chi^2 found on the grid (relative to the line)."""
    A = np.linspace(0, lb.A_MAX, nA); mu = np.linspace(*lb.MU_RANGE, nmu)
    ry = lb.PPERP @ y
    U = lb.PPERP @ lb.bump(lb.X[:, None], mu[None, :])
    uy = ry @ U / lb.SIGMA**2                                   # (nmu,)
    UU = U.T @ U / lb.SIGMA**2                                  # (nmu, nmu)
    AA1, AA2 = np.meshgrid(A, A, indexing="ij")                 # (nA, nA)
    wA = np.trapezoid(np.eye(nA), A, axis=0)                    # trapezoid weights along A
    wmu = np.trapezoid(np.eye(nmu), mu, axis=0)
    total, best = 0.0, np.inf
    for i in range(nmu):                                        # loop over mu_1, vectorise the rest
        dc = (-2 * AA1[None] * uy[i] - 2 * AA2[None] * uy[:, None, None]
              + AA1[None] ** 2 * UU[i, i] + 2 * AA1[None] * AA2[None] * UU[i, :, None, None]
              + AA2[None] ** 2 * np.diag(UU)[:, None, None])    # (nmu_2, nA, nA)
        best = min(best, dc.min())
        f = np.exp(-0.5 * dc)
        total += wmu[i] * np.einsum("kab,a,b,k->", f, wA, wA, wmu)
    return np.log(total / (lb.A_MAX * lb.L_MU) ** 2), best


out, lnp = {}, {}
sets = {"one": lb.make_data(rng_for("ch05", "22_line_or_bump", 0), True),
        "two": two_bump_data(rng_for("ch05", "25_how_many_bumps", 0))}
for key, y in sets.items():
    lnZ0 = lb.ln_evidence_line(y)
    lnB1, A, mu, dc1 = lb.ln_bayes_factor(y, return_grid=True)
    lnB2, best2 = ln_B_two(y)
    lnZ = np.array([lnZ0, lnZ0 + lnB1, lnZ0 + lnB2])
    lnp[key] = lnZ - logsumexp(lnZ)
    chi = np.array([lb.chi2min_line(y), lb.chi2min_line(y) + dc1.min(), lb.chi2min_line(y) + best2])
    k = np.array([2, 4, 6])
    bic = chi + k * np.log(lb.N_DATA); aic = chi + 2 * k
    tag = "One" if key == "one" else "Two"
    for M in range(3):
        nm = ["Zero", "One", "Two"][M]
        out[f"FiveCHm{tag}lnZ{nm}"] = lnZ[M]
        out[f"FiveCHm{tag}P{nm}"] = np.exp(lnp[key][M])
        out[f"FiveCHm{tag}Chi{nm}"] = chi[M]
        out[f"FiveCHm{tag}BIC{nm}"] = bic[M]
        out[f"FiveCHm{tag}AIC{nm}"] = aic[M]
    out[f"FiveCHm{tag}lnBtwoone"] = lnZ[2] - lnZ[1]
    out[f"FiveCHm{tag}lnBonezero"] = lnZ[1] - lnZ[0]
    out[f"FiveCHm{tag}BICtwoone"] = -0.5 * (bic[2] - bic[1])
    out[f"FiveCHm{tag}BIConezero"] = -0.5 * (bic[1] - bic[0])

# convergence check of the 4-D grid on the two-bump data
lnB2_fine, _ = ln_B_two(sets["two"], nA=101, nmu=201)
out["FiveCHmGridErr"] = abs(lnB2_fine - (out["FiveCHmTwolnZTwo"] - out["FiveCHmTwolnZZero"]))


# ---- one data set of each kind could be a fluke: repeat on fresh data sets
def best_dchi2(y, nmu=401):
    """Smallest chi^2 minus the line's, for one bump and for two bumps (amplitudes >= 0), exact
    in the amplitudes and on a fine grid in the positions."""
    mu = np.linspace(*lb.MU_RANGE, nmu)
    U = lb.PPERP @ lb.bump(lb.X[:, None], mu[None, :])
    u = (lb.PPERP @ y) @ U / lb.SIGMA**2
    Q = U.T @ U / lb.SIGMA**2
    q = np.diag(Q)
    one = -np.maximum(u, 0.0) ** 2 / q                          # best single bump at each position
    det = q[:, None] * q[None, :] - Q**2
    with np.errstate(divide="ignore", invalid="ignore"):
        A1 = (q[None, :] * u[:, None] - Q * u[None, :]) / det    # 2x2 normal equations, pair (i, j)
        A2 = (q[:, None] * u[None, :] - Q * u[:, None]) / det
        two = -(A1 * u[:, None] + A2 * u[None, :])
    ok = (det > 1e-9 * q[:, None] * q[None, :]) & (A1 >= 0) & (A2 >= 0)
    return one.min(), min(np.where(ok, two, np.inf).min(), one.min())   # boundary = one bump


R_LOOP = 100
pick = {"one": np.zeros(3, int), "two": np.zeros(3, int)}         # how often the evidence picks M
aic_two = bic_two = ev_two = 0
for key, make in (("one", lambda g: lb.make_data(g, True)), ("two", two_bump_data)):
    for r in range(R_LOOP):
        y = make(rng_for("ch05", "25_how_many_bumps_loop_" + key, r))
        lnB1 = lb.ln_bayes_factor(y, nA=401, nmu=1001)
        lnB2 = ln_B_two(y, nA=41, nmu=101)[0]
        pick[key][np.argmax([0.0, lnB1, lnB2])] += 1
        if key == "one":
            ev_two += lnB2 > lnB1                                     # evidence: two bumps over one
            d1, d2 = best_dchi2(y)
            aic_two += (d2 + 2 * 2) < d1                          # AIC: 2 per extra parameter
            bic_two += (d2 + 2 * np.log(lb.N_DATA)) < d1          # BIC: ln N per extra parameter
out.update({"FiveCHmLoopR": R_LOOP,
            "FiveCHmLoopOneEvOne": int(pick["one"][1]), "FiveCHmLoopOneEvTwo": int(ev_two),
            "FiveCHmLoopTwoEvTwo": int(pick["two"][2]),
            "FiveCHmLoopOneAicTwo": int(aic_two), "FiveCHmLoopOneBicTwo": int(bic_two)})

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"wspace": 0.35})
y = sets["two"]
axes[0].errorbar(lb.X, y, yerr=lb.SIGMA, fmt="o", ms=3, color="0.35", elinewidth=0.6)
xx = np.linspace(0, 10, 400)
axes[0].plot(xx, lb.TRUE["a"] + lb.TRUE["b"] * xx + TWO["A1"] * lb.bump(xx, TWO["mu1"])
             + TWO["A2"] * lb.bump(xx, TWO["mu2"]), color="k", ls="--", lw=1.1, label="true mean")
axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$y$"); axes[0].legend(fontsize=7, loc="upper left")
axes[0].set_title("data with two bumps", fontsize=9)
w = 0.38
for c, (key, v), dx in zip(SERIES, lnp.items(), [-w / 2, w / 2]):
    axes[1].bar(np.arange(3) + dx, v / np.log(10), width=w, color=c,
                label="one-bump data" if key == "one" else "two-bump data")
axes[1].set_xticks(range(3)); axes[1].set_xlabel("number of bumps $M$")
axes[1].set_ylabel(r"$\log_{10} p(M\,|\,\mathrm{data})$"); axes[1].legend(fontsize=7, loc="lower right")
savefig(fig, "ch05", "how_many_bumps")

out.update({"FiveCTwoMuOne": TWO["mu1"], "FiveCTwoMuTwo": TWO["mu2"], "FiveCTwoA": TWO["A1"]})
save_numbers("ch05", "25_how_many_bumps", out)
for k, v in out.items():
    print(f"{k:26s} {v:.4g}")
