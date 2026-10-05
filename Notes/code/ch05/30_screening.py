"""30_screening.py -- breast-cancer screening, test by test.

Question: a woman has a screening mammogram, then an ultrasound, then a biopsy.  After each
result, what is the probability that she has cancer?  Each posterior is the prior for the
next test.  All numbers are illustrative:
  prevalence 0.01;  mammogram sens 0.87, spec 0.90;  ultrasound sens 0.80, spec 0.90;
  biopsy sens 0.95, spec 0.99.
Computes: the posterior after every sequence of results (2^3 paths), the likelihood ratios
LR+ = sens/(1-spec) and LR- = (1-sens)/spec, the effect of a shared cause of false positives
(a benign feature that makes both imaging tests positive), and the numbers of the reader's
problem (two tests A and B, repeated positives, correlated false positives).  Every exact
number is checked by simulating a population of women and filtering on the results.
Writes: figures/ch05/screening_paths.pdf, figures/ch05/screening_squares.pdf,
        results/ch05/30_screening.tex
"""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import setup, savefig, save_numbers, rng_for, SERIES, INK2
from lib_area import box_diagram, posterior

import numpy as np
import matplotlib.pyplot as plt

rng = rng_for("ch05", "30_screening")
setup()

prev = 0.01
tests = [("mammogram", 0.87, 0.90), ("ultrasound", 0.80, 0.90), ("biopsy", 0.95, 0.99)]


def update(p, sens, spec, result):
    """One test: multiply by the likelihood of the result under {cancer, no cancer}, renormalise."""
    likes = [sens, 1 - spec] if result == "+" else [1 - sens, spec]
    return posterior([p, 1 - p], likes)[2][0]


def lr(sens, spec):
    return sens / (1 - spec), (1 - sens) / spec


# ---------- every sequence of results ----------
paths = {}
for res in itertools.product("+-", repeat=3):
    p, traj = prev, [prev]
    for (name, se, sp), r in zip(tests, res):
        p = update(p, se, sp, r)
        traj.append(p)
    paths["".join(res)] = traj

# ---------- shared cause of false positives for the two imaging tests ----------
f = 0.05                                     # healthy women with a benign feature: both images positive
rM = (0.10 - f) / (1 - f)                    # remaining random false-positive rates, chosen so that
rU = (0.10 - f) / (1 - f)                    # each test alone keeps its false-positive rate 0.10
pPP_h = f + (1 - f) * rM * rU
pPP_d = 0.87 * 0.80
odds0 = prev / (1 - prev)
odds_corr = odds0 * pPP_d / pPP_h
post_corr = odds_corr / (1 + odds_corr)

# ---------- Monte Carlo: a population of women, filtered on the results ----------
N = 4_000_000
cancer = rng.random(N) < prev
benign = (~cancer) & (rng.random(N) < f)    # used only for the correlated variant
results = []
for (name, se, sp) in tests:
    u = rng.random(N)
    results.append(np.where(cancer, u < se, u < 1 - sp))
mc = {}
for key in ["+", "++", "+-", "+++"]:
    mask = np.ones(N, bool)
    for k, r in enumerate(key):
        mask &= results[k] if r == "+" else ~results[k]
    mc[key] = cancer[mask].mean()
# correlated variant: benign feature -> both imaging tests positive; others independent at rate r
uM, uU = rng.random(N), rng.random(N)
M_c = np.where(cancer, uM < 0.87, benign | (uM < rM))
U_c = np.where(cancer, uU < 0.80, benign | (uU < rU))
mc_corr = cancer[M_c & U_c].mean()

# ---------- the reader's problem ----------
pi_ = 0.005
sA, cA = 0.90, 0.92          # test A: sensitivity, specificity
sB, cB = 0.85, 0.95          # test B
o0 = pi_ / (1 - pi_)
LApos, LAneg = lr(sA, cA)
LBpos, LBneg = lr(sB, cB)
prob = lambda o: o / (1 + o)
oApos = o0 * LApos
oAB = oApos * LBpos
oAb = oApos * LBneg
n_needed = int(np.ceil(np.log(19 / o0) / np.log(LApos)))
post_n = [prob(o0 * LApos**n) for n in range(1, n_needed + 1)]
# correlated false positives: a fraction g of healthy women carry a feature that makes both A and B
# positive; the others get independent false positives at rates chosen to keep each test's own rate
g = 0.03
qA = (1 - cA - g) / (1 - g)
qB = (1 - cB - g) / (1 - g)
pAB_h = g + (1 - g) * qA * qB
pAb_h = (1 - g) * qA * (1 - qB)
oAB_c = o0 * sA * sB / pAB_h
oAb_c = o0 * sA * (1 - sB) / pAb_h
# MC check of the problem (independent and correlated)
cP = rng.random(N) < pi_
featP = (~cP) & (rng.random(N) < g)
a1, b1 = rng.random(N), rng.random(N)
A_i = np.where(cP, a1 < sA, a1 < 1 - cA)
B_i = np.where(cP, b1 < sB, b1 < 1 - cB)
A_c = np.where(cP, a1 < sA, featP | (a1 < qA))
B_c = np.where(cP, b1 < sB, featP | (b1 < qB))
mcP_AB, mcP_Ab = cP[A_i & B_i].mean(), cP[A_i & ~B_i].mean()
mcP_ABc, mcP_Abc = cP[A_c & B_c].mean(), cP[A_c & ~B_c].mean()

# ---------- figure 1: the posterior along every sequence of results ----------
fig, ax = plt.subplots(figsize=(6.6, 3.8))
logit = lambda p: np.log10(p / (1 - p))
for key, traj in paths.items():
    for k in range(3):
        sub = key[:k + 1]
        col = SERIES[0] if sub[-1] == "+" else SERIES[1]
        ax.plot([k, k + 1], [logit(traj[k]), logit(traj[k + 1])], color=col, lw=1.2,
                marker="o", ms=3.5, zorder=3)
ends = sorted(paths.items(), key=lambda kv: logit(kv[1][-1]))
ylab = [logit(traj[-1]) for _, traj in ends]
for i in range(1, len(ylab)):            # keep end labels at least 0.32 decades apart
    ylab[i] = max(ylab[i], ylab[i - 1] + 0.32)
for (key, traj), yl in zip(ends, ylab):
    ax.text(3.08, yl, key.replace("-", "\u2212") + f": {traj[-1]:.3g}", va="center", fontsize=7.5)
ticks = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
ax.set_yticks([logit(t) for t in ticks])
ax.set_yticklabels([f"{t:g}" for t in ticks])
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["prior", "mammogram", "ultrasound", "biopsy"])
ax.set_xlim(-0.15, 3.75)
ax.set_ylabel("P(cancer)  (log-odds scale)")
ax.plot([], [], color=SERIES[0], label="positive result")
ax.plot([], [], color=SERIES[1], label="negative result")
ax.legend(fontsize=8, loc="upper left")
savefig(fig, "ch05", "screening_paths")

# ---------- figure 2: the sample space before each test, along the path + + + ----------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
p = prev
for k, ((name, se, sp), ax) in enumerate(zip(tests, axes)):
    # strips as wide as the current prior; the box is "this test is positive"
    inset = dict(x="box", bounds=[0.42, 0.42, 0.55, 0.5]) if p < 0.05 else None
    box_diagram(ax, [p, 1 - p], [se, 1 - sp], ["C", "no C"], "+",
                title=f"{name} +", fontsize=7, inset=inset)
    pn = update(p, se, sp, "+")
    ax.text(0.75, -0.24, f"prior {p:.3g}  ->  posterior {pn:.3g}", ha="center", va="top",
            fontsize=7.5)
    ax.set_ylim(-0.36, 1.14)
    p = pn
fig.subplots_adjust(wspace=0.1)
savefig(fig, "ch05", "screening_squares")

LRMp, LRMm = lr(0.87, 0.90)
LRUp, LRUm = lr(0.80, 0.90)
LRBp, LRBm = lr(0.95, 0.99)
save_numbers("ch05", "30_screening", {
    "FiveAScrLRMp": f"{LRMp:.2f}", "FiveAScrLRMm": f"{LRMm:.3f}",
    "FiveAScrLRUp": f"{LRUp:.2f}", "FiveAScrLRUm": f"{LRUm:.3f}",
    "FiveAScrLRBp": f"{LRBp:.1f}", "FiveAScrLRBm": f"{LRBm:.4f}",
    "FiveAScrP": f"{paths['+++'][1]:.4f}", "FiveAScrPP": f"{paths['+++'][2]:.3f}",
    "FiveAScrPM": f"{paths['+-+'][2]:.4f}", "FiveAScrPPP": f"{paths['+++'][3]:.4f}",
    "FiveAScrPPM": f"{paths['++-'][3]:.4f}", "FiveAScrPMP": f"{paths['+-+'][3]:.3f}",
    "FiveAScrM": f"{paths['-++'][1]:.5f}",
    "FiveAScrMCP": f"{mc['+']:.4f}", "FiveAScrMCPP": f"{mc['++']:.3f}",
    "FiveAScrMCPM": f"{mc['+-']:.4f}", "FiveAScrMCPPP": f"{mc['+++']:.4f}",
    "FiveAScrCorrHH": f"{pPP_h:.4f}", "FiveAScrCorrLR": f"{pPP_d / pPP_h:.1f}",
    "FiveAScrCorrPost": f"{post_corr:.3f}", "FiveAScrCorrMC": f"{mc_corr:.3f}",
    "FiveAPrbLAp": f"{LApos:.2f}", "FiveAPrbLBp": f"{LBpos:.1f}", "FiveAPrbLBm": f"{LBneg:.4f}",
    "FiveAPrbOzero": f"{o0:.6f}",
    "FiveAPrbA": f"{prob(oApos):.4f}", "FiveAPrbAB": f"{prob(oAB):.3f}", "FiveAPrbAb": f"{prob(oAb):.5f}",
    "FiveAPrbOAB": f"{oAB:.4f}", "FiveAPrbOAb": f"{oAb:.6f}",
    "FiveAPrbN": n_needed, "FiveAPrbNminus": f"{post_n[-2]:.3f}", "FiveAPrbNpost": f"{post_n[-1]:.4f}",
    "FiveAPrbqA": f"{qA:.5f}", "FiveAPrbqB": f"{qB:.5f}",
    "FiveAPrbHAB": f"{pAB_h:.5f}", "FiveAPrbHAb": f"{pAb_h:.5f}",
    "FiveAPrbABc": f"{prob(oAB_c):.3f}", "FiveAPrbAbc": f"{prob(oAb_c):.4f}",
    "FiveAPrbLRABc": f"{sA * sB / pAB_h:.1f}", "FiveAPrbLRABi": f"{LApos * LBpos:.0f}",
    "FiveAPrbMCAB": f"{mcP_AB:.3f}", "FiveAPrbMCAb": f"{mcP_Ab:.5f}",
    "FiveAPrbMCABc": f"{mcP_ABc:.3f}", "FiveAPrbMCAbc": f"{mcP_Abc:.4f}",
})
