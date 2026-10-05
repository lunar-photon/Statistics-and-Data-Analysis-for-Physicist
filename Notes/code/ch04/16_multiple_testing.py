"""16_multiple_testing.py -- many tests at once: family-wise error, Bonferroni and Benjamini-Hochberg.

Question: if we run m tests, each at level alpha, how often does at least one fire when every
null is true?  How do Bonferroni (control the chance of any false rejection) and
Benjamini-Hochberg (control the expected fraction of false rejections) trade discoveries
against false alarms, when some of the nulls are false?

Computes: Kruschke's experimentwise rate for 36 comparisons; Wasserman's Example 10.28 (ten
p-values) under Bonferroni and BH; a simulation of m = 1000 one-sided z-tests of which m1 = 100
have a real effect, repeated 2000 times, giving the family-wise error rate, the false discovery
proportion and the number of true discoveries of three procedures.

Writes: figures/ch04/bh_procedure.pdf, results/ch04/16_multiple_testing.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch04", "16_multiple_testing")
setup()
alpha = 0.05

# ---------- experimentwise error (Kruschke 11.4.1) ----------
c = 36
alpha_ew = 1 - (1 - alpha) ** c
alpha_pc_bonf = alpha / c
alpha_ew_bonf = 1 - (1 - alpha_pc_bonf) ** c
alpha_pc_sidak = 1 - (1 - alpha) ** (1 / c)

# ---------- Wasserman Example 10.28 ----------
p_ex = np.array([0.00017, 0.00448, 0.00671, 0.00907, 0.01220,
                 0.33626, 0.39341, 0.53882, 0.58125, 0.98617])

def bonferroni(p, a=alpha):
    return p < a / p.size

def bh(p, a=alpha, dependent=False):
    """Benjamini-Hochberg: largest i with P_(i) < i a / (C_m m); reject all P <= that P_(i)."""
    m = p.size
    cm = np.sum(1.0 / np.arange(1, m + 1)) if dependent else 1.0
    ps = np.sort(p)
    below = ps < np.arange(1, m + 1) * a / (cm * m)
    if not below.any():
        return np.zeros(m, bool)
    T = ps[np.nonzero(below)[0].max()]
    return p <= T

n_bonf_ex = int(bonferroni(p_ex).sum())
n_bh_ex = int(bh(p_ex).sum())
n_raw_ex = int((p_ex < alpha).sum())

# ---------- simulation: 1000 tests, 100 real effects ----------
m, m1, shift, R = 1000, 100, 3.0, 2000
is_real = np.zeros(m, bool); is_real[:m1] = True
stats_acc = {k: {"V": [], "S": [], "R": []} for k in ("raw", "bonf", "bh")}
p_show = None
for r in range(R):
    z = rng.standard_normal(m) + shift * is_real
    p = stats.norm.sf(z)                               # one-sided p-values
    if r == 0:
        p_show = p.copy()
    for name, rej in (("raw", p < alpha), ("bonf", bonferroni(p)), ("bh", bh(p))):
        V = np.sum(rej & ~is_real); S = np.sum(rej & is_real)
        stats_acc[name]["V"].append(V); stats_acc[name]["S"].append(S)
        stats_acc[name]["R"].append(V + S)

summ = {}
for name, d in stats_acc.items():
    V, S, Rr = map(np.array, (d["V"], d["S"], d["R"]))
    fdp = np.where(Rr > 0, V / np.maximum(Rr, 1), 0.0)
    summ[name] = dict(fwer=np.mean(V > 0), fdr=fdp.mean(), S=S.mean(), V=V.mean())

# all nulls true: family-wise error by simulation (m = 1000)
Rn = 5000
fw_raw = fw_bonf = fw_bh = 0
for r in range(Rn):
    p = rng.uniform(size=m)
    fw_raw += (p < alpha).any(); fw_bonf += bonferroni(p).any(); fw_bh += bh(p).any()
fw_raw, fw_bonf, fw_bh = fw_raw / Rn, fw_bonf / Rn, fw_bh / Rn

# ---------- figure: the BH procedure on the first simulated family ----------
ps = np.sort(p_show)
k = np.arange(1, m + 1)
line = k * alpha / m
below = ps < line
iT = np.nonzero(below)[0].max()
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axs[0]
ax.loglog(k, ps, ".", color="0.35", ms=2.5, label="ordered p-values")
ax.loglog(k, line, color=SERIES[0], label=r"BH line $i\alpha/m$")
ax.axhline(alpha, color=SERIES[1], ls="--", lw=1.0, label=r"uncorrected $\alpha$")
ax.axhline(alpha / m, color=SERIES[2], ls="-.", lw=1.0, label=r"Bonferroni $\alpha/m$")
ax.plot(k[iT], ps[iT], "o", mfc="none", color="k", ms=7)
ax.set_xlabel("rank $i$"); ax.set_ylabel(r"$P_{(i)}$")
ax.set_ylim(1e-9, 1.5); ax.legend(fontsize=7, loc="lower right")
ax = axs[1]
names = ["uncorrected", "Bonferroni", "BH"]
keys = ["raw", "bonf", "bh"]
xs = np.arange(3)
ax.bar(xs - 0.2, [summ[q]["S"] for q in keys], 0.4, color=SERIES[0], label="true discoveries")
ax.bar(xs + 0.2, [summ[q]["V"] for q in keys], 0.4, color=SERIES[1], label="false discoveries")
ax.set_xticks(xs); ax.set_xticklabels(names)
ax.set_ylabel(f"mean count (m = {m}, {m1} real)")
ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
savefig(fig, "ch04", "bh_procedure")

save_numbers("ch04", "16_multiple_testing", {
    "FourBMtC": c, "FourBMtAlphaEW": f"{alpha_ew:.2f}", "FourBMtBonfPC": f"{alpha_pc_bonf:.5f}",
    "FourBMtBonfEW": f"{alpha_ew_bonf:.4f}", "FourBMtSidakPC": f"{alpha_pc_sidak:.5f}",
    "FourBMtExRaw": n_raw_ex, "FourBMtExBonf": n_bonf_ex, "FourBMtExBH": n_bh_ex,
    "FourBMtM": m, "FourBMtMone": m1, "FourBMtShift": f"{shift:.0f}", "FourBMtReps": R,
    "FourBMtRawS": f"{summ['raw']['S']:.1f}", "FourBMtRawV": f"{summ['raw']['V']:.1f}",
    "FourBMtRawFDR": f"{summ['raw']['fdr']:.3f}", "FourBMtRawFWER": f"{summ['raw']['fwer']:.3f}",
    "FourBMtBonfS": f"{summ['bonf']['S']:.1f}", "FourBMtBonfV": f"{summ['bonf']['V']:.2f}",
    "FourBMtBonfFDR": f"{summ['bonf']['fdr']:.4f}", "FourBMtBonfFWER": f"{summ['bonf']['fwer']:.3f}",
    "FourBMtBHS": f"{summ['bh']['S']:.1f}", "FourBMtBHV": f"{summ['bh']['V']:.1f}",
    "FourBMtBHFDR": f"{summ['bh']['fdr']:.4f}", "FourBMtBHFWER": f"{summ['bh']['fwer']:.3f}",
    "FourBMtBHBound": f"{(m - m1) / m * alpha:.3f}",
    "FourBMtNullRaw": f"{fw_raw:.3f}", "FourBMtNullBonf": f"{fw_bonf:.4f}", "FourBMtNullBH": f"{fw_bh:.4f}",
    "FourBMtNullReps": Rn,
})
print(f"alpha_EW(36) = {alpha_ew:.4f}; Bonferroni per-comparison {alpha_pc_bonf:.5f} -> EW {alpha_ew_bonf:.4f}")
print(f"Ex 10.28: raw {n_raw_ex}, Bonferroni {n_bonf_ex}, BH {n_bh_ex}")
for q in keys:
    print(q, {k2: round(v, 4) for k2, v in summ[q].items()})
print(f"all-null FWER: raw {fw_raw:.3f} bonf {fw_bonf:.4f} bh {fw_bh:.4f}")
