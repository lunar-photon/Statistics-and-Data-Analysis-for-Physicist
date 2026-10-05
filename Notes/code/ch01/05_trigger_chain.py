"""A two-level trigger as a probability tree: the law of total probability.

Question: a fraction s of collisions contains a real muon (S), the rest do not
(B). Level-1 (L1) accepts S with probability e1S and B with e1B; the high-level
trigger (HLT) then accepts with e2S or e2B *given* that L1 fired. What fraction
of all collisions is written to disk, and how much of it comes through each
path of the tree?

Computes: the exact answer as a sum over tree paths, and a simulation that
follows each collision down the tree; the contribution of each path.
Writes: figures/ch01/1a_trigger_chain.pdf, results/ch01/1a_trigger_chain.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, SERIES

import numpy as np
import matplotlib.pyplot as plt

s = 0.01                    # P(S): collision contains a real muon
e1S, e1B = 0.95, 0.02       # P(L1 | S), P(L1 | B)
e2S, e2B = 0.90, 0.10       # P(HLT | L1, S), P(HLT | L1, B)

# --- exact: multiply along each path, add the paths that end in "accepted"
path_S = s * e1S * e2S
path_B = (1 - s) * e1B * e2B
p_acc = path_S + path_B
purity = path_S / p_acc      # used again in the Bayes section

# --- simulation: follow every collision down the tree
rng = rng_for("ch01", "1a_trigger_chain")
N = 2_000_000
is_S = rng.random(N) < s
l1 = np.where(is_S, rng.random(N) < e1S, rng.random(N) < e1B)
hlt = l1 & np.where(is_S, rng.random(N) < e2S, rng.random(N) < e2B)
mc_acc = hlt.mean()
mc_path_S = (hlt & is_S).mean()
mc_path_B = (hlt & ~is_S).mean()
mc_purity = is_S[hlt].mean()        # conditioning = filtering on "accepted"

setup(6.0, 3.4)
fig, ax = plt.subplots()
labels = ["exact (sum over paths)", f"simulation, $N=2\\times10^6$"]
xs = np.arange(2)
bS = np.array([path_S, mc_path_S])
bB = np.array([path_B, mc_path_B])
ax.bar(xs, bS * 100, width=0.5, color=SERIES[0], label=r"path $S\to$L1$\to$HLT")
ax.bar(xs, bB * 100, width=0.5, bottom=bS * 100, color=SERIES[1],
       label=r"path $B\to$L1$\to$HLT")
ax.set_xticks(xs, labels)
ax.set_ylabel("accepted collisions (%)")
ax.set_ylim(0, 1.5)
ax.legend(loc="upper center", ncol=2, fontsize=8)
fig.tight_layout()
savefig(fig, "ch01", "1a_trigger_chain")

save_numbers("ch01", "1a_trigger_chain", {
    "OneATrigPathS": f"{path_S:.5f}", "OneATrigPathB": f"{path_B:.5f}",
    "OneATrigAcc": f"{p_acc:.5f}", "OneATrigPurity": f"{purity:.4f}",
    "OneATrigMCAcc": f"{mc_acc:.5f}", "OneATrigMCPurity": f"{mc_purity:.4f}",
    "OneATrigMCPathS": f"{mc_path_S:.5f}", "OneATrigMCPathB": f"{mc_path_B:.5f}",
    "OneATrigNacc": int(hlt.sum()),
})
print(p_acc, mc_acc, purity, mc_purity)
