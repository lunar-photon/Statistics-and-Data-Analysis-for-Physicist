"""The whole pipeline on a coin: model -> data -> summary -> sampling distribution -> inference.

Question: once the toss sequence is compressed into the summary S = k (number of heads),
how do the sampling distribution P(k | p) and hence the posterior for p behave as the
number of tosses grows?  And did the compression lose anything?
Computes: (a) the likelihood of two *different* sequences with the same k, to show they are
identical functions of p (k is all that matters); (b) the flat-prior posterior for p after
n = 10, 100, 1000 tosses of a coin with true p = 0.8, and its width.
Writes:   figures/ch00/pipeline_coin.pdf, results/ch00/03_pipeline_coin.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from common import setup, savefig, save_numbers, rng_for, SERIES

P_TRUE = 0.8
setup(7.0, 3.0)
rng = rng_for("ch00", "03_pipeline_coin")
p = np.linspace(0, 1, 2001)
dp = p[1] - p[0]


def seq_likelihood(x, p):
    """P(x_1..x_n | p) = prod p^x_i (1-p)^(1-x_i) for a 0/1 sequence x."""
    return np.prod([p**xi * (1 - p) ** (1 - xi) for xi in x], axis=0)


# (a) two different sequences, both with 8 heads in 10
seqA = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
seqB = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1])
LA, LB = seq_likelihood(seqA, p), seq_likelihood(seqB, p)
max_diff = np.max(np.abs(LA - LB))

# (b) posteriors from one long run of tosses, looked at after n = 10, 100, 1000
tosses = rng.random(1000) < P_TRUE
fig, (a1, a2) = plt.subplots(1, 2)
a1.plot(p, LA, color=SERIES[0], label="HHHHHHHHTT")
a1.plot(p, LB, color=SERIES[1], ls=(0, (4, 3)), label="THHTHHHHHH")
a1.set(xlabel="$p$", ylabel="$P(\\mathrm{sequence}\\mid p)$",
       title="two sequences, same $k=8$: same likelihood")
a1.legend(fontsize=8)
out = {"PipeMaxDiff": max_diff}
names = {10: "Ten", 100: "Hundred", 1000: "Thousand"}
for i, n in enumerate([10, 100, 1000]):
    kk = int(tosses[:n].sum())
    logpost = kk * np.log(np.clip(p, 1e-300, None)) + (n - kk) * np.log(np.clip(1 - p, 1e-300, None))
    post = np.exp(logpost - logpost.max())
    post /= post.sum() * dp                      # flat prior: posterior = normalised likelihood
    mean = np.sum(p * post) * dp
    sd = np.sqrt(np.sum((p - mean) ** 2 * post) * dp)
    a2.plot(p, post, color=SERIES[i], label=f"$n={n}$, $k={kk}$")
    out[f"PipeK{names[n]}"] = kk
    out[f"PipeSd{names[n]}"] = sd
    out[f"PipeSdTh{names[n]}"] = np.sqrt((kk / n) * (1 - kk / n) / n)
a2.axvline(P_TRUE, color="k", ls="--", lw=1.2)
a2.set(xlabel="$p$", ylabel="$p(p\\mid k)$, flat prior", title="inference sharpens with $n$")
a2.legend(fontsize=8)
savefig(fig, "ch00", "pipeline_coin")
# exact flat-prior (Beta) posterior sd at n = 10, for the k this run produced
k10 = out["PipeKTen"]
out["PipeSdExactTen"] = f"{np.sqrt((k10 + 1) * (10 - k10 + 1) / (12**2 * 13)):.3f}"
save_numbers("ch00", "03_pipeline_coin", out)
