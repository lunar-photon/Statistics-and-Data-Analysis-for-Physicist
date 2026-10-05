"""10_convergence.py -- how many simulations does the Monte Carlo need?

Question: as the number of simulated skies N grows, how fast do the Monte Carlo estimates of the
bias and of the error bar settle, and how many skies give the error bar to 10 percent?
Computes: from the 2000 skies of 05_mc_run, running estimates (in the multipole bin 400-424) of the
mean of C_hat/C - 1 with its 1/sqrt(N) envelope, and of sigma(C_hat), whose relative error should
fall as 1/sqrt(2(N-1)); the same for 200 reshuffled orderings to show the spread.
Writes: figures/ch08/convergence.pdf, results/ch08/10_convergence.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from camb_fiducial import load_fiducial
import lib_cmbsim as cs

setup()
ell_all, cl_all = load_fiducial()
exp = cs.Experiment()
L = exp.lmax
sims = cs.cached("mc_fullsky", lambda: (_ for _ in ()).throw(RuntimeError("run 05_mc_run.py first")))
cl = cl_all[: L + 1]
chat = (sims["auto"].astype(float) - exp.nl(L)) / exp.transfer(L) ** 2
nsim = chat.shape[0]
a, b = 400, 425                                       # one bin of 25 multipoles
x = chat[:, a:b].mean(1) / cl[a:b].mean() - 1          # fractional error of the binned estimate
sig_true = np.sqrt(cs.var_fullsky(cl_all, exp)[a:b].sum()) / (b - a) / cl[a:b].mean()

rng = rng_for("ch08", "10_convergence")
Ns = np.unique(np.geomspace(5, nsim, 40).astype(int))
nperm = 200
mean_runs = np.empty((nperm, Ns.size)); sd_runs = np.empty((nperm, Ns.size))
for k in range(nperm):
    xp = rng.permutation(x)
    for j, n in enumerate(Ns):
        mean_runs[k, j] = xp[:n].mean()
        sd_runs[k, j] = xp[:n].std(ddof=1)

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
for k in range(5):
    axs[0].semilogx(Ns, 100 * mean_runs[k], color=SERIES[0], lw=0.7, alpha=0.6)
theory_line(axs[0], Ns, 100 * sig_true / np.sqrt(Ns), label=r"$\pm\sigma/\sqrt{N}$")
axs[0].semilogx(Ns, -100 * sig_true / np.sqrt(Ns), color="k", ls="--", lw=1.4)
axs[0].axhline(0, color="0.6", lw=0.8)
axs[0].set_xlabel(r"number of simulated skies $N$"); axs[0].set_ylabel(r"running bias estimate [%]")
axs[0].legend(fontsize=7.5)
rel = sd_runs / sig_true - 1
axs[1].loglog(Ns[:-1], rel.std(0)[:-1], "o", ms=3, color=SERIES[1], label="spread over orderings")
theory_line(axs[1], Ns, 1 / np.sqrt(2 * (Ns - 1)), label=r"$1/\sqrt{2(N-1)}$")
axs[1].axhline(0.1, color="0.6", ls=":", lw=1)
axs[1].set_xlabel(r"$N$"); axs[1].set_ylabel(r"relative error of $\hat\sigma$")
axs[1].legend(fontsize=7.5)
fig.tight_layout()
savefig(fig, "ch08", "convergence")

n10 = int(np.ceil(1 + 1 / (2 * 0.1 ** 2)))
j100 = int(np.argmin(np.abs(Ns - 100)))       # well below 2000: the reshuffled subsamples barely overlap
n1pc = int(np.ceil((sig_true / 0.001) ** 2))
save_numbers("ch08", "10_convergence", {
    "EightASigBin": f"{100 * sig_true:.2f}",
    "EightANTenPct": n10,
    "EightANOnePermil": n1pc,
    "EightARelErrN": f"{100 * rel.std(0)[j100]:.1f}",
    "EightARelErrNval": int(Ns[j100]),
    "EightARelErrNth": f"{100 / np.sqrt(2 * (Ns[j100] - 1)):.1f}",
})
print("sig bin", sig_true, "n10", n10, "n for 0.1% bias", n1pc, "check", Ns[j100], rel.std(0)[j100], 1 / np.sqrt(2 * (Ns[j100] - 1)))
