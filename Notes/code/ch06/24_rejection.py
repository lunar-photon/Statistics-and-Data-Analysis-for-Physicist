"""24_rejection.py -- accept-reject sampling, and the acceptance rate 1/M.

Question: how do we sample a density f whose CDF we cannot invert?  Draw a
candidate V from an envelope density g with f <= M g, draw U uniform, and keep
V if U < f(V) / (M g(V)).  The kept values follow f, a fraction 1/M of the
candidates is kept, and the number of candidates per kept value is geometric
with mean M.

Three targets:
  1. the e+e- -> mu+mu- angular distribution f(x) = (3/8)(1 + x^2), x = cos(theta),
     in a box of height 3/4 on [-1, 1] (M = 3/2, acceptance 2/3);
  2. the half-normal f(x) = sqrt(2/pi) exp(-x^2/2), x > 0, under the exponential
     envelope g(x) = exp(-x) with M = sqrt(2 e / pi);
  3. Beta(2.7, 6.3) in the unit box of height c = max f = 2.669 (acceptance 1/c).

Writes: figures/ch06/rejection.pdf, figures/ch06/rejection_trials.pdf,
        results/ch06/24_rejection.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

setup(9.0, 3.0)
rng = rng_for("ch06", "24_rejection")
N = 200_000
out = {"SixBRjN": N}

# ---------------------------------------------------------------- 1. muon-pair angle in a box
f_mu = lambda x: 0.375 * (1.0 + x**2)
xc = 2.0 * rng.random(N) - 1.0                 # candidate, uniform on [-1, 1]
uc = 0.75 * rng.random(N)                      # height, uniform on [0, f_max]
acc1 = uc < f_mu(xc)
out.update(SixBRjMuAcc=acc1.mean(), SixBRjMuMeanSq=np.mean(xc[acc1] ** 2))   # exact E x^2 = 2/5

# ---------------------------------------------------------------- 2. half-normal under an exponential
M2 = np.sqrt(2.0 * np.e / np.pi)
v = -np.log(1.0 - rng.random(N))               # candidate from g(x) = e^{-x} (inverse CDF)
u = rng.random(N)
ratio = np.sqrt(2.0 / np.pi) * np.exp(-v**2 / 2) / (M2 * np.exp(-v))
acc2 = u < ratio
hn = v[acc2]
out.update(SixBRjHnM=M2, SixBRjHnInvM=1 / M2, SixBRjHnAcc=acc2.mean(),
           SixBRjHnMean=hn.mean(), SixBRjHnExactMean=np.sqrt(2 / np.pi))

# trial counts: candidates needed per accepted value
idx = np.flatnonzero(acc2)
trials = np.diff(np.concatenate(([-1], idx)))
out.update(SixBRjTrialsMean=trials.mean())

# ---------------------------------------------------------------- 3. Beta(2.7, 6.3) in a box
a, b = 2.7, 6.3
mode = (a - 1) / (a + b - 2)
cmax = stats.beta.pdf(mode, a, b)
vb, ub = rng.random(N), rng.random(N)
acc3 = ub < stats.beta.pdf(vb, a, b) / cmax
out.update(SixBRjBetaC=cmax, SixBRjBetaAcc=acc3.mean(), SixBRjBetaInvC=1 / cmax,
           SixBRjBetaMean=vb[acc3].mean(), SixBRjBetaExactMean=a / (a + b))

# ---------------------------------------------------------------- figure 1: geometry + histograms
fig, ax = plt.subplots(1, 3)
m = 3000
ax[0].scatter(xc[:m][acc1[:m]], uc[:m][acc1[:m]], s=1.5, color=SERIES[0], label="accepted")
ax[0].scatter(xc[:m][~acc1[:m]], uc[:m][~acc1[:m]], s=1.5, color=SERIES[1], label="rejected")
xx = np.linspace(-1, 1, 200)
theory_line(ax[0], xx, f_mu(xx), label=r"$f(x)$")
ax[0].set(xlabel=r"$x=\cos\theta$", ylabel=r"$u$", title=r"box of height $3/4$")
ax[0].legend(loc="upper center", fontsize=7, ncol=3, markerscale=4)
ax[0].set_ylim(0, 1.0)
ax[1].hist(xc[acc1], bins=40, density=True, color=SERIES[0], alpha=0.55, label="accepted")
theory_line(ax[1], xx, f_mu(xx), label=r"$\frac{3}{8}(1+x^2)$")
ax[1].set(xlabel=r"$x=\cos\theta$", ylabel="density", title="muon-pair angle", ylim=(0, 0.9))
ax[1].legend(fontsize=8, loc="upper center")
zz = np.linspace(0, 4.5, 300)
ax[2].hist(hn, bins=np.linspace(0, 4.5, 46), density=True, color=SERIES[0], alpha=0.55, label="accepted")
theory_line(ax[2], zz, np.sqrt(2 / np.pi) * np.exp(-zz**2 / 2), label="half-normal")
ax[2].plot(zz, M2 * np.exp(-zz), color=SERIES[2], lw=1.3, label=r"envelope $Mg$")
ax[2].set(xlabel=r"$x$", ylabel="density", title="exponential envelope")
ax[2].legend(fontsize=8)
fig.tight_layout()
savefig(fig, "ch06", "rejection")

# ---------------------------------------------------------------- figure 2: trials are geometric
setup(4.6, 3.0)
fig, ax = plt.subplots()
kk = np.arange(1, 9)
freq = np.array([np.mean(trials == k) for k in kk])
ax.bar(kk, freq, color=SERIES[0], alpha=0.7, label="simulation")
p = 1 / M2
ax.plot(kk, p * (1 - p) ** (kk - 1), "k_", ms=14, mew=1.6, label=r"geometric, $p=1/M$")
ax.set(xlabel="candidates needed per accepted value", ylabel="probability", yscale="log")
ax.legend()
fig.tight_layout()
savefig(fig, "ch06", "rejection_trials")
save_numbers("ch06", "24_rejection", out)
print(out)
