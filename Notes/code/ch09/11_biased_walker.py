"""From the unbiased random walker to the biased one.

Question: a walker proposes Gaussian steps of size s on the real line.  What does it sample if
(a) it always accepts (the unbiased random walk, the "drunkard"), (b) it accepts only uphill
steps (the "hill climber"), (c) it accepts with min(1, f(y)/f(x)) (Metropolis), and
(d) it accepts with min(1, [f(y)/f(x)]^beta) for other exponents beta?
Target: a two-bump density f(x) = 0.3 N(-2, 0.6^2) + 0.7 N(2, 0.8^2).
Writes figures/ch09/biased_walker.pdf, figures/ch09/tempered_family.pdf,
results/ch09/11_biased_walker.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line

W1, M1, S1, W2, M2, S2 = 0.3, -2.0, 0.6, 0.7, 2.0, 0.8


def f(x):
    """The target density (normalised here only so that we can draw it; the walker never uses it)."""
    g = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return W1 * g(M1, S1) + W2 * g(M2, S2)


def logf(x):
    """log f, computed without underflow far from the bumps."""
    a = np.log(W1 / (S1 * np.sqrt(2 * np.pi))) - 0.5 * ((x - M1) / S1) ** 2
    b = np.log(W2 / (S2 * np.sqrt(2 * np.pi))) - 0.5 * ((x - M2) / S2) ** 2
    return np.logaddexp(a, b)


def walk(beta, n, s, x0, rng):
    """Propose x -> x + s z; accept with min(1, (f(y)/f(x))**beta).  beta=0: always accept,
    beta=inf: accept only if f(y) >= f(x)."""
    x = x0
    out = np.empty(n + 1); out[0] = x
    acc = 0
    kicks = s * rng.standard_normal(n)
    u = rng.random(n)
    for t in range(n):
        y = x + kicks[t]
        log_r = logf(y) - logf(x)                  # log of R = f(y)/f(x)
        if np.isinf(beta):
            ok = log_r >= 0.0                      # hill climber: uphill only
        else:
            ok = np.log(u[t]) < beta * log_r       # u < R^beta  (beta = 0: always)
        if ok:
            x = y; acc += 1
        out[t + 1] = x
    return out, acc / n


rng = rng_for("ch09", "11_biased_walker")
n, s, x0 = 50_000, 1.5, 0.0
drunk, a_d = walk(0.0, n, s, x0, rng)
metro, a_m = walk(1.0, n, s, x0, rng)
climb, a_c = walk(np.inf, n, s, x0, rng)

# tempered family: the walker with exponent beta samples f^beta (normalised numerically)
grid = np.linspace(-6, 6, 2401)
dg = grid[1] - grid[0]
betas = [0.3, 1.0, 3.0]
tempered = {b: walk(b, 200_000, 2.0, 0.0, rng)[0][2000:] for b in betas}

# --- numbers ---------------------------------------------------------------------------------
burn = 1000
m_post = metro[burn:]
frac_right = np.mean(m_post > 0)
exact_mean = W1 * M1 + W2 * M2
exact_var = W1 * (S1**2 + M1**2) + W2 * (S2**2 + M2**2) - exact_mean**2
nums = {
    "NineBBwN": f"{n:,}".replace(",", r"\,"),
    "NineBBwStep": s,
    "NineBBwAccDrunk": a_d,
    "NineBBwAccMetro": a_m,
    "NineBBwAccClimb": a_c,
    "NineBBwDrunkEnd": drunk[-1],
    "NineBBwDrunkScale": s * np.sqrt(n),
    "NineBBwClimbEnd": f"{climb[-1]:.3f}",
    "NineBBwFracRight": frac_right,
    "NineBBwMean": m_post.mean(),
    "NineBBwMeanExact": exact_mean,
    "NineBBwVar": m_post.var(),
    "NineBBwVarExact": exact_var,
}
for b in betas:
    fb = f(grid) ** b
    fb /= fb.sum() * dg
    pr = np.sum(fb[grid > 0]) * dg
    key = {0.3: "Low", 1.0: "One", 3.0: "Three"}[b]
    nums[f"NineBBwRight{key}"] = np.mean(tempered[b] > 0)
    nums[f"NineBBwRightExact{key}"] = pr
save_numbers("ch09", "11_biased_walker", nums)

# --- figure 1: three walkers ------------------------------------------------------------------
setup(7.2, 5.0)
fig, axes = plt.subplots(3, 2, gridspec_kw={"width_ratios": [2.3, 1]}, sharey="row")
rows = [(drunk, "(a) always accept: the unbiased walker", a_d),
        (metro, r"(b) accept with $\min(1,f(y)/f(x))$: Metropolis", a_m),
        (climb, "(c) accept only uphill: the hill climber", a_c)]
T = 3000
for k, (ch, title, a) in enumerate(rows):
    ax, axh = axes[k]
    ax.plot(np.arange(T + 1), ch[:T + 1], color=SERIES[k], lw=0.5)
    ax.set_title(title + f"   (acceptance {a:.2f})", loc="left", fontsize=9)
    ax.set_ylabel(r"$x_t$")
    if k == 2:
        ax.set_xlabel("step $t$")
    lo, hi = (ch[:T + 1].min() - 0.5, ch[:T + 1].max() + 0.5) if k == 0 else (-5, 5)
    ax.set_ylim(lo, hi)
    yy = np.linspace(lo, hi, 400)
    data = ch[:T + 1] if k == 0 else ch[burn:]     # the drunkard has no long-run histogram
    axh.hist(data, bins=80, range=(lo, hi), density=True, orientation="horizontal",
             color=SERIES[k], alpha=0.55)
    if k > 0:
        axh.plot(f(yy), yy, "k--", lw=1.2)
    axh.set_xlabel("density" if k == 2 else "")
fig.tight_layout()
savefig(fig, "ch09", "biased_walker")

# --- figure 2: the tempered family samples f^beta ---------------------------------------------
setup(7.0, 2.6)
fig, axes = plt.subplots(1, 3, sharey=True)
for ax, b, c in zip(axes, betas, SERIES):
    fb = f(grid) ** b
    fb /= fb.sum() * dg
    ax.hist(tempered[b], bins=90, range=(-6, 6), density=True, color=c, alpha=0.55)
    theory_line(ax, grid, fb, label=rf"$f^{{{b:g}}}$, normalised")
    ax.set_title(rf"accept with $\min(1,R^{{{b:g}}})$", fontsize=9)
    ax.set_xlabel("$x$")
    ax.legend(loc="upper left", fontsize=7)
axes[0].set_ylabel("density")
fig.tight_layout()
savefig(fig, "ch09", "tempered_family")
