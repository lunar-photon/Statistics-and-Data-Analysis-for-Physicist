"""Is this number real?  Numerical noise, Monte Carlo noise, and the checks that tell them apart.

Question:  when a script prints a number, how many of its digits can we trust, and how do we check?
Computes:  (1) the float64 facts (machine epsilon, 0.1 + 0.2), and two cancellations with their
               cures: 1 - cos x for small x, and the one-pass variance of data with a large offset;
           (2) the error of forward and central finite-difference derivatives of exp(x) at x = 1
               against the step h, with the predicted best steps sqrt(eps) and eps^(1/3);
           (3) what quad, brentq and minimize report when they struggle: a narrow peak quad
               cannot see, a bracket with no sign change, a minimiser stopped early;
           (4) Cholesky of a sample covariance from fewer simulations than bins (it fails);
           (5) a Monte Carlo tail probability for several seeds and sizes, and the scatter of a
               difference of two Monte Carlo estimates with independent and with common random numbers;
           (6) trapezoid integrals at halving steps, and Richardson extrapolation.
Writes:    results/chT0/08_trust.tex, results/chT0/08_trust_out.txt,
           figures/chT0/08_trust_digits.pdf, figures/chT0/08_trust_mc.pdf
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, stats
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line, NOTES

nums, lines = {}, []


def say(text):
    """Print a line and keep it, so the text can show exactly what was printed."""
    print(text)
    lines.append(text)


# (1) floating point ------------------------------------------------------------------------------
fi = np.finfo(np.float64)
say(f"eps = {fi.eps:.4e}   largest = {fi.max:.4e}   smallest normal = {fi.tiny:.4e}")
say(f"1.0 + eps == 1.0 -> {1.0 + fi.eps == 1.0};   1.0 + eps/2 == 1.0 -> {1.0 + fi.eps / 2 == 1.0}")
say(f"0.1 + 0.2 == 0.3 -> {0.1 + 0.2 == 0.3};   np.isclose -> {np.isclose(0.1 + 0.2, 0.3)}")
nums.update(TzEps=fi.eps, TzMaxFloat=fi.max, TzTiny=fi.tiny)

# cancellation 1: (1 - cos x) / x^2  ->  1/2 as x -> 0;  the cure: 1 - cos x = 2 sin^2(x/2)
xs = np.logspace(-9, -1, 81)
naive = (1 - np.cos(xs)) / xs ** 2
good = 2 * np.sin(xs / 2) ** 2 / xs ** 2
exact = 0.5 - xs**2 / 24 + xs**4 / 720 - xs**6 / 40320   # Taylor series: accurate to 1e-14 for x <= 0.1
x1 = 1e-5
say(f"x = 1e-5: (1 - cos x)/x^2 = {(1 - np.cos(x1)) / x1**2:.10f},  2 sin^2(x/2)/x^2 = "
    f"{2 * np.sin(x1 / 2)**2 / x1**2:.10f}")
x2 = 1e-8
say(f"x = 1e-8: naive = {(1 - np.cos(x2)) / x2**2},  rewritten = {2 * np.sin(x2 / 2)**2 / x2**2:.10f}")
nums.update(TzCancNaive=f"{(1 - np.cos(x1)) / x1**2:.10f}", TzCancGood=f"{2 * np.sin(x1 / 2)**2 / x1**2:.10f}",
            TzCancNaiveTiny=f"{(1 - np.cos(x2)) / x2**2:.1f}")

# cancellation 2: variance of readings with a large common offset
rng = rng_for("chT0", "08_trust")
d = 1e8 + rng.standard_normal(10_000)                 # e.g. times in ns since some epoch
one_pass = np.mean(d ** 2) - np.mean(d) ** 2          # E[x^2] - E[x]^2: two huge, nearly equal numbers
two_pass = np.mean((d - d.mean()) ** 2)               # subtract the mean first, then square
say(f"variance of 1e8 + N(0,1): one-pass = {one_pass:.4f},  two-pass = {two_pass:.4f},  np.var = {np.var(d):.4f}")
nums.update(TzVarOnePass=f"{one_pass:.2f}", TzVarTwoPass=f"{two_pass:.4f}")

# (2) finite differences ---------------------------------------------------------------------------
x0 = 1.0
f = np.exp
fp = np.exp(x0)
hs = np.logspace(-13, -0.5, 200)
x0s = np.linspace(0.5, 1.5, 101)                     # many points, so that lucky roundings average out
fwd = np.sqrt(np.mean([((f(x + hs) - f(x)) / hs / np.exp(x) - 1) ** 2 for x in x0s], axis=0))
cen = np.sqrt(np.mean([((f(x + hs) - f(x - hs)) / (2 * hs) / np.exp(x) - 1) ** 2 for x in x0s], axis=0))
eps = fi.eps
h_fwd = 2 * np.sqrt(eps)                              # sqrt(4 eps |f| / |f''|) with f = f'' = e
h_cen = (3 * eps) ** (1 / 3)                          # (3 eps |f| / |f'''|)^(1/3) with f = f''' = e
err_fwd_min, err_cen_min = 2 * np.sqrt(eps), (3 * eps) ** (2 / 3) / 2   # predicted smallest relative errors
say(f"best h: forward {hs[np.argmin(fwd)]:.1e} (predicted {h_fwd:.1e}),  central {hs[np.argmin(cen)]:.1e} "
    f"(predicted {h_cen:.1e})")
say(f"smallest error: forward {fwd.min():.1e} (predicted {err_fwd_min:.1e}),  central {cen.min():.1e} "
    f"(predicted {err_cen_min:.1e})")
# the plateau check: the central derivative at steps a factor 10 apart
plateau = [(h, (f(x0 + h) - f(x0 - h)) / (2 * h)) for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12)]
for h, dv in plateau:
    say(f"  h = {h:.0e}:  central difference = {dv:.12f}")
nums.update(TzHfwdPred=h_fwd, TzHcenPred=h_cen, TzHfwdBest=hs[np.argmin(fwd)], TzHcenBest=hs[np.argmin(cen)],
            TzErrFwdMin=fwd.min(), TzErrCenMin=cen.min(), TzErrFwdPred=err_fwd_min, TzErrCenPred=err_cen_min,
            TzExactE=f"{fp:.12f}")

# (3) black boxes that can fail quietly or loudly --------------------------------------------------
peak = lambda x: stats.norm.pdf(x, 37.3, 0.01)         # a narrow peak, total area exactly 1
val, est = integrate.quad(peak, 0, 100)
val3, est3 = integrate.quad(peak, 0, 100, epsabs=1e-13, epsrel=1e-12)    # tighter tolerance alone
val4, est4 = integrate.quad(peak, 0, 100, points=[37.3])                  # one break point at the peak
val5, est5 = integrate.quad(peak, 0, 100, points=[37.25, 37.35])          # an interval as wide as the peak
val2, est2 = integrate.quad(peak, 37.3 - 0.1, 37.3 + 0.1)                 # integrate where the peak lives
say(f"quad narrow peak on [0, 100]: {val:.3e} (claimed error {est:.1e});  tighter tolerance: {val3:.3e};  "
    f"points=[37.3]: {val4:.3e};  points=[37.25, 37.35]: {val5:.10f} (claimed {est5:.1e});  "
    f"on [37.2, 37.4]: {val2:.10f} (claimed {est2:.1e})")
nums.update(TzQuadMiss=val, TzQuadMissErr=est, TzQuadHit=f"{val2:.10f}", TzQuadHitErr=est2, TzQuadTight=val3,
            TzQuadOnePoint=val4, TzQuadTwoPoints=f"{val5:.10f}")

try:
    optimize.brentq(lambda x: x ** 2 - 2, 0.0, 1.0)
except ValueError as err:
    say(f"brentq on [0, 1] for x^2 - 2: ValueError: {err}")
root = optimize.brentq(lambda x: x ** 2 - 2, 0.0, 2.0)
say(f"brentq on [0, 2]: {root:.15f}   (sqrt 2 = {np.sqrt(2):.15f})")
nums.update(TzRootTwo=f"{root:.12f}")

rosen = optimize.rosen                                 # a curved valley, minimum at (1, 1, 1, 1, 1)
x_start = np.zeros(5)
short = optimize.minimize(rosen, x_start, method="Nelder-Mead", options=dict(maxiter=200))
full = optimize.minimize(rosen, x_start, method="BFGS", jac=optimize.rosen_der)
say(f"Nelder-Mead, 200 iterations: success = {short.success}, message = '{short.message}', "
    f"x = {np.round(short.x, 3)}, |grad| = {np.linalg.norm(optimize.rosen_der(short.x)):.2e}")
say(f"BFGS with gradient: success = {full.success}, x = {np.round(full.x, 6)}, "
    f"|grad| = {np.linalg.norm(optimize.rosen_der(full.x)):.2e}")
nums.update(TzNMsuccess=str(short.success), TzNMgrad=np.linalg.norm(optimize.rosen_der(short.x)),
            TzNMfun=short.fun, TzBFGSgrad=np.linalg.norm(optimize.rosen_der(full.x)))

# (4) a covariance from too few simulations --------------------------------------------------------
p = 50                                                 # number of bins in a data vector
for nsim in (30, 200):
    sims = rng.standard_normal((nsim, p))
    Cs = np.cov(sims, rowvar=False)
    lam = np.linalg.eigvalsh(Cs)
    try:
        np.linalg.cholesky(Cs)
        ok = "succeeds"
    except np.linalg.LinAlgError:
        ok = "fails (LinAlgError)"
    say(f"{nsim} simulations of {p} bins: smallest eigenvalue {lam.min():.1e}, rank {np.linalg.matrix_rank(Cs)}, "
        f"cholesky {ok}")
    nums[f"TzCholMin{'Few' if nsim == 30 else 'Many'}"] = lam.min()
    nums[f"TzCholRank{'Few' if nsim == 30 else 'Many'}"] = int(np.linalg.matrix_rank(Cs))

# (5) Monte Carlo noise ----------------------------------------------------------------------------
q_exact = stats.norm.sf(2.0)                            # P(Z > 2) = 0.02275
Ns = np.logspace(2, 6, 9).astype(int)
est_seeds = np.array([[np.mean(rng_for("chT0", "08_trust", s).standard_normal(N) > 2.0) for N in Ns]
                      for s in range(1, 9)])           # 8 seeds x 9 sizes
sig = np.sqrt(q_exact * (1 - q_exact) / Ns)             # binomial standard error of a fraction
N4 = 10_000
e4 = est_seeds[:, list(Ns).index(N4)]
say(f"P(Z>2) exact {q_exact:.5f};  N = 10^4, eight seeds: {np.round(e4, 4)};  predicted spread "
    f"{np.sqrt(q_exact * (1 - q_exact) / N4):.5f}, observed {e4.std(ddof=1):.5f}")
nums.update(TzQexact=f"{q_exact:.5f}", TzQsigFour=f"{np.sqrt(q_exact * (1 - q_exact) / N4):.5f}",
            TzQobsFour=f"{e4.std(ddof=1):.5f}", TzQfirst=f"{e4[0]:.4f}",
            TzQzFirst=f"{(e4[0] - q_exact) / np.sqrt(q_exact * (1 - q_exact) / N4):.2f}")

# common random numbers: how P(X > 2) changes when the mean moves from 0 to 0.1, X ~ N(mu, 1)
delta, M, R = 0.1, 10_000, 400
diff_ind, diff_crn = [], []
r5 = rng_for("chT0", "08_trust", 99)
for _ in range(R):
    z1, z2 = r5.standard_normal(M), r5.standard_normal(M)
    diff_ind.append(np.mean(z2 + delta > 2) - np.mean(z1 > 2))    # fresh draws for the second setting
    diff_crn.append(np.mean(z1 + delta > 2) - np.mean(z1 > 2))    # the same draws, shifted
diff_ind, diff_crn = np.array(diff_ind), np.array(diff_crn)
true_diff = stats.norm.sf(2 - delta) - stats.norm.sf(2)
say(f"difference P(X>2 | mu=0.1) - P(X>2 | mu=0) = {true_diff:.5f}; scatter over {R} repeats: independent "
    f"{diff_ind.std():.5f}, common random numbers {diff_crn.std():.5f}")
nums.update(TzCrnTrue=f"{true_diff:.5f}", TzCrnInd=f"{diff_ind.std():.5f}", TzCrnCrn=f"{diff_crn.std():.5f}",
            TzCrnGain=f"{(diff_ind.std() / diff_crn.std())**2:.0f}", TzCrnM=M, TzCrnR=R,
            TzCrnZind=f"{true_diff / diff_ind.std():.1f}", TzCrnZcrn=f"{true_diff / diff_crn.std():.1f}",
            TzCrnOneInd=f"{diff_ind[0]:.5f}", TzCrnOneCrn=f"{diff_crn[0]:.5f}",
            TzCrnFracIndBelow=f"{100 * np.mean(diff_ind / diff_ind.std() < 3):.0f}",   # % of single comparisons below 3 sigma
            TzCrnFracCrnBelow=f"{100 * np.mean(diff_crn / diff_crn.std() < 3):.0f}")

# (6) convergence and Richardson extrapolation ----------------------------------------------------
def trap(n):
    """Trapezoid rule for the integral of sin x from 0 to pi (exactly 2) with n intervals."""
    x = np.linspace(0, np.pi, n + 1)
    return np.trapezoid(np.sin(x), x)


for n in (8, 16, 32):
    say(f"trapezoid n = {n:2d}: {trap(n):.10f}   error {trap(n) - 2:.2e}")
rich = (4 * trap(16) - trap(8)) / 3
say(f"Richardson (4 T16 - T8)/3 = {rich:.10f}   error {rich - 2:.2e}")
nums.update(TzTrapEight=f"{trap(8):.6f}", TzTrapSixteen=f"{trap(16):.6f}", TzTrapErrEight=trap(8) - 2,
            TzTrapErrSixteen=trap(16) - 2, TzTrapRatio=f"{(trap(8) - 2) / (trap(16) - 2):.3f}",
            TzRich=f"{rich:.8f}", TzRichErr=rich - 2)

# figures ------------------------------------------------------------------------------------------
setup(8.4, 3.3)
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.loglog(xs, np.maximum(np.abs(naive - exact) / exact, 1e-17), color=SERIES[1], label=r"$(1-\cos x)/x^2$")
ax0.loglog(xs, np.maximum(np.abs(good - exact) / exact, 1e-17), color=SERIES[0],
           label=r"$2\sin^2(x/2)/x^2$")
theory_line(ax0, xs, eps / xs ** 2, label=r"$\epsilon/x^2$")
ax0.set_ylim(1e-17, 10)
ax0.set_xlabel("$x$")
ax0.set_ylabel("relative error")
ax0.legend(fontsize=7, loc="upper right")
ax1.loglog(hs, fwd, color=SERIES[1], label="forward difference")
ax1.loglog(hs, cen, color=SERIES[0], label="central difference")
ax1.axvline(h_fwd, color=SERIES[1], ls=":", lw=1)
ax1.axvline(h_cen, color=SERIES[0], ls=":", lw=1)
theory_line(ax1, hs, eps / hs, label=r"round-off $\epsilon/h$")
ax1.loglog(hs, hs ** 2 / 6, color="0.5", lw=1, ls="-.", label=r"truncation $h^2/6$")
ax1.set_ylim(1e-12, 1e2)
ax1.set_xlabel("step $h$")
ax1.set_ylabel(r"relative error of $f'(1)$, $f=e^x$")
ax1.legend(fontsize=7, loc="upper center")
fig.tight_layout()
savefig(fig, "chT0", "08_trust_digits")

fig, (ax0, ax1) = plt.subplots(1, 2)
for s in range(est_seeds.shape[0]):
    ax0.semilogx(Ns, est_seeds[s], "o-", ms=2.5, lw=0.8, color=SERIES[0], alpha=0.6)
ax0.fill_between(Ns, q_exact - sig, q_exact + sig, color="0.6", alpha=0.35, lw=0, label=r"$\pm\sigma/\sqrt{N}$")
ax0.axhline(q_exact, color="k", ls="--", lw=1.2, label="exact")
ax0.set_xlabel("number of draws $N$")
ax0.set_ylabel(r"estimate of $P(Z>2)$")
ax0.legend(fontsize=7)
bins = np.linspace(-0.004, 0.017, 60)
ax1.hist(diff_ind, bins=bins, color=SERIES[1], alpha=0.6, label="independent draws")
ax1.hist(diff_crn, bins=bins, color=SERIES[0], alpha=0.6, label="common random numbers")
ax1.axvline(true_diff, color="k", ls="--", lw=1.2, label="exact difference")
ax1.set_xlabel(r"estimated change of $P(X>2)$, $\mu: 0\to0.1$")
ax1.set_ylabel("repeats")
ax1.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "chT0", "08_trust_mc")

(NOTES / "results" / "chT0" / "08_trust_out.txt").write_text("\n".join(lines) + "\n")
save_numbers("chT0", "08_trust", nums)
