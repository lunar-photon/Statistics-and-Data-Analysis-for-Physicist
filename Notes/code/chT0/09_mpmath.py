"""When 16 digits are not enough: float64 against mpmath on three small problems.

Question:  where does double precision break, how does arbitrary precision (mpmath) fix it, how do
           we choose the number of digits, and what does the extra precision cost?
Computes:  (1) the Gaussian upper tail P(Z > z) for z = 1 ... 40 four ways: 1 - cdf in float64,
               scipy's sf in float64, scipy's logsf, and mpmath's erfc at 50 digits;
           (2) the polynomial (x - 1)^7, written out as x^7 - 7x^6 + ... - 1 and evaluated near x = 1
               in float64, in mpmath at 15 and at 30 digits, and in the factored form;
           (3) the "d versus 2d" test: the same expanded polynomial at dps = d and 2d for several d,
               and the number of digits on which the two agree;
           (4) the time per evaluation of exp in NumPy (vectorised), Python's math, and mpmath.
Writes:    results/chT0/09_mpmath.tex, results/chT0/09_mpmath_out.txt, figures/chT0/09_mpmath.pdf
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy import stats
from common import setup, savefig, save_numbers, SERIES, NOTES

nums, lines = {}, []


def say(text):
    print(text)
    lines.append(text)


# (1) Gaussian tails ------------------------------------------------------------------------------
mp.mp.dps = 50
zs = np.arange(1.0, 40.5, 0.5)
ref = np.array([float(mp.erfc(mp.mpf(z) / mp.sqrt(2)) / 2) for z in zs])          # may underflow to 0 as a float
ref_log10 = np.array([float(mp.log10(mp.erfc(mp.mpf(z) / mp.sqrt(2)) / 2)) for z in zs])
naive = 1.0 - stats.norm.cdf(zs)
sf = stats.norm.sf(zs)
logsf10 = stats.norm.logsf(zs) / np.log(10)
for z in (5, 8, 9, 20, 37, 38, 39, 40):
    i = list(zs).index(z)
    say(f"z = {z:2d}: 1-cdf = {naive[i]:.4e}   sf = {sf[i]:.4e}   log10 sf = {logsf10[i]:9.4f}   "
        f"mpmath log10 = {ref_log10[i]:9.4f}")
first_zero_naive = zs[np.argmax(naive == 0)]
first_zero_sf = zs[np.argmax(sf == 0)]
i5 = list(zs).index(5.0)
nums.update(TzNaiveZero=f"{first_zero_naive:.1f}", TzSfZero=f"{first_zero_sf:.1f}",
            TzNaiveRelFive=abs(naive[i5] - ref[i5]) / ref[i5], TzSfRelFive=abs(sf[i5] - ref[i5]) / ref[i5],
            TzLogTenForty=f"{ref_log10[-1]:.2f}", TzLogsfForty=f"{logsf10[-1]:.2f}")

# (2) cancellation that a rewrite cannot easily remove: the expanded (x - 1)^7 near its root -------
coef = [1, -7, 21, -35, 35, -21, 7, -1]                 # x^7 - 7 x^6 + 21 x^5 - ... - 1


def expanded(x):
    """Horner's rule on the expanded coefficients, in whatever number type x is."""
    out = 0
    for c in coef:
        out = out * x + c
    return out


xg = np.linspace(1 - 0.012, 1 + 0.012, 801)              # narrow enough that the noise band is visible
p64 = expanded(xg)                                        # float64
pfac = (xg - 1) ** 7                                      # factored: no cancellation
x_test = 1.01
mp.mp.dps = 15
p_mp15 = expanded(mp.mpf(x_test))
mp.mp.dps = 30
p_mp30 = expanded(mp.mpf(x_test))
p_true = (mp.mpf("1.01") - 1) ** 7                         # 1e-14 exactly, if x is exactly 1.01
say(f"x = 1.01: float64 expanded = {expanded(x_test):.6e}   factored = {(x_test - 1)**7:.6e}")
say(f"          mpmath 15 digits = {mp.nstr(p_mp15, 8)}   30 digits = {mp.nstr(p_mp30, 8)}   "
    f"exact (x = 1.01 as decimal) = {mp.nstr(p_true, 8)}")
nums.update(TzPolyFloat=float(expanded(x_test)), TzPolyFact=(x_test - 1) ** 7, TzPolyMpThirty=float(p_mp30))

# (3) the d versus 2d test --------------------------------------------------------------------------
xs_dec = "1.01"
rows = []
for dps in (10, 15, 20, 30):
    mp.mp.dps = dps
    a = expanded(mp.mpf(xs_dec))
    mp.mp.dps = 2 * dps
    b = expanded(mp.mpf(xs_dec))
    mp.mp.dps = 4 * dps
    agree = max(float(-mp.log10(abs((a - b) / b))), 0.0) if a != b else float(4 * dps)
    rows.append((dps, a, b, agree))
    say(f"dps = {dps:2d}: {mp.nstr(a, 12):>18}   dps = {2 * dps:2d}: {mp.nstr(b, 12):>18}   agree to "
        f"{agree:5.1f} digits")
mp.mp.dps = 50
xm = mp.mpf(xs_dec)
kappa = sum(abs(c) * xm ** (7 - k) for k, c in enumerate(coef)) / abs(expanded(xm))
say(f"condition number of the expanded sum at x = 1.01: {mp.nstr(kappa, 4)}  (log10 = {float(mp.log10(kappa)):.1f})")
nums.update(TzPolyKappa=f"{float(kappa):.2e}", TzPolyLogKappa=f"{float(mp.log10(kappa)):.1f}")
nums.update(TzAgreeFifteen=f"{rows[1][3]:.1f}", TzAgreeThirty=f"{rows[3][3]:.1f}",
            TzAgreeTen=f"{rows[0][3]:.1f}", TzAgreeTwenty=f"{rows[2][3]:.1f}")

# (4) cost ---------------------------------------------------------------------------------------
mp.mp.dps = 15
x = np.linspace(0.0, 1.0, 100_000)
t0 = time.perf_counter(); np.exp(x); t_np = (time.perf_counter() - t0) / x.size
t0 = time.perf_counter(); [math.exp(v) for v in x[:20_000]]; t_math = (time.perf_counter() - t0) / 20_000
t0 = time.perf_counter(); [mp.exp(v) for v in x[:20_000]]; t_mp15 = (time.perf_counter() - t0) / 20_000
mp.mp.dps = 50
t0 = time.perf_counter(); [mp.exp(v) for v in x[:20_000]]; t_mp50 = (time.perf_counter() - t0) / 20_000
say(f"time per exp: numpy {1e9 * t_np:.1f} ns, math {1e9 * t_math:.0f} ns, mpmath (15 digits) {1e9 * t_mp15:.0f} ns, "
    f"mpmath (50 digits) {1e9 * t_mp50:.0f} ns")
nums.update(TzTnp=f"{1e9 * t_np:.1f}", TzTmath=f"{1e9 * t_math:.0f}", TzTmpFifteen=f"{1e9 * t_mp15:.0f}",
            TzTmpFifty=f"{1e9 * t_mp50:.0f}", TzMpOverNp=f"{t_mp15 / t_np:.0f}", TzMpOverMath=f"{t_mp15 / t_math:.0f}")

# figure ---------------------------------------------------------------------------------------------
setup(8.4, 3.3)
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.plot(zs, ref_log10, color="k", lw=2.4, alpha=0.35, label="mpmath, 50 digits")
ok = naive > 0
ax0.plot(zs[ok], np.log10(naive[ok]), "o", ms=3, color=SERIES[1], label="1 - cdf (float64)")
ok = sf > 0
ax0.plot(zs[ok], np.log10(sf[ok]), "-", color=SERIES[0], label="sf (float64)")
ax0.plot(zs, logsf10, ":", color=SERIES[2], lw=1.6, label="logsf (float64)")
ax0.axhline(np.log10(np.finfo(float).tiny), color="0.5", ls="--", lw=0.8)
ax0.text(1.5, np.log10(np.finfo(float).tiny) + 8, "smallest normal float", fontsize=7, color="0.4")
ax0.set_xlabel("$z$ (in standard deviations)")
ax0.set_ylabel(r"$\log_{10} P(Z>z)$")
ax0.legend(fontsize=7, loc="upper right")
ax1.plot(xg, p64, color=SERIES[1], lw=0.7, label="expanded, float64")
ax1.plot(xg, pfac, color="k", lw=1.4, label=r"factored $(x-1)^7$")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$p(x)$")
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax1.legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "chT0", "09_mpmath")

(NOTES / "results" / "chT0" / "09_mpmath_out.txt").write_text("\n".join(lines) + "\n")
save_numbers("chT0", "09_mpmath", nums)
