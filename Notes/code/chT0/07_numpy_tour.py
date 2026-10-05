"""NumPy in the amount the book needs: arrays, indexing, masks, broadcasting, reductions, grids,
linear algebra and the FFT, each shown on a few numbers and printed.

Question:  what does each NumPy idea used by the book's scripts actually do, and is it true that
           np.linalg.solve is safer than multiplying by np.linalg.inv?
Computes:  one short block per idea, printed as "expression = value"; for linear algebra, the
           residual |A x - b| of x = solve(A, b) and of x = inv(A) @ b for Hilbert matrices of
           growing size (condition numbers from 20 to 10^17), and the time of each for n = 2000;
           for the FFT, the power spectrum of a noisy 50 Hz tone and a check of Parseval's theorem.
Writes:    results/chT0/07_<topic>.txt (the printed lines of each block), results/chT0/07_numpy_tour.tex,
           figures/chT0/07_numpy_tour.pdf
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
from common import setup, savefig, save_numbers, rng_for, SERIES, NOTES

np.set_printoptions(legacy="1.25", precision=4, suppress=True)   # plain printing: 0.5, not np.float64(0.5)
rng = rng_for("chT0", "07_numpy_tour")
OUT = NOTES / "results" / "chT0"
OUT.mkdir(parents=True, exist_ok=True)
printed = {}
current = None
nums = {}


def topic(name):
    """Start a new block: everything said from now on is filed under this name."""
    global current
    current = name
    printed[name] = []


def say(text):
    """Print a line and keep it for the text."""
    print(text)
    printed[current].append(text)


# [arrays] ----------------------------------------------------------------------------------------
topic("arrays")
g = np.array([9.80, 9.71, 9.86, 9.83])          # four readings of g: a 1-d array
say(f"{g.shape = }   {g.dtype = }   {g.ndim = }")
counts = np.array([3, 0, 2])                     # whole numbers -> an integer array
say(f"{counts.dtype = }")
X = np.zeros((2, 3))                             # 2 rows, 3 columns, filled with 0.0
say(f"{X.shape = }   {X.size = }")
say(f"{np.arange(5) = }")                        # 0, 1, ..., 4: stops before 5
say(f"{np.linspace(0, 1, 5) = }")                # 5 points, both ends included
say(f"{np.logspace(0, 3, 4) = }")                # 10^0 ... 10^3, 4 points
say(f"{g * 2 = }")                               # arithmetic acts on every element
say(f"{np.sqrt(g) = }")

# [indexing] --------------------------------------------------------------------------------------
topic("indexing")
v = np.arange(10) * 10                           # [0, 10, 20, ..., 90]
say(f"{v[0] = }   {v[-1] = }   {v[2:5] = }")
say(f"{v[::3] = }   {v[::-1][:3] = }")
A = np.arange(12).reshape(3, 4)                  # 3 rows, 4 columns
say(f"A =\n{A}")
say(f"{A[1, 2] = }   {A[1] = }   {A[:, 0] = }")
say(f"A[0:2, 1:3] =\n{A[0:2, 1:3]}")

# [masks] -----------------------------------------------------------------------------------------
topic("masks")
x = np.array([0.3, -1.2, 2.5, 0.8, -0.1])
m = x > 0                                        # one True/False per element
say(f"{m = }")
say(f"{x[m] = }   {m.sum() = }   {m.mean() = }")  # keep the True ones; count them; their fraction
say(f"{np.where(x > 0, x, 0.0) = }")             # x where positive, else 0
say(f"{x[(x > 0) & (x < 1)] = }")                # and: &,  or: |,  not: ~  (brackets needed)

# [broadcasting] ----------------------------------------------------------------------------------
topic("broadcasting")
r = np.array([0.0, 1.0, 3.0])
say(f"{r[:, None].shape = }   {r[None, :].shape = }")
D = r[:, None] - r[None, :]                      # (3,1) - (1,3) -> (3,3): every difference r_i - r_j
say(f"r[:, None] - r[None, :] =\n{D}")
A = np.arange(12.0).reshape(3, 4)
say(f"{(A - A.mean(axis=0)).mean(axis=0) = }")   # subtract each column's mean: shape (4,) fits (3,4)

# [reductions] ------------------------------------------------------------------------------------
topic("reductions")
E = np.array([[9.80, 9.71, 9.86, 9.83],          # 3 experiments (rows) x 4 readings (columns)
              [9.79, 9.85, 9.77, 9.81],
              [9.90, 9.74, 9.82, 9.80]])
say(f"{E.mean() = :.4f}")                        # all 12 numbers
say(f"{E.mean(axis=1) = }")                      # along each row: one mean per experiment
say(f"{E.mean(axis=0) = }")                      # down each column: one mean per reading slot
say(f"{E.std(axis=1, ddof=1) = }")               # sample standard deviation of each experiment
say(f"{E.max() = }   {E.argmax() = }   {np.cumsum([1, 2, 3]) = }")

# [grids] -----------------------------------------------------------------------------------------
topic("grids")
xs = np.linspace(-1, 1, 3)
ys = np.linspace(0, 2, 3)
XX, YY = np.meshgrid(xs, ys, indexing="ij")      # XX[i, j] = xs[i],  YY[i, j] = ys[j]
say(f"XX =\n{XX}")
say(f"YY =\n{YY}")
say(f"np.hypot(XX, YY) =\n{np.hypot(XX, YY)}")

# [linalg] ----------------------------------------------------------------------------------------
topic("linalg")
C = np.array([[4.0, 1.8], [1.8, 1.0]])           # a covariance matrix
b = np.array([1.0, 2.0])
L = np.linalg.cholesky(C)                        # lower triangular, C = L L^T
say(f"L =\n{L}")
say(f"{np.allclose(L @ L.T, C) = }")
lam, V = np.linalg.eigh(C)                       # eigenvalues (ascending) and eigenvectors (columns)
say(f"{lam = }")
say(f"{np.linalg.solve(C, b) = }")               # x with C x = b
say(f"{np.linalg.inv(C) @ b = }")                # the same here: C is well conditioned
say(f"{np.linalg.cond(C) = :.1f}")

# solve versus inv on badly conditioned matrices: the Hilbert matrix H_ij = 1/(i + j + 1)
ns = np.arange(2, 15)
cond, res_solve, res_inv, err_solve = [], [], [], []
for n in ns:
    H = hilbert(n)
    x_true = np.ones(n)
    bb = H @ x_true
    xs_ = np.linalg.solve(H, bb)
    xi_ = np.linalg.inv(H) @ bb
    scale = np.linalg.norm(H, 2) * np.linalg.norm(xs_)
    cond.append(np.linalg.cond(H))
    res_solve.append(np.linalg.norm(H @ xs_ - bb) / scale)
    res_inv.append(np.linalg.norm(H @ xi_ - bb) / np.linalg.norm(H, 2) / np.linalg.norm(xi_))
    err_solve.append(np.linalg.norm(xs_ - x_true) / np.linalg.norm(x_true))
cond, res_solve, res_inv, err_solve = map(np.array, (cond, res_solve, res_inv, err_solve))
i10 = list(ns).index(10)
say(f"Hilbert n = 10: cond = {cond[i10]:.1e}, residual solve = {res_solve[i10]:.1e}, "
    f"inv = {res_inv[i10]:.1e}, error of x = {err_solve[i10]:.1e}")

# time: one solve against forming the inverse and multiplying, n = 2000
n = 2000
G = rng.standard_normal((n, n))
S = G @ G.T + n * np.eye(n)                       # a well-conditioned covariance-like matrix
rhs = rng.standard_normal(n)


def best(f, rep=3):
    t = []
    for _ in range(rep):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


t_solve = best(lambda: np.linalg.solve(S, rhs))
t_inv = best(lambda: np.linalg.inv(S) @ rhs)
t_chol = best(lambda: np.linalg.cholesky(S))
say(f"n = 2000: solve {1e3 * t_solve:.0f} ms, inv @ b {1e3 * t_inv:.0f} ms, cholesky {1e3 * t_chol:.0f} ms")
nums.update(TzHilbCond=cond[i10], TzHilbResSolve=res_solve[i10], TzHilbResInv=res_inv[i10],
            TzHilbErr=err_solve[i10], TzTsolve=f"{1e3 * t_solve:.0f}", TzTinv=f"{1e3 * t_inv:.0f}",
            TzTchol=f"{1e3 * t_chol:.0f}", TzInvOverSolve=f"{t_inv / t_solve:.1f}")

# [fft] -------------------------------------------------------------------------------------------
topic("fft")
fs, T = 1000.0, 2.0                              # sampling rate (Hz) and duration (s)
t = np.arange(int(fs * T)) / fs                  # 2000 sample times
sig = 0.5 * np.sin(2 * np.pi * 50.0 * t) + rng.standard_normal(t.size)   # weak tone in loud noise
F = np.fft.rfft(sig)                              # complex amplitudes at f = 0 ... fs/2
f = np.fft.rfftfreq(t.size, d=1 / fs)             # the frequency of each entry
power = np.abs(F) ** 2
say(f"{t.size = }   {F.size = }   {f[1] = }   {f[-1] = }")
say(f"peak at f = {f[np.argmax(power[1:]) + 1]} Hz")
full = np.fft.fft(sig)
say(f"Parseval: sum x^2 = {np.sum(sig ** 2):.3f},  sum |X|^2 / N = {np.sum(np.abs(full) ** 2) / t.size:.3f}")
nums.update(TzFftPeak=f"{f[np.argmax(power[1:]) + 1]:.1f}", TzFftN=t.size, TzFftBins=F.size,
            TzFftSnr=f"{power[np.argmax(power[1:]) + 1] / np.median(power[1:]):.0f}")

# figure: left, residuals of solve and inv against the condition number; right, the tone's spectrum
setup(8.4, 3.2)
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.loglog(cond, res_solve, "o-", color=SERIES[0], label="solve(A, b)")
ax0.loglog(cond, res_inv, "s-", color=SERIES[1], label="inv(A) @ b")
ax0.loglog(cond, err_solve, "^:", color=SERIES[2], label="error of $x$ (solve)")
ax0.loglog(cond, 1.1e-16 * cond, color="0.4", ls="--", lw=1, label=r"$\epsilon\,\kappa(A)$")
ax0.set_xlabel(r"condition number $\kappa(A)$ (Hilbert, $n=2\ldots14$)")
ax0.set_ylabel(r"relative size")
ax0.set_ylim(1e-19, 1e3)
ax0.legend(fontsize=7, loc="upper left")
ax1.semilogy(f[1:], power[1:] / np.median(power[1:]), color=SERIES[0], lw=0.8)
ax1.set_xlabel("frequency $f$ (Hz)")
ax1.set_ylabel(r"$|X(f)|^2$ / median")
ax1.annotate("50 Hz tone", (50, power[100] / np.median(power[1:])), xytext=(120, 300),
             textcoords="data", fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
fig.tight_layout()
savefig(fig, "chT0", "07_numpy_tour")

for name, lines in printed.items():
    (OUT / f"07_{name}.txt").write_text("\n".join(lines) + "\n")
save_numbers("chT0", "07_numpy_tour", nums)
