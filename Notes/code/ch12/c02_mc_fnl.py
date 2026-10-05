"""c02_mc_fnl.py -- the Monte Carlo of the local model: summaries of Gaussian and non-Gaussian maps.

Question: what is the sampling distribution P(S | eps) of each summary S (power spectrum,
skewness, Euler characteristic, Betti curves, persistence-diagram histograms) for the
local-type model u = W * R_eps * [g + eps (g^2 - 1)] of lib_fnl, whose power spectrum does not
depend on eps?
Computes, on 256 x 256 maps:
  set "null"  : 1000 maps at eps = 0            -> mean and covariance under the null
  set "check" : 500 more maps at eps = 0        -> calibration of p-values (independent)
  set "grid"  : 500 seeds x eps in {0, +-0.025, +-0.05, +-0.1}, the SAME Gaussian g for every
                eps of a seed (common random numbers) -> derivatives, means for the posterior,
                power of the tests
and, for the grid maps, the three skewness parameters of Matsubara's formula.
Writes: data/ch12/c_fnl_mc.npz (cached; delete it to recompute)
"""
import sys
import time
import pathlib
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import rng_for, DATA
from lib_fnl import Model, summaries, KEYS, standardise
from lib_topo import spectral_gradient

N, SMOOTH = 256, 2.0
EPS_GRID = np.array([-0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.1])
N_NULL, N_CHECK, N_GRID = 1000, 500, 500
OUT = DATA / "ch12" / "c_fnl_mc.npz"
model = Model(N, SMOOTH)


def skew_params(u):
    """S0 sigma0, S1 sigma0, S2 sigma0 of Matsubara (T3a eq. skewpar) for one map."""
    x = standardise(u)
    gx, gy = spectral_gradient(x)
    g2 = gx ** 2 + gy ** 2
    k0 = 2 * np.pi * np.fft.fftfreq(N)[:, None]
    k1 = 2 * np.pi * np.fft.rfftfreq(N)[None, :]
    lap = np.fft.irfft2(-(k0 ** 2 + k1 ** 2) * np.fft.rfft2(x), s=x.shape)
    s1sq = g2.mean()
    return np.array([np.mean(x ** 3),
                     -0.75 * np.mean(x ** 2 * lap) / s1sq,
                     -3.0 * np.mean(g2 * lap) / s1sq ** 2,
                     s1sq / 2])                           # last entry: lambda per pixel^2


def one(args):
    stream, eps_list = args
    rng = rng_for("ch12", "c02_mc_fnl", stream)
    g = model.gauss(rng)                                  # one Gaussian field per seed
    res = []
    for eps in eps_list:                                  # common random numbers across eps
        u = model.draw(eps, g=g)
        s = summaries(u, model)
        s["sk"] = skew_params(u)
        res.append(s)
    return res


def run():
    jobs = ([(i, [0.0]) for i in range(N_NULL)]
            + [(10_000 + i, [0.0]) for i in range(N_CHECK)]
            + [(20_000 + i, list(EPS_GRID)) for i in range(N_GRID)])
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(one, jobs, chunksize=8)
    print(f"{len(jobs)} seeds in {time.time() - t0:.0f} s")
    keys = KEYS + ["sk"]
    null = out[:N_NULL]
    check = out[N_NULL:N_NULL + N_CHECK]
    grid = out[N_NULL + N_CHECK:]
    arrays = {}
    for k in keys:
        arrays["null_" + k] = np.array([r[0][k] for r in null])
        arrays["check_" + k] = np.array([r[0][k] for r in check])
        arrays["grid_" + k] = np.array([[r[j][k] for j in range(len(EPS_GRID))] for r in grid])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, eps_grid=EPS_GRID, N=N, smooth=SMOOTH, **arrays)
    print("[data]", OUT)


if __name__ == "__main__":
    if OUT.exists():
        print("cached:", OUT)
    else:
        run()
