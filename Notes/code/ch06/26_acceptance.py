"""26_acceptance.py -- the geometric acceptance of a detector, by Monte Carlo.

Question: a point source emits particles isotropically; a flat square detector of
side a faces it at distance d.  What fraction of the particles hits it?

Isotropic directions come from the inverse-CDF method: cos(theta) = 2 u1 - 1,
phi = 2 pi u2 (the solid angle element is d(cos theta) d(phi)).  A particle
travelling up (cos theta > 0) crosses the plane z = d at (x, y) = d tan(theta)
(cos phi, sin phi); it is accepted if |x|, |y| < a/2.

  1. Verification: the analytic answer is Omega / 4 pi with
     Omega = 4 arcsin(a^2 / (a^2 + 4 d^2)); for d = a/2 it is exactly 1/6
     (one face of a cube seen from its centre).
  2. Research-style: the same detector with a dead central strip of width w and
     a circular hole of radius rho -- no closed form, the Monte Carlo number is
     the answer, with its binomial error.

Writes: figures/ch06/acceptance.pdf, results/ch06/26_acceptance.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt

setup(9.0, 3.2)
rng = rng_for("ch06", "26_acceptance")
N = 1_000_000
out = {"SixBAcN": N}


def isotropic(n):
    cth = 2.0 * rng.random(n) - 1.0             # cos(theta) uniform on [-1, 1]
    phi = 2.0 * np.pi * rng.random(n)           # phi uniform on [0, 2 pi)
    return cth, phi


def hits(cth, phi, a, d):
    """Crossing point with the plane z = d for upward particles (NaN otherwise)."""
    up = cth > 0
    sth = np.sqrt(1.0 - cth**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(up, d * sth / cth, np.nan)
    return rho * np.cos(phi), rho * np.sin(phi)


def omega_square(a, d):
    return 4.0 * np.arcsin(a**2 / (a**2 + 4.0 * d**2))


a = 10.0
rows = []
for d, tag in [(5.0, "Cube"), (10.0, "Far"), (20.0, "VeryFar")]:
    cth, phi = isotropic(N)
    x, y = hits(cth, phi, a, d)
    inside = (np.abs(x) < a / 2) & (np.abs(y) < a / 2)
    p = inside.mean()
    exact = omega_square(a, d) / (4 * np.pi)
    se = np.sqrt(p * (1 - p) / N)
    out.update({f"SixBAc{tag}D": d, f"SixBAc{tag}": p, f"SixBAc{tag}Exact": exact,
                f"SixBAc{tag}Se": se, f"SixBAc{tag}Pull": (p - exact) / se})

# research-style geometry: dead strip |x| < w/2 and a hole of radius r0 at the centre
d, w, r0 = 5.0, 1.0, 1.5
cth, phi = isotropic(N)
x, y = hits(cth, phi, a, d)
inside = (np.abs(x) < a / 2) & (np.abs(y) < a / 2)
live = inside & (np.abs(x) > w / 2) & (x**2 + y**2 > r0**2)
p = live.mean()
out.update(SixBAcDeadW=w, SixBAcHoleR=r0, SixBAcLive=p, SixBAcLiveSe=np.sqrt(p * (1 - p) / N),
           SixBAcLiveRatio=p / inside.mean())

fig, ax = plt.subplots(1, 2)
m = 20000
sel = inside[:m]
ax[0].scatter(x[:m][sel & live[:m]], y[:m][sel & live[:m]], s=0.8, color=SERIES[0], label="detected")
ax[0].scatter(x[:m][sel & ~live[:m]], y[:m][sel & ~live[:m]], s=0.8, color=SERIES[1], label="dead region")
ax[0].set(xlabel=r"$x$ on detector [cm]", ylabel=r"$y$ [cm]", aspect="equal",
          xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), title="")
ax[0].legend(loc="upper center", fontsize=7, markerscale=6, ncol=2, bbox_to_anchor=(0.5, 1.16), frameon=False)
ds = np.linspace(1, 40, 60)
meas = []
for dd in ds:
    c, f = isotropic(100_000)
    xx, yy = hits(c, f, a, dd)
    meas.append(np.mean((np.abs(xx) < a / 2) & (np.abs(yy) < a / 2)))
ax[1].plot(ds, meas, "o", ms=3, color=SERIES[0], label=r"Monte Carlo, $10^5$ each")
theory_line(ax[1], ds, omega_square(a, ds) / (4 * np.pi), label=r"$\Omega/4\pi$")
ax[1].set(xlabel=r"distance $d$ [cm]", ylabel="acceptance", yscale="log",
          title=r"square detector, $a=10$ cm")
ax[1].legend()
fig.tight_layout()
savefig(fig, "ch06", "acceptance")
save_numbers("ch06", "26_acceptance", out)
print(out)
