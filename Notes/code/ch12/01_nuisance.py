"""01_nuisance.py -- which summaries of an excursion set survive a nuisance transformation?

Question: take one Gaussian random field and push it through four different "nuisance"
operations: a smooth one-to-one remapping of the plane (a toy version of lensing), a
monotonic change of the field values, added pixel noise, and extra smoothing.  Which of the
numbers area, boundary length, beta0, beta1, chi of the excursion set {phi > nu} change?

Computes: the five numbers at nu = 1 for the original field and each transformed one.
Writes:   figures/ch12/nuisance_maps.pdf, results/ch12/01_nuisance.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch07"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from common import setup, savefig, save_numbers, rng_for
from lib_fields import gaussian_field
from lib_topo import n_components, euler_cubical, spectral_gradient

setup()
rng = rng_for("ch12", "01_nuisance")
N, R, NU = 256, 6.0, 1.0
P = lambda k: np.exp(-(k * R) ** 2)            # white noise smoothed on R pixels
phi = gaussian_field(P, N, N, 2, rng)
phi = (phi - phi.mean()) / phi.std()


def summary(u, nu):
    """Area fraction, boundary length (pixels), beta0, beta1, chi of {u > nu}."""
    m = u > nu
    gx, gy = spectral_gradient(u)
    dnu = 0.05
    length = (np.hypot(gx, gy) * (np.abs(u - nu) < dnu / 2)).sum() / dnu
    b0 = n_components(m, 8, periodic=True)       # the map lives on a torus (FFT box)
    chi = euler_cubical(m, 8, periodic=True)
    return m.mean(), length, b0, b0 - chi, chi   # beta1 = beta0 - chi (no set wraps the torus)


# 1. a smooth one-to-one remapping x -> x + s(x): s is a smooth periodic displacement
#    (largest gradient well below 1, so the map cannot fold the plane onto itself)
s = [gaussian_field(lambda k: np.exp(-(k * 45.0) ** 2), N, N, 2, rng) for _ in range(2)]
s = [14.0 * si / si.std() for si in s]                         # rms displacement 14 pixels
g0, g1 = np.gradient(s[0]), np.gradient(s[1])
jac = np.min((1 + g0[0]) * (1 + g1[1]) - g0[1] * g1[0])     # smallest det(1 + grad s) > 0: no folding
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
remap = ndimage.map_coordinates(phi, [ii + s[0], jj + s[1]], order=3, mode="grid-wrap")
# 2. a monotonic change of values, with the threshold moved along: f(phi) > f(nu)
f = lambda x: np.exp(x) + x ** 3
mono = f(phi)
# 3. white pixel noise of rms 0.3 sigma
noisy = phi + 0.3 * rng.standard_normal(phi.shape)
# 4. extra smoothing: the field seen through a wider beam (Gaussian of 6 pixels)
smooth = ndimage.gaussian_filter(phi, 6.0, mode="wrap")
smooth = smooth / smooth.std()

rows = {
    "Orig": summary(phi, NU),
    "Remap": summary(remap, NU),
    "Mono": None,
    "Noise": summary(noisy, NU),
    "Smooth": summary(smooth, NU),
}
# the monotone case: the set {f(phi) > f(nu)} is compared pixel by pixel with {phi > nu}
mset = mono > f(NU)
same = bool(np.array_equal(mset, phi > NU))
b0 = n_components(mset, 8, periodic=True)
chi = euler_cubical(mset, 8, periodic=True)
rows["Mono"] = (mset.mean(), rows["Orig"][1], b0, b0 - chi, chi)   # same pixels, same length

nums = {"ElNuisN": N, "ElNuisR": int(R), "ElNuisJac": round(float(jac), 2),
        "ElNuisMonoSame": "yes" if same else "no"}
for name, (a, l, b0, b1, chi) in rows.items():
    nums[f"ElNuis{name}Area"] = round(100 * a, 1)
    nums[f"ElNuis{name}Len"] = int(round(l))
    nums[f"ElNuis{name}Bzero"] = b0
    nums[f"ElNuis{name}Bone"] = b1
    nums[f"ElNuis{name}Chi"] = chi
    print(f"{name:7s} area {100*a:5.1f}%  length {l:7.0f}  b0 {b0:4d}  b1 {b1:3d}  chi {chi:4d}")
# the changes quoted in the text
o, r, n, s = rows["Orig"], rows["Remap"], rows["Noise"], rows["Smooth"]
nums.update({"ElNuisRemapDA": round(100 * abs(r[0] - o[0]), 1),          # area change, percentage points
             "ElNuisRemapDL": int(round(100 * abs(r[1] / o[1] - 1))),     # length change, per cent
             "ElNuisNoiseX": int(n[2] // max(o[2], 1)),                    # factor on the number of pieces
             "ElNuisSmoothLost": int(round(100 * (1 - s[2] / max(o[2], 1))))})   # per cent of pieces lost
save_numbers("ch12", "01_nuisance", nums)

fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.8))
for ax, (u, title, key) in zip(axs, [(phi, "original", "Orig"), (remap, "smooth remapping", "Remap"),
                                     (noisy, "added noise", "Noise"), (smooth, "extra smoothing", "Smooth")]):
    ax.imshow(u > NU, cmap="Blues", origin="lower", interpolation="nearest")
    a, l, b0, b1, chi = rows[key]
    ax.set_title(f"{title}\n" + rf"$\beta_0={b0},\ \beta_1={b1},\ \chi={chi}$", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
savefig(fig, "ch12", "nuisance_maps")
