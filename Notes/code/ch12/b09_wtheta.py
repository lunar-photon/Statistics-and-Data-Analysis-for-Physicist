"""b09_wtheta.py -- the angular correlation function of polarization singularities (Huterer-Vachaspati, their eqs. 4-6).

Question: apart from the count and the charge spread, how are the singularities of a Gaussian
E-mode map arranged on the sky?  Are they clustered or spread out compared with random points,
and does the answer depend on the signs of the two charges involved?  Huterer and Vachaspati (2005)
found w(theta) consistent with zero beyond about 10 degrees, negative for equal charges and
positive for opposite charges closer than about 10 degrees.
Computes: on periodic flat 40 x 40 degree patches with C_ell^EE (their cosmology, no reionization)
cut at l_max = 100, the catalogue of singularities of each of 60 maps; the Peebles-Hauser estimator
w = DD/RR - 1 in bins of separation, for all pairs, equal-charge pairs and opposite-charge pairs.
RR is the expected number of pairs for points thrown at random on the same torus (exact:
the fraction of the torus inside the annulus), so no random catalogue is needed.
Writes: figures/ch12/b_wtheta.pdf, results/ch12/b09_wtheta.tex
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES, DATA
from lib_persist import flat_qu, winding

rng = rng_for("ch12", "b09_wtheta")
ee = np.load(DATA / "ch12" / "b_hv_spectra.npz")["EE_noreion"]   # written by b07_hv_counts.py

SIDE, NPIX, LMAX, NSIM = 40.0, 256, 100, 60
pix = SIDE / NPIX                                               # degrees per pixel
edges = np.arange(0.0, 20.01, 2.5)                              # separation bins [deg]
area = SIDE ** 2
frac = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / area        # share of the torus in each annulus

w_all, w_same, w_opp = [], [], []
for s in range(NSIM):
    Q, U, _ = flat_qu(ee, NPIX, SIDE, rng, lmax=LMAX)
    w = winding(Q, U)
    i, j = np.nonzero(w)
    pos = (np.column_stack([i, j]) + 0.5) * pix
    q = w[i, j]
    d = np.abs(pos[:, None, :] - pos[None, :, :])
    d = np.minimum(d, SIDE - d)                                 # periodic separations
    r = np.hypot(d[..., 0], d[..., 1])
    iu = np.triu_indices(len(q), k=1)
    rr, sign = r[iu], q[iu[0]] * q[iu[1]]
    npl, nmi = (q > 0).sum(), (q < 0).sum()
    n = len(q)
    pairs_all = n * (n - 1) / 2
    pairs_same = npl * (npl - 1) / 2 + nmi * (nmi - 1) / 2
    pairs_opp = npl * nmi
    for store, mask, npairs in ((w_all, np.ones_like(sign, bool), pairs_all),
                                (w_same, sign > 0, pairs_same), (w_opp, sign < 0, pairs_opp)):
        dd = np.histogram(rr[mask], edges)[0]
        store.append(dd / (npairs * frac) - 1)                  # Peebles-Hauser: DD/RR - 1

w_all, w_same, w_opp = (np.array(a) for a in (w_all, w_same, w_opp))
mid = 0.5 * (edges[1:] + edges[:-1])
m = {k: (a.mean(axis=0), a.std(axis=0) / np.sqrt(NSIM)) for k, a in
     (("all", w_all), ("same", w_same), ("opp", w_opp))}
for k in m:
    print(k, np.round(m[k][0], 3), "+-", np.round(m[k][1], 3))

setup(4.6, 3.0)
fig, ax = plt.subplots()
for k, lab, c in (("same", "equal charges", SERIES[0]), ("opp", "opposite charges", SERIES[1]),
                  ("all", "all pairs", "0.4")):
    ax.errorbar(mid, m[k][0], m[k][1], fmt="o-", ms=3, lw=0.9, capsize=2, color=c, label=lab)
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.set_xlabel(r"separation $\theta$ [deg]")
ax.set_ylabel(r"$w(\theta)$")
ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch12", "b_wtheta")

far = mid > 10
save_numbers("ch12", "b09_wtheta", {
    "PbWtNsim": NSIM,
    "PbWtSame": m["same"][0][0], "PbWtOpp": m["opp"][0][0], "PbWtAll": m["all"][0][0],
    "PbWtSameErr": m["same"][1][0], "PbWtOppErr": m["opp"][1][0], "PbWtAllErr": m["all"][1][0],
    "PbWtFarMax": np.abs(np.r_[m["same"][0][far], m["opp"][0][far], m["all"][0][far]]).max(),
    "PbWtFarSigma": np.abs(np.r_[m["same"][0][far] / m["same"][1][far],
                                 m["opp"][0][far] / m["opp"][1][far],
                                 m["all"][0][far] / m["all"][1][far]]).max(),
})
