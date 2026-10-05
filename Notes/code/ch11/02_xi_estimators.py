"""02_xi_estimators.py -- four pair-count estimators of xi(r) in a survey with edges and holes.

Question: for a clustered point set seen through an irregular window, which combination of the
pair counts DD, DR, RR gives xi(r) with the smallest scatter, and is any of them biased?
Computes: a 2-D lognormal field (side 1000 Mpc/h, 256^2 cells, known xi); a survey window
(a 700 x 700 square with five holes and a notch); NSIM Poisson samples of the field inside the
window; a fresh random catalogue (NR_FAC times denser) for every mock, and, for comparison, one
catalogue reused for all mocks; DD, DR, RR with scipy's cKDTree;
the natural DD/RR - 1, Davis-Peebles DD/DR - 1, Hamilton DD RR/DR^2 - 1 and Landy-Szalay
(DD - 2DR + RR)/RR estimators; their mean and scatter against the true xi, the integral
constraint xi_bar predicted from the window, and the Poisson floor. The same estimators on the
full periodic box (no edges) as a control.
Writes: figures/ch11/xi_survey.pdf, figures/ch11/xi_estimators.pdf, results/ch11/02_xi_estimators.tex,
data/ch11/xi_estimators.npz
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES, DATA
from lib_lss import xi_box_from_pk, Lognormal, poisson_sample, points_from_counts, RadialBins, rgrid

L, N, D2 = 1000.0, 256, 2
import os
NSIM, NSIM_PER = int(os.environ.get("NSIM", 300)), int(os.environ.get("NSIMP", 150))
NBAR = 0.012                      # points per (Mpc/h)^2 inside the window
NR_FAC = 5
EDGES = np.arange(5.0, 160.0, 10.0)
rng = rng_for("ch11", "02_xi_estimators")

# ---- the field: a smooth 2-D spectrum with a turnover, normalised to grid variance SIG2
K0, SIG2 = 0.03, 1.0
Pshape = lambda k: (k / K0) / (1 + (k / K0) ** 2) ** 1.8
xi0 = xi_box_from_pk(Pshape, N, L, D2)
xi_target = xi0 * SIG2 / xi0.flat[0]
ln = Lognormal(xi_target, N, L, D2)
rb = RadialBins(N, L, D2, EDGES)
xi_true = rb(xi_target)

# ---- the window: square [150, 850]^2, five circular holes, a rectangular notch
D = L / N
xc = (np.arange(N) + 0.5) * D
X, Y = np.meshgrid(xc, xc, indexing="ij")
W = ((X > 150) & (X < 850) & (Y > 150) & (Y < 850)).astype(float)
for (cx, cy, rad) in [(300, 320, 45), (560, 260, 30), (700, 620, 60), (420, 640, 25), (620, 440, 20)]:
    W[(X - cx) ** 2 + (Y - cy) ** 2 < rad ** 2] = 0
W[(X > 150) & (X < 400) & (Y > 760)] = 0
area = W.sum() * D * D

# integral constraint: xi_bar = (1/A^2) sum over cell pairs in the window of xi (periodic box)
Wk = np.fft.fftn(W)
Q = np.real(np.fft.ifftn(np.abs(Wk) ** 2)) / W.sum() ** 2
xi_bar = float(np.sum(Q * xi_target))

# ---- random catalogues, uniform inside the window
nr = int(NR_FAC * NBAR * area)


def counts(t1, t2, edges):
    c = t1.count_neighbors(t2, edges)     # ordered pairs with separation <= edge
    return np.diff(c).astype(float)


def randoms(rng):
    cand = rng.random((int(nr / (W.mean()) * 1.3), 2)) * L
    inw = W[(cand[:, 0] // D).astype(int), (cand[:, 1] // D).astype(int)] > 0
    t = cKDTree(cand[inw][:nr])
    return t, counts(t, t, EDGES) / (nr * (nr - 1))


tr_fix, rr_fix = randoms(rng)          # one catalogue reused for every mock (the comparison)


def estimators(dd, dr, rr):
    return np.array([dd / rr - 1, dd / dr - 1, dd * rr / dr ** 2 - 1, (dd - 2 * dr + rr) / rr])


NAMES = ["natural $DD/RR-1$", "Davis--Peebles $DD/DR-1$", "Hamilton $DD\\,RR/DR^2-1$", "Landy--Szalay"]
t0 = time.time()
est, est_fix, ddpairs = [], [], []
for s in range(NSIM):
    delta = ln.draw(rng)
    c = poisson_sample(1 + delta, NBAR, L, rng, window=W)
    g = points_from_counts(c, L, rng)
    ng = len(g)
    tg = cKDTree(g)
    ddc = counts(tg, tg, EDGES)
    dd = ddc / (ng * (ng - 1))
    tr_r, rr = randoms(rng)              # a fresh random catalogue for this mock
    dr = counts(tg, tr_r, EDGES) / (ng * nr)
    est.append(estimators(dd, dr, rr))
    dr_fix = counts(tg, tr_fix, EDGES) / (ng * nr)
    est_fix.append(estimators(dd, dr_fix, rr_fix))
    ddpairs.append(ddc / 2)
    if s == 0:
        g_example = g
est = np.array(est)                    # (nsim, 4, nbins)
est_fix = np.array(est_fix)
print(f"masked sims: {time.time()-t0:.0f} s")

# ---- control: full periodic box, no edges (randoms over the whole box, periodic distances)
nbar_p = NBAR * area / L ** 2           # same number of points as the survey
nr_p = int(NR_FAC * nbar_p * L ** 2)
est_p = []
for s in range(NSIM_PER):
    delta = ln.draw(rng)
    c = poisson_sample(1 + delta, nbar_p, L, rng)
    g = points_from_counts(c, L, rng) % L
    ng = len(g)
    tg = cKDTree(g, boxsize=L)
    tr_p = cKDTree(rng.random((nr_p, 2)) * L, boxsize=L)
    rr_p = counts(tr_p, tr_p, EDGES) / (nr_p * (nr_p - 1))
    dd = counts(tg, tg, EDGES) / (ng * (ng - 1))
    dr = counts(tg, tr_p, EDGES) / (ng * nr_p)
    est_p.append(estimators(dd, dr, rr_p))
est_p = np.array(est_p)
print(f"total: {time.time()-t0:.0f} s")
np.savez(DATA / "ch11" / "xi_estimators.npz", est=est, est_fix=est_fix, est_p=est_p, xi_true=xi_true, edges=EDGES)

rmid = rb.rmean
mean, sd = est.mean(0), est.std(0, ddof=1)
mean_fix = est_fix.mean(0)
sd_p = est_p.std(0, ddof=1)
pois = (1 + xi_true) / np.sqrt(np.mean(ddpairs, axis=0))       # Poisson floor (1+xi)/sqrt(DD)
xi_ic = (1 + xi_true) / (1 + xi_bar) - 1                         # integral-constraint expectation

# ---- figure 1: the survey and one realisation
setup(6.0, 3.0)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.1))
dshow = ln.draw(rng_for("ch11", "02_xi_estimators", 7))
axs[0].imshow(np.log1p(dshow).T, origin="lower", extent=[0, L, 0, L], cmap="Blues")
axs[0].contour(xc, xc, W.T, levels=[0.5], colors=SERIES[1], linewidths=1.0)
axs[0].set_title(r"field $\ln(1+\delta)$ and the window")
axs[1].scatter(g_example[:, 0], g_example[:, 1], s=0.3, color=SERIES[0], rasterized=True)
axs[1].set_title(f"one catalogue: {len(g_example)} points")
for a in axs:
    a.set_xlim(0, L); a.set_ylim(0, L); a.set_aspect("equal"); a.set_xticks([0, 500, 1000]); a.set_yticks([0, 500, 1000])
    a.grid(False)
savefig(fig, "ch11", "xi_survey")

# ---- figure 2: bias and scatter of the four estimators
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.3))
off = [-1.5, -0.5, 0.5, 1.5]
for i in range(4):
    axs[0].errorbar(rmid + off[i], (mean[i] - xi_true) / sd[3], yerr=None, fmt="o-", ms=3, lw=1,
                    color=SERIES[i], label=NAMES[i])
axs[0].plot(rmid + off[0], (mean_fix[0] - xi_true) / sd[3], "o:", ms=3, lw=1, mfc="none", color=SERIES[0],
            label="natural, one catalogue for all mocks")
axs[0].plot(rmid, (xi_ic - xi_true) / sd[3], "k--", lw=1.2, label=r"$(1+\xi)/(1+\bar\xi)-1-\xi$")
axs[0].axhline(0, color="0.5", lw=0.6)
axs[0].set_xlabel(r"$r\ [h^{-1}\mathrm{Mpc}]$")
axs[0].set_ylabel(r"$(\langle\hat\xi\rangle-\xi)/\sigma_{\rm LS}$")
axs[0].set_title("bias, in units of the LS scatter")
axs[0].set_ylim(top=max(axs[0].get_ylim()[1], 0.0) + 0.45 * np.ptp(axs[0].get_ylim()))   # room for the legend
axs[0].legend(fontsize=6, loc="upper left", ncol=1, framealpha=0.9)
for i in range(4):
    axs[1].semilogy(rmid, sd[i], "o-", ms=3, lw=1, color=SERIES[i])
    axs[1].semilogy(rmid, sd_p[i], ":", lw=1.2, color=SERIES[i])
axs[1].semilogy(rmid, pois, "k--", lw=1.2, label=r"Poisson $(1+\xi)/\sqrt{DD}$")
axs[1].plot([], [], "k-", lw=1, label="survey with edges")
axs[1].plot([], [], "k:", lw=1.2, label="periodic box, no edges")
axs[1].set_xlabel(r"$r\ [h^{-1}\mathrm{Mpc}]$")
axs[1].set_ylabel(r"standard deviation of $\hat\xi(r)$")
axs[1].set_title("scatter")
axs[1].legend(fontsize=6.5, loc="lower left")
fig.tight_layout()
savefig(fig, "ch11", "xi_estimators")

j = np.argmin(np.abs(rmid - 100))     # a large-scale bin
j2 = np.argmin(np.abs(rmid - 30))
save_numbers("ch11", "02_xi_estimators", {
    "XiNsim": NSIM, "XiNsimPer": NSIM_PER, "XiNbar": f"{NBAR:g}", "XiNg": int(round(NBAR * area)),
    "XiNr": nr, "XiArea": f"{area/1e3:.0f}", "XiBar": f"{xi_bar:.4f}",
    "XiRbig": f"{rmid[j]:.0f}", "XiTrueBig": f"{xi_true[j]:.4f}", "XiRmid": f"{rmid[j2]:.0f}",
    "XiSdNatBig": f"{sd[0, j]:.4f}", "XiSdDPBig": f"{sd[1, j]:.4f}", "XiSdHamBig": f"{sd[2, j]:.4f}",
    "XiSdLSBig": f"{sd[3, j]:.4f}", "XiSdPoisBig": f"{pois[j]:.4f}",
    "XiRatioNatBig": f"{sd[0, j]/sd[3, j]:.2f}", "XiRatioDPBig": f"{sd[1, j]/sd[3, j]:.2f}",
    "XiRatioHamBig": f"{sd[2, j]/sd[3, j]:.2f}",
    "XiRatioNatMid": f"{sd[0, j2]/sd[3, j2]:.2f}", "XiRatioDPMid": f"{sd[1, j2]/sd[3, j2]:.2f}",
    "XiPerRatioNatBig": f"{sd_p[0, j]/sd_p[3, j]:.2f}", "XiPerRatioDPBig": f"{sd_p[1, j]/sd_p[3, j]:.2f}",
    "XiMeanLSBig": f"{mean[3, j]:.4f}", "XiICBig": f"{xi_ic[j]:.4f}",
    "XiMeanNatBig": f"{mean[0, j]:.4f}", "XiMeanHamBig": f"{mean[2, j]:.4f}", "XiMeanDPBig": f"{mean[1, j]:.4f}",
    "XiErrMeanBig": f"{sd[3, j]/np.sqrt(NSIM):.4f}", "XiErrMeanNatBig": f"{sd[0, j]/np.sqrt(NSIM):.4f}",
    "XiErrMeanDPBig": f"{sd[1, j]/np.sqrt(NSIM):.4f}",
    "XiFixNatBig": f"{mean_fix[0, j]:.4f}", "XiFixDPBig": f"{mean_fix[1, j]:.4f}",
    "XiFixHamBig": f"{mean_fix[2, j]:.4f}", "XiFixLSBig": f"{mean_fix[3, j]:.4f}", "XiTrueMid": f"{xi_true[j2]:.3f}",
    "XiClip": f"{100*ln.clipped:.2f}",
})
