"""03_fkp.py -- the FKP power-spectrum estimator in a survey whose density falls with distance.

Question: when the mean galaxy density n_bar(x) varies across a survey, does weighting each
position by w = 1/(1 + n_bar P0) shrink the scatter of the estimated P(k) by the amount the
Feldman-Kaiser-Peacock (FKP) calculation predicts, and what does the window do to the mean?
Computes: NSIM lognormal galaxy fields (side 1000 Mpc/h, 128^3 cells, b = 1.5 times the z = 0
linear spectrum); a spherical survey around the box centre with n_bar(r) = n0 exp(-(r/r0)^2) out
to r_max; Poisson counts; the FKP field F = w (n_g - n_bar)/sqrt(int n_bar^2 w^2) for w = 1 and
for the FKP weight; P_hat = |F_k|^2 - P_shot shell-averaged; the exact window-convolved
expectation; the predicted variance ratio from the FKP variance formula; V_eff(k).
Writes: figures/ch11/fkp_survey.pdf, figures/ch11/fkp_pk.pdf, results/ch11/03_fkp.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES
from lib_lss import linear_pk_z0, xi_box_from_pk, Lognormal, poisson_sample, KBins, rgrid

L, N, B = 1000.0, 128, 1.5
NSIM = 100
N0, R0, RMAX, P0 = 3e-3, 200.0, 450.0, 1e4
rng = rng_for("ch11", "03_fkp")

k_tab, pk_tab, _, _ = linear_pk_z0()
P = lambda k: B ** 2 * np.interp(k, k_tab, pk_tab)
xi_grid = xi_box_from_pk(P, N, L, 3)
ln = Lognormal(xi_grid, N, L, 3)
D = L / N
dV = D ** 3

# selection function: distance from the box centre, measured to cell centres
c = (np.arange(N) + 0.5) * D - L / 2
X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
nbar = np.where(r < RMAX, N0 * np.exp(-(r / R0) ** 2), 0.0)

weights = {"w = 1": np.where(nbar > 0, 1.0, 0.0), "FKP": np.where(nbar > 0, 1 / (1 + nbar * P0), 0.0)}
edges = np.arange(0.01, 0.31, 0.02)
kb = KBins(N, L, 3, edges)
k = kb.kmean
ptrue = kb.avg(np.fft.rfftn(xi_grid).real * dV)          # the grid spectrum, shell-averaged


def fkp_setup(w):
    norm = np.sum(nbar ** 2 * w ** 2) * dV
    pshot = np.sum(nbar * w ** 2) * dV / norm               # FKP eq. (2.1.9) with infinitely many randoms
    # exact expectation: FT of xi(r) times the window autocorrelation Q(r) of n_bar w
    g = np.fft.rfftn(nbar * w)
    Q = np.fft.irfftn(np.abs(g) ** 2, s=nbar.shape) * dV / norm      # Q(0) = 1
    pconv = kb.avg(np.fft.rfftn(Q * xi_grid).real * dV)
    # FKP variance (their 2.3.2) evaluated at each shell's P: (2pi)^3/V_k * int n^4 w^4 (1+1/nP)^2 / norm^2,
    # times 2 because the shell holds every independent mode twice (k and -k carry the same |F_k|^2)
    vk = 4 * np.pi * k ** 2 * np.diff(edges)
    sel = nbar > 0
    frac = np.array([np.sum(nbar[sel] ** 4 * w[sel] ** 4 * (1 + 1 / (nbar[sel] * p)) ** 2) * dV / norm ** 2
                     for p in ptrue]) * 2 * (2 * np.pi) ** 3 / vk
    return norm, pshot, pconv, frac


setups = {name: fkp_setup(w) for name, w in weights.items()}
est = {name: [] for name in weights}
for s in range(NSIM):
    delta = ln.draw(rng)
    ng = poisson_sample(1 + delta, 1.0, L, rng, window=nbar) / dV     # galaxies per unit volume
    for name, w in weights.items():
        norm, pshot, _, _ = setups[name]
        F = w * (ng - nbar) / np.sqrt(norm)
        pk = np.abs(np.fft.rfftn(F) * dV) ** 2
        est[name].append(kb.avg(pk) - pshot)
est = {n: np.array(v) for n, v in est.items()}

# effective volume V_eff(k) = int [n P/(1 + n P)]^2 d^3x  (FKP; Tegmark 1997)
sel = nbar > 0
veff = np.array([np.sum((nbar[sel] * p / (1 + nbar[sel] * p)) ** 2) * dV for p in ptrue])
vgeo = sel.sum() * dV
ngal = nbar.sum() * dV

setup(6.4, 3.0)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
rr = np.linspace(0, RMAX, 300)
nr = N0 * np.exp(-(rr / R0) ** 2)
axs[0].semilogy(rr, nr * P0, color=SERIES[0], label=r"$\bar n(r)\,P_0$")
axs[0].semilogy(rr, 1 / (1 + nr * P0), color=SERIES[1], label=r"$w_{\rm FKP}=1/(1+\bar nP_0)$")
axs[0].semilogy(rr, nr * P0 / (1 + nr * P0) ** 1, color=SERIES[2], label=r"$\bar n w_{\rm FKP}\,P_0$")
axs[0].axhline(1, color="0.5", lw=0.6)
axs[0].set_xlabel(r"distance from observer $r\ [h^{-1}\mathrm{Mpc}]$")
axs[0].legend(fontsize=7)
axs[0].set_title("selection and weight")
sl = np.s_[N // 2]
im = axs[1].imshow(np.log10(nbar[sl] + 1e-6).T, origin="lower", extent=[-L / 2, L / 2, -L / 2, L / 2],
                   cmap="Blues", vmin=-5.5, vmax=np.log10(N0))
axs[1].set_title(r"slice of $\log_{10}\bar n(\mathbf{x})$")
axs[1].grid(False)
fig.colorbar(im, ax=axs[1], fraction=0.046)
fig.tight_layout()
savefig(fig, "ch11", "fkp_survey")

fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.1))
for i, name in enumerate(weights):
    m = est[name].mean(0)
    e = est[name].std(0, ddof=1) / np.sqrt(NSIM)
    axs[0].errorbar(k * (1 + 0.01 * i), m / ptrue, e / ptrue, fmt="o", ms=3, color=SERIES[i], label=f"mean, {name}")
    axs[0].plot(k, setups[name][2] / ptrue, color=SERIES[i], lw=1, ls="--")
axs[0].axhline(1, color="k", lw=0.6)
axs[0].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
axs[0].set_ylabel(r"$\langle\hat P\rangle/P$")
axs[0].set_title("mean (dashed: window-convolved $P$)")
axs[0].legend(fontsize=7, loc="lower right")
for i, name in enumerate(weights):
    sd = est[name].std(0, ddof=1)
    axs[1].plot(k, sd / ptrue, "o", ms=3, color=SERIES[i], label=f"measured, {name}")
    axs[1].plot(k, np.sqrt(setups[name][3]), color=SERIES[i], lw=1)
axs[1].plot([], [], "k-", lw=1, label="Gaussian prediction")
axs[1].set_yscale("log")
axs[1].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
axs[1].set_ylabel(r"$\sigma(\hat P)/P$")
axs[1].set_title("scatter")
axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch11", "fkp_pk")

sd1, sdF = est["w = 1"].std(0, ddof=1), est["FKP"].std(0, ddof=1)
j = np.argmin(np.abs(k - 0.1))
jl = np.argmin(np.abs(k - 0.05))
save_numbers("ch11", "03_fkp", {
    "FkpNsim": NSIM, "FkpNzero": f"{N0:g}", "FkpRzero": f"{R0:g}", "FkpRmax": f"{RMAX:g}", "FkpPzero": f"{P0:g}",
    "FkpNgal": f"{ngal/1e3:.0f}", "FkpVgeo": f"{vgeo/1e9:.2f}",
    "FkpVeffK": f"{veff[j]/1e9:.2f}", "FkpVeffKl": f"{veff[jl]/1e9:.2f}",
    "FkpKa": f"{k[jl]:.2f}", "FkpKb": f"{k[j]:.2f}",
    "FkpRatioA": f"{sdF[jl]/sd1[jl]:.2f}", "FkpRatioB": f"{sdF[j]/sd1[j]:.2f}",
    "FkpPredA": f"{np.sqrt(setups['FKP'][3][jl]/setups['w = 1'][3][jl]):.2f}",
    "FkpPredB": f"{np.sqrt(setups['FKP'][3][j]/setups['w = 1'][3][j]):.2f}",
    "FkpShotOne": f"{setups['w = 1'][1]:.0f}", "FkpShotF": f"{setups['FKP'][1]:.0f}",
    "FkpMeanB": f"{est['FKP'].mean(0)[j]/setups['FKP'][2][j]:.3f}",
    "FkpExcessA": f"{sd1[jl]/ptrue[jl]/np.sqrt(setups['w = 1'][3][jl]):.2f}",
    "FkpExcessB": f"{sd1[j]/ptrue[j]/np.sqrt(setups['w = 1'][3][j]):.2f}",
    "FkpConvLow": f"{setups['FKP'][2][0]/ptrue[0]:.2f}", "FkpKlow": f"{k[0]:.2f}",
})
