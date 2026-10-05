"""05_bao_fisher.py -- how well can a survey measure the acoustic scale? A Fisher forecast.

Question: for a survey of given volume, galaxy density and bias, what error on the dilation alpha
does the Fisher matrix promise, and how does that compare with (i) our mock surveys of
04_bao_mocks.py, (ii) the first detection (Eisenstein et al. 2005) and (iii) DESI DR1's
bright-galaxy measurement (DESI 2024 VI, Table 1)?
Computes: the BAO-only Fisher information on alpha,
    F = int dk k^2/(4 pi^2) V_eff(k) [d ln P/d alpha]^2,   d ln P/d alpha = -k O'(k)/(1 + O(k)),
with O = (P_w - P_nw)/P_nw the damped wiggles; V_eff(k) = int [nP/(1+nP)]^2 dV; Eisenstein's
survey rebuilt from his numbers (3816 deg^2, 0.16 < z < 0.47, nbar = 1e-4 to z = 0.36 then
falling to 2.5e-5 at z = 0.47) and his effective volumes checked; D_V/r_d of the Planck 2018
fiducial model against the DESI DR1 distances (D_V from D_M, D_H and their correlation).
Writes: figures/ch11/bao_fisher.pdf, figures/ch11/bao_desi.pdf, results/ch11/05_bao_fisher.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from common import setup, savefig, save_numbers, SERIES, DATA, theory_line
from lib_lss import linear_pk_z0, nowiggle_pk, growth, E_of_z, OMM, H

C_KMS = 299792.458
k_tab, pk_tab, sigma8, rdrag = linear_pk_z0()
k = np.geomspace(2e-3, 0.5, 4000)
p_lin = np.interp(np.log(k), np.log(k_tab), pk_tab)
p_nw = nowiggle_pk(k, k_tab, pk_tab)


def dlnp_dalpha(sig_nl):
    """d ln P(k/alpha)/d alpha at alpha = 1 with the broad band held fixed: -k O'(k)/(1 + O)."""
    O = (p_lin / p_nw - 1) * np.exp(-0.5 * (k * sig_nl) ** 2)
    return -k * np.gradient(O, k) / (1 + O)


def sigma_alpha(veff_of_P, P, sig_nl, kmin=2e-3, kmax=0.3):
    """Fisher error on alpha. veff_of_P maps the galaxy power P(k) to V_eff(k) [(Mpc/h)^3]."""
    sel = (k >= kmin) & (k <= kmax)
    integrand = k ** 2 / (4 * np.pi ** 2) * veff_of_P(P) * dlnp_dalpha(sig_nl) ** 2
    F = np.trapezoid(integrand[sel], k[sel])
    return 1 / np.sqrt(F), integrand


# ---- (i) the mock box of 04_bao_mocks.py
m = np.load(DATA / "ch11" / "bao_mocks.npz")
L, nbar, b, z_m, sig_m = float(m["L"]), float(m["nbar"]), float(m["b"]), float(m["z"]), float(m["signl"])
D_m, f_m = growth(z_m)
beta = f_m[0] / b
P_box = (b * D_m[0]) ** 2 * (1 + 2 * beta / 3 + beta ** 2 / 5) * p_lin
veff_box = lambda P: L ** 3 * (nbar * P / (1 + nbar * P)) ** 2
s_box, integ_box = sigma_alpha(veff_box, P_box, sig_m, kmin=2 * np.pi / L, kmax=np.pi * 96 / L)
sd_mock = float(np.std(m["a_w"], ddof=1))
med_mock = float(np.median(m["s_w"]))


# ---- (ii) Eisenstein et al. (2005): rebuild the survey from the numbers in his Section 2
def chi_of_z(z, om):
    zz = np.linspace(0, z, 2000)
    return C_KMS / 100 * np.trapezoid(1 / np.sqrt(om * (1 + zz) ** 3 + 1 - om), zz)   # Mpc/h


fsky = 3816 / (4 * np.pi * (180 / np.pi) ** 2)
zs = np.linspace(0.16, 0.47, 400)
chis = np.array([chi_of_z(z, 0.3) for z in zs])
dV = 4 * np.pi * fsky * chis ** 2 * np.gradient(chis, zs)                  # (Mpc/h)^3 per unit z
n_z = np.where(zs < 0.36, 1e-4, 1e-4 * (2.5e-5 / 1e-4) ** ((zs - 0.36) / (0.47 - 0.36)))
V_eis = np.trapezoid(dV, zs)
N_eis = np.trapezoid(n_z * dV, zs)
veff_eis = lambda P: np.trapezoid((n_z[None, :] * np.atleast_1d(P)[:, None]
                                   / (1 + n_z[None, :] * np.atleast_1d(P)[:, None])) ** 2 * dV[None, :], zs, axis=1)
veff_checks = veff_eis(np.array([1e4, 4e4, 1e5])) / 1e9
P_eis = (1.8 / sigma8) ** 2 * p_lin                  # galaxy power with sigma_8 = 1.8, as in his Fig. 1
s_eis, integ_eis = sigma_alpha(veff_eis, P_eis, 8.0)

# ---- (iii) DESI DR1 BGS: V_eff = 1.7 Gpc^3 from Table 1, taken to hold at every k
veff_desi = lambda P: 1.7e9 * H ** 3 * np.ones_like(P)          # Gpc^3 -> (Mpc/h)^3
s_desi_rec, integ_desi = sigma_alpha(veff_desi, p_lin, 4.0)
s_desi_pre, _ = sigma_alpha(veff_desi, p_lin, 8.0)
desi_frac = 0.15 / 7.93


# ---- the Planck 2018 fiducial D_V/r_d against the DESI DR1 distances
def dists(z):
    dm = np.array([chi_of_z(zz, OMM) for zz in np.atleast_1d(z)]) / H        # Mpc
    dh = C_KMS / (100 * H * E_of_z(np.atleast_1d(z)))
    return dm, dh, (np.atleast_1d(z) * dm ** 2 * dh) ** (1 / 3)


desi = [  # tracer, z_eff, DM/rd, sig, DH/rd, sig, r, DV/rd, sig   (DESI 2024 VI, Table 1)
    ("BGS", 0.295, None, None, None, None, None, 7.93, 0.15),
    ("LRG1", 0.510, 13.62, 0.25, 20.98, 0.61, -0.445, None, None),
    ("LRG2", 0.706, 16.85, 0.32, 20.08, 0.60, -0.420, None, None),
    ("LRG3+ELG1", 0.930, 21.71, 0.28, 17.88, 0.35, -0.389, None, None),
    ("ELG2", 1.317, 27.79, 0.69, 13.82, 0.42, -0.444, None, None),
    ("QSO", 1.491, None, None, None, None, None, 26.07, 0.67),
    ("Lya", 2.330, 39.71, 0.94, 8.52, 0.17, -0.477, None, None),
]
pts = []
for name, z, dm, sdm, dh, sdh, rr, dv, sdv in desi:
    if dv is None:
        dv = (z * dm ** 2 * dh) ** (1 / 3)
        # delta method on ln D_V = (2 ln D_M + ln D_H + ln z)/3, with the D_M-D_H correlation r
        a, c = sdm / dm, sdh / dh
        sdv = dv * np.sqrt(4 * a ** 2 + c ** 2 + 4 * rr * a * c) / 3
    pred = dists(z)[2][0] / rdrag
    pts.append((name, z, dv, sdv, pred))

zz = np.linspace(0.05, 2.6, 300)
dv_curve = dists(zz)[2] / rdrag
setup(6.4, 3.0)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
axs[0].plot(zz, dv_curve / zz ** (2 / 3), color="k", ls="--", lw=1.3, label=r"Planck 2018 $\Lambda$CDM")
for i, (name, z, dv, sdv, pred) in enumerate(pts):
    axs[0].errorbar(z, dv / z ** (2 / 3), sdv / z ** (2 / 3), fmt="o", ms=4, color=SERIES[i % len(SERIES)], label=name)
axs[0].set_xlabel("redshift $z$")
axs[0].set_ylabel(r"$D_V/(r_d\,z^{2/3})$")
axs[0].legend(fontsize=6.5, ncol=2, loc="lower right")
pull = [(dv - pred) / sdv for (_, _, dv, sdv, pred) in pts]
axs[1].axhline(0, color="0.5", lw=0.6)
for i, (name, z, dv, sdv, pred) in enumerate(pts):
    axs[1].errorbar(z, 100 * (dv / pred - 1), 100 * sdv / pred, fmt="o", ms=4, color=SERIES[i % len(SERIES)])
axs[1].set_xlabel("redshift $z$")
axs[1].set_ylabel(r"$D_V/D_V^{\rm Planck}-1$ [%]")
fig.tight_layout()
savefig(fig, "ch11", "bao_desi")

# ---- where the information comes from: dF/dln k for the mock box, for three damping lengths
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
axs[0].semilogx(k, p_lin / p_nw - 1, color="0.5", lw=1, label=r"$\Sigma_{\rm nl}=0$")
for i, s in enumerate([4.0, 8.0]):
    axs[0].semilogx(k, (p_lin / p_nw - 1) * np.exp(-0.5 * (k * s) ** 2), color=SERIES[i], lw=1.4,
                    label=rf"$\Sigma_{{\rm nl}}={s:g}\,h^{{-1}}$Mpc")
axs[0].set_xlim(0.01, 0.5)
axs[0].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
axs[0].set_ylabel(r"wiggles $O(k)=P/P_{\rm nw}-1$")
axs[0].legend(fontsize=7)
for i, s in enumerate([0.0, 4.0, 8.0]):
    _, integ = sigma_alpha(veff_box, P_box, s, kmin=2 * np.pi / L, kmax=0.5)
    axs[1].semilogx(k, k * integ / 1e3, color=["0.5", SERIES[0], SERIES[1]][i], lw=1.4)
axs[1].set_xlim(0.01, 0.5)
axs[1].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
axs[1].set_ylabel(r"$\mathrm{d}F_{\alpha\alpha}/\mathrm{d}\ln k\ [10^3]$")
for a in axs:  # plain tick labels on the log axis, no crowded minor labels
    a.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_formatter(plt.NullFormatter())
fig.tight_layout()
savefig(fig, "ch11", "bao_fisher")

bgs = pts[0]
lrg2 = pts[2]
save_numbers("ch11", "05_bao_fisher", {
    "FishBox": f"{100*s_box:.1f}", "FishMockSd": f"{100*sd_mock:.1f}", "FishMockMed": f"{100*med_mock:.1f}",
    "FishVeis": f"{V_eis/1e9:.2f}", "FishNeis": f"{N_eis:.0f}",
    "FishVeffA": f"{veff_checks[0]:.2f}", "FishVeffB": f"{veff_checks[1]:.2f}", "FishVeffC": f"{veff_checks[2]:.2f}",
    "FishEis": f"{100*s_eis:.1f}", "FishDesiRec": f"{100*s_desi_rec:.1f}", "FishDesiPre": f"{100*s_desi_pre:.1f}",
    "FishDesiFrac": f"{100*desi_frac:.1f}", "FishDesiVh": f"{1.7*H**3:.2f}",
    "FishBgsPred": f"{bgs[4]:.2f}", "FishBgsPull": f"{(bgs[2]-bgs[4])/bgs[3]:.1f}",
    "FishLrgTwoDv": f"{lrg2[2]:.2f}", "FishLrgTwoSd": f"{lrg2[3]:.2f}", "FishLrgTwoPred": f"{lrg2[4]:.2f}",
    "FishRd": f"{rdrag:.2f}", "FishMaxPull": f"{max(abs(p) for p in pull):.1f}",
    "FishChiTwo": f"{sum(p**2 for p in pull):.1f}", "FishNpts": len(pull),
    "FishChiP": f"{chi2_dist.sf(sum(p**2 for p in pull), len(pull)):.2f}",
    "FishMaxP": f"{1 - (1 - 2 * norm_dist.sf(max(abs(p) for p in pull))) ** len(pull):.2f}",
})
for p in pts:
    print(f"{p[0]:10s} z={p[1]:.3f}  DV/rd={p[2]:.2f}+-{p[3]:.2f}  Planck {p[4]:.2f}")
print(f"box {s_box:.4f} mocks sd {sd_mock:.4f} med {med_mock:.4f}; Eis V={V_eis/1e9:.2f} N={N_eis:.0f} "
      f"Veff={veff_checks} sig={s_eis:.4f}; DESI rec {s_desi_rec:.4f} pre {s_desi_pre:.4f} vs {desi_frac:.4f}")
