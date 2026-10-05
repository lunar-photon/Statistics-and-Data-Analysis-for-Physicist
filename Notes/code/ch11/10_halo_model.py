"""10_halo_model.py -- the halo model power spectrum of matter and of galaxies.

Question: if all matter sits in haloes with NFW profiles, Sheth-Tormen abundances and
peak-background-split biases, how well does P(k) = P_1h + P_2h reproduce the non-linear matter
power spectrum of N-body simulations (here: CAMB's halofit), and what does a halo occupation
distribution do to the galaxy power spectrum and correlation function?
Computes, at z = 0 with the CAMB linear P(k):
  the NFW transform u(k|M) (closed form, checked against direct integration), the PS and ST
  biases and the consistency integral int dM n (M/rho) b = 1, the one- and two-halo terms (with
  the mass below M_min assigned to the lowest-mass bin), their sum against halofit;
  a central + Poisson-satellite HOD: nbar_g, b_g, satellite fraction, P_gg = P_1h + P_2h, xi_gg(r);
  a Monte Carlo check of the pair count <N(N-1)|M> = 2 <N_s> + <N_s>^2/<N_c>.
Writes: figures/ch11/halo_uk_bias.pdf, figures/ch11/halo_model_pk.pdf, figures/ch11/hod_pk.pdf,
results/ch11/10_halo_model.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line, DATA
from lib_web import (RHO_M, DELTA_C, DELTA_HALO, F_ps, F_st, bias_ps, bias_st, dndlnM, m_star,
                     concentration, r_halo, u_nfw, K_TAB, PK_TAB)
from lib_lss import xi_from_pk

rng = rng_for("ch11", "10_halo_model")
pk_nl = np.load(DATA / "chT1" / "pk.npz")["pk_nl"][0]

# ---------------------------------------------------------------- ingredients on a mass grid
lnM = np.linspace(np.log(1e7), np.log(3e15), 500)
M = np.exp(lnM)
n_st, nu, sig = dndlnM(M, "st")                   # dn/dlnM
n_ps, _, _ = dndlnM(M, "ps")
b_st, b_ps = bias_st(nu), bias_ps(nu)
mst = m_star()
c = concentration(M, mst)
k = np.geomspace(1e-3, 30.0, 300)
u = u_nfw(k, M, c)                                 # (nM, nk)

# check the closed form against a direct integral for one halo
Mt, ct = 1e14, float(concentration(1e14, mst))
rs = r_halo(Mt) / ct
r = np.linspace(1e-5, r_halo(Mt), 200001)
rho_r = 1 / ((r / rs) * (1 + r / rs) ** 2)
mass = np.trapezoid(4 * np.pi * r ** 2 * rho_r, r)
k_chk = np.array([0.3, 1.0, 3.0])
u_num = [np.trapezoid(4 * np.pi * r ** 2 * rho_r * np.sinc(kk * r / np.pi), r) / mass for kk in k_chk]
u_cf = u_nfw(k_chk, Mt, ct)[0]
u_err = float(np.max(np.abs(np.array(u_num) / u_cf - 1)))

# consistency integrals over the grid (they would be exactly 1 if the grid reached M -> 0)
mass_int = float(np.trapezoid(n_st * M / RHO_M, lnM))
bias_int = float(np.trapezoid(n_st * M / RHO_M * b_st, lnM))
bias_int_ps = float(np.trapezoid(n_ps * M / RHO_M * b_ps, lnM))

# ---------------------------------------------------------------- matter: one- and two-halo terms
P_lin = np.interp(k, K_TAB, PK_TAB)
P_hf = np.interp(k, K_TAB, pk_nl)
w1 = n_st * (M / RHO_M) ** 2                        # weight per dlnM of the one-halo term
P1h = np.trapezoid(w1[:, None] * u ** 2, lnM, axis=0)
I2 = np.trapezoid((n_st * M / RHO_M * b_st)[:, None] * u, lnM, axis=0) + (1 - bias_int) * u[0]
P2h = P_lin * I2 ** 2
P_hm = P1h + P2h
# the same model with the halo edge at 200 rho_m instead of the virial overdensity
P1h_200 = np.trapezoid(w1[:, None] * u_nfw(k, M, c, delta=200.0) ** 2, lnM, axis=0)
D2 = lambda p: k ** 3 * p / (2 * np.pi ** 2)
ratio = P_hm / P_hf
at = lambda kk, arr: float(np.interp(np.log(kk), np.log(k), arr))
k_eq = float(k[np.argmin(np.abs(np.log(P1h / P2h)))])      # where the two terms are equal
worst = int(np.argmax(np.abs(ratio - 1) * (k > 0.05) * (k < 10)))
# which masses give the one-halo power at k = 1 h/Mpc
ik1 = np.argmin(np.abs(k - 1.0))
contrib = w1 * u[:, ik1] ** 2
cdf = np.cumsum(contrib) / np.sum(contrib)
m_lo, m_hi = np.exp(np.interp(0.1, cdf, lnM)), np.exp(np.interp(0.9, cdf, lnM))
lmw = float(np.exp(np.trapezoid(contrib * lnM, lnM) / np.trapezoid(contrib, lnM)))

# ---------------------------------------------------------------- galaxies: a simple HOD
HOD = dict(lMmin=12.8, sig=0.5, lM0=12.6, lM1=14.0, alpha=1.1)    # illustrative values, Zheng 2005 form
Nc = 0.5 * (1 + erf((np.log10(M) - HOD["lMmin"]) / HOD["sig"]))
lam = (np.clip(M - 10 ** HOD["lM0"], 0, None) / 10 ** HOD["lM1"]) ** HOD["alpha"]
Ns = Nc * lam                                       # satellites only where there is a central
Ntot = Nc + Ns
nbar_g = float(np.trapezoid(n_st * Ntot, lnM))
fsat = float(np.trapezoid(n_st * Ns, lnM) / nbar_g)
b_g = float(np.trapezoid(n_st * b_st * Ntot, lnM) / nbar_g)
M_eff = float(np.trapezoid(n_st * M * Ntot, lnM) / nbar_g)
P1h_g = np.trapezoid((n_st * (2 * Ns))[:, None] * u + (n_st * Nc * lam ** 2)[:, None] * u ** 2, lnM, axis=0) / nbar_g ** 2
P2h_g = P_lin * (np.trapezoid((n_st * b_st)[:, None] * (Nc[:, None] + Ns[:, None] * u), lnM, axis=0) / nbar_g) ** 2
P_gg = P1h_g + P2h_g
r_x = np.geomspace(0.1, 60, 120)
kk_f = np.geomspace(1e-3, 30.0, 4000)
xi_gg = xi_from_pk(kk_f, np.exp(np.interp(np.log(kk_f), np.log(k), np.log(P_gg))), r_x, damp=0.05)
xi_mm = xi_from_pk(kk_f, np.exp(np.interp(np.log(kk_f), np.log(k), np.log(P_hf))), r_x, damp=0.05)
sel = (r_x > 0.2) & (r_x < 10)
gam_g = -np.polyfit(np.log(r_x[sel]), np.log(xi_gg[sel]), 1)[0]
r0_g = float(np.exp(np.interp(0.0, np.log(xi_gg[::-1]), np.log(r_x[::-1]))))
# Monte Carlo check of the pair count in one halo mass
iM = np.argmin(np.abs(M - 1e14))
nh = 2000000
cen = rng.random(nh) < Nc[iM]
sat = np.where(cen, rng.poisson(lam[iM], nh), 0)
Nh = cen.astype(int) + sat
pairs_mc = float(np.mean(Nh * (Nh - 1)))
pairs_se = float(np.std(Nh * (Nh - 1)) / np.sqrt(nh))
pairs_th = float(2 * Ns[iM] + Nc[iM] * lam[iM] ** 2)

# ---- figure 1: u(k|M) and the biases
setup(6.8, 2.8)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for i, mm in enumerate([1e11, 1e12, 1e13, 1e14, 1e15]):
    j = np.argmin(np.abs(M - mm))
    axs[0].loglog(k, u[j], color=SERIES[i], label=rf"$10^{{{int(np.log10(mm))}}}$")
axs[0].set_ylim(1e-2, 1.5); axs[0].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$"); axs[0].set_ylabel(r"$u(k|M)$")
axs[0].legend(fontsize=7, title=r"$M\ [h^{-1}M_\odot]$", title_fontsize=7, loc="lower left")
axs[1].semilogx(M, b_ps, color=SERIES[0], ls=":", label="Press--Schechter")
axs[1].semilogx(M, b_st, color=SERIES[1], label="Sheth--Tormen")
axs[1].axhline(1, color="0.6", lw=0.8)
axs[1].set_xlim(1e9, 3e15); axs[1].set_ylim(0, 8)
axs[1].set_xlabel(r"$M\ [h^{-1}M_\odot]$"); axs[1].set_ylabel(r"bias $b(M)$")
axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch11", "halo_uk_bias")

# ---- figure 2: the halo model against halofit
fig, axs = plt.subplots(2, 1, figsize=(5.4, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
axs[0].loglog(k, D2(P_lin), color="0.55", ls="-.", lw=1.1, label="linear")
axs[0].loglog(k, D2(P1h), color=SERIES[2], ls=":", label="one-halo")
axs[0].loglog(k, D2(P2h), color=SERIES[3], ls="--", label="two-halo")
axs[0].loglog(k, D2(P_hm), color=SERIES[0], label="halo model")
axs[0].loglog(k, D2(P_hf), color="k", lw=0.9, label="halofit")
axs[0].set_ylim(1e-3, 3e3); axs[0].set_ylabel(r"$\Delta^2(k)=k^3P/2\pi^2$")
axs[0].legend(fontsize=7, loc="upper left")
axs[1].semilogx(k, ratio, color=SERIES[0])
axs[1].axhline(1, color="0.6", lw=0.8)
axs[1].set_ylim(0.6, 1.3); axs[1].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$"); axs[1].set_ylabel("halo model / halofit")
fig.tight_layout()
savefig(fig, "ch11", "halo_model_pk")

# ---- figure 3: HOD and the galaxy correlation function
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
axs[0].loglog(M, Nc, color=SERIES[0], ls="--", label=r"$\langle N_c|M\rangle$")
axs[0].loglog(M, Ns, color=SERIES[1], ls=":", label=r"$\langle N_s|M\rangle$")
axs[0].loglog(M, Ntot, color="k", lw=1.0, label=r"$\langle N|M\rangle$")
axs[0].set_xlim(1e11, 3e15); axs[0].set_ylim(1e-2, 50)
axs[0].set_xlabel(r"$M\ [h^{-1}M_\odot]$"); axs[0].set_ylabel("galaxies per halo")
axs[0].legend(fontsize=7)
axs[1].loglog(r_x, xi_gg, color=SERIES[0], label="galaxies (HOD)")
axs[1].loglog(r_x, xi_mm, color="k", lw=0.9, label="matter (halofit)")
theory_line(axs[1], r_x, (r_x / r0_g) ** (-gam_g), label=rf"power law $\gamma={gam_g:.2f}$")
axs[1].set_ylim(1e-3, 3e3); axs[1].set_xlim(0.1, 60)
axs[1].set_xlabel(r"$r\ [h^{-1}\mathrm{Mpc}]$"); axs[1].set_ylabel(r"$\xi(r)$")
axs[1].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch11", "hod_pk")

save_numbers("ch11", "10_halo_model", {
    "HmMstar": mst / 1e12, "HmCstar": 9.0, "HmDvir": DELTA_HALO, "HmCfourteen": ct, "HmUerr": u_err,
    "HmRvirFourteen": float(r_halo(1e14)), "HmRsFourteen": float(r_halo(1e14) / ct),
    "HmUkOneFourteen": float(u_nfw(np.array([1.0]), 1e14, ct)[0, 0]),
    "HmUkOneTwelve": float(u_nfw(np.array([1.0]), 1e12, concentration(1e12, mst))[0, 0]),
    "HmMassInt": mass_int, "HmBiasInt": bias_int, "HmBiasIntPS": bias_int_ps, "HmMmin": 1e7,
    "HmBtwelve": float(np.interp(np.log(1e12), lnM, b_st)), "HmBfourteen": float(np.interp(np.log(1e14), lnM, b_st)),
    "HmBfifteen": float(np.interp(np.log(1e15), lnM, b_st)),
    "HmBtwelvePS": float(np.interp(np.log(1e12), lnM, b_ps)), "HmBfifteenPS": float(np.interp(np.log(1e15), lnM, b_ps)),
    "HmBmin": float(b_st.min()), "HmBminPS": float(b_ps.min()),
    "HmRatioPtZeroOne": at(0.01, ratio), "HmRatioPtOne": at(0.1, ratio), "HmRatioPtThree": at(0.3, ratio),
    "HmRatioOne": at(1.0, ratio), "HmRatioThree": at(3.0, ratio), "HmRatioTen": at(10.0, ratio),
    "HmRatioTenTwoHundred": float(np.interp(np.log(10.0), np.log(k), (P1h_200 + P2h) / P_hf)), "HmKworst": float(k[worst]), "HmRatioWorst": float(ratio[worst]), "HmKeq": k_eq,
    "HmOneHaloConst": float(P1h[0]), "HmMlo": m_lo / 1e13, "HmMhi": m_hi / 1e13,
    "HodLMmin": HOD["lMmin"], "HodSig": HOD["sig"], "HodLMzero": HOD["lM0"], "HodLMone": HOD["lM1"],
    "HodAlpha": HOD["alpha"], "HodNbar": nbar_g, "HodFsat": fsat, "HodBias": b_g, "HodMeff": M_eff / 1e13,
    "HodGamma": float(gam_g), "HodRzero": r0_g, "HodShot": 1 / nbar_g, "HodOneHaloConst": float(P1h_g[0]),
    "HodPairsMC": pairs_mc, "HodPairsTh": pairs_th, "HodPairsSE": pairs_se, "HodNhalo": nh, "HodNcFourteen": float(Nc[iM]), "HodLamFourteen": float(lam[iM]),
})
print("u_err", u_err, "mass/bias int", mass_int, bias_int, bias_int_ps, "ratios", [at(x, ratio) for x in (0.01, 0.1, 0.3, 1, 3, 10)],
      "keq", k_eq, "worst", k[worst], ratio[worst], "HOD", nbar_g, fsat, b_g, M_eff, gam_g, r0_g, pairs_mc, pairs_th, m_lo, m_hi)
