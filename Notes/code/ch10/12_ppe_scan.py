"""12_ppe_scan.py -- which power-law dephasing can LISA see in one intermediate mass-ratio inspiral?

Question: for the benchmark IMRI (1e3 + 1.4 Msun at D_L = 76 Mpc, last five years before the ISCO, 1 Hz cut,
Robson et al. sensitivity with 4-year confusion noise), with the parameters
    theta = (ln Mc, ln eta, tc, phic, ln A, beta),    Psi = Psi_1.5PN + beta u^b,  u = pi G Mc f/c^3,
and the vacuum (beta = 0) as the fiducial model:
 (1) how does the marginalised error sigma_beta depend on the exponent b, and how much larger is it than the
     error with everything else known?  (singular where u^b coincides with a GR phase function)
 (2) how does beta correlate with ln Mc, ln eta and tc as a function of b?
 (3) what is the smallest phase shift at the start of the band, |dPsi(f_lo)| = 3 sigma_beta u_lo^b, that would
     be detected at 3 sigma, and where do the physical effects of 10_env_phases.py sit relative to it?
 (4) check: sigma_beta from the Schur complement equals 1/(rho sqrt(residual variance)) of u^b regressed on the
     GR derivatives with the weight w(f).
Writes: figures/ch10/ppe_sigma.pdf, ppe_corr.pdf, ppe_map.pdf, results/ch10/12_ppe_scan.tex, data/ch10/ppe_scan.npz
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, DATA
import lib_gw10 as L

setup()
m1, m2, DL = 1e3, 1.4, 76 * L.MPC
Mc, eta, M = L.chirp_mass(m1, m2), L.sym_ratio(m1, m2), m1 + m2
fisco = L.f_isco(M)
f_lo = L.f_of_tau(L.tau_of_f(fisco, Mc) + 5 * L.YEAR, Mc)
f_hi = min(1.0, fisco)
f = L.logfgrid(f_lo, f_hi, 5000)
Sn = L.Sn_robson(f, "4yr")
amp2 = L.amp_avg(f, Mc, DL) ** 2
rho = np.sqrt(L.snr2(f, amp2, Sn))
u_lo = L.u_of(f_lo, Mc)
nums = dict(TenBppSNR=rho, TenBppMc=Mc, TenBppEta=eta, TenBppflo=1e3 * f_lo, TenBppulo=u_lo)

# GR-only errors (5 parameters)
F5 = L.fisher_from_dlogh(f, amp2, Sn, L.phase_derivs(f, Mc, eta))
e5 = L.marg_err(F5)
nums.update(TenBppGRlnMc=e5[0], TenBppGRlneta=e5[1], TenBppGRtc=e5[2], TenBppGRphic=e5[3], TenBppGRlnA=e5[4])

bs = np.linspace(-6.5, 1.6, 815) + 1e-4            # avoid sitting exactly on the GR exponents
sig, sigc, Dlo, cMc, ceta, ctc, cphi, resid, cMc4 = ([] for _ in range(9))
w = 4 * amp2 / Sn / rho**2                           # weight with int w df = 1
E = np.vstack([L.dpsi_pn(f, Mc, eta)[0], L.dpsi_pn(f, Mc, eta)[1], 2 * np.pi * f, -np.ones_like(f)])
for b in bs:
    F = L.ppe_fisher(f, Mc, eta, DL, Sn, b)
    C = L.inv_scaled(F)
    s = np.sqrt(np.diag(C))
    sig.append(s[5]); sigc.append(1 / np.sqrt(F[5, 5]))
    Dlo.append(3 * s[5] * u_lo**b)
    R = C / np.outer(s, s)
    cMc.append(R[5, 0]); ceta.append(R[5, 1]); ctc.append(R[5, 2]); cphi.append(R[5, 3])
    keep = [0, 2, 3, 4, 5]                            # the same with eta known
    cMc4.append(L.corr(F[np.ix_(keep, keep)])[4, 0])
    # residual of u^b after weighted least squares on the GR phase derivatives (scaled to avoid round-off)
    g = L.u_of(f, Mc) ** b
    sc = np.sqrt(L.integ(f, w * g * g))
    gs = g / sc
    Es = E / np.sqrt(L.integ(f, w * E * E))[:, None]
    A = np.array([[L.integ(f, w * ei * ej) for ej in Es] for ei in Es])
    y = np.array([L.integ(f, w * ei * gs) for ei in Es])
    coef = np.linalg.solve(A, y)
    r = gs - coef @ Es
    resid.append(sc * np.sqrt(L.integ(f, w * r * r)))
sig, sigc, Dlo, cMc, ceta, ctc, cphi, resid, cMc4 = map(np.array, (sig, sigc, Dlo, cMc, ceta, ctc, cphi, resid, cMc4))
schur_check = np.max(np.abs(sig * rho * resid - 1)[(np.abs(bs + 5 / 3) > 0.05) & (np.abs(bs) > 0.05) & (np.abs(bs - 1) > 0.05)])
nums["TenBppSchurCheck"] = schur_check

def at(b0, arr):
    return np.interp(b0, bs, arr)

for nm, b0 in (("Four", -4.0), ("Dfa", -34 / 9), ("Bd", -7 / 3), ("Mg", -1.0), ("Edd", -13 / 3), ("Two", -2.0)):
    nums[f"TenBppD{nm}"] = at(b0, Dlo)
    nums[f"TenBppRatio{nm}"] = at(b0, sig / sigc)
    nums[f"TenBppcMc{nm}"] = at(b0, cMc)
    nums[f"TenBppctc{nm}"] = at(b0, ctc)
    nums[f"TenBppceta{nm}"] = at(b0, ceta)
    nums[f"TenBppcMcfix{nm}"] = at(b0, cMc4)
ilo = bs < -2.5
nums["TenBppDmin"] = Dlo[ilo].min()
nums["TenBppbDmin"] = bs[ilo][np.argmin(Dlo[ilo])]
nums["TenBppcMcfixEight"] = L.corr(L.ppe_fisher(f, Mc, eta, DL, Sn, -8.0)[np.ix_([0, 2, 3, 4, 5], [0, 2, 3, 4, 5])])[4, 0]

# the physical effects (from 10_env_phases.py)
P = np.load(DATA / "ch10" / "env_points.npz", allow_pickle=True)
pts = list(zip(P["keys"], P["b"], P["dpsi"], P["labels"]))
for key, b, dp, lab in pts:
    nm = {"dfA": "DfA", "dfB": "DfB", "acc": "Acc", "pull": "Pull", "edd": "Edd", "mig": "Mig"}[str(key)]
    nums[f"TenBppSig{nm}"] = abs(dp) / (at(b, Dlo) / 3)        # predicted |beta| / sigma_beta
save_numbers("ch10", "12_ppe_scan", L.tidy(nums))
np.savez(DATA / "ch10" / "ppe_scan.npz", bs=bs, sig=sig, sigc=sigc, Dlo=Dlo, rho=rho, ulo=u_lo)
print({k: v for k, v in nums.items()})

GRB = [(-5 / 3, r"$\mathcal{M}$"), (-1, r"$\eta$ (1PN)"), (-2 / 3, "1.5PN"), (0, r"$\phi_c$"), (1, r"$t_c$")]
# ---- Fig 1: sigma_beta (as detectable phase at f_lo, 1 sigma) marginal vs conditional
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.semilogy(bs, sig * u_lo**bs, color=SERIES[0], label=r"marginalised: $\sigma_\beta u_{\rm lo}^{\,b}$")
ax.semilogy(bs, sigc * u_lo**bs, color=SERIES[1], ls="--", label=r"all other parameters known")
for b0, lab in GRB:
    ax.axvline(b0, color="0.6", lw=0.7, ls=":")
    ax.text(b0, 1.4e-3, lab, rotation=90, fontsize=7, ha="right", va="bottom", color="0.35")
ax.set_ylim(1e-3, 3e3); ax.set_xlim(bs[0], bs[-1])
ax.set_xlabel("exponent $b$ of the phase correction $\\beta u^b$")
ax.set_ylabel(r"1$\sigma$ phase shift at $f_{\rm lo}$ [rad]")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
savefig(fig, "ch10", "ppe_sigma")

# ---- Fig 2: correlations
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.plot(bs, cMc, color=SERIES[0], label=r"corr$(\beta,\ln\mathcal{M})$")
ax.plot(bs, cMc4, color=SERIES[0], ls="--", lw=1.1, label=r"corr$(\beta,\ln\mathcal{M})$, $\eta$ known")
ax.plot(bs, ceta, color=SERIES[2], label=r"corr$(\beta,\ln\eta)$")
ax.plot(bs, ctc, color=SERIES[1], label=r"corr$(\beta,t_c)$")
ax.plot(bs, cphi, color=SERIES[3], lw=1, label=r"corr$(\beta,\phi_c)$")
for b0, lab in GRB:
    ax.axvline(b0, color="0.6", lw=0.7, ls=":")
ax.set_xlim(bs[0], bs[-1]); ax.set_ylim(-1.05, 1.05)
ax.set_xlabel("exponent $b$"); ax.set_ylabel("correlation coefficient")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
fig.tight_layout()
savefig(fig, "ch10", "ppe_corr")

# ---- Fig 3: detectability map
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.fill_between(bs, Dlo, 1e8, color=SERIES[2], alpha=0.12)
ax.semilogy(bs, Dlo, color="k", lw=1.6, label=r"3$\sigma$ threshold $|\delta\Psi(f_{\rm lo})|=3\sigma_\beta u_{\rm lo}^{\,b}$")
mk = "osD^vP"
for k, (key, b, dp, lab) in enumerate(pts):
    ax.plot(b, abs(dp), mk[k], color=SERIES[k], ms=7, label=str(lab))
for b0, lab in GRB:
    ax.axvline(b0, color="0.6", lw=0.7, ls=":")
ax.text(-5.9, 2e6, "detectable", fontsize=9, color=SERIES[2])
ax.text(-5.9, 1e-2, "not detectable", fontsize=9, color="0.4")
ax.set_xlim(-6.0, 1.5); ax.set_ylim(1e-3, 1e7)
ax.set_xlabel("exponent $b$"); ax.set_ylabel(r"$|\delta\Psi(f_{\rm lo})|$ [rad]")
ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
fig.tight_layout()
savefig(fig, "ch10", "ppe_map")

# ---- Fig 4: the same threshold in physical units -- smallest spike density detectable by friction, vs distance
DLs = np.geomspace(10, 1000, 60)                       # Mpc
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for k, gam in enumerate((1.5, 2.0, 7 / 3)):
    n = (11 - 2 * gam) / 3
    b = -5 / 3 - n
    rho_ref = 1e10
    dpsi_ref = abs(L.dpsi_powerlaw(np.array([f_lo]), Mc, L.eps_df(np.array([f_lo]), m1, m2, rho_ref, gam), n)[0])
    thr = at(b, Dlo) * DLs / 76.0                       # 3 sigma threshold scales with distance (sigma ~ 1/rho)
    rmin = rho_ref * thr / dpsi_ref
    # density at which friction equals radiation at f_lo (beyond it the power law is not a small correction)
    r_eq = rho_ref / L.eps_df(np.array([f_lo]), m1, m2, rho_ref, gam)[0]
    ax.loglog(DLs, rmin, color=SERIES[k], label=rf"$\gamma={gam:.3g}$  ($b={b:.2f}$)")
    ax.axhline(r_eq, color=SERIES[k], ls=":", lw=0.9)
    nm = {1.5: "A", 2.0: "B", 7 / 3: "C"}[gam]
    nums[f"TenBppRhoMin{nm}"] = np.interp(76.0, DLs, rmin)
    nums[f"TenBppRhoEq{nm}"] = r_eq
ax.plot([76], [5.448e15], "*", color="k", ms=9, label="Coogan et al. benchmark, 76 Mpc")
ax.set_xlabel(r"$D_L$ [Mpc]"); ax.set_ylabel(r"smallest detectable $\rho_6$ [$M_\odot\,$pc$^{-3}$]")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
fig.tight_layout()
savefig(fig, "ch10", "ppe_rhomin")
save_numbers("ch10", "12_ppe_scan", L.tidy(nums))
