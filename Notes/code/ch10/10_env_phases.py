"""10_env_phases.py -- how large is the Fourier-phase shift of each environmental effect for one LISA source?

Question: for the benchmark intermediate mass-ratio inspiral (1e3 + 1.4 Msun, observed for the last five years
before the ISCO, band cut at 1 Hz), what is eps(f) = P_env/P_GW and the resulting phase shift
dPsi(f) = -(2 pi f^2/fdot_V) eps/((n+8/3)(n+5/3)) for
  dynamical friction in spikes rho = rho6 (r6/r)^gamma, gamma = 7/3 and 3/2 (rho6 of Coogan et al. 2022),
  collisionless accretion in the 7/3 spike, the pull of the enclosed dark matter (order of magnitude),
  Eddington-limited gas accretion onto the central black hole, type-I migration in a thin disc (Barausse 2014)?
It checks the closed form against a direct double integral of d^2 Psi/df^2 = 2 pi/fdot for one case, records the
frequency where friction equals radiation, and saves the phase shifts at the start of the band for the map of 12.
Writes: figures/ch10/env_phases.pdf, results/ch10/10_env_phases.tex, data/ch10/env_points.npz
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from common import setup, savefig, save_numbers, SERIES, DATA
import lib_gw10 as L

setup()
m1, m2 = 1e3, 1.4
Mc, M = L.chirp_mass(m1, m2), m1 + m2
fisco = L.f_isco(M)
f_lo = L.f_of_tau(L.tau_of_f(fisco, Mc) + 5 * L.YEAR, Mc)
f_hi = min(1.0, fisco)
f = L.logfgrid(f_lo, f_hi, 3000)
RHO6 = 5.448e15                                   # Coogan et al. Table I, astrophysical benchmark

effects = {  # name: (eps(f), n, b, label)
    "dfA": (L.eps_df(f, m1, m2, RHO6, 7 / 3), (11 - 2 * 7 / 3) / 3, "friction, spike $\\gamma=7/3$"),
    "dfB": (L.eps_df(f, m1, m2, RHO6, 1.5), (11 - 3) / 3, "friction, spike $\\gamma=3/2$"),
    "acc": (L.eps_acc(f, m1, m2, RHO6, 7 / 3), (9 - 2 * 7 / 3) / 3, "DM accretion, $\\gamma=7/3$"),
    "pull": (L.eps_pull(f, m1, m2, RHO6, 7 / 3), 2 * (3 - 7 / 3) / 3, "enclosed DM mass, $\\gamma=7/3$"),
    "edd": (L.eps_edd(f, m1, m2, fedd=1.0), 8 / 3, "Eddington gas accretion onto $m_1$"),
    "mig": (L.eps_migI(f, m1, m2, fedd=0.1), 7 / 3, "type-I migration, thin disc"),
}
nums = dict(TenBenvMc=Mc, TenBenvflo=1e3 * f_lo, TenBenvfhi=f_hi, TenBenvfisco=fisco)
pts = {}
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2))
for k, (key, (eps, n, lab)) in enumerate(effects.items()):
    b = -5 / 3 - n
    dpsi = L.dpsi_powerlaw(f, Mc, eps, n)
    pts[key] = (b, dpsi[0], eps[0], lab)
    ax[0].loglog(f, eps, color=SERIES[k], label=lab)
    ax[1].loglog(f, np.abs(dpsi), color=SERIES[k])
    nm = {"dfA": "DfA", "dfB": "DfB", "acc": "Acc", "pull": "Pull", "edd": "Edd", "mig": "Mig"}[key]
    nums[f"TenBenvEps{nm}"] = eps[0]
    nums[f"TenBenvDpsi{nm}"] = abs(dpsi[0])
    nums[f"TenBenvB{nm}"] = b
ax[0].axhline(1, color="k", lw=0.8, ls=":")
ax[0].set_xlabel("$f$ [Hz]"); ax[0].set_ylabel(r"$\epsilon=P_{\rm env}/P_{\rm GW}$")
ax[1].axhline(1, color="k", lw=0.8, ls=":")
ax[1].set_xlabel("$f$ [Hz]"); ax[1].set_ylabel(r"$|\delta\Psi(f)|$ [rad]")
fig.legend(*ax[0].get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.17, 1, 1))
savefig(fig, "ch10", "env_phases")

# where friction (gamma = 7/3) equals radiation
feq = brentq(lambda x: L.eps_df(np.array([x]), m1, m2, RHO6, 7 / 3)[0] - 1, 1e-4, 1.0)
nums["TenBenvfeq"] = 1e3 * feq

# check the closed form against the exact double integral for a weak case (friction at 1e-6 of the density)
eps = L.eps_df(f, m1, m2, RHO6 * 1e-6, 7 / 3)
g = 2 * np.pi / L.fdot_vac(f, Mc) * (1 / (1 + eps) - 1)            # d^2 dPsi / df^2
A = -cumulative_trapezoid((f * g)[::-1], f[::-1], initial=0)[::-1]   # int_f^fhi f' g
B = -cumulative_trapezoid(g[::-1], f[::-1], initial=0)[::-1]         # int_f^fhi g
num = A - f * B                                                       # dPsi with dPsi(fhi) = dPsi'(fhi) = 0
n = (11 - 2 * 7 / 3) / 3
cf = L.dpsi_powerlaw(f, Mc, eps, n)
# the two differ by a + b f (absorbed by phic, tc): remove the best linear fit before comparing
X = np.vstack([np.ones_like(f), f]).T
res = num - cf
coef = np.linalg.lstsq(X, res, rcond=None)[0]
nums["TenBenvClosedErr"] = np.max(np.abs(res - X @ coef)) / np.max(np.abs(cf))
save_numbers("ch10", "10_env_phases", L.tidy(nums))
(DATA / "ch10").mkdir(parents=True, exist_ok=True)
np.savez(DATA / "ch10" / "env_points.npz", keys=list(pts), b=[pts[k][0] for k in pts],
         dpsi=[pts[k][1] for k in pts], eps=[pts[k][2] for k in pts], labels=[pts[k][3] for k in pts],
         flo=f_lo, fhi=f_hi)
print(nums)
