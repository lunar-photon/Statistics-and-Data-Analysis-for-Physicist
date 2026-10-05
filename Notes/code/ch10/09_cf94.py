"""09_cf94.py -- does our gravitational-wave Fisher code reproduce Cutler & Flanagan (1994), Table I?

Question: with the restricted 1.5PN phase (their eq. 42), their analytic advanced-LIGO noise curve (eq. 4),
f from 10 Hz to the ISCO frequency, and errors normalised to S/N = 10, what are the rms errors on phi_c, t_c,
the chirp mass and the reduced mass, and the correlation c_{M mu}, for their five mass pairs?
We compute the Fisher matrix in (ln Mc, ln eta, tc, phic, ln A), convert to (ln Mc, ln mu) with
ln mu = ln Mc + (2/5) ln eta, and compare. We also show the weight w(f) = 4|h|^2/(S_n rho^2) for a NS-NS
binary, and check the two "bandwidth" formulas for the timing error.
Writes: results/ch10/09_cf94.tex, figures/ch10/cf94_weight.pdf
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
import lib_gw10 as L

setup()
PAPER = {  # (m1, m2): dphic, dtc [ms], dMc/Mc [%], dmu/mu [%], c_Mmu   (CF94 Table I, S/N = 10)
    (2.0, 1.0): (1.31, 0.721, 0.0038, 0.39, 0.899),
    (1.4, 1.4): (1.28, 0.713, 0.0040, 0.41, 0.906),
    (10.0, 1.4): (1.63, 1.01, 0.020, 0.54, 0.927),
    (15.0, 5.0): (2.02, 1.44, 0.113, 1.5, 0.954),
    (10.0, 10.0): (1.98, 1.43, 0.16, 1.9, 0.958),
}


def errors(m1, m2, snr=10.0):
    Mc, eta = L.chirp_mass(m1, m2), L.sym_ratio(m1, m2)
    f = L.logfgrid(10.0, L.f_isco(m1 + m2), 6000)
    Sn = L.Sn_cf94(f)
    F = L.fisher_from_dlogh(f, f ** (-7 / 3), Sn, L.phase_derivs(f, Mc, eta))
    rho2 = F[4, 4]                         # (h|h) = Gamma_{lnA lnA}
    F *= snr**2 / rho2                     # normalise to the requested S/N
    C = np.linalg.inv(F)
    J = np.array([[1, 0], [1, 0.4]])       # (ln Mc, ln mu) from (ln Mc, ln eta)
    Cm = J @ C[:2, :2] @ J.T
    return dict(phic=np.sqrt(C[3, 3]), tc=1e3 * np.sqrt(C[2, 2]), Mc=100 * np.sqrt(Cm[0, 0]),
                mu=100 * np.sqrt(Cm[1, 1]), c=Cm[0, 1] / np.sqrt(Cm[0, 0] * Cm[1, 1]), F=F, f=f, Sn=Sn)


nums, rows = {}, []
letters = "ABCDE"
for k, ((m1, m2), pap) in enumerate(PAPER.items()):
    e = errors(m1, m2)
    ours = (e["phic"], e["tc"], e["Mc"], e["mu"], e["c"])
    rows.append(((m1, m2), pap, ours))
    for nm, v in zip(["Phic", "Tc", "Mc", "Mu", "Corr"], ours):
        nums[f"TenBcf{nm}{letters[k]}"] = v
    print(f"{m1:5.1f}+{m2:4.1f}  paper {pap}  ours " + " ".join(f"{v:.4g}" for v in ours))
ratios = np.array([[o / p for p, o in zip(pap, ours)] for _, pap, ours in rows])
nums["TenBcfMaxDev"] = 100 * np.max(np.abs(ratios - 1))
print("max deviation %", nums["TenBcfMaxDev"])

# -------- the weight function and the bandwidth formulas for 1.4+1.4
e = errors(1.4, 1.4)
f, Sn = e["f"], e["Sn"]
w = f ** (-7 / 3) / Sn
w /= L.integ(f, w)
fbar = L.integ(f, w * f)
f2 = L.integ(f, w * f * f)
sig_tc_alone = 1 / (2 * np.pi * 10 * np.sqrt(f2))                 # only tc unknown
sig_tc_phic = 1 / (2 * np.pi * 10 * np.sqrt(f2 - fbar**2))          # tc and phic unknown
nums.update(TenBcfFbar=fbar, TenBcfFrms=np.sqrt(f2 - fbar**2), TenBcfTcAlone=1e3 * sig_tc_alone,
            TenBcfTcPhic=1e3 * sig_tc_phic, TenBcfFpeak=f[np.argmax(w * f)])
# check sigma_tc with (tc, phic) only from the Fisher submatrix
Fsub = e["F"][2:4, 2:4]
nums["TenBcfTcPhicFisher"] = 1e3 * np.sqrt(np.linalg.inv(Fsub)[0, 0])
save_numbers("ch10", "09_cf94", L.tidy(nums))

fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(f, w * f, color=SERIES[0], label=r"$f\,w(f)$: share of $\rho^2$ per $\ln f$")
ax.axvline(fbar, color=SERIES[1], ls=":", lw=1.2, label=r"$\bar f$")
ax.axvspan(fbar - np.sqrt(f2 - fbar**2), fbar + np.sqrt(f2 - fbar**2), color=SERIES[1], alpha=0.12,
           label=r"$\bar f\pm\sigma_f$")
ax.set_xscale("log")
ax.set_xlabel(r"$f$ [Hz]")
ax.set_ylabel(r"$f\,w(f)$")
ax.legend(loc="upper right")
savefig(fig, "ch10", "cf94_weight")
