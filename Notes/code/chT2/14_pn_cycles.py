"""How many wave cycles does each post-Newtonian term of the phase contribute?

Question: the 1PN and 1.5PN corrections to the binding energy and the flux change the inspiral by
a few per cent. Over a million cycles, how many cycles is that, and is it negligible?
Computes: N = (5/(32 pi eta)) int_{v1}^{v2} v^{-6} (1 + a v^2 + b v^3) dv, the number of GW cycles
between two frequencies, split into its Newtonian, 1PN and 1.5PN parts, with
a = 743/336 + 11 eta/4 and b = -4 pi, for the book's benchmark binaries in their observation
windows and for a LIGO neutron-star binary for comparison.
Writes: figures/chT2/pn_cycles.pdf, results/chT2/14_pn_cycles.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
from lib_lisa import chirp_mass, f_isco, tau_of_f, f_of_tau, TSUN, YEAR


def v_of_f(f, M):
    """PN velocity parameter v = (pi G M f / c^3)^{1/3}, M = total mass in solar masses."""
    return (np.pi * M * TSUN * f) ** (1 / 3)


def cycle_parts(f1, f2, m1, m2):
    """Newtonian, 1PN and 1.5PN contributions to the number of cycles between f1 < f2."""
    M = m1 + m2
    eta = m1 * m2 / M**2
    a = 743 / 336 + 11 / 4 * eta
    b = -4 * np.pi
    v1, v2 = v_of_f(f1, M), v_of_f(f2, M)
    pre = 5 / (32 * np.pi * eta)
    N0 = pre * (v1**-5 - v2**-5) / 5
    N1 = pre * a * (v1**-3 - v2**-3) / 3
    N15 = pre * b * (v1**-2 - v2**-2) / 2
    return N0, N1, N15, v1, v2


systems = {}
# EMRI and IMRI: the four years before the ISCO (Newtonian clock, as in 01_chirp.py)
for key, (m1, m2) in {"Emri": (1e4, 1.0), "Imri": (1e3, 1.4)}.items():
    Mc = chirp_mass(m1, m2)
    fend = f_isco(m1 + m2)
    fstart = f_of_tau(tau_of_f(fend, Mc) + 4 * YEAR, Mc)
    systems[key] = (fstart, fend, m1, m2)
# GW150914-like binary: from five years to one year before merger (LISA band)
Mc = chirp_mass(36, 29)
systems["GW"] = (f_of_tau(5 * YEAR, Mc), f_of_tau(1 * YEAR, Mc), 36.0, 29.0)
# LIGO binary neutron star: 10 Hz to the ISCO
systems["Ns"] = (10.0, f_isco(2.8), 1.4, 1.4)

def tex(v):
    """Cycle counts as LaTeX: 3 significant figures and a power of ten from 10^5 on, else an integer."""
    if abs(v) >= 1e5:
        m, e = f"{v:.2e}".split("e")
        return rf"{m}\times10^{{{int(e)}}}"
    return f"{v:,.0f}".replace(",", r"\,")


nums = {}
rows = []
for key, (f1, f2, m1, m2) in systems.items():
    N0, N1, N15, v1, v2 = cycle_parts(f1, f2, m1, m2)
    rows.append((key, N0, N1, N15))
    print(f"{key:5s} f {f1:.4g}->{f2:.4g} Hz  v {v1:.3f}->{v2:.3f}  N0={N0:.4g}  N1PN={N1:.4g}  N1.5PN={N15:.4g}")
    nums[f"TwPN{key}N"] = tex(N0)
    nums[f"TwPN{key}One"] = tex(N1)
    nums[f"TwPN{key}Onefive"] = tex(N15)
    nums[f"TwPN{key}vone"] = v1
    nums[f"TwPN{key}vtwo"] = v2

setup(6.0, 3.4)
fig, ax = plt.subplots()
names = {"Emri": "EMRI\n$10^4+1$", "Imri": "IMRI\n$10^3+1.4$", "GW": "GW150914-like\n(LISA)",
         "Ns": "NS-NS\n(LIGO, 10 Hz)"}
xs = np.arange(len(rows))
wd = 0.26
for j, (lab, col) in enumerate([("Newtonian", SERIES[0]), ("1PN", SERIES[1]), ("1.5PN (tail)", SERIES[2])]):
    vals = [abs(r[1 + j]) for r in rows]
    ax.bar(xs + (j - 1) * wd, vals, wd, color=col, label=lab)
ax.axhline(1.0, color="0.4", lw=0.8, ls=":", label="one cycle")
ax.set_yscale("log")
ax.set_ylim(0.3, 3e8)
ax.set_xticks(xs)
ax.set_xticklabels([names[r[0]] for r in rows], fontsize=8)
ax.set_ylabel("cycles contributed (absolute value)")
ax.legend(loc="upper right", ncol=2, fontsize=8)
savefig(fig, "chT2", "pn_cycles")
save_numbers("chT2", "14_pn_cycles", nums)
