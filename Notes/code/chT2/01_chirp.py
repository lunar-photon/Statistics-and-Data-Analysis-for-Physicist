"""How fast does a Newtonian binary chirp, and how many cycles does LISA see?

Question: for the two benchmark systems used later in the book (a 1e4 + 1 Msun EMRI and a
1e3 + 1.4 Msun IMRI) and for a GW150914-like binary, what are the chirp mass, the frequency
at the innermost stable circular orbit, the frequency four years before that orbit, and the
number of gravitational-wave cycles in those four years?
Computes: the Newtonian formulas tau(f), f(tau), N(f1, f2) of lib_lisa; checks the
adiabatic condition fdot/f^2 << 1 at the last orbit.
Writes: figures/chT2/chirp_tracks.pdf, results/chT2/01_chirp.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES
from lib_lisa import (chirp_mass, f_isco, tau_of_f, f_of_tau, cycles, fdot_vac, YEAR)

setup(6.0, 3.6)
T4 = 4 * YEAR
systems = {  # name: (m1, m2) in solar masses
    "EMRI": (1e4, 1.0),       # benchmark of Kim et al. 2023
    "IMRI": (1e3, 1.4),       # benchmark of Eda et al. 2015 / Coogan et al. 2022
}
nums = {}
fig, ax = plt.subplots()
for (name, (m1, m2)), col in zip(systems.items(), SERIES):
    Mc, M = chirp_mass(m1, m2), m1 + m2
    fi = f_isco(M)
    t_isco = tau_of_f(fi, Mc)                       # time left at the ISCO frequency
    f4 = f_of_tau(t_isco + T4, Mc)                  # frequency 4 years before the ISCO
    N4 = cycles(f4, fi, Mc)
    adiab = fdot_vac(fi, Mc) / fi**2                # fractional change of f per cycle
    key = {"EMRI": "Emri", "IMRI": "Imri"}[name]
    nums.update({f"{key}Mc": Mc, f"{key}fisco": fi, f"{key}ffouryr": f4 * 1e3,
                 f"{key}Ncyc": N4, f"{key}adiab": adiab})
    # frequency track versus time left before the ISCO
    tau = np.logspace(2, np.log10(30 * YEAR), 400)
    f = f_of_tau(tau + t_isco, Mc)
    ax.loglog(tau / YEAR, f, color=col, label=rf"{name}: ${m1:.0e}+{m2:g}\,M_\odot$".replace("e+0", r"\times10^"))
    ax.plot([4], [f4], "o", color=col)
    ax.axhline(fi, color=col, lw=0.8, ls=":")
ax.axhspan(1e-4, 1.0, color="0.9", zorder=-1)
ax.text(5e-6, 0.55, "LISA band (below 1 Hz)", color="0.35", fontsize=8)
ax.set_xlabel("time left before the ISCO [yr]")
ax.set_ylabel("GW frequency $f$ [Hz]")
ax.set_xlim(1e2 / YEAR, 30)
ax.set_ylim(1e-3, 10)
ax.legend(loc="upper right")
savefig(fig, "chT2", "chirp_tracks")

# GW150914-like binary seen by LISA (Robson et al.: 16 -> 29 mHz in the last 4 of 5 years)
Mc150914 = chirp_mass(36.0, 29.0)
nums["GWMc"] = Mc150914
nums["GWfFive"] = f_of_tau(5 * YEAR, Mc150914) * 1e3
nums["GWfOne"] = f_of_tau(1 * YEAR, Mc150914) * 1e3
nums["GWNcyc"] = cycles(f_of_tau(5 * YEAR, Mc150914), f_of_tau(1 * YEAR, Mc150914), Mc150914)
for k, v in nums.items():
    print(f"{k:12s} {v:.6g}")
save_numbers("chT2", "01_chirp", {"Tw" + k: v for k, v in nums.items()})   # "Tw" = chapter T2 prefix
