"""What does LISA's noise look like, and how loud are the benchmark inspirals above it?

Question: which noise source limits LISA at each frequency, and what signal-to-noise ratio do
the book's benchmark binaries reach in a four-year observation?
Computes: the sky-averaged sensitivity S_n(f) of Robson, Cornish & Liu (2019) split into optical
metrology, test-mass acceleration and Galactic confusion; the characteristic strain
h_c = 2 f |h~| of three inspirals (angle-averaged, |h~|^2 = (4/5) A^2) against the noise
amplitude h_n = sqrt(f S_n), and their SNR  rho^2 = int (h_c/h_n)^2 d ln f.
Writes: figures/chT2/lisa_noise.pdf, figures/chT2/lisa_tracks.pdf, results/chT2/03_lisa_noise.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, INK
from lib_lisa import (lisa_components, Sn_lisa, F_STAR, chirp_mass, f_isco, tau_of_f, f_of_tau,
                      amp_spa, snr_sky_avg, MPC, YEAR)

setup(6.0, 3.7)
f = np.logspace(-5, 0, 4000)
oms, acc, conf = lisa_components(f)
Sn = oms + acc + conf

# ---- figure 1: the three noise pieces (amplitude spectral densities)
fig, ax = plt.subplots()
ax.loglog(f, np.sqrt(acc), color=SERIES[1], label="test-mass acceleration")
ax.loglog(f, np.sqrt(oms), color=SERIES[0], label="optical metrology")
ax.loglog(f, np.sqrt(conf + 1e-60), color=SERIES[2], label="Galactic confusion (4 yr)")
ax.loglog(f, np.sqrt(Sn), color=INK, lw=2.0, label=r"total $\sqrt{S_n}$")
ax.axvline(F_STAR, color="0.5", lw=0.8, ls=":")
ax.text(F_STAR, 1.01, r"$f_*$", color="0.35", transform=ax.get_xaxis_transform(), ha="center", va="bottom")
ax.set_ylim(1e-21, 1e-13)
ax.set_xlim(1e-5, 1)
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel(r"amplitude spectral density [Hz$^{-1/2}$]")
ax.legend(loc="upper right")
savefig(fig, "chT2", "lisa_noise")

i_min = np.argmin(Sn)
cross = f[np.argmin(np.abs(np.log(acc / oms)) + (f < 1e-4) * 1e9)]   # acceleration = metrology
conf_band = f[conf > (oms + acc)]

# ---- figure 2: three inspirals over the noise (characteristic strain)
T4 = 4 * YEAR
sources = [  # label, m1, m2, D_L [Mpc], duration of the observed stretch, end
    ("EMRI $10^4+1\\,M_\\odot$", 1e4, 1.0, None),
    ("IMRI $10^3+1.4\\,M_\\odot$, 75 Mpc", 1e3, 1.4, 75.0),
    ("GW150914-like, 410 Mpc", 36.0, 29.0, 410.0),
]
fig, ax = plt.subplots()
ax.loglog(f, np.sqrt(f * Sn), color=INK, lw=2.0, label=r"noise $h_n=\sqrt{f S_n}$")
nums = {"fstar": F_STAR * 1e3, "ASDmin": np.sqrt(Sn[i_min]), "fASDmin": f[i_min] * 1e3,
        "fAccOms": cross * 1e3, "fConfLo": conf_band.min() * 1e3, "fConfHi": conf_band.max() * 1e3}
for (lab, m1, m2, D), col in zip(sources, SERIES):
    Mc, M = chirp_mass(m1, m2), m1 + m2
    if m1 > 100:     # inspiral observed during the last 4 years before the ISCO
        f_end = min(f_isco(M), 1.0)
        f_beg = f_of_tau(tau_of_f(f_isco(M), Mc) + T4, Mc)
    else:            # stellar binary 5 years before merger, observed for 4 years
        f_beg, f_end = f_of_tau(5 * YEAR, Mc), f_of_tau(1 * YEAR, Mc)
    ff = np.logspace(np.log10(f_beg), np.log10(f_end), 2000)
    if D is None:    # EMRI: distance at which the 4-year SNR equals 14
        snr1 = snr_sky_avg(ff, amp_spa(ff, Mc, 1000 * MPC))
        D = 1000 * snr1 / 14.0
        nums["EmriDL"] = D
        lab = lab + f", {D:.0f} Mpc"
    A = amp_spa(ff, Mc, D * MPC)
    snr = snr_sky_avg(ff, A)
    hc = 2 * ff * np.sqrt(4 / 5) * A
    ax.loglog(ff, hc, color=col, lw=2.2, label=lab)
    ax.plot(ff[0], hc[0], "o", color=col)
    key = {1e4: "Emri", 1e3: "Imri", 36.0: "GWlike"}[m1]
    nums[f"{key}SNR"] = snr
    print(f"{lab:40s} f: {f_beg:.4g} -> {f_end:.4g} Hz   SNR = {snr:.4g}")
ax.set_xlim(1e-5, 1)
ax.set_ylim(1e-22, 1e-17)
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel("characteristic strain")
ax.legend(loc="upper right", fontsize=8)
savefig(fig, "chT2", "lisa_tracks")
for k, v in nums.items():
    print(f"{k:10s} {v:.5g}")
save_numbers("chT2", "03_lisa_noise", {"Tw" + k: v for k, v in nums.items()})   # "Tw" = chapter T2 prefix
