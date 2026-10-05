"""Does the stationary-phase formula really give the Fourier transform of a chirp?

Question: for a chirp h(t) = A(t) cos phi(t) whose frequency sweeps slowly, is the Fourier
transform  h~(f) = int h(t) e^{-2 pi i f t} dt  equal to  (1/2) A(t_f) fdot^{-1/2} e^{-i Psi(f)},
with  Psi(f) = 2 pi f t_f - phi(t_f) - pi/4  and t_f the instant at which the chirp has frequency f?
Computes: a Newtonian chirp (chirp mass 10 Msun, 20 Hz -> 1000 Hz, sampled at 4096 Hz, ends
tapered by short cosine ramps), its FFT, and the SPA prediction; compares modulus and phase in the band 30-150 Hz.
(A stellar-mass chirp lasts seconds, so the check is cheap; the same formula is applied to
LISA sources lasting years, where an FFT of the full signal would be expensive.)
Writes: figures/chT2/spa_check.pdf, results/chT2/02_spa_check.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from common import setup, savefig, save_numbers, SERIES, theory_line
from lib_lisa import TSUN, tau_of_f, fdot_vac

setup(6.4, 4.6)
Mc = 10.0
m = Mc * TSUN                      # chirp mass as a time [s]
f0, f1, fs = 20.0, 1000.0, 4096.0
tau0, tau1 = tau_of_f(f0, Mc), tau_of_f(f1, Mc)
t = np.arange(-tau0, -tau1, 1 / fs)            # time from coalescence (t_c = 0)
tau = -t
f_t = (1 / (8 * np.pi * m)) * (5 * m / tau) ** (3 / 8)     # f(t), Newtonian
phi_t = -2 * (tau / (5 * m)) ** (5 / 8)                     # GW phase, phi_c = 0
A_t = (f_t / f0) ** (2 / 3)                                 # amplitude ~ f^{2/3}
# cosine ramps: 0.3 s at the start (f near 20 Hz), 2 ms at the end (f > 360 Hz),
# so the band 30-150 Hz used for the comparison is untouched by the window
win = np.ones_like(t)
r0, r1 = t < t[0] + 0.3, t > t[-1] - 2e-3
win[r0] = 0.5 * (1 - np.cos(np.pi * (t[r0] - t[0]) / 0.3))
win[r1] = 0.5 * (1 - np.cos(np.pi * (t[-1] - t[r1]) / 2e-3))
h = A_t * np.cos(phi_t) * win

# FFT with the book's convention  h~(f) = sum h(t_n) e^{-2 pi i f t_n} dt
n = 1 << int(np.ceil(np.log2(t.size)) + 1)
H = np.fft.rfft(h, n) / fs * np.exp(-2j * np.pi * np.fft.rfftfreq(n, 1 / fs) * t[0])
f = np.fft.rfftfreq(n, 1 / fs)

# SPA prediction
band = (f > 30) & (f < 150)
fb = f[band]
tf = -tau_of_f(fb, Mc)
phif = -2 * (tau_of_f(fb, Mc) / (5 * m)) ** (5 / 8)
Psi = 2 * np.pi * fb * tf - phif - np.pi / 4
Hspa = 0.5 * (fb / f0) ** (2 / 3) / np.sqrt(fdot_vac(fb, Mc)) * np.exp(-1j * Psi)

ratio = np.abs(H[band]) / np.abs(Hspa)
dphase = np.angle(H[band] / Hspa)              # phase difference, wrapped to (-pi, pi]
print("amplitude ratio: median", np.median(ratio), " max |1-ratio|", np.max(np.abs(1 - ratio)))
print("phase difference: rms", np.sqrt(np.mean(dphase**2)), " max", np.max(np.abs(dphase)))
print("phase accumulated in band:", Psi.max() - Psi.min(), "rad")

fig, axs = plt.subplots(3, 1, gridspec_kw=dict(height_ratios=[1, 1.3, 0.9]))
axs[0].plot(t, h, lw=0.5, color=SERIES[0])
axs[0].set_xlabel("time before coalescence $t$ [s]")
axs[0].set_ylabel("$h(t)$")
axs[1].loglog(f[(f > 15) & (f < 300)], np.abs(H[(f > 15) & (f < 300)]), color=SERIES[0], label="FFT of the chirp")
theory_line(axs[1], fb, np.abs(Hspa), label="stationary phase")
axs[1].set_xlabel("$f$ [Hz]")
axs[1].set_ylabel(r"$|\tilde h(f)|$")
axs[1].legend()
axs[1].xaxis.set_major_formatter(ScalarFormatter())
axs[1].xaxis.set_minor_formatter(NullFormatter())
axs[1].set_xticks([20, 50, 100, 200])
axs[2].plot(fb, dphase, color=SERIES[1])
axs[2].set_ylim(-0.05, 0.05)
axs[2].set_xlabel("$f$ [Hz]")
axs[2].set_ylabel(r"phase error [rad]")
fig.tight_layout()
savefig(fig, "chT2", "spa_check")
save_numbers("chT2", "02_spa_check", {
    "TwSpaAmpErr": 100 * np.max(np.abs(1 - ratio)),
    "TwSpaPhaseRms": np.sqrt(np.mean(dphase**2)),
    "TwSpaPsiBand": Psi.max() - Psi.min(),
    "TwSpaDuration": tau0 - tau1,
})
