"""One inference problem, three detectors.

Question: what does "data = model prediction + noise" look like for (a) a collider
counting events in bins (Poisson), (b) a CMB sky summarised by its power spectrum
(Gaussian random field), (c) a gravitational-wave strain record (Gaussian noise with a PSD)?
Computes: (a) Poisson counts over a falling background plus a small bump, and the
maximum-likelihood signal strength from a scan of the Poisson log-likelihood;
(b) simulated full-sky power spectra C_hat_l = C_l chi^2_{2l+1}/(2l+1) around the fiducial
Planck spectrum; (c) coloured Gaussian noise drawn in the frequency domain with
E|n_k|^2 = T S_n(f_k)/2, its periodogram 2|d_k|^2/T, and a weak monochromatic signal.
Writes:   figures/ch00/three_detectors.pdf, results/ch00/05_three_detectors.tex
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
from camb_fiducial import load_fiducial

setup(7.2, 2.6)
out = {}

# ---------------- (a) collider: Poisson counts in 30 mass bins ----------------
rng = rng_for("ch00", "05_three_detectors", 0)
m = np.linspace(100, 160, 31); mc = 0.5 * (m[1:] + m[:-1])
bkg = 400 * np.exp(-(mc - 100) / 25)                        # expected background per bin
sig_shape = 60 * np.exp(-0.5 * ((mc - 125) / 2.5) ** 2)      # expected signal for strength mu=1
n_obs = rng.poisson(bkg + 1.0 * sig_shape)                   # the data: one Poisson draw per bin
mus = np.linspace(0, 3, 3001)
# ln L(mu) = sum_i [ n_i ln(lambda_i) - lambda_i ]  (+ const, the ln n_i! term)
lam = bkg[None, :] + mus[:, None] * sig_shape[None, :]
lnL = np.sum(n_obs * np.log(lam) - lam, axis=1)
i_best = np.argmax(lnL)
inside = mus[lnL >= lnL.max() - 0.5]                         # Delta lnL = 1/2 interval
out.update(ColMuHat=mus[i_best], ColMuLo=inside.min(), ColMuHi=inside.max(), ColNtot=int(n_obs.sum()))

# ---------------- (b) CMB: C_hat_l scatter from cosmic variance ----------------
rng = rng_for("ch00", "05_three_detectors", 1)
ell, cl = load_fiducial("TT")
L = np.arange(2, 1501); C = cl[L]
chat = C * rng.chisquare(2 * L + 1) / (2 * L + 1)          # full sky, no noise: 2l+1 real modes per l
Dfac = L * (L + 1) / (2 * np.pi)
ratio = chat / C
low = (L >= 2) & (L <= 30)
out.update(CmbMeanRatio=ratio.mean(),
           CmbScatterLow=np.std(ratio[low] - 1), CmbScatterHigh=np.std(ratio[(L > 1000)] - 1))

# ---------------- (c) GW: Gaussian noise with a PSD, in the frequency domain ----------------
rng = rng_for("ch00", "05_three_detectors", 2)
T, fs = 64.0, 256.0                                          # seconds, samples per second
f = np.fft.rfftfreq(int(T * fs), 1 / fs)[1:]                 # positive frequencies, drop f=0
f0, S0 = 20.0, 1.0
Sn = S0 * ((f / f0) ** -4 + 1 + (f / f0) ** 2)               # a toy one-sided PSD
sd = np.sqrt(T * Sn / 4)                                     # Re and Im each get half of T S_n/2
n_k = sd * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
h_k = np.zeros_like(n_k); f_sig = 40.0
h_k[np.argmin(abs(f - f_sig))] = 12 * np.sqrt(T * Sn[np.argmin(abs(f - f_sig))])   # a weak line
d_k = n_k + h_k
pgram = 2 * np.abs(d_k) ** 2 / T                             # periodogram: E[pgram] = S_n off the line
noline = np.abs(f - f_sig) > 1
out.update(GwMeanRatio=np.mean(pgram[noline] / Sn[noline]), GwNbins=f.size)

fig, ax = plt.subplots(1, 3)
ax[0].errorbar(mc, n_obs, yerr=np.sqrt(n_obs), fmt="o", ms=3, color=SERIES[0], label="counts $n_i$")
theory_line(ax[0], mc, bkg, label="background")
ax[0].plot(mc, bkg + mus[i_best] * sig_shape, color=SERIES[1], label="best fit")
ax[0].set(xlabel="invariant mass (GeV)", ylabel="events per bin", title="collider: Poisson counts")
ax[0].legend(fontsize=7)
ax[1].plot(L, Dfac * chat, ".", ms=1.5, color=SERIES[0], label="$\\hat D_\\ell$, one sky")
theory_line(ax[1], L, Dfac * C, label="$D_\\ell$ theory")
ax[1].set(xlabel="multipole $\\ell$", ylabel="$D_\\ell$ ($\\mu$K$^2$)", title="CMB: $\\hat C_\\ell$ scatter")
ax[1].legend(fontsize=7)
vis = f >= 5                                                  # show the sensitive band only
ax[2].loglog(f[vis], pgram[vis], color=SERIES[0], lw=0.4, label="periodogram")
theory_line(ax[2], f[vis], Sn[vis], label="$S_n(f)$")
ax[2].set(xlabel="frequency (Hz)", ylabel="power (arb.)", title="GW: noise with a PSD")
ax[2].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "ch00", "three_detectors")
save_numbers("ch00", "05_three_detectors", out)
