"""16_whiten_chi2.py -- three checks that go with the matched filter: is the noise white after whitening, how much
does the filter gain from fitting the amplitude, and does the chirp's band-by-band shape separate a signal from a
noise burst?  Same set-up as 08_detection.py (analytic advanced-detector noise, 10+10 Msun chirp, T = 16 s).

Check 1: for stationary Gaussian noise the periodogram |n_k|^2 / (T/2 S_n) of every bin is exponentially distributed
  with unit mean (a chi^2_2 / 2), so the whitened modes are unit-variance complex Gaussians. Test with a KS test.
Check 2: maximising the likelihood ratio over the signal amplitude A gives ln Lambda_max = z^2/2 with z = (d|h)/rho.
Check 3: split the band into p pieces that each carry the same share rho^2/p of the signal-to-noise ratio. At the
  best lag, chi2 = p sum_i |x_i/rho - x/(p rho)|^2 follows chi^2_{2p-2} for noise or a true chirp; a short narrow-band
  burst with the same optimal SNR does not.
Check 4: the overlap (match) of two chirps whose chirp masses differ by dlnM, against the Fisher expansion
  1 - M = (1/2) dlnM^2 Gamma_MM / rho^2.
Writes: figures/ch10/det_chi2.pdf, results/ch10/16_whiten_chi2.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
import lib_gw10 as L

setup()
rng = rng_for("ch10", "16_whiten_chi2")

fs, T = 1024.0, 16.0
N = int(fs * T)
dt, df = 1 / fs, 1 / T
fk = np.fft.rfftfreq(N, dt)
band = (fk >= 15.0) & (fk <= 500.0)
Sn = np.full_like(fk, np.inf)
Sn[band] = L.Sn_cf94(fk[band])
sig2 = 0.5 * T * Sn


def noise_fd(n):
    s = np.sqrt(np.where(band, sig2, 0.0) / 2)
    return s * (rng.standard_normal((n, fk.size)) + 1j * rng.standard_normal((n, fk.size)))


def inner(a, b):
    return 4 * df * np.real(np.sum((a * np.conj(b))[..., band] / Sn[band], axis=-1))


m1 = m2 = 10.0
Mc = L.chirp_mass(m1, m2)
fhi = L.f_isco(m1 + m2)
tc = 11.0
inband = (fk >= 20.0) & (fk <= fhi)


def chirp(Mc_, rho=None):
    h = np.zeros_like(fk, dtype=complex)
    h[inband] = fk[inband] ** (-7 / 6) * np.exp(-1j * L.psi_pn(fk[inband], Mc_, 0.25, tc=tc, order=0))
    if rho is not None:
        h *= rho / np.sqrt(inner(h, h))
    return h


h0 = chirp(Mc, rho=5.0)
rho = np.sqrt(inner(h0, h0))
nums = {}

# ---- check 1: periodograms and whitened modes
n = noise_fd(400)
u = (np.abs(n[:, band]) ** 2 / sig2[band]).ravel()          # should be Exp(1)
ks = stats.kstest(u, "expon")
nums.update(TenBwhMean=u.mean(), TenBwhVar=u.var(), TenBwhKSp=ks.pvalue)
wre = (n[:, band].real / np.sqrt(sig2[band] / 2)).ravel()
nums.update(TenBwhReSd=wre.std(), TenBwhReKurt=stats.kurtosis(wre))

# ---- check 2: amplitude maximisation
d = noise_fd(2000) + 1.0 * h0
z = inner(d, h0) / rho
Ahat = z / rho
lnL = lambda A: A * inner(d, h0) - 0.5 * A**2 * rho**2
nums.update(TenBampDiff=np.max(np.abs(lnL(Ahat) - z**2 / 2)))
nums.update(TenBampAhatMean=Ahat.mean(), TenBampAhatSd=Ahat.std() * rho)

# ---- check 3: chi^2 consistency over p bands
P = 8
cum = np.cumsum(np.where(band, 4 * np.abs(h0) ** 2 / np.where(band, Sn, 1), 0.0)) * df
edges = np.searchsorted(cum, np.linspace(0, cum[-1], P + 1)[1:-1])
bands = [(a, b) for a, b in zip(np.r_[0, edges], np.r_[edges, fk.size])]


def filters(dd):
    """complex filter at all lags, total and per band"""
    out = []
    for lo, hi in bands + [(0, fk.size)]:
        q = np.zeros_like(dd)
        q[:, lo:hi] = np.where(band[lo:hi], dd[:, lo:hi] * np.conj(h0[lo:hi]) / np.where(band, Sn, 1)[lo:hi], 0)
        full = np.concatenate([q, np.zeros((q.shape[0], N - q.shape[1]))], axis=1)
        out.append(4 * df * np.fft.ifft(full, axis=1) * N)
    return out[:-1], out[-1]


def chi2_of(dd):
    xb, xt = filters(dd)
    lag = np.argmax(np.abs(xt), axis=1)
    idx = np.arange(dd.shape[0])
    xt_l = xt[idx, lag]
    c = np.zeros(dd.shape[0])
    for xi in xb:
        c += np.abs(xi[idx, lag] / rho - xt_l / (P * rho)) ** 2
    return P * c, np.abs(xt_l) / rho


NS = 600
chi_n, zn = chi2_of(noise_fd(NS))
chi_s, zs = chi2_of(noise_fd(NS) + 2.0 * h0)             # a true chirp, rho = 10
# burst: Gaussian-windowed tone near the time the chirp passes 60 Hz, scaled to the chirp's optimal SNR (10)
tg = tc - L.tau_of_f(60.0, Mc)
t = np.arange(N) * dt
hb_t = np.exp(-0.5 * ((t - tg) / 0.04) ** 2) * np.cos(2 * np.pi * 60.0 * t)
hb = np.fft.rfft(hb_t) * dt
hb = hb * (10.0 / np.sqrt(inner(hb, hb)))
chi_g, zg = chi2_of(noise_fd(NS) + hb)
nums.update(TenBchiP=P, TenBchiDof=2 * P - 2,
            TenBchiNoiseMean=chi_n.mean(), TenBchiSigMean=chi_s.mean(), TenBchiBurstMed=np.median(chi_g),
            TenBchiNoiseZ=np.median(zn), TenBchiSigZ=np.median(zs), TenBchiBurstZ=np.median(zg),
            TenBchiThr=stats.chi2.isf(1e-3, 2 * P - 2), TenBchiBurstPass=100 * np.mean(chi_g < stats.chi2.isf(1e-3, 2 * P - 2)),
            TenBchiSigPass=100 * np.mean(chi_s < stats.chi2.isf(1e-3, 2 * P - 2)))

# ---- check 4: match against the Fisher metric
hM = chirp(Mc, rho=5.0)
dpsi, _ = L.dpsi_pn(fk[inband], Mc, 0.25, order=0)
D = np.zeros_like(fk, dtype=complex)
D[inband] = -1j * dpsi * hM[inband]
Gam = inner(D, D)
rows = []
for dl in (1e-4, 3e-4, 1e-3, 3e-3):
    h2 = chirp(Mc * np.exp(dl), rho=None)
    h2 *= np.sqrt(inner(hM, hM) / inner(h2, h2))
    # keep the amplitude law of the template: only the phase changes with Mc, so scale to equal rho
    M = inner(hM, h2) / (np.sqrt(inner(hM, hM)) * np.sqrt(inner(h2, h2)))
    rows.append((dl, 1 - M, 0.5 * dl**2 * Gam / rho**2))
for (dl, a, b), nm in zip(rows, ("A", "B", "C", "D")):
    nums[f"TenBmatch{nm}"] = a
    nums[f"TenBmetric{nm}"] = b
nums["TenBmatchGamma"] = Gam
nums["TenBmatchDl"] = rows[2][0]
save_numbers("ch10", "16_whiten_chi2", L.tidy(nums))

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
xs = np.linspace(0, 45, 300)
bins = np.linspace(0, 45, 40)
ax[0].hist(chi_n, bins=bins, density=True, color=SERIES[0], alpha=0.55, label="noise")
ax[0].hist(chi_s, bins=bins, density=True, color=SERIES[1], alpha=0.55, label="chirp, $\\rho=10$")
ax[0].hist(chi_g, bins=bins, density=True, color=SERIES[2], alpha=0.55, label="burst, same $|x|$")
theory_line(ax[0], xs, stats.chi2.pdf(xs, 2 * P - 2), label=r"$\chi^2_{2p-2}$")
ax[0].axvline(stats.chi2.isf(1e-3, 2 * P - 2), color="k", lw=0.8)
ax[0].set_xlabel(r"$\chi^2$ at the best lag"); ax[0].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
dls = np.array([r[0] for r in rows])
ax[1].loglog(dls, [r[1] for r in rows], "o", color=SERIES[0], label="$1-$match, computed")
theory_line(ax[1], dls, 0.5 * dls**2 * Gam / rho**2, label=r"$\frac{1}{2}\delta^2\Gamma/\rho^2$")
ax[1].set_xlabel(r"$\delta\ln M_c$"); ax[1].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, frameon=False)
fig.tight_layout()
savefig(fig, "ch10", "det_chi2")
print({k: v for k, v in nums.items()})
