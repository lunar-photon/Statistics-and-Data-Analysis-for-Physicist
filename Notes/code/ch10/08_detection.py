"""08_detection.py -- noise alone, or a signal in addition to noise?  Matched filtering in coloured Gaussian noise.

Question 1: a detector output d(t) is either noise n(t) (H0) or noise plus a chirp, n + h (H1). How is the
  optimal statistic distributed under each hypothesis, and how much does not knowing the arrival time and
  phase cost?  We simulate 2000 noise realisations with a known one-sided PSD S_n(f) (the analytic
  advanced-detector shape of Cutler & Flanagan 1994, so that a whole 10+10 Msun chirp fits in 16 s of data),
  generate each Fourier mode as a complex Gaussian with E|n_k|^2 = (T/2) S_n(f_k), and compute
    z  = (d|h)/sqrt((h|h))                                  (template, time and phase known)
    zmax = max over arrival time and phase of |(d|h_t0)|/sqrt((h|h))   (FFT over all lags)
  under H0 and H1 (optimal SNR rho = 5), with their ROC curves.
Question 2: if the PSD is itself estimated from M off-source segments, does the plug-in Gaussian likelihood
  give honest error bars on the signal amplitude, and does the Student-t likelihood obtained by marginalising
  the unknown PSD (prior 1/S) repair it?  Coverage of the 68% interval over 2000 experiments, M = 2..32.
Writes: figures/ch10/det_timeseries.pdf, det_stats.pdf, det_psdcov.pdf, results/ch10/08_detection.tex
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
from common import setup, savefig, save_numbers, rng_for, SERIES, theory_line
import lib_gw10 as L

setup()
rng = rng_for("ch10", "08_detection")

fs, T = 1024.0, 16.0
N = int(fs * T)
dt, df = 1 / fs, 1 / T
fk = np.fft.rfftfreq(N, dt)
band = (fk >= 15.0) & (fk <= 500.0)
Sn = np.full_like(fk, np.inf)
Sn[band] = L.Sn_cf94(fk[band])
sig2 = 0.5 * T * Sn                                    # E |n_k|^2 = (T/2) S_n(f_k)


def noise_fd(n):
    """n realisations of the noise Fourier modes (complex Gaussian, independent across k)."""
    s = np.sqrt(np.where(band, sig2, 0.0) / 2)
    return s * (rng.standard_normal((n, fk.size)) + 1j * rng.standard_normal((n, fk.size)))


def inner(a, b):
    """(a|b) = 4 Re sum a b* / S_n df  over the band."""
    return 4 * df * np.real(np.sum((a * np.conj(b))[..., band] / Sn[band], axis=-1))


# ---------------------------------------------------------------- the signal: Newtonian SPA chirp, 10+10 Msun
m1 = m2 = 10.0
Mc = L.chirp_mass(m1, m2)
fhi = L.f_isco(m1 + m2)
tc = 11.0
inband = (fk >= 20.0) & (fk <= fhi)
h0 = np.zeros_like(fk, dtype=complex)
h0[inband] = fk[inband] ** (-7 / 6) * np.exp(-1j * L.psi_pn(fk[inband], Mc, 0.25, tc=tc, order=0))
RHO = 5.0
h0 *= RHO / np.sqrt(inner(h0, h0))
nums = dict(TenBdetRho=RHO, TenBdetT=T, TenBdetfs=fs, TenBdetfhi=fhi,
            TenBdetDur=L.tau_of_f(20.0, Mc))


# ---------------------------------------------------------------- Fig: one realisation in time
n1 = noise_fd(1)[0]
d_t = np.fft.irfft(n1 + h0, n=N) / dt
h_t = np.fft.irfft(h0, n=N) / dt
t = np.arange(N) * dt
fw, Pw = signal.welch(d_t, fs=fs, nperseg=2048)
# whitened data: divide each Fourier mode by sqrt(S_n), band-limit
white = np.zeros_like(fk, dtype=complex)
white[band] = (n1 + h0)[band] / np.sqrt(Sn[band])
w_t = np.fft.irfft(white, n=N)
whiteh = np.zeros_like(white)
whiteh[band] = h0[band] / np.sqrt(Sn[band])
wh_t = np.fft.irfft(whiteh, n=N)
nums["TenBdetStrainRatio"] = np.std(d_t) / np.max(np.abs(h_t))

fig, ax = plt.subplots(3, 1, figsize=(6.2, 6.4))
ax[0].plot(t, d_t, color=SERIES[0], lw=0.5, label="data $d(t)=n(t)+h(t)$")
ax[0].plot(t, h_t, color=SERIES[1], lw=0.9, label="signal $h(t)$")
ax[0].set_xlim(0, T); ax[0].set_xlabel("$t$ [s]"); ax[0].set_ylabel("strain")
ax[0].legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False, fontsize=8)
ax[1].loglog(fw[1:], Pw[1:], color=SERIES[0], lw=1, label="Welch estimate from this $d(t)$")
theory_line(ax[1], fk[band], Sn[band], label=r"true $S_n(f)$")
ax[1].set_xlim(8, 512); ax[1].set_xlabel("$f$ [Hz]"); ax[1].set_ylabel(r"PSD [Hz$^{-1}$]")
ax[1].legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False, fontsize=8)
ax[2].plot(t, w_t / np.std(w_t), color=SERIES[0], lw=0.4, label="whitened data")
ax[2].plot(t, wh_t / np.std(w_t), color=SERIES[1], lw=1.0, label="whitened signal")
ax[2].set_xlim(tc - 6.5, tc + 0.6); ax[2].set_xlabel("$t$ [s]"); ax[2].set_ylabel("whitened")
ax[2].legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
savefig(fig, "ch10", "det_timeseries")

# ---------------------------------------------------------------- statistics under H0 and H1
NS = 2000
z0, z1, m0, m1s = [], [], [], []
norm = np.sqrt(inner(h0, h0))
for chunk in range(10):
    n, n_b = noise_fd(NS // 10), noise_fd(NS // 10)        # independent noise for the two hypotheses
    for d, zl, ml in ((n, z0, m0), (n_b + h0, z1, m1s)):
        zl.append(inner(d, h0) / norm)
        # complex filter over all lags: x(t0) = 4 sum d h* / S df e^{+2 pi i f t0}; |x| maximises over the phase
        q = np.zeros_like(d)
        q[:, band] = d[:, band] * np.conj(h0[band]) / Sn[band]
        x = 4 * df * np.fft.ifft(np.concatenate([q, np.zeros((q.shape[0], N - q.shape[1]))], axis=1), axis=1) * N
        ml.append(np.max(np.abs(x), axis=1) / norm)
z0, z1, m0, m1s = (np.concatenate(a) for a in (z0, z1, m0, m1s))
nums.update(TenBdetZzeroMean=z0.mean(), TenBdetZzeroSd=z0.std(), TenBdetZoneMean=z1.mean(), TenBdetZoneSd=z1.std(),
            TenBdetMzeroMed=np.median(m0), TenBdetMoneMed=np.median(m1s))
pfa = 1e-2
thr_z = stats.norm.isf(pfa)
thr_m = np.quantile(m0, 1 - pfa)
nums.update(TenBdetThrZ=thr_z, TenBdetThrM=thr_m, TenBdetPdZ=np.mean(z1 > thr_z), TenBdetPdZth=stats.norm.sf(thr_z - RHO),
            TenBdetPdM=np.mean(m1s > thr_m), TenBdetNlags=N)
# trials-factor estimate: independent lags ~ N_eff; P(max |z| > u) ~ N_eff exp(-u^2/2)
neff = pfa / np.exp(-thr_m**2 / 2)
nums["TenBdetNeff"] = int(float(f"{neff:.1g}"))      # one digit: the 1% quantile rests on ~20 null values

fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
xs = np.linspace(-4, 9, 400)
ax[0].hist(z0, bins=50, density=True, color=SERIES[0], alpha=0.6, label="$H_0$")
ax[0].hist(z1, bins=50, density=True, color=SERIES[1], alpha=0.6, label="$H_1$")
theory_line(ax[0], xs, stats.norm.pdf(xs), label=r"$N(0,1)$, $N(\rho,1)$")
ax[0].plot(xs, stats.norm.pdf(xs - RHO), color="k", ls="--", lw=1.4)
ax[0].axvline(thr_z, color="k", lw=0.8)
ax[0].set_xlabel("$z$ (all known)"); ax[0].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, columnspacing=0.8, handlelength=1.2)
xs2 = np.linspace(0, 9, 400)
ax[1].hist(m0, bins=50, density=True, color=SERIES[0], alpha=0.6, label="$H_0$")
ax[1].hist(m1s, bins=50, density=True, color=SERIES[1], alpha=0.6, label="$H_1$")
ax[1].axvline(thr_m, color="k", lw=0.8)
ax[1].set_xlabel(r"$z_{\max}$ (time, phase unknown)"); ax[1].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, columnspacing=0.8, handlelength=1.2)
# ROC
grid = np.linspace(-3, 10, 600)
ax[2].plot([np.mean(z0 > g) for g in grid], [np.mean(z1 > g) for g in grid], color=SERIES[0], label="$z$")
ax[2].plot([np.mean(m0 > g) for g in grid], [np.mean(m1s > g) for g in grid], color=SERIES[2], label=r"$z_{\max}$")
pf = np.geomspace(1e-4, 1, 200)
ax[2].plot(pf, stats.norm.sf(stats.norm.isf(pf) - RHO), color="k", ls="--", lw=1.2, label="theory, $z$")
ax[2].set_xscale("log"); ax[2].set_xlim(5e-4, 1); ax[2].set_ylim(0, 1.02)
ax[2].set_xlabel("false-alarm probability"); ax[2].set_ylabel("detection probability")
ax[2].legend(fontsize=7, loc="lower right")
fig.tight_layout()
savefig(fig, "ch10", "det_stats")

# ---------------------------------------------------------------- PSD estimated from M segments: amplitude coverage
NE = 2000
Agrid = np.linspace(-1.5, 3.5, 161)          # amplitude in units of the true one (true A = 1)
hb = h0[band]
s_true = sig2[band]
cov = {}
for M in (2, 4, 8, 16, 32):
    hitG = hitT = 0
    widthG, widthT, estG, estT = [], [], [], []
    for chunk in range(NE // 100):
        nb = 100
        on = np.sqrt(s_true / 2) * (rng.standard_normal((nb, hb.size)) + 1j * rng.standard_normal((nb, hb.size)))
        d = on + hb
        off = np.sqrt(s_true / 2) * (rng.standard_normal((nb, M, hb.size)) + 1j * rng.standard_normal((nb, M, hb.size)))
        shat = np.mean(np.abs(off) ** 2, axis=1)                  # estimate of E|n_k|^2 per bin
        # Gaussian likelihood with the estimate plugged in:  A_hat = (d|h)/(h|h),  sigma = 1/sqrt(h|h)
        hh = np.sum(np.abs(hb) ** 2 / shat, axis=1) * 2
        dh = np.sum(np.real(d * np.conj(hb)) / shat, axis=1) * 2
        Ahat, sA = dh / hh, 1 / np.sqrt(hh)
        hitG += np.sum(np.abs(Ahat - 1) < sA)
        widthG.append(sA)
        estG.append(Ahat)
        # Student-t likelihood: ln L(A) = -(M+1) sum ln(1 + |d - A h|^2/(M shat))
        lnL = np.empty((nb, Agrid.size))
        for j, A in enumerate(Agrid):
            lnL[:, j] = -(M + 1) * np.sum(np.log1p(np.abs(d - A * hb) ** 2 / (M * shat)), axis=1)
        p = np.exp(lnL - lnL.max(axis=1, keepdims=True))
        cdf = np.cumsum(p, axis=1); cdf /= cdf[:, -1:]
        lo = np.array([np.interp(0.15865, c_, Agrid) for c_ in cdf])
        hi = np.array([np.interp(0.84135, c_, Agrid) for c_ in cdf])
        hitT += np.sum((lo < 1) & (hi > 1))
        widthT.append((hi - lo) / 2)
        estT.append(np.array([np.interp(0.5, c_, Agrid) for c_ in cdf]))
    cov[M] = (hitG / NE, hitT / NE, np.mean(np.concatenate(widthG)), np.mean(np.concatenate(widthT)),
              np.std(np.concatenate(estG)), np.std(np.concatenate(estT)))
    print(M, cov[M])
names = {2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 32: "ThirtyTwo"}
for M, (cg, ct, wg, wt, sg, st) in cov.items():
    nums[f"TenBsdG{names[M]}"] = sg * RHO
    nums[f"TenBsdT{names[M]}"] = st * RHO
    nums[f"TenBcovG{names[M]}"] = 100 * cg
    nums[f"TenBcovT{names[M]}"] = 100 * ct
    nums[f"TenBwidG{names[M]}"] = wg * RHO
    nums[f"TenBwidT{names[M]}"] = wt * RHO
nums["TenBcovMCerr"] = 100 * np.sqrt(0.68 * 0.32 / NE)
save_numbers("ch10", "08_detection", L.tidy(nums))

fig, ax = plt.subplots(figsize=(5.2, 3.0))
Ms = np.array(sorted(cov))
ax.plot(Ms, [100 * cov[m][0] for m in Ms], "o-", color=SERIES[1], label="Gaussian, estimated PSD plugged in")
ax.plot(Ms, [100 * cov[m][1] for m in Ms], "s-", color=SERIES[0], label="Student-$t$ (PSD marginalised)")
ax.axhline(68.27, color="k", ls="--", lw=1.2, label="nominal 68.3%")
ax.fill_between([1.5, 40], 68.27 - 2 * nums["TenBcovMCerr"], 68.27 + 2 * nums["TenBcovMCerr"], color="0.85")
ax.set_xscale("log", base=2); ax.set_xlim(1.7, 38)
ax.set_xlabel("number $M$ of off-source segments used to estimate $S_n$")
ax.set_ylabel("coverage of the 68% interval [%]")
ax.legend(fontsize=8, loc="lower right")
savefig(fig, "ch10", "det_psdcov")
