"""30_unfolding.py -- unfolding a smeared spectrum with a response matrix built by Monte Carlo.

Question: a detector smears and loses events, so the measured histogram n is the true one
mu pushed through a response matrix R (nu = R mu).  How do we get mu back, and how honest
are the error bars of the answer?

  1. Truth: two peaks on a flat floor, y in [0, 1], 20 bins, mu_tot = 20000 events.
     Detector: Gaussian smearing x = y + N(0, sigma^2), sigma = 1.5 bin widths; events
     smeared outside [0, 1] are lost (so the edge bins have efficiency < 1).
  2. R_ij = P(observed in bin i | true in bin j) from 2x10^6 simulated events, generated
     flat in y (the analyser does not know the truth); its singular values against the
     Gaussian-kernel formula exp(-q^2 sigma^2 / 2), q = pi k / L.
  3. Naive inverse mu-hat = R^{-1} n: unbiased, covariance R^{-1} V R^{-T}.
  4. Tikhonov (curvature) regularisation: minimise
       (n - R mu)^T V^{-1} (n - R mu) + tau |C mu|^2,  C = second differences,
     a linear estimator mu-hat = A_tau n; exact bias (A_tau R - 1) mu and covariance
     A_tau V A_tau^T against tau -> minimum mean squared error; L-curve from one data set.
  5. Iterative Bayesian unfolding (D'Agostini 1995), flat starting guess; mean squared
     error against the number of iterations from toys.
  6. Bin-by-bin correction factors C_i = mu_i^MC / nu_i^MC from the flat simulation.
  7. Coverage by toys: for each method, the fraction of toy experiments in which the
     quoted 1-sigma error bar of each bin contains the true value.

Writes: figures/ch06/unfold_inverse.pdf, figures/ch06/unfold_response.pdf,
        figures/ch06/unfold_regularised.pdf, figures/ch06/unfold_coverage.pdf,
        results/ch06/30_unfolding.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, rng_for, theory_line, SERIES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = rng_for("ch06", "30_unfolding")
out = {}

# ---------------------------------------------------------------- 1. truth and detector
M = 20                                   # true bins = observed bins
edges = np.linspace(0.0, 1.0, M + 1)
cen = 0.5 * (edges[1:] + edges[:-1])
Delta = edges[1] - edges[0]
sigma = 1.5 * Delta                      # Gaussian resolution
mu_tot = 20_000


def cdf_true(y):                         # 20% flat + two Gaussian peaks of 40% each
    return (0.2 * y + 0.4 * stats.norm.cdf(y, 0.3, 0.08) + 0.4 * stats.norm.cdf(y, 0.7, 0.08))


p_true = np.diff(cdf_true(edges)) / (cdf_true(1.0) - cdf_true(0.0))
mu = mu_tot * p_true                     # true histogram (expectation values)

# ---------------------------------------------------------------- 2. response matrix by Monte Carlo
N_mc = 2_000_000
y_mc = rng.random(N_mc)                  # simulated truth, flat
x_mc = y_mc + sigma * rng.standard_normal(N_mc)
H, _, _ = np.histogram2d(x_mc, y_mc, bins=[edges, edges])   # H[i, j]: observed i, true j
gen_j = np.histogram(y_mc, bins=edges)[0]
R = H / gen_j                            # R_ij = N(obs i, true j) / N(true j)
eff = R.sum(axis=0)                      # column sums = efficiency of each true bin
nu = R @ mu                              # expected observed histogram

# response from a simulation that follows the true shape, to see the model dependence
y_sh = np.interp(rng.random(N_mc), cdf_true(np.linspace(0, 1, 4001)) / cdf_true(1.0),
                 np.linspace(0, 1, 4001))
x_sh = y_sh + sigma * rng.standard_normal(N_mc)
H2, _, _ = np.histogram2d(x_sh, y_sh, bins=[edges, edges])
R_shape = H2 / np.histogram(y_sh, bins=edges)[0]
out.update(SixBUfM=M, SixBUfMuTot=r"20\,000", SixBUfNmc=r"2\times10^{6}", SixBUfSigmaBins=1.5,
           SixBUfStay=100 * np.round(R[M // 2, M // 2], 3),
           SixBUfOne=100 * np.round(R[M // 2 + 1, M // 2] + R[M // 2 - 1, M // 2], 3),
           SixBUfTwo=100 * np.round(1 - R[M // 2 - 1:M // 2 + 2, M // 2].sum(), 3),
           SixBUfEffEdge=np.round(eff[0], 3), SixBUfEffMid=np.round(eff[M // 2], 3),
           SixBUfModelDep=np.round(100 * np.max(np.abs(R_shape @ mu - nu) / nu), 2),
           SixBUfMCrelerr=np.round(100 / np.sqrt(N_mc / M * R[M // 2, M // 2]), 2))

# singular values against the Gaussian kernel formula
s = np.linalg.svd(R, compute_uv=False)
k = np.arange(M)
q = np.pi * k / 1.0                      # wavenumber of the k-th cosine mode on [0, L], L = 1
lam_formula = np.exp(-0.5 * (q * sigma) ** 2)
out.update(SixBUfSmax=np.round(s[0], 3), SixBUfSmin=s[-1], SixBUfCond=s[0] / s[-1],
           SixBUfLamMinFormula=lam_formula[-1])

# ---------------------------------------------------------------- 3. one experiment, naive inverse
n_obs = rng.poisson(nu)
Rinv = np.linalg.inv(R)
mu_inv = Rinv @ n_obs
U_inv = Rinv @ np.diag(n_obs) @ Rinv.T
sd_inv = np.sqrt(np.diag(U_inv))
rho_inv = U_inv[M // 2, M // 2 + 1] / (sd_inv[M // 2] * sd_inv[M // 2 + 1])
out.update(SixBUfNrelLo=np.round(100 / np.sqrt(nu.max()), 1), SixBUfNrelHi=np.round(100 / np.sqrt(nu.min()), 1),
           SixBUfInvRelMid=np.round(sd_inv[M // 2] / mu[M // 2], 1), SixBUfInvSdMid=int(round(sd_inv[M // 2], -2)),
           SixBUfInvRho=np.round(rho_inv, 3), SixBUfMuMid=int(round(mu[M // 2])),
           SixBUfInvAmp=int(round(sd_inv[M // 2] / np.sqrt(nu[M // 2]), -1)))

setup(9.0, 5.2)
fig, ax = plt.subplots(2, 2, sharex=True)
for a, h, lab in [(ax[0, 0], mu, r"(a) true $\mu_j$"), (ax[0, 1], nu, r"(b) expected $\nu_i=\sum_j R_{ij}\mu_j$"),
                  (ax[1, 0], n_obs, r"(c) one data set $n_i$")]:
    a.stairs(h, edges, color=SERIES[0], lw=1.4)
    a.set_title(lab, loc="left")
    a.set_ylim(0, 2300)
ax[1, 0].errorbar(cen, n_obs, yerr=np.sqrt(n_obs), fmt="none", ecolor=SERIES[0], lw=1)
ax[1, 1].stairs(mu, edges, color=SERIES[2], lw=1.2, label=r"true $\mu$")
ax[1, 1].errorbar(cen, mu_inv, yerr=sd_inv, fmt="o", ms=3, color=SERIES[1], label=r"$R^{-1}n$")
ax[1, 1].set_title(r"(d) naive inverse $\hat\mu=R^{-1}n$", loc="left")
ax[1, 1].legend(fontsize=8, loc="lower right")
for a in ax[1]:
    a.set_xlabel("$x$ (or $y$)")
ax[0, 0].set_ylabel("events per bin")
ax[1, 0].set_ylabel("events per bin")
fig.tight_layout()
savefig(fig, "ch06", "unfold_inverse")

# ---------------------------------------------------------------- 4. Tikhonov (curvature) regularisation
C = np.zeros((M - 2, M))
for i in range(M - 2):
    C[i, i:i + 3] = [1.0, -2.0, 1.0]     # second difference mu_i - 2 mu_{i+1} + mu_{i+2}
G = C.T @ C


def tikhonov_matrix(tau, Vdiag):
    """A_tau with mu-hat = A_tau n, for data covariance diag(Vdiag)."""
    W = R.T / Vdiag                       # R^T V^{-1}
    return np.linalg.solve(W @ R + tau * G, W)


taus = np.logspace(-8, 0, 161)
bias2, var = [], []
for tau in taus:
    A = tikhonov_matrix(tau, nu)          # exact: built with the true covariance diag(nu)
    b = (A @ R - np.eye(M)) @ mu
    U = A @ np.diag(nu) @ A.T
    bias2.append(np.mean(b ** 2)); var.append(np.mean(np.diag(U)))
bias2, var = np.array(bias2), np.array(var)
mse = bias2 + var
i_best = np.argmin(mse)
tau_mse = taus[i_best]

# L-curve from the one data set: residual chi^2 against roughness |C mu|^2
Vd = np.maximum(n_obs, 1)
res, rough, sols = [], [], []
for tau in taus:
    m_hat = tikhonov_matrix(tau, Vd) @ n_obs
    res.append(np.sum((n_obs - R @ m_hat) ** 2 / Vd)); rough.append(np.sum((C @ m_hat) ** 2))
    sols.append(m_hat)
lr, le = np.log(np.array(res)), np.log(np.array(rough))
d1r, d1e = np.gradient(lr), np.gradient(le)
d2r, d2e = np.gradient(d1r), np.gradient(d1e)
curv = (d1r * d2e - d2r * d1e) / (d1r ** 2 + d1e ** 2) ** 1.5
inner = slice(10, len(taus) - 10)
i_L = np.arange(len(taus))[inner][np.argmax(curv[inner])]
tau_L = taus[i_L]
out.update(SixBUfTauMSE=tau_mse, SixBUfTauL=tau_L,
           SixBUfMSEmin=int(round(mse[i_best])), SixBUfBiasAtMin=np.round(np.sqrt(bias2[i_best]), 1),
           SixBUfSdAtMin=np.round(np.sqrt(var[i_best]), 1),
           SixBUfMSEinv=np.round(np.mean(np.diag(U_inv)), 0),
           SixBUfRmsInv=int(round(np.sqrt(np.mean(np.diag(Rinv @ np.diag(nu) @ Rinv.T))), -2)),
           SixBUfRmsMin=np.round(np.sqrt(mse[i_best]), 1),
           SixBUfRmsPoisson=np.round(np.sqrt(np.mean(mu)), 1))
A_best = tikhonov_matrix(tau_mse, Vd)
mu_tik = A_best @ n_obs
sd_tik = np.sqrt(np.diag(A_best @ np.diag(Vd) @ A_best.T))
out.update(SixBUfTikChiTrue=np.round(np.sum((mu_tik - mu) ** 2 / sd_tik ** 2), 1))

# ---------------------------------------------------------------- 5. iterative Bayesian unfolding


def ibu(n, iters, prior=None):
    """D'Agostini iterations; n may be (..., M).  Returns the estimate after `iters` steps."""
    p = np.full(M, 1.0 / M) if prior is None else prior / prior.sum()
    m_hat = np.broadcast_to(p * n.sum(axis=-1, keepdims=True), n.shape).copy()
    for _ in range(iters):
        nu_hat = m_hat @ R.T                              # sum_j R_ij mu_j
        m_hat = m_hat / eff * ((n / nu_hat) @ R)          # mu_j/eps_j sum_i R_ij n_i / nu_i
    return m_hat


iters_grid = np.array([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200, 500])
toys = rng.poisson(nu, size=(1000, M)).astype(float)
mse_ibu = []
for it in iters_grid:
    est = ibu(toys, it)
    mse_ibu.append(np.mean((est - mu) ** 2))
mse_ibu = np.array(mse_ibu)
k_best = int(iters_grid[np.argmin(mse_ibu)])
mu_ibu = ibu(n_obs.astype(float), k_best)
boot = ibu(rng.poisson(n_obs, size=(400, M)).astype(float), k_best)
sd_ibu = boot.std(axis=0, ddof=1)
out.update(SixBUfKbest=k_best, SixBUfMSEibu=int(round(mse_ibu.min())),
           SixBUfRmsIbu=np.round(np.sqrt(mse_ibu.min()), 1),
           SixBUfItLast=int(iters_grid[-1]), SixBUfMSEibuLast=int(round(mse_ibu[-1], -2)))

# ---------------------------------------------------------------- 6. bin-by-bin correction factors
nu_mc_flat = R @ np.full(M, 1.0)        # flat MC truth, one unit per bin
Cfac = 1.0 / nu_mc_flat                  # mu^MC / nu^MC with mu^MC = 1 per bin
bias_cf = Cfac * nu - mu
out.update(SixBUfCfBiasPeak=np.round(100 * bias_cf[np.argmax(mu)] / mu[np.argmax(mu)], 1),
           SixBUfCfBiasDip=np.round(100 * bias_cf[M // 2] / mu[M // 2], 1))

setup(9.0, 5.4)
fig, ax = plt.subplots(2, 2)
ax[0, 0].stairs(mu, edges, color="k", lw=1.0, label=r"true $\mu$")
ax[0, 0].errorbar(cen - 0.006, mu_tik, yerr=sd_tik, fmt="o", ms=3, color=SERIES[0], label="Tikhonov, min. MSE")
ax[0, 0].errorbar(cen + 0.006, mu_ibu, yerr=sd_ibu, fmt="s", ms=3, color=SERIES[1], label=f"Bayesian, {k_best} iterations")
ax[0, 0].set(xlabel="$y$", ylabel="events per bin", ylim=(0, 3300))
ax[0, 0].legend(fontsize=7, loc="upper center", ncol=2, columnspacing=0.8, handletextpad=0.3)
ax[0, 0].set_title("(a) one data set unfolded", loc="left")
ax[0, 1].loglog(taus, var, color=SERIES[0], label="variance")
ax[0, 1].loglog(taus, bias2, color=SERIES[1], label="bias$^2$")
ax[0, 1].loglog(taus, mse, color="k", lw=1.2, label="MSE")
ax[0, 1].axvline(tau_mse, color=SERIES[2], lw=0.8, ls=":")
ax[0, 1].axvline(tau_L, color=SERIES[4], lw=0.8, ls="--")
ax[0, 1].set(xlabel=r"regularisation strength $\tau$", ylabel="per bin (events$^2$)", ylim=(1, 1e8))
ax[0, 1].legend(fontsize=7, loc="upper right")
ax[0, 1].set_title("(b) Tikhonov: bias against variance", loc="left")
ax[1, 0].loglog(res, rough, color=SERIES[0])
ax[1, 0].plot(res[i_L], rough[i_L], "o", color=SERIES[4], label=r"L-curve corner")
ax[1, 0].plot(res[i_best], rough[i_best], "s", color=SERIES[2], label="minimum MSE")
ax[1, 0].set(xlabel=r"misfit $\chi^2(\hat\mu_\tau)$", ylabel=r"roughness $|C\hat\mu_\tau|^2$")
ax[1, 0].legend(fontsize=7)
ax[1, 0].set_title("(c) L-curve of the one data set", loc="left")
ax[1, 1].loglog(iters_grid, mse_ibu, "o-", color=SERIES[1], ms=3)
ax[1, 1].axvline(k_best, color=SERIES[2], lw=0.8, ls=":")
ax[1, 1].set(xlabel="iterations", ylabel="MSE per bin (events$^2$)")
ax[1, 1].set_title("(d) Bayesian iterations: MSE", loc="left")
fig.tight_layout()
savefig(fig, "ch06", "unfold_regularised")

# ---------------------------------------------------------------- 7. coverage by toys
Ntoy = 2000
toys = rng.poisson(nu, size=(Ntoy, M)).astype(float)
cov = {}
# naive inverse, errors from diag(n)
est = toys @ Rinv.T
sd = np.sqrt(toys @ (Rinv ** 2).T)       # diagonal of R^{-1} diag(n) R^{-T}
cov["inverse"] = np.mean(np.abs(est - mu) <= sd, axis=0)
mean_inv = est.mean(axis=0)
out.update(SixBUfInvBiasPull=np.round(np.max(np.abs(mean_inv - mu) / (est.std(axis=0) / np.sqrt(Ntoy))), 2))
# Tikhonov at tau_MSE, V from each toy's own counts
c_t = np.zeros((Ntoy, M))
for t in range(Ntoy):
    Vt = np.maximum(toys[t], 1)
    A = tikhonov_matrix(tau_mse, Vt)
    c_t[t] = np.abs(A @ toys[t] - mu) <= np.sqrt(np.einsum("ik,k,ik->i", A, Vt, A))
cov["Tikhonov"] = c_t.mean(axis=0)
# Bayesian iterations, errors from Poisson resampling of each toy
c_b = np.zeros((Ntoy, M))
for t in range(0, Ntoy, 100):
    block = toys[t:t + 100]
    e = ibu(block, k_best)
    rs = ibu(rng.poisson(np.repeat(block[:, None, :], 100, axis=1)).astype(float), k_best)
    c_b[t:t + 100] = np.abs(e - mu) <= rs.std(axis=1, ddof=1)
cov["Bayesian"] = c_b.mean(axis=0)
# correction factors, error C_i sqrt(n_i)
cov["correction"] = np.mean(np.abs(Cfac * toys - mu) <= Cfac * np.sqrt(toys), axis=0)
nominal = stats.norm.cdf(1) - stats.norm.cdf(-1)
out.update(SixBUfNtoy=Ntoy, SixBUfNominal=np.round(100 * nominal, 1),
           SixBUfCovInvMean=np.round(100 * cov["inverse"].mean(), 1),
           SixBUfCovTikMean=np.round(100 * cov["Tikhonov"].mean(), 1),
           SixBUfCovTikMin=np.round(100 * cov["Tikhonov"].min(), 1),
           SixBUfCovIbuMean=np.round(100 * cov["Bayesian"].mean(), 1),
           SixBUfCovIbuMin=np.round(100 * cov["Bayesian"].min(), 1),
           SixBUfCovCfMean=np.round(100 * cov["correction"].mean(), 1),
           SixBUfCovCfMin=np.round(100 * cov["correction"].min(), 1),
           SixBUfCovTikPeak=np.round(100 * cov["Tikhonov"][np.argmax(mu)], 1),
           SixBUfCovIbuPeak=np.round(100 * cov["Bayesian"][np.argmax(mu)], 1),
           SixBUfCovSe=np.round(100 * np.sqrt(nominal * (1 - nominal) / Ntoy), 1))

setup(9.0, 3.0)
fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.25]})
ax[0].semilogy(k + 1, s, "o", color=SERIES[0], ms=4, label="singular values of $R$")
theory_line(ax[0], k + 1, lam_formula, label=r"$e^{-q_k^2\sigma^2/2}$")
ax[0].set(xlabel="mode number $k+1$", ylabel="gain of the mode")
ax[0].legend(fontsize=8)
im = ax[1].imshow(R, origin="lower", extent=[0, 1, 0, 1], cmap="Blues")
ax[1].set(xlabel="true $y$ (bin $j$)", ylabel="observed $x$ (bin $i$)")
ax[1].grid(False)
fig.colorbar(im, ax=ax[1], label=r"$R_{ij}$")
fig.tight_layout()
savefig(fig, "ch06", "unfold_response")

setup(9.0, 3.0)
fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1.0]})
for c, (name, lab) in zip([SERIES[0], SERIES[1], SERIES[2], SERIES[3]],
                          [("inverse", r"naive $R^{-1}n$"), ("Tikhonov", "Tikhonov, min. MSE"),
                           ("Bayesian", f"Bayesian, {k_best} it."), ("correction", "correction factors")]):
    ax[0].plot(cen, 100 * cov[name], "o-", ms=3, lw=1, color=c, label=lab)
ax[0].axhline(100 * nominal, color="k", ls="--", lw=1)
ax[0].set(xlabel="$y$", ylabel=r"coverage of $\pm1\sigma$ (%)", ylim=(0, 125))
ax[0].legend(fontsize=7, loc="upper center", ncol=2)
ax[1].stairs(mu, edges, color="k", lw=1.0, label=r"true $\mu$")
ax[1].plot(cen, ibu(nu, k_best), "s", ms=3, color=SERIES[1], label=r"Bayesian on $\nu$ (no noise)")
ax[1].plot(cen, tikhonov_matrix(tau_mse, nu) @ nu, "o", ms=3, color=SERIES[0], label=r"Tikhonov on $\nu$")
ax[1].set(xlabel="$y$", ylabel="events per bin", ylim=(0, 3000))
ax[1].legend(fontsize=7, loc="upper center")
fig.tight_layout()
savefig(fig, "ch06", "unfold_coverage")

save_numbers("ch06", "30_unfolding", out)
print(out)
print("coverage per bin:", {k_: np.round(v_, 3) for k_, v_ in cov.items()})
print("ibu mse:", dict(zip(iters_grid.tolist(), np.round(mse_ibu, 0).tolist())))
