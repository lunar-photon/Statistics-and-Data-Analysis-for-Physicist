"""Metropolis--Hastings sampler and chain diagnostics, written to be read.

Used by the scripts of parts 9b and 9c, and imported by 10b and 12c:

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ch09"))
    from lib_mcmc import metropolis, mh, acf, tau_int, ess, mcse, gelman_rubin, corner

Conventions
-----------
* A target is given by its LOG density ``logp(x)`` up to an additive constant: only
  differences logp(y) - logp(x) are ever used, so the normalising constant (the evidence of a
  posterior, the partition function of a Boltzmann weight) never has to be known.
* ``logp`` returns -inf outside the support; such proposals are rejected and the chain stays.
* A chain is an array of shape (n + 1, D): the starting point and n further states.
  A rejected proposal REPEATS the current state; the repeats are part of the sample.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------------------------------
# samplers
# ------------------------------------------------------------------------------------------
def metropolis(logp, x0, n, step, rng, cov=None):
    """Random-walk Metropolis with Gaussian proposal y = x + step * L z,  z ~ N(0, 1).

    L is the Cholesky factor of ``cov`` (identity if None).  The proposal is symmetric,
    q(y|x) = q(x|y), so the acceptance probability is min(1, p(y)/p(x)).
    Returns (chain, logp_chain, acceptance_fraction).
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    D = x.size
    L = np.eye(D) if cov is None else np.linalg.cholesky(np.atleast_2d(cov))
    chain = np.empty((n + 1, D))
    lp_chain = np.empty(n + 1)
    lp = logp(x)
    chain[0], lp_chain[0] = x, lp
    kicks = step * rng.standard_normal((n, D)) @ L.T     # all proposal increments at once
    log_u = np.log(rng.random(n))                          # all uniform draws at once
    accepted = 0
    for t in range(n):
        y = x + kicks[t]                                   # 1. propose: the unbiased walker
        lp_y = logp(y)
        if log_u[t] < lp_y - lp:                           # 2. the bias: accept with min(1, p(y)/p(x))
            x, lp = y, lp_y
            accepted += 1
        chain[t + 1], lp_chain[t + 1] = x, lp              # 3. a rejection repeats x
    return chain, lp_chain, accepted / n


def mh(logp, x0, n, propose, log_q, rng):
    """General Metropolis--Hastings.

    propose(x, rng) -> y draws a candidate;  log_q(a, b) = log q(b | a) is the log density of
    proposing b from a.  Acceptance probability min(1, p(y) q(x|y) / (p(x) q(y|x))).
    Returns (chain, logp_chain, acceptance_fraction).
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    chain = np.empty((n + 1, x.size))
    lp_chain = np.empty(n + 1)
    lp = logp(x)
    chain[0], lp_chain[0] = x, lp
    accepted = 0
    for t in range(n):
        y = np.atleast_1d(propose(x, rng))
        lp_y = logp(y)
        log_r = lp_y - lp + log_q(y, x) - log_q(x, y)      # Hastings ratio, in logs
        if np.log(rng.random()) < log_r:
            x, lp = y, lp_y
            accepted += 1
        chain[t + 1], lp_chain[t + 1] = x, lp
    return chain, lp_chain, accepted / n


# ------------------------------------------------------------------------------------------
# diagnostics for one scalar series x_1..x_N (e.g. one column of a chain after burn-in)
# ------------------------------------------------------------------------------------------
def acf(x, maxlag=None):
    """Estimated autocorrelation rho_k = C_k / C_0 for k = 0..maxlag (FFT, zero padded)."""
    x = np.asarray(x, dtype=float)
    N = x.size
    xc = x - x.mean()
    nfft = 1 << (2 * N - 1).bit_length()                   # padding avoids circular wrap-around
    f = np.fft.rfft(xc, nfft)
    c = np.fft.irfft(f * np.conj(f), nfft)[:N] / N         # C_k = (1/N) sum_t xc_t xc_{t+k}
    rho = c / c[0]
    return rho if maxlag is None else rho[: maxlag + 1]


def tau_int(x, c=5.0):
    """Integrated autocorrelation time tau = 1 + 2 sum_{k=1}^{M} rho_k with Sokal's window:
    the smallest M such that M >= c * tau(M).  Returns (tau, M)."""
    rho = acf(x)
    taus = 2.0 * np.cumsum(rho) - 1.0                      # taus[M] = 1 + 2 sum_{k=1}^{M} rho_k
    ok = np.arange(rho.size) >= c * taus
    M = int(np.argmax(ok)) if ok.any() else rho.size - 1
    return float(taus[M]), M


def ess(x, c=5.0):
    """Effective sample size N / tau."""
    return np.asarray(x).size / tau_int(x, c)[0]


def mcse(x, c=5.0):
    """Monte Carlo standard error of the chain mean, sd * sqrt(tau / N)."""
    x = np.asarray(x, dtype=float)
    return x.std(ddof=1) * np.sqrt(tau_int(x, c)[0] / x.size)


def gelman_rubin(chains, split=False):
    """Potential scale reduction factor R-hat = sqrt(V / W) for M chains of one quantity.

    chains: array (M, N).  W = mean within-chain variance, B/N = variance of the chain means,
    V = (N-1)/N W + (1 + 1/M) B/N.  With split=True each chain is cut in two halves first,
    so that a chain that is still drifting also inflates R-hat.
    """
    ch = np.asarray(chains, dtype=float)
    if split:
        h = ch.shape[1] // 2
        ch = np.concatenate([ch[:, :h], ch[:, h:2 * h]], axis=0)
    M, N = ch.shape
    means = ch.mean(axis=1)
    W = ch.var(axis=1, ddof=1).mean()
    B_over_N = means.var(ddof=1)
    V = (N - 1) / N * W + (1 + 1 / M) * B_over_N
    return float(np.sqrt(V / W))


# ------------------------------------------------------------------------------------------
# a triangle ("corner") plot written by hand
# ------------------------------------------------------------------------------------------
def _hpd_levels(H, fracs=(0.68, 0.95)):
    """Heights of a 2-D histogram that enclose the given fractions of the samples."""
    h = np.sort(H.ravel())[::-1]
    cum = np.cumsum(h) / h.sum()
    return sorted(h[np.searchsorted(cum, f)] for f in fracs)


def corner(samples, labels, color="#2a78d6", truths=None, bins=40, fig=None, ranges=None,
           label=None):
    """Triangle plot: 1-D histograms on the diagonal, 68% and 95% contours below it.

    samples: (n, D).  Call twice with the same ``fig`` to overlay two sample sets.
    """
    import matplotlib.pyplot as plt
    s = np.asarray(samples)
    D = s.shape[1]
    if fig is None:
        fig, _ = plt.subplots(D, D, figsize=(2.1 * D, 2.1 * D))
    axes = np.array(fig.axes).reshape(D, D)
    if ranges is None:
        ranges = [np.percentile(s[:, i], [0.2, 99.8]) for i in range(D)]
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(s[:, i], bins=bins, range=ranges[i], density=True, histtype="step",
                        color=color, lw=1.4, label=label)
                ax.set_yticks([])
            else:
                H, xe, ye = np.histogram2d(s[:, j], s[:, i], bins=bins // 2,
                                           range=[ranges[j], ranges[i]])
                xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                lev = _hpd_levels(H)
                ax.contour(xc, yc, H.T, levels=lev + [H.max() + 1], colors=color,
                           linewidths=[0.9, 1.4])
            if truths is not None:
                ax.axvline(truths[j], color="k", lw=0.6, ls=":")
                if i != j:
                    ax.axhline(truths[i], color="k", lw=0.6, ls=":")
            ax.set_xlim(ranges[j])
            if i != j:
                ax.set_ylim(ranges[i])
            if i == D - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j > 0:
                ax.set_yticklabels([])
    return fig
