"""lib_mcmc9c.py -- the biased random walker and its diagnostics, as used in part 9c.

A compact, self-contained version of the chapter's Metropolis-Hastings toolkit:

  mh(logpost, x0, cov, n, rng)        one random-walk Metropolis chain (Gaussian proposal)
  run_chains(...)                     several chains from dispersed starting points
  rhat(chains)                        Gelman-Rubin potential scale reduction (last halves)
  acf(x), tau_int(x), ess(x)          autocorrelation, integrated autocorrelation time, ESS
  hpd_levels(H, probs)                density levels enclosing given probability (for contours)
  triangle(...)                       a corner / triangle plot written by hand with matplotlib
  gauss_ellipse(ax, mean, cov, ...)   the 1- and 2-sigma ellipses of a Gaussian (Fisher, analytic)

Conventions: a chain is an array of shape (n_steps, n_params); several chains are stacked as
(n_chains, n_steps, n_params).  logpost returns ln p(theta | d) up to a constant, or -inf
outside the prior support.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ the sampler
def mh(logpost, x0, cov, n, rng, scale=None):
    """Random-walk Metropolis: propose y = x + L z (z ~ N(0,1), L L^T = scale^2 cov),
    accept with probability min(1, p(y)/p(x)).  The proposal is symmetric, so q cancels.

    Returns (chain, logp, acceptance fraction).  scale defaults to 2.38/sqrt(d).
    """
    x = np.array(x0, dtype=float)
    d = x.size
    s = 2.38 / np.sqrt(d) if scale is None else scale
    L = np.linalg.cholesky(np.atleast_2d(cov)) * s
    chain = np.empty((n, d))
    lp = np.empty(n)
    lx = logpost(x)
    z = rng.standard_normal((n, d)) @ L.T          # all proposal steps drawn at once
    logu = np.log(rng.random(n))                   # all accept-reject coins drawn at once
    acc = 0
    for t in range(n):
        y = x + z[t]
        ly = logpost(y)
        if logu[t] < ly - lx:                      # the biasing step: uphill always, downhill sometimes
            x, lx = y, ly
            acc += 1
        chain[t] = x
        lp[t] = lx
    return chain, lp, acc / n


def run_chains(logpost, starts, cov, n, rng, scale=None):
    """One chain per starting point; returns chains (k, n, d), logps (k, n), acceptance (k,)."""
    out = [mh(logpost, s, cov, n, rng, scale) for s in starts]
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]),
            np.array([o[2] for o in out]))


# ------------------------------------------------------------------ diagnostics
def rhat(chains):
    """Gelman-Rubin R for each parameter from the last halves of k chains of length n.

    W = mean within-chain variance, B/n = variance of the chain means,
    V = (n-1)/n W + B/n,  R = sqrt(V / W).
    """
    c = np.asarray(chains)
    c = c[:, c.shape[1] // 2:, :]
    k, n, _ = c.shape
    means = c.mean(axis=1)
    W = c.var(axis=1, ddof=1).mean(axis=0)
    B_over_n = means.var(axis=0, ddof=1)
    V = (n - 1) / n * W + B_over_n
    return np.sqrt(V / W)


def acf(x, maxlag=None):
    """Normalised autocorrelation rho(k) of a 1-D series, via FFT (zero padding)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    f = np.fft.rfft(x, n=2 * n)
    r = np.fft.irfft(f * np.conj(f))[:n]
    r /= r[0]
    return r if maxlag is None else r[: maxlag + 1]


def tau_int(x, c=5.0):
    """Integrated autocorrelation time tau = 1 + 2 sum_{k>=1} rho(k), summed up to the first
    window M with M >= c tau(M) (Sokal's automatic window)."""
    r = acf(x)
    taus = 2.0 * np.cumsum(r) - 1.0
    m = np.arange(taus.size)
    w = np.where(m >= c * taus)[0]
    M = w[0] if w.size else taus.size - 1
    return float(taus[M])


def ess(x):
    """Effective sample size N / tau_int."""
    return x.size / tau_int(x)


def summary(samples, names=None):
    """Mean, standard deviation and correlation matrix of a sample array (N, d)."""
    s = np.asarray(samples)
    return s.mean(0), s.std(0, ddof=1), np.corrcoef(s.T)


# ------------------------------------------------------------------ contours and plots
def hpd_levels(H, probs=(0.68, 0.95)):
    """Heights h_p of a (normalised-histogram) density H such that the region H >= h_p holds
    probability p.  Sort the cells from the highest down and accumulate their mass."""
    flat = np.sort(H.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return [flat[np.searchsorted(cum, p)] for p in probs]


def _smooth2d(H, s=1.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(H, s)


def contour2d(ax, x, y, color, bins=50, smooth=1.2, probs=(0.68, 0.95), ranges=None,
              filled=True, lw=1.2, ls="-", label=None, weights=None):
    """68/95 per cent contours of the 2-D marginal of samples (x, y), from a smoothed histogram."""
    rx = ranges[0] if ranges else (np.min(x), np.max(x))
    ry = ranges[1] if ranges else (np.min(y), np.max(y))
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[rx, ry], weights=weights)
    H = _smooth2d(H, smooth)
    lev = hpd_levels(H, probs)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    import matplotlib.colors as mc
    if filled:
        rgb = np.array(mc.to_rgb(color))
        cols = [tuple(1 - 0.25 * (1 - rgb)), tuple(1 - 0.55 * (1 - rgb))]
        ax.contourf(xc, yc, H.T, levels=[lev[1], lev[0], H.max() * 1.01], colors=cols, alpha=0.9)
    cs = ax.contour(xc, yc, H.T, levels=[lev[1], lev[0]], colors=[color], linewidths=lw,
                    linestyles=ls)
    if label is not None:
        ax.plot([], [], color=color, lw=lw, ls=ls, label=label)
    return cs


def hist1d(ax, x, color, bins=50, ranges=None, lw=1.3, ls="-", label=None, weights=None):
    """Normalised 1-D marginal histogram drawn as a line (peak scaled to 1)."""
    h, e = np.histogram(x, bins=bins, range=ranges, weights=weights, density=True)
    from scipy.ndimage import gaussian_filter1d
    h = gaussian_filter1d(h, 0.8)
    c = 0.5 * (e[1:] + e[:-1])
    ax.plot(c, h / h.max(), color=color, lw=lw, ls=ls, label=label)


def gauss_ellipse(ax, mean, cov, nsig=(1.52, 2.49), color="k", ls="--", lw=1.1, label=None):
    """Ellipses of a 2-D Gaussian enclosing 68 and 95 per cent (Delta chi^2 = 2.30, 6.18).

    nsig are the square roots of those Delta chi^2 values."""
    w, v = np.linalg.eigh(np.asarray(cov))
    t = np.linspace(0, 2 * np.pi, 200)
    circ = np.array([np.cos(t), np.sin(t)])
    for i, k in enumerate(nsig):
        e = v @ (np.sqrt(w)[:, None] * circ) * k
        ax.plot(mean[0] + e[0], mean[1] + e[1], color=color, ls=ls, lw=lw,
                label=label if i == 0 else None)


def gauss_1d(ax, mean, sd, xr, color="k", ls="--", lw=1.1):
    x = np.linspace(*xr, 300)
    ax.plot(x, np.exp(-0.5 * ((x - mean) / sd) ** 2), color=color, ls=ls, lw=lw)


def triangle(sample_sets, names, colors, labels=None, truths=None, ranges=None, gaussians=None,
             fig=None, bins=50, smooth=1.2, size=2.1, filled=None):
    """Hand-written triangle plot.

    sample_sets: list of arrays (N_i, d); names: axis labels; colors: one per set;
    gaussians: optional list of (mean, cov, color, label) drawn as dashed ellipses / curves;
    ranges: list of (lo, hi) per parameter (default: union of 0.1-99.9 percentiles).
    Diagonal: 1-D marginals (peak 1).  Lower triangle: 68/95 per cent 2-D contours.
    """
    import matplotlib.pyplot as plt
    d = len(names)
    if ranges is None:
        allx = np.vstack([s[:, :d] for s in sample_sets])
        lo, hi = np.percentile(allx, 0.1, axis=0), np.percentile(allx, 99.9, axis=0)
        pad = 0.08 * (hi - lo)
        ranges = list(zip(lo - pad, hi + pad))
    if fig is None:
        fig = plt.figure(figsize=(size * d, size * d))
    axs = np.empty((d, d), dtype=object)
    filled = filled if filled is not None else [True] * len(sample_sets)
    for i in range(d):
        for j in range(i + 1):
            ax = fig.add_subplot(d, d, i * d + j + 1)
            axs[i, j] = ax
            for k, s in enumerate(sample_sets):
                lab = labels[k] if (labels and i == 0 and j == 0) else None
                if i == j:
                    hist1d(ax, s[:, i], colors[k], bins=bins, ranges=ranges[i], label=lab)
                else:
                    contour2d(ax, s[:, j], s[:, i], colors[k], bins=bins, smooth=smooth,
                              ranges=[ranges[j], ranges[i]], filled=filled[k])
            if gaussians:
                for (m, C, col, glab) in gaussians:
                    C = np.asarray(C)
                    if i == j:
                        gauss_1d(ax, m[i], np.sqrt(C[i, i]), ranges[i], color=col)
                    else:
                        sub = C[np.ix_([j, i], [j, i])]
                        gauss_ellipse(ax, [m[j], m[i]], sub, color=col)
            if truths is not None:
                ax.axvline(truths[j], color="0.45", lw=0.6, ls=":")
                if i != j:
                    ax.axhline(truths[i], color="0.45", lw=0.6, ls=":")
            ax.set_xlim(ranges[j])
            if i != j:
                ax.set_ylim(ranges[i])
            else:
                ax.set_ylim(0, 1.12)
                ax.set_yticks([])
            if i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[j])
            if j > 0 or i == 0:
                if i != j or i == 0:
                    ax.set_yticklabels([]) if i != 0 else None
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            ax.tick_params(labelsize=7)
            for lab in ax.get_xticklabels():
                lab.set_rotation(35)
    return fig, axs
