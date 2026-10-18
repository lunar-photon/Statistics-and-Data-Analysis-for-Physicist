"""lib_t5a.py -- small helpers shared by the scripts 01-06 of code/chT5 (location estimators).

ml_location(x, logf):   maximum-likelihood location of each row of x for a noise log-density logf,
                        by a grid search around the median refined with a parabola.
huber_location(x, k):   Huber M-estimate of each row of x, found by bisection on the monotone
                        estimating equation sum_i psi_k((x_i - mu)/s) = 0 (s = MAD scale).
trimmed_mean(x, alpha): mean of each row after dropping the alpha fraction at each end.
"""
import numpy as np


def ml_location(x, logf, half=0.8, ngrid=321, chunk=500):
    """Grid search over mu in median +- half, then the vertex of the parabola through the best grid
    point and its two neighbours (the log-likelihood is smooth near its peak)."""
    x = np.atleast_2d(x)
    med = np.median(x, axis=1)
    out = np.empty(len(x))
    offs = np.linspace(-half, half, ngrid)
    h = offs[1] - offs[0]
    for lo in range(0, len(x), chunk):
        xs, m0 = x[lo:lo + chunk], med[lo:lo + chunk]
        mu = m0[:, None] + offs[None, :]                         # candidate values of mu
        ll = logf(xs[:, None, :] - mu[:, :, None]).sum(axis=2)    # log-likelihood on the grid
        j = np.clip(ll.argmax(axis=1), 1, ngrid - 2)
        r = np.arange(len(xs))
        a, b, c = ll[r, j - 1], ll[r, j], ll[r, j + 1]
        den = a - 2 * b + c
        shift = np.where(den < 0, 0.5 * (a - c) / np.where(den < 0, den, -1.0), 0.0)
        out[lo:lo + chunk] = mu[r, j] + h * np.clip(shift, -1, 1)
    return out


def mad_scale(x):
    """Median absolute deviation, rescaled so that it estimates sigma for Gaussian data."""
    med = np.median(x, axis=-1, keepdims=True)
    return np.median(np.abs(x - med), axis=-1) / 0.6744897501960817


def huber_location(x, k=1.345, scale=None, iters=60):
    """Root of sum_i psi_k((x_i-mu)/s) = 0 by bisection. psi_k(u) = clip(u, -k, k) is
    non-decreasing in u, so the sum is non-increasing in mu and the root is bracketed by the
    smallest and largest reading. 60 halvings shrink the bracket by 2^-60."""
    x = np.atleast_2d(x)
    s = mad_scale(x) if scale is None else np.broadcast_to(scale, (len(x),))
    lo, hi = x.min(axis=1), x.max(axis=1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = np.clip((x - mid[:, None]) / s[:, None], -k, k).sum(axis=1)
        lo = np.where(g > 0, mid, lo)        # sum still positive: the root lies above mid
        hi = np.where(g > 0, hi, mid)
    return 0.5 * (lo + hi)


def trimmed_mean(x, alpha=0.1):
    x = np.sort(np.atleast_2d(x), axis=1)
    g = int(np.floor(alpha * x.shape[1]))
    return x[:, g:x.shape[1] - g].mean(axis=1)
