"""Small finite-state Markov-chain toolkit used by the scripts of part 9a.

A chain on states 0..K-1 is given by its transition matrix P (rows sum to 1,
P[i, j] = probability of jumping from i to j).  Distributions are ROW vectors,
and one step of the chain maps p -> p @ P.
"""
from __future__ import annotations

import numpy as np


def check_stochastic(P):
    P = np.asarray(P, dtype=float)
    assert np.all(P >= 0) and np.allclose(P.sum(axis=1), 1.0), "rows must be pmfs"
    return P


def stationary(P):
    """Left eigenvector of P with eigenvalue 1, normalised to sum to one."""
    P = check_stochastic(P)
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    return pi / pi.sum()


def evolve(p0, P, t):
    """Distributions p_0, p_1, ..., p_t (array of shape (t+1, K)): p_{s+1} = p_s P."""
    out = [np.asarray(p0, dtype=float)]
    for _ in range(t):
        out.append(out[-1] @ P)
    return np.array(out)


def tv(p, q):
    """Total-variation distance 1/2 sum_i |p_i - q_i| (works along the last axis)."""
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=-1)


def doeblin(P):
    """Doeblin coefficient delta = sum_j min_i P[i, j]: the probability mass every row shares."""
    return float(np.asarray(P).min(axis=0).sum())


def step_many(states, P, rng):
    """Move many independent walkers one step: walker in state i draws its next state from row i."""
    cum = np.cumsum(P, axis=1)
    u = rng.random(states.shape[0])
    return np.minimum((u[:, None] > cum[states]).sum(axis=1), P.shape[0] - 1)


def walk(P, x0, n, rng):
    """One trajectory X_0 = x0, X_1, ..., X_n of the chain."""
    cum = np.cumsum(P, axis=1)
    x = np.empty(n + 1, dtype=int)
    x[0] = x0
    u = rng.random(n)
    for t in range(n):
        x[t + 1] = np.searchsorted(cum[x[t]], u[t], side="right")
    return np.minimum(x, P.shape[0] - 1)
