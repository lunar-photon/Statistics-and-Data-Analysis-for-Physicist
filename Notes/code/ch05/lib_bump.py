"""lib_bump.py -- a straight line, or a straight line plus a Gaussian bump?

Shared by the 5c scripts (22, 24, 26, 27, 28).  Nothing here is random except
make_data(), which takes a numpy Generator.

Model M0 (line):  y_i = a + b x_i + noise
Model M1 (bump):  y_i = a + b x_i + A exp(-(x_i - mu)^2 / (2 W^2)) + noise
noise ~ N(0, SIGMA^2) independent, W known.

Priors (uniform boxes, part of each model's definition):
  a in [0, 20], b in [-2, 2], A in [0, A_MAX], mu in [0, 10].

The line parameters (a, b) enter linearly, so for fixed (A, mu) their integral
is a Gaussian integral done exactly.  What is left is a 2-D integral over
(A, mu), done on a fine grid.  That gives the evidence to high accuracy and is
the reference for every other method.
"""
import numpy as np

N_DATA, SIGMA, W = 40, 1.0, 0.5
X = np.linspace(0.0, 10.0, N_DATA)
A_RANGE, B_RANGE = (0.0, 20.0), (-2.0, 2.0)
A_MAX, MU_RANGE = 10.0, (0.0, 10.0)
V_AB = (A_RANGE[1] - A_RANGE[0]) * (B_RANGE[1] - B_RANGE[0])      # prior area of (a, b)
L_MU = MU_RANGE[1] - MU_RANGE[0]

TRUE = dict(a=5.0, b=0.3, A=2.5, mu=6.2)

DESIGN = np.column_stack([np.ones_like(X), X])                    # columns 1 and x
_G = DESIGN.T @ DESIGN / SIGMA**2                                  # Fisher matrix of (a, b)
PPERP = np.eye(N_DATA) - DESIGN @ np.linalg.solve(DESIGN.T @ DESIGN, DESIGN.T)


def bump(x, mu, w=W):
    return np.exp(-0.5 * ((x - mu) / w) ** 2)


def model(theta, x=X):
    a, b, A, mu = theta
    return a + b * x + A * bump(x, mu)


def make_data(rng, with_bump=True):
    t = TRUE
    mean = t["a"] + t["b"] * X + (t["A"] * bump(X, t["mu"]) if with_bump else 0.0)
    return mean + SIGMA * rng.standard_normal(N_DATA)


def chi2(theta, y):
    return np.sum(((y - model(theta)) / SIGMA) ** 2)


def ln_norm():
    """ln of the Gaussian normalisation (2 pi sigma^2)^(-N/2)."""
    return -0.5 * N_DATA * np.log(2 * np.pi * SIGMA**2)


def ln_ab_integral():
    """ln of  integral exp(-(chi2 - chi2_min)/2) da db / V_ab  (exact Gaussian, box assumed wide)."""
    return np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(_G)) - np.log(V_AB)


def chi2min_line(y):
    r = PPERP @ y
    return r @ r / SIGMA**2


def ln_evidence_line(y):
    return ln_norm() - 0.5 * chi2min_line(y) + ln_ab_integral()


def dchi2_grid(y, A, mu):
    """chi2_min over (a, b) at each (A, mu), minus the line's chi2_min.  Shape (len(A), len(mu))."""
    ry = PPERP @ y
    U = PPERP @ bump(X[:, None], mu[None, :])                     # projected bump shapes, (N, n_mu)
    uy = ry @ U / SIGMA**2                                        # (n_mu,)
    uu = np.sum(U * U, axis=0) / SIGMA**2
    return -2 * A[:, None] * uy[None, :] + A[:, None] ** 2 * uu[None, :]


def ln_bayes_factor(y, a_max=A_MAX, mu_fixed=None, nA=2001, nmu=2001, return_grid=False):
    """ln B_10 = ln Z(bump) - ln Z(line), priors A ~ U[0, a_max], mu ~ U[MU_RANGE] (or mu fixed)."""
    A = np.linspace(0.0, a_max, nA)
    mu = np.array([mu_fixed]) if mu_fixed is not None else np.linspace(*MU_RANGE, nmu)
    dc = dchi2_grid(y, A, mu)
    f = np.exp(-0.5 * dc)                                         # likelihood ratio to the best line
    inner = f[:, 0] if mu_fixed is not None else np.trapezoid(f, mu, axis=1) / L_MU
    lnB = np.log(np.trapezoid(inner, A) / a_max)
    if return_grid:
        return lnB, A, mu, dc
    return lnB


def ln_evidence_bump(y, **kw):
    return ln_evidence_line(y) + ln_bayes_factor(y, **kw)


def ln_post_unnorm(theta, y, a_max=A_MAX):
    """ln [likelihood x prior] for the bump model, -inf outside the prior box."""
    a, b, A, mu = theta
    if not (A_RANGE[0] <= a <= A_RANGE[1] and B_RANGE[0] <= b <= B_RANGE[1]
            and 0.0 <= A <= a_max and MU_RANGE[0] <= mu <= MU_RANGE[1]):
        return -np.inf
    return ln_norm() - 0.5 * chi2(theta, y) - np.log(V_AB * a_max * L_MU)
