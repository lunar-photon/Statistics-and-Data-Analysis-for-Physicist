"""29_counting_evidence.py -- background only, or background plus signal?  A counting experiment.

Question: a counter with a known background rate b = 2 (expected counts in the
run) records n = 5 events.  Model M0: the counts are Poisson(b).  Model M1: they
are Poisson(b + s) with an unknown signal s, uniform on [0, s_max].  How
probable was n = 5 under each model, and which model do the data favour?

Computes
  * Z0 = Poisson(n | b),
  * Z1 = (1/s_max) * integral_0^s_max Poisson(n | b + s) ds, exactly through
    integral_a^inf e^{-u} u^n / n! du = P(Poisson(a) <= n), and by quadrature,
  * the Bayes factor B10, the best-fit likelihood ratio and the Occam factor,
  * the same for s_max = 100.

Writes: results/ch05/29_counting_evidence.tex
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import save_numbers

import numpy as np
from scipy import stats, integrate

b, n = 2.0, 5


def Z1(s_max):
    exact = (stats.poisson.cdf(n, b) - stats.poisson.cdf(n, b + s_max)) / s_max
    quad = integrate.quad(lambda s: stats.poisson.pmf(n, b + s), 0, s_max)[0] / s_max
    return exact, quad


Z0 = stats.poisson.pmf(n, b)
Z1a, Z1a_q = Z1(10.0)
Z1b, _ = Z1(100.0)
s_hat = n - b                                    # maximum-likelihood signal
L_hat = stats.poisson.pmf(n, b + s_hat)
save_numbers("ch05", "29_counting_evidence", {
    "FiveCCntZzero": Z0, "FiveCCntZone": Z1a, "FiveCCntZoneQuad": Z1a_q,
    "FiveCCntB": Z1a / Z0, "FiveCCntLhat": L_hat, "FiveCCntLR": L_hat / Z0,
    "FiveCCntOcc": (Z1a / Z0) / (L_hat / Z0),
    "FiveCCntCdfb": stats.poisson.cdf(n, b), "FiveCCntCdfbs": stats.poisson.cdf(n, b + 10.0),
    "FiveCCntZoneWide": Z1b, "FiveCCntBWide": Z1b / Z0,
    "FiveCCntPost": (Z1a / Z0) / (1 + Z1a / Z0),
})
print(Z0, Z1a, Z1a_q, Z1a / Z0, L_hat / Z0, Z1b / Z0)
