"""The spectral test: how far apart are the hyperplanes that hold an LCG's points?

Question: the t-tuples (u_n, ..., u_{n+t-1}) of an LCG lie on families of
parallel hyperplanes.  For the best such family (the widest spacing), how wide
is the gap d_t, and how does it compare with the best any lattice with m points
per unit volume could do?

Computes: the dual lattice L*_t = {h in Z^t : h_1 + a h_2 + ... + a^{t-1} h_t = 0 mod m},
a reduced basis of it (LLL, exact rational arithmetic), its shortest vector h
(search over small combinations of the reduced basis), nu_t = |h| and
d_t = 1/nu_t, for t = 2..6 and several classic multipliers.  Also Marsaglia's
bound (t! m)^(1/t) on the number of hyperplanes, and the ideal spacing
gamma_t^(-1/2) m^(-1/t) from the Hermite constants.
Writes: figures/ch06/spectral.pdf, results/ch06/05_spectral_test.tex
"""
import sys, pathlib, itertools, math
from fractions import Fraction
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import setup, savefig, save_numbers, SERIES

import numpy as np
import matplotlib.pyplot as plt


def dual_basis(a, m, t):
    B = [[m] + [0] * (t - 1)]
    for j in range(1, t):
        v = [0] * t
        v[0] = (-pow(a, j, m)) % m
        v[j] = 1
        B.append(v)
    return B


def lll(B, delta=Fraction(3, 4)):
    """Textbook LLL reduction with exact fractions (fine for t <= 6)."""
    B = [list(map(int, b)) for b in B]
    n = len(B)
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))

    def gso(B):
        Bs, mu = [], [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            v = [Fraction(x) for x in B[i]]
            for j in range(i):
                mu[i][j] = Fraction(dot(B[i], Bs[j])) / dot(Bs[j], Bs[j])
                v = [x - mu[i][j] * y for x, y in zip(v, Bs[j])]
            Bs.append(v)
        return Bs, mu

    Bs, mu = gso(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q:
                B[k] = [x - q * y for x, y in zip(B[k], B[j])]
                Bs, mu = gso(B)
        if dot(Bs[k], Bs[k]) >= (delta - mu[k][k - 1] ** 2) * dot(Bs[k - 1], Bs[k - 1]):
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Bs, mu = gso(B)
            k = max(k - 1, 1)
    return B


def shortest(a, m, t, span=3):
    R = lll(dual_basis(a, m, t))
    best = None
    for c in itertools.product(range(-span, span + 1), repeat=t):
        if not any(c):
            continue
        h = [sum(ci * R[i][j] for i, ci in enumerate(c)) for j in range(t)]
        n2 = sum(x * x for x in h)
        if best is None or n2 < best[0]:
            best = (n2, h)
    n2, h = best
    assert sum(hj * pow(a, j, m) for j, hj in enumerate(h)) % m == 0
    if h[next(i for i, x in enumerate(h) if x)] < 0:
        h = [-x for x in h]
    return math.sqrt(n2), h


GENS = {
    "RANDU": (65539, 2**31),
    "MINSTD": (16807, 2**31 - 1),
    "L'Ecuyer 40692": (40692, 2147483399),
    "Lambert 1229": (1229, 2048),
}
HERMITE = {2: (4 / 3) ** 0.5, 3: 2 ** (1 / 3), 4: 2 ** 0.5, 5: 8 ** (1 / 5), 6: (64 / 3) ** (1 / 6)}
ts = [2, 3, 4, 5, 6]
res = {}
for name, (a, m) in GENS.items():
    res[name] = [shortest(a, m, t) for t in ts]
    for t, (nu, h) in zip(ts, res[name]):
        print(f"{name:16s} t={t} nu={nu:12.2f} d={1/nu:.3e} h={h} planes<={sum(map(abs, h))}")

m31 = 2**31
ideal = [1 / (HERMITE[t] ** 0.5 * m31 ** (1 / t)) for t in ts]
bound32 = {t: (math.factorial(t) * 2**32) ** (1 / t) for t in (3, 10)}
print("Marsaglia bound, m=2^32:", bound32)

setup(5.6, 3.2)
fig, ax = plt.subplots()
for j, name in enumerate(["RANDU", "MINSTD", "L'Ecuyer 40692"]):
    ax.semilogy(ts, [1 / nu for nu, _ in res[name]], "o-", color=SERIES[j], label=name)
ax.semilogy(ts, ideal, "k--", lw=1.2, label=r"best possible for $m=2^{31}$")
ax.set_xticks(ts)
ax.set_xlabel("dimension $t$")
ax.set_ylabel(r"widest gap between hyperplanes $d_t$")
ax.legend()
savefig(fig, "ch06", "spectral")

r3 = res["RANDU"][1]
save_numbers("ch06", "05_spectral_test", {
    "SixASpecRanduThreeNu": f"{r3[0]:.2f}", "SixASpecRanduThreeD": f"{1/r3[0]:.3f}",
    "SixASpecRanduTwoD": 1 / res["RANDU"][0][0],
    "SixASpecMinstdTwoD": 1 / res["MINSTD"][0][0], "SixASpecMinstdThreeD": 1 / res["MINSTD"][1][0],
    "SixASpecRanduThreeIdeal": ideal[1],
    "SixASpecLecTwoD": 1 / res["L'Ecuyer 40692"][0][0],
    "SixASpecLamTwoNu": f"{res['Lambert 1229'][0][0]:.2f}",
    "SixASpecLamTwoH": f"({res['Lambert 1229'][0][1][0]},{res['Lambert 1229'][0][1][1]})",
    "SixASpecMinstdSixD": 1 / res["MINSTD"][4][0],
    "SixAMarsBoundThree": f"{bound32[3]:.0f}", "SixAMarsBoundTen": f"{bound32[10]:.0f}",
})
