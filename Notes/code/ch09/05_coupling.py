"""Coupling: two walkers who forget their different starts the moment they meet.

Question: walker A starts on a rainy day, walker B starts from the stationary distribution pi of
the three-state weather chain.  Once they stand in the same state we let them move together.
How does the probability that they have NOT yet met compare with the total-variation distance
between A's distribution and pi?  (Coupling inequality: TV(t) <= P(not met by t).)
Two couplings: (i) independent moves until they meet; (ii) the Doeblin coupling, in which with
probability delta both walkers draw their next state from the SAME shared part of the rows.
Writes figures/ch09/coupling.pdf and results/ch09/05_coupling.tex.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, rng_for, SERIES
from lib_markov import stationary, evolve, tv, doeblin, step_many

P = np.array([[0.6, 0.2, 0.2], [0.3, 0.7, 0.0], [0.6, 0.0, 0.4]])
pi = stationary(P)
delta = doeblin(P)
nu = P.min(axis=0) / delta                      # the shared part of every row, as a pmf
R = (P - delta * nu) / (1 - delta)              # what is left of each row, renormalised
rng = rng_for("ch09", "05_coupling")
T, W = 25, 200_000


def met_fraction(doeblin_coupling):
    a = np.full(W, 2)                                   # A: certain rain
    b = rng.choice(3, size=W, p=pi)                     # B: already stationary
    met = a == b
    not_met = [1.0 - met.mean()]
    for _ in range(T):
        if doeblin_coupling:
            shared = rng.random(W) < delta              # both draw from nu: same state, they meet
            common = rng.choice(3, size=W, p=nu)
            a_new, b_new = step_many(a, R, rng), step_many(b, R, rng)
            a_new = np.where(shared, common, a_new)
            b_new = np.where(shared, common, b_new)
        else:
            a_new, b_new = step_many(a, P, rng), step_many(b, P, rng)
        b_new = np.where(met, a_new, b_new)             # once met, they move together
        a, b = a_new, b_new
        met |= a == b
        not_met.append(1.0 - met.mean())
    return np.array(not_met)


ind = met_fraction(False)
doe = met_fraction(True)
exact = tv(evolve(np.array([0, 0, 1.0]), P, T), pi)
t = np.arange(T + 1)

setup(6.0, 3.2)
fig, ax = plt.subplots()
ax.semilogy(t, ind, "-", color=SERIES[0], label="P(not met), independent moves")
ax.semilogy(t, doe, "-", color=SERIES[1], label="P(not met), Doeblin coupling")
ax.semilogy(t, (1 - delta) ** t * (1 - pi[2]), ":", color="k", lw=1.4, label=r"$(1-\delta)^t\,(1-\pi_{\rm R})$")
ax.semilogy(t, exact, "--", color="k", lw=1.4, label=r"exact $\|p_t-\pi\|_{\rm TV}$")
ax.set_ylim(1e-6, 1.5)
ax.set_xlabel("day $t$")
ax.set_ylabel("probability")
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, "ch09", "coupling")

save_numbers("ch09", "05_coupling", {
    "NineACoupIndTen": round(ind[10], 4), "NineACoupDoeTen": round(doe[10], 4),
    "NineACoupTVTen": round(exact[10], 4), "NineACoupBoundTen": round((1 - delta) ** 10 * (1 - pi[2]), 4),
    "NineACoupWalkers": f"{W:,}".replace(",", r"\,"),
})
print("t=10: independent", ind[10], "doeblin", doe[10], "TV", exact[10], "bound", (1 - delta) ** 10 * (1 - pi[2]))
print("nu", nu, "R", R)
