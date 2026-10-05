"""Reproducible randomness: the same script name always gives the same random numbers.

Question: if we rerun a simulation next month, will every number quoted in the text come
out identical?  And are two differently named scripts statistically independent?
Computes: two generators built from the same (chapter, name) and one from a different name;
compares their first draws; estimates pi by throwing darts, as a tiny Monte Carlo whose
answer is then quoted in the text through a results macro.
Writes:   results/ch00/06_reproducible_seeds.tex  (no figure)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from common import save_numbers, rng_for

r1 = rng_for("ch00", "06_reproducible_seeds")     # seed = hash of "ch00/06_reproducible_seeds"
r2 = rng_for("ch00", "06_reproducible_seeds")     # same name -> same seed -> same stream
r3 = rng_for("ch00", "some_other_script")          # different name -> unrelated stream
a, b, c = r1.random(5), r2.random(5), r3.random(5)
print("same name     :", np.round(a, 4), np.round(b, 4))
print("different name:", np.round(c, 4))

# a one-line Monte Carlo: fraction of darts in the unit quarter-disc -> pi/4
N = 1_000_000
x, y = r1.random(N), r1.random(N)
frac = np.mean(x**2 + y**2 < 1.0)
pi_hat = 4 * frac
pi_err = 4 * np.sqrt(frac * (1 - frac) / N)       # binomial error of a fraction, times 4

save_numbers("ch00", "06_reproducible_seeds", {
    "SeedIdentical": "yes" if np.array_equal(a, b) else "no",
    "SeedFirstDraw": a[0], "SeedOtherFirstDraw": c[0],
    "PiHat": pi_hat, "PiErr": pi_err, "PiN": N,
})
