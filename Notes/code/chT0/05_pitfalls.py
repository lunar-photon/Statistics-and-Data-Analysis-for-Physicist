"""Seven small traps, each shown by running it: what Python actually does, printed.

Question:  what really happens in the cases that surprise newcomers: comparing floats,
           integers inside float formulas, slices that share memory, ranges that stop one
           short, arrays of the wrong shape, a list as a default argument, and a missing seed?
Computes:  each trap in two or three lines, and the printed result.
Writes:    results/chT0/05_pitfalls.tex, results/chT0/05_pitfalls_out.txt (the printed lines)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from common import save_numbers, rng_for, NOTES

lines = []


def say(text):
    """Print a line and keep it, so the text can show exactly what was printed."""
    print(text)
    lines.append(text)


# 1. floating-point equality
say("# 1. floats")
say(f"0.1 + 0.2 == 0.3          -> {0.1 + 0.2 == 0.3}")
say(f"0.1 + 0.2 - 0.3           -> {0.1 + 0.2 - 0.3:.3e}")
say(f"np.isclose(0.1 + 0.2, 0.3) -> {np.isclose(0.1 + 0.2, 0.3)}")

# 2. integers stay integers inside an integer array
say("# 2. integer arrays")
a = np.array([1, 2, 3])
a[0] = 0.7
say(f"a = np.array([1, 2, 3]); a[0] = 0.7  -> a = {a}, a.dtype = {a.dtype}")
say(f"7 / 2 -> {7 / 2},   7 // 2 -> {7 // 2}")

# 3. a slice is a view: changing it changes the original
say("# 3. views and copies")
x = np.zeros(5)
y = x[1:3]
y[:] = 9.0
say(f"x = np.zeros(5); y = x[1:3]; y[:] = 9  -> x = {x}")
x = np.zeros(5)
y = x[1:3].copy()
y[:] = 9.0
say(f"same with y = x[1:3].copy()           -> x = {x}")

# 4. off by one: ranges and slices stop before the end
say("# 4. where ranges stop")
say(f"list(range(5))            -> {list(range(5))}")
say(f"len(np.arange(0, 1, 0.1)) -> {len(np.arange(0, 1, 0.1))}   (1.0 is not included)")
say(f"len(np.linspace(0, 1, 11)) -> {len(np.linspace(0, 1, 11))}  (both ends included)")
v = np.arange(10)
say(f"v = np.arange(10); v[2:5] -> {v[2:5]},  v[-1] -> {v[-1]}")

# 5. broadcasting: shapes (3,) and (3,1) make a 3 x 3 table, (3,) and (4,) do not fit
say("# 5. shapes")
r = np.array([1.0, 2.0, 3.0])
say(f"(r - r[:, None]).shape    -> {(r - r[:, None]).shape}")
try:
    r + np.ones(4)
except ValueError as err:
    say(f"r + np.ones(4)            -> ValueError: {err}")

# 6. a list as a default argument is created once and then shared by every call
say("# 6. mutable default argument")


def append_bad(value, store=[]):
    store.append(value)
    return store


def append_good(value, store=None):
    store = [] if store is None else store
    store.append(value)
    return store


append_bad(1)
say(f"append_bad(1); append_bad(2)   -> {append_bad(2)}")
append_good(1)
say(f"append_good(1); append_good(2) -> {append_good(2)}")

# 7. without a seed, every run gives different numbers
say("# 7. seeds")
u1 = np.random.default_rng().random()
u2 = np.random.default_rng().random()
g1 = rng_for("chT0", "05_pitfalls").random()
g2 = rng_for("chT0", "05_pitfalls").random()
say(f"default_rng() twice        -> equal: {u1 == u2}")
say(f"rng_for(same name) twice   -> equal: {g1 == g2}  ({g1:.6f})")

(NOTES / "results" / "chT0").mkdir(parents=True, exist_ok=True)
(NOTES / "results" / "chT0" / "05_pitfalls_out.txt").write_text("\n".join(lines) + "\n")
save_numbers("chT0", "05_pitfalls", {
    "TzFloatGap": 0.1 + 0.2 - 0.3,
    "TzSeedDraw": f"{g1:.6f}",
})
