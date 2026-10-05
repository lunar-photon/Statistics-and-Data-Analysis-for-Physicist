"""Plain Python in the amount the book needs: values, containers, loops, conditions, functions.

Question:  what does each piece of core Python syntax used by the book's scripts actually do?
Computes:  one short block per idea (numbers, strings, f-strings, lists, tuples, dictionaries,
           for, if, while, functions, lambda, comprehensions, imports); each block prints what
           it made, as "expression = value".
Writes:    results/chT0/06_<topic>.txt (the printed lines of each block, shown in the text)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from common import NOTES

np.set_printoptions(legacy="1.25")       # print NumPy numbers plainly: 0.5, not np.float64(0.5)
OUT = NOTES / "results" / "chT0"
OUT.mkdir(parents=True, exist_ok=True)
printed = {}                             # topic -> the lines printed under it
current = None


def topic(name):
    """Start a new block: everything said from now on is filed under this name."""
    global current
    current = name
    printed[name] = []


def say(text):
    """Print a line and keep it for the text."""
    print(text)
    printed[current].append(text)


# [numbers] ---------------------------------------------------------------------------------------
topic("numbers")
n = 20                                   # an integer: a count
p = 0.2                                  # a float: a real number, about 16 significant digits
say(f"{n * p = }")
say(f"{n / 8 = }")                       # / always gives a float
say(f"{n // 8 = }   {n % 8 = }")         # whole-number division, and the remainder
say(f"{2 ** 10 = }   {1.5e-3 * 2 = }")   # powers, and scientific notation
say(f"{1_000_000 = }")                   # underscores only make long numbers readable
say(f"{type(n) = }   {type(p) = }")

# [strings] ---------------------------------------------------------------------------------------
topic("strings")
name = "binomial"                        # text goes between quotes, single or double
say(f"{len(name) = }   {name.upper() = }")
say(f"{'02_' + name = }")                # + joins two strings
say(f"{name[0] = }   {name[-1] = }   {name[:3] = }")

# [fstrings] --------------------------------------------------------------------------------------
topic("fstrings")
tau_hat, sig = 1.0234567, 0.1447
say(f"tau = {tau_hat:.3f} +/- {sig:.2f}")       # :.3f -> three digits after the point
say(f"p-value = {2.87e-7:.2e}")                  # :.2e -> scientific notation
say(f"N = {1000000:,}")                          # :,   -> thousands separators
say(f"$n={n},\\ p={p}$")                         # a plot label; a backslash is written twice

# [lists] -----------------------------------------------------------------------------------------
topic("lists")
ns = [1, 2, 5, 30]                       # a list: an ordered sequence that can grow
say(f"{ns[0] = }   {ns[-1] = }   {len(ns) = }")
say(f"{ns[1:3] = }")                     # a slice: positions 1 and 2, stopping before 3
ns.append(100)                           # add one item at the end
say(f"{ns = }")
results = []                             # the usual pattern: start empty, append in a loop
for k in ns:
    results.append(k * k)
say(f"{results = }")

# [tuples] ----------------------------------------------------------------------------------------
topic("tuples")
case = (20, 0.2)                         # a tuple: a fixed group of values
n, p = case                              # unpacking: one name for each entry
say(f"{n = }   {p = }")
cases = [(5, 0.5), (20, 0.1)]            # a list of tuples, as in the binomial script
for n, p in cases:                       # unpack each tuple as the loop meets it
    say(f"n = {n}, p = {p}, mean n p = {n * p}")

# [dicts] -----------------------------------------------------------------------------------------
topic("dicts")
fid = {"H0": 67.36, "ombh2": 0.02237, "ns": 0.9649}    # a dictionary: names -> values
say(f"{fid['H0'] = }")
fid["tau"] = 0.0544                      # add a new entry
say(f"{len(fid) = }   {'tau' in fid = }")
for key, value in fid.items():           # walk through (name, value) pairs
    say(f"{key:>6} = {value}")

# [loops] -----------------------------------------------------------------------------------------
topic("loops")
total = 0
for k in range(1, 6):                    # k = 1, 2, 3, 4, 5  (stops before 6)
    total += k                           # the same as total = total + k
say(f"{total = }")
for i, n in enumerate([5, 10, 20]):      # enumerate: the position and the item
    say(f"i = {i}, n = {n}")
for n, p in zip([5, 10], [0.5, 0.1]):    # zip: walk two lists side by side
    say(f"n = {n}, p = {p}")

# [ifs] -------------------------------------------------------------------------------------------
topic("ifs")
for z in [0.5, 3.4, 5.3]:
    if z >= 5:
        verdict = "discovery"
    elif z >= 3:
        verdict = "evidence"
    else:
        verdict = "nothing yet"
    say(f"z = {z}: {verdict}")
say(f"{(2 < 3) and (3 < 1) = }   {not (3 < 1) = }")

# [while] -----------------------------------------------------------------------------------------
topic("while")
n = 0
while 0.5 ** n > 1e-3:                   # keep halving while still above one in a thousand
    n += 1
say(f"{n = }   {0.5 ** n = }")

# [functions] -------------------------------------------------------------------------------------
topic("functions")


def gaussian(x, mu=0.0, sigma=1.0):
    """Normal density at x; mu and sigma have default values."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def mean_and_error(x):
    """Two answers at once: the sample mean and its standard error."""
    return np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x))


say(f"{gaussian(0.0) = :.4f}")                   # defaults used: mu = 0, sigma = 1
say(f"{gaussian(1.0, sigma=2.0) = :.4f}")        # sigma given by name, mu left at 0
m, e = mean_and_error(np.array([9.80, 9.71, 9.86, 9.83]))
say(f"m = {m:.3f} +/- {e:.3f}")
square = lambda t: t ** 2                        # a one-line function without a name
say(f"{square(3) = }")

# [comprehensions] --------------------------------------------------------------------------------
topic("comprehensions")
squares = [k ** 2 for k in range(5)]             # a list built by a loop in one line
evens = [k for k in range(10) if k % 2 == 0]     # ... keeping only some items
lengths = {s: len(s) for s in ["pi", "K", "p"]}  # the same for a dictionary
say(f"{squares = }")
say(f"{evens = }")
say(f"{lengths = }")
say(f"{sum(k ** 2 for k in range(5)) = }")      # inside a function call, no brackets needed

# [imports] ---------------------------------------------------------------------------------------
topic("imports")
import math                                      # a whole module: use as math.something
from math import comb                            # one name from a module
say(f"{math.pi = }   {comb(10, 3) = }")
here = pathlib.Path(__file__).resolve()          # the full path of this script
say(f"{here.name = }")
say(f"{here.parents[0].name = }   {here.parents[1].name = }")

# write every block's printed lines to its own file
for name, lines in printed.items():
    (OUT / f"06_{name}.txt").write_text("\n".join(lines) + "\n")
