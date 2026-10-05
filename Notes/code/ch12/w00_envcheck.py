"""Which optional packages does the environment running the chapter-11 scripts provide?

Question: the persistence code needs either gudhi or numba.  Prints which of gudhi, numba,
ripser, persim and scikit-image import.  Writes nothing.
"""
import importlib
for m in ("gudhi", "numba", "ripser", "persim", "skimage", "scipy", "healpy"):
    try:
        mod = importlib.import_module(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
