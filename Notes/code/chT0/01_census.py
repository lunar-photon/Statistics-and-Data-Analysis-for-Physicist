"""How much Python does the book actually use?  A census of every script, read but never run.

Question:  which pieces of Python syntax, which modules and which functions do the book's
           scripts use, and how many distinct function names cover 50, 80, 90 and 99 per
           cent of all the calls they make?
Computes:  parses every .py file under code/ (except this chapter's own folder) with the
           ast module, which turns source text into a tree of statements without executing
           anything; counts (a) syntax features, as the number of files that use each one,
           (b) imported modules, (c) every function or method call, named the way a reader
           sees it (np.linspace, plt.subplots, ax.plot, rng.normal, len, ...); (d) how often
           the scripts check themselves (assert, np.isclose, np.allclose) and where their
           random numbers come from (rng_for, an unseeded default_rng).
Writes:    figures/chT0/01_census_coverage.pdf, figures/chT0/01_census_syntax.pdf,
           figures/chT0/01_census_topcalls.pdf, results/chT0/01_census.tex,
           results/chT0/01_census_checks.tex, results/chT0/01_census_top.tex,
           data/chT0/census.json (the full tables, for reading)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ast
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from common import setup, savefig, save_numbers, SERIES, NOTES, DATA

CODE = NOTES / "code"
files = sorted(p for p in CODE.rglob("*.py")
               if "chT0" not in p.parts and "__pycache__" not in p.parts)

# the short names the book always uses for the big libraries
SHORT = {"numpy": "np", "matplotlib.pyplot": "plt", "healpy": "hp", "scipy": "scipy",
         "camb": "camb", "math": "math"}
BOOK_MODULES = {p.stem for p in files}          # common, camb_fiducial, lib_fields, ...


def dotted(node):
    """Turn the function part of a call into text: np.linalg.solve, ax.plot, len."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = dotted(node.value)
        return f"{inner}.{node.attr}" if inner else f"obj.{node.attr}"
    return None                                  # e.g. axes[0].plot -> an object, handled above


# ---------------------------------------------------------------------------------------------
# pass 1: the names of every function the book defines itself (its own vocabulary)
book_defs = set()               # functions and classes defined anywhere in the book
book_methods = set()            # functions defined inside the book's own classes
trees = {}
n_lines = 0
for f in files:
    src = f.read_text()
    n_lines += src.count("\n")
    tree = ast.parse(src)
    trees[f] = tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            book_defs.add(node.name)
        if isinstance(node, ast.ClassDef):
            book_methods.update(m.name for m in node.body if isinstance(m, ast.FunctionDef))

# ---------------------------------------------------------------------------------------------
# pass 2: syntax features (per file), imports (per file) and calls (every call counted)
FEATURES = {
    "assignment  x = ...":             lambda n: isinstance(n, ast.Assign),
    "function call  f(x)":             lambda n: isinstance(n, ast.Call),
    "import":                          lambda n: isinstance(n, (ast.Import, ast.ImportFrom)),
    "keyword argument  f(x, n=3)":     lambda n: isinstance(n, ast.keyword) and n.arg is not None,
    "indexing  x[i]":                  lambda n: isinstance(n, ast.Subscript) and not isinstance(n.slice, ast.Slice)
                                                 and not (isinstance(n.slice, ast.Tuple) and any(isinstance(e, ast.Slice) for e in n.slice.elts)),
    "slicing  x[a:b]":                 lambda n: isinstance(n, ast.Subscript) and (isinstance(n.slice, ast.Slice)
                                                 or (isinstance(n.slice, ast.Tuple) and any(isinstance(e, ast.Slice) for e in n.slice.elts))),
    "for loop":                        lambda n: isinstance(n, ast.For),
    "if / else":                       lambda n: isinstance(n, ast.If),
    "def (own function)":              lambda n: isinstance(n, ast.FunctionDef),
    "return":                          lambda n: isinstance(n, ast.Return),
    "f-string  f\"{x}\"":              lambda n: isinstance(n, ast.JoinedStr),
    "list  [a, b]":                    lambda n: isinstance(n, ast.List),
    "tuple unpacking  a, b = ...":     lambda n: isinstance(n, (ast.Assign, ast.For)) and isinstance(
                                                 n.targets[0] if isinstance(n, ast.Assign) else n.target, ast.Tuple),
    "dict  {key: value}":              lambda n: isinstance(n, ast.Dict),
    "comparison giving True/False":    lambda n: isinstance(n, ast.Compare),
    "comprehension  [f(x) for x in]":  lambda n: isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)),
    "x += ...":                        lambda n: isinstance(n, ast.AugAssign),
    "inline if  a if c else b":        lambda n: isinstance(n, ast.IfExp),
    "matrix product  A @ B":           lambda n: isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult),
    "while loop":                      lambda n: isinstance(n, ast.While),
    "lambda":                          lambda n: isinstance(n, ast.Lambda),
    "with ... :":                      lambda n: isinstance(n, ast.With),
    "try / except":                    lambda n: isinstance(n, ast.Try),
    "class":                           lambda n: isinstance(n, ast.ClassDef),
    "decorator  @...":                 lambda n: isinstance(n, (ast.FunctionDef, ast.ClassDef)) and len(n.decorator_list) > 0,
    "assert":                          lambda n: isinstance(n, ast.Assert),
    "*args / **kwargs":                lambda n: isinstance(n, ast.arguments) and (n.vararg is not None or n.kwarg is not None),
    "yield (generator)":               lambda n: isinstance(n, (ast.Yield, ast.YieldFrom)),
    "global":                          lambda n: isinstance(n, ast.Global),
}
feat_files = Counter()          # in how many files does each feature appear
feat_uses = Counter()           # how many times in total
mod_files = Counter()           # in how many files is each module imported
calls = Counter()               # every call, by the name a reader sees
own_calls = Counter()           # calls of functions the book defines itself
first_file = {}                 # first file (in folder order) that makes each call
file_calls = {}                 # file -> the set of distinct library names it calls
file_own = {}                   # file -> the set of the book's own functions it calls
bare_rng = [0]                  # np.random.default_rng() called with no seed

for f, tree in trees.items():
    alias = {}                  # local name -> canonical short name, e.g. stats -> scipy.stats
    seen = set()
    mods = set()
    file_calls[f] = set()
    file_own[f] = set()
    for node in ast.walk(tree):
        for name, test in FEATURES.items():
            if test(node):
                feat_uses[name] += 1
                seen.add(name)
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                mods.add(a.name if a.name == "matplotlib.pyplot" else top)
                if a.asname:                       # import numpy as np
                    alias[a.asname] = SHORT.get(a.name, a.name)
                else:                              # import scipy.special -> scipy.special.gammaln
                    alias[top] = SHORT.get(top, top)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            mods.add(node.module if node.module.startswith("matplotlib") else top)
            for a in node.names:
                full = f"{node.module}.{a.name}"
                if top in BOOK_MODULES or node.module in BOOK_MODULES:
                    alias[a.asname or a.name] = f"{node.module}.{a.name}"
                else:
                    alias[a.asname or a.name] = SHORT.get(full, full)
    for m in mods:
        mod_files[m] += 1
    for name in seen:
        feat_files[name] += 1
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        if name is None:
            name = "obj." + node.func.attr if isinstance(node.func, ast.Attribute) else "(other)"
        head, _, rest = name.partition(".")
        if head in alias:                          # a module or an imported function
            name = alias[head] + ("." + rest if rest else "")
        elif rest:                                 # a method on some object: ax.plot, x.sum
            name = "obj." + rest.split(".")[-1]
        last = name.split(".")[-1]
        own = (name.split(".")[0] in BOOK_MODULES
               or (head not in alias and not rest and last in book_defs)
               or (name.startswith("obj.") and last in book_methods))
        if own:
            own_calls[name.split(".")[-1]] += 1
            file_own[f].add(name.split(".")[-1])
            continue
        calls[name] += 1
        file_calls[f].add(name)
        if name == "np.random.default_rng" and not node.args and not node.keywords:
            bare_rng[0] += 1                       # a generator with no seed: different every run
        first_file.setdefault(name, str(f.relative_to(NOTES)))

# ---------------------------------------------------------------------------------------------
# coverage: how many distinct names cover a given fraction of all library calls
ranked = calls.most_common()
counts = np.array([c for _, c in ranked])
cum = np.cumsum(counts) / counts.sum()
def names_for(frac):
    return int(np.searchsorted(cum, frac) + 1)   # first rank at which cum >= frac
N50, N80, N90, N99 = (names_for(q) for q in (0.5, 0.8, 0.9, 0.99))
n_files = len(files)
rare = counts <= 3                               # names called at most three times in the whole book
summary = [f"{n_files} files, {n_lines} lines, {counts.sum()} library calls, {len(ranked)} distinct names",
           f"names covering 50/80/90/99 %: {N50} {N80} {N90} {N99}",
           f"names used at most 3 times: {rare.sum()}, carrying {100 * counts[rare].sum() / counts.sum():.1f}% of calls"]
for line in summary:
    print(line)

# --- figure 1: the coverage curve ---------------------------------------------------------------
setup(6.0, 3.6)
fig, ax = plt.subplots()
rank = np.arange(1, len(cum) + 1)
ax.semilogx(rank, 100 * cum, color=SERIES[0])
for q, n, c in zip((50, 80, 90, 99), (N50, N80, N90, N99), SERIES[1:5]):
    ax.plot([n, n], [0, q], color=c, lw=1.0, ls=":")
    ax.plot([1, n], [q, q], color=c, lw=1.0, ls=":")
    ax.plot(n, q, "o", color=c)
    ax.annotate(f"{n} names: {q}%", (n, q), xytext=(-6, 5), textcoords="offset points", ha="right", fontsize=8, color=c)
ax.set_xlim(1, len(cum) * 1.1)
ax.set_ylim(0, 105)
ax.set_xlabel("number of distinct function names, most used first")
ax.set_ylabel("share of all calls in the book (%)")
savefig(fig, "chT0", "01_census_coverage")

# --- figure 2: syntax features, as the share of files using each --------------------------------
order = sorted(FEATURES, key=lambda k: feat_files[k])
share = np.array([100 * feat_files[k] / n_files for k in order])
setup(6.0, 5.6)
fig, ax = plt.subplots()
colors = [SERIES[0] if s >= 20 else SERIES[1] for s in share]
ax.barh(range(len(order)), share, color=colors)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8, family="monospace")
for i, s in enumerate(share):
    ax.text(s + 1, i, f"{s:.0f}%", va="center", fontsize=7)
ax.axvline(20, color="0.5", lw=0.8, ls="--")
ax.set_xlim(0, 112)
ax.set_xlabel("share of the book's scripts that use it (%)")
ax.grid(axis="y", visible=False)
savefig(fig, "chT0", "01_census_syntax")

# --- figure 3: the forty most used names, coloured by library --------------------------------
def family(name):
    if name.startswith("np."):
        return 0
    if name.startswith("plt.") or name.startswith("obj.set_") or name in (
            "obj.plot", "obj.hist", "obj.legend", "obj.text", "obj.axhline", "obj.axvline",
            "obj.fill_between", "obj.errorbar", "obj.imshow", "obj.loglog", "obj.semilogy",
            "obj.semilogx", "obj.subplots", "obj.tight_layout", "obj.colorbar", "obj.scatter",
            "obj.bar", "obj.annotate", "obj.step", "obj.contour", "obj.grid", "obj.axis"):
        return 1
    if name.startswith("scipy"):
        return 2
    if name.startswith("obj."):
        return 3
    return 4
top = ranked[:40]
setup(6.0, 7.2)
fig, ax = plt.subplots()
labs = [n.replace("obj.", "x.") for n, _ in top][::-1]
vals = [c for _, c in top][::-1]
cols = [SERIES[family(n)] for n, _ in top][::-1]
ax.barh(range(len(top)), vals, color=cols)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(labs, fontsize=7.5, family="monospace")
ax.set_xlabel("number of calls in the whole book")
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=SERIES[i], label=l) for i, l in enumerate(
    ["NumPy", "Matplotlib", "SciPy", "method of an object (array, list, generator)", "built into Python"])],
    loc="lower right", fontsize=8)
savefig(fig, "chT0", "01_census_topcalls")

# --- numbers for the text --------------------------------------------------------------------------
def pct(k):
    return round(100 * feat_files[k] / n_files)
np_calls = sum(c for n, c in ranked if n.startswith("np."))
plt_like = sum(c for n, c in ranked if family(n) == 1)
sp_calls = sum(c for n, c in ranked if n.startswith("scipy"))
common_calls = sum(own_calls[k] for k in ("setup", "savefig", "save_numbers", "rng_for", "theory_line"))
save_numbers("chT0", "01_census", {
    "TzFiles": n_files, "TzLines": n_lines,
    "TzCalls": int(counts.sum()), "TzNames": len(ranked),
    "TzNfifty": N50, "TzNeighty": N80, "TzNninety": N90, "TzNninetynine": N99,
    "TzRareNames": int(rare.sum()), "TzRareShare": round(100 * counts[rare].sum() / counts.sum(), 1),
    "TzOwnCalls": int(sum(own_calls.values())), "TzCommonCalls": int(common_calls),
    "TzNumpyShare": round(100 * np_calls / counts.sum()),
    "TzPlotShare": round(100 * plt_like / counts.sum()),
    "TzScipyShare": round(100 * sp_calls / counts.sum()),
    "TzPctFor": pct("for loop"), "TzPctDef": pct("def (own function)"),
    "TzPctIf": pct("if / else"), "TzPctFstring": pct("f-string  f\"{x}\""),
    "TzPctKw": pct("keyword argument  f(x, n=3)"), "TzPctSlice": pct("slicing  x[a:b]"),
    "TzPctComp": pct("comprehension  [f(x) for x in]"), "TzPctDict": pct("dict  {key: value}"),
    "TzPctUnpack": pct("tuple unpacking  a, b = ..."), "TzPctWhile": pct("while loop"),
    "TzPctLambda": pct("lambda"), "TzPctClass": pct("class"), "TzPctDecorator": pct("decorator  @..."),
    "TzPctWith": pct("with ... :"), "TzPctTry": pct("try / except"), "TzPctMatmul": pct("matrix product  A @ B"),
    "TzPctYield": pct("yield (generator)"), "TzPctGlobal": pct("global"),
    "TzPctNumpy": round(100 * mod_files["numpy"] / n_files),
    "TzPctPlt": round(100 * mod_files["matplotlib.pyplot"] / n_files),
    "TzPctScipy": round(100 * mod_files["scipy"] / n_files),
    "TzPctHealpy": round(100 * mod_files["healpy"] / n_files),
    "TzPctCamb": round(100 * mod_files["camb"] / n_files),
    "TzPctNumba": round(100 * mod_files["numba"] / n_files),
    "TzPctMultiproc": round(100 * (mod_files["multiprocessing"] + mod_files["concurrent"]) / n_files),
    "TzPctTime": round(100 * mod_files["time"] / n_files),
})

# --- checks and seeds: how often the book tests itself, and where its random numbers come from ---
RANDOM_METHODS = {"standard_normal", "normal", "uniform", "random", "integers", "poisson", "exponential",
                  "choice", "permutation", "multivariate_normal", "binomial", "gamma", "beta",
                  "chisquare", "standard_cauchy", "shuffle", "rvs"}
draws = [f for f in files if any(n.split(".")[-1] in RANDOM_METHODS for n in file_calls[f])]
seeded = [f for f in draws if "rng_for" in file_own[f]]
libs = [f for f in draws if f not in seeded and f.stem.startswith("lib_")]   # take an rng argument
per_file = np.array([len(file_calls[f]) for f in files])
check_calls = sum(calls[n] for n in ("np.isclose", "np.allclose", "np.testing.assert_allclose",
                                     "np.array_equal", "math.isclose"))
check_files = sum(1 for f in files if file_calls[f] & {"np.isclose", "np.allclose",
                                                      "np.testing.assert_allclose", "np.array_equal"})
legacy_seed = sum(calls[n] for n in ("np.random.seed", "np.random.normal", "np.random.rand",
                                     "np.random.randn", "np.random.uniform"))
save_numbers("chT0", "01_census_checks", {
    "TzAsserts": feat_uses["assert"], "TzAssertFiles": feat_files["assert"],
    "TzCheckCalls": int(check_calls), "TzCheckFiles": int(check_files),
    "TzRngForCalls": own_calls["rng_for"], "TzRngForFiles": sum(1 for f in files if "rng_for" in file_own[f]),
    "TzDrawFiles": len(draws), "TzSeededDrawFiles": len(seeded), "TzLibDrawFiles": len(libs),
    "TzBareRng": bare_rng[0],
    "TzDefaultRng": calls["np.random.default_rng"], "TzLegacySeed": int(legacy_seed),
    "TzMedianNames": int(np.median(per_file)), "TzMaxNames": int(per_file.max()),
    "TzInv": calls["np.linalg.inv"], "TzSolve": calls["np.linalg.solve"],
    "TzChol": calls["np.linalg.cholesky"], "TzEigh": calls["np.linalg.eigh"],
    "TzFft": sum(c for n, c in ranked if n.startswith("np.fft.")),
    "TzPerfCounter": calls["time.perf_counter"], "TzTimeTime": calls["time.time"],
})
words = ["One", "Two", "Three", "Four", "Five"]
top5 = {}
for w, (n, c) in zip(words, ranked[:5]):
    top5[f"TzTopName{w}"] = r"\texttt{" + n.replace("obj.", "x.").replace("_", r"\_") + "}"
    top5[f"TzTopCount{w}"] = c
top5["TzAppendCalls"] = calls["obj.append"]
top5["TzTopFiveSum"] = int(sum(c for _, c in ranked[:5]))
top5["TzTopFiveShare"] = round(100 * sum(c for _, c in ranked[:5]) / counts.sum(), 1)
save_numbers("chT0", "01_census_top", top5)
print("random-drawing files:", len(draws), "of which seeded with rng_for:", len(seeded))
print("unseeded drawing files:", [str(f.relative_to(NOTES)) for f in draws if f not in seeded][:30])

# --- what the script printed, kept so that the text can show it -------------------------------------
(NOTES / "results" / "chT0" / "01_census_out.txt").write_text("\n".join(summary) + "\n")

# --- the full tables, for reading -----------------------------------------------------------------
out = DATA / "chT0"
out.mkdir(parents=True, exist_ok=True)
(out / "census.json").write_text(json.dumps({
    "files": n_files, "lines": n_lines,
    "features_files": dict(feat_files.most_common()), "features_uses": dict(feat_uses.most_common()),
    "modules_files": dict(mod_files.most_common()),
    "calls": ranked, "own_calls": own_calls.most_common(60), "first_file": first_file,
    "coverage": {"50": N50, "80": N80, "90": N90, "99": N99},
}, indent=1))
print("modules:", mod_files.most_common(25))
print("features:", feat_files.most_common())
print("top 80 calls:", ranked[:80])
print("own calls:", own_calls.most_common(20))
